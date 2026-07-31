"""Matrix-free, preconditioned solve of one l channel of the Fock operator.

Exchange is non-local, so the Fock matrix is dense and the tridiagonal
eigensolver that powers radial_solver.py cannot be used. Nor can a dense one:
the grid runs to N ~ 1e4 to 5e4 and a dense symmetric matrix at N = 2e4 is
3.2 GB before any factorization. Matrix-free iteration is mandatory here, not a
preference.

LOBPCG needs a preconditioner. The top of the finite-difference kinetic
spectrum is 2/h^2, which is 8e4 hartree at h = 0.005 and 2e6 at h = 0.001,
against valence level spacings of order 1 hartree; unpreconditioned iteration
stagnates. The preconditioner is free: the LOCAL part of the Fock operator is
still tridiagonal, so shifting it below the lowest sought eigenvalue makes it
positive definite and its inverse is a banded Cholesky solve in O(N). It works
because the local part carries all the high-frequency content while exchange is
a smooth integral kernel with a fast-decaying spectrum.

The mesh owns the discretization. Everything here takes a `RadialMesh` and asks
it for the local Hamiltonian and its quadrature, rather than assuming a
constant step - that is what lets the same SCF run on the uniform grid this
module was written against and on the exponential mesh the heavier atoms need.
The eigenproblem is solved in the mesh's S variable, not in P, because the
Euclidean dot product on S is the physical integral P^2 dr and LOBPCG knows
only the Euclidean one. Exchange is an integral operator in P, so the matvec
carries its argument to P and its result back; the transformation is a
diagonal scaling, so the operator stays symmetric.

Two-electron integrals stay in P on the raw radii: they are trapezoid sums
over r, which are correct on any increasing grid, and keeping them there means
the two total-energy routes are quadratured identically and their agreement
still tests the angular coefficients rather than the mesh.

Returns plain arrays, not Quantity or Field: these are intermediate eigenpairs
of one channel, and hf_atom.py attaches provenance when it reports an atom.

`exchange=False` turns Hartree-Fock into Hartree - the counterfactual model in
which electrons repel each other but are not indistinguishable. Exchange enters
this module in exactly three places and the flag has to reach all three, because
they are not three copies of one calculation:

    _fock_parts        the non-local operator the channel solve diagonalizes,
    orbital_energy     the <P| K |P> expectation that makes eps_a,
    _interaction_energy  the F_k (k>0) and G_k terms of the energy functional.

What does NOT move with it is the (q_a - 1) factor in `direct_potential`. That
is an electron declining to repel itself, which is classical electrostatics and
true in either model; folding it into the exchange bucket would put a
self-interaction error inside a number reported as the exchange energy.

Half-applying the flag is caught for free: hf_atom.py's two total-energy routes
run through the functional and through the operator respectively, so a mismatch
raises HFConvergenceError on every atom rather than returning a plausible wrong
energy.

Hartree atomic units. P = r R(r), normalized as integral P^2 dr = 1.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded, eigh_tridiagonal
from scipy.sparse.linalg import LinearOperator, lobpcg

from atomsim.analytic.wigner import wigner_3j
from atomsim.numerics.hf_terms import (
    ExchangeOperator,
    Subshell,
    direct_potential,
    exchange_apply,
    exchange_operator,
)
from atomsim.numerics.mesh import RadialMesh
from atomsim.numerics.slater import slater_f, slater_g

__all__ = [
    "ChannelSolution",
    "HFConvergenceError",
    "SCFSolution",
    "fock_operator",
    "kinetic_and_potential",
    "local_expectation",
    "local_hamiltonian_bands",
    "one_electron_integral",
    "orbital_energy",
    "scf",
    "solve_channel",
    "total_energy_direct",
    "total_energy_from_orbitals",
]

# Exchange turned off, as an operator rather than as an absence. Frozen and
# empty, so it is shared by every channel of every Hartree solve and applies as
# a zero without a special case anywhere downstream.
_NO_EXCHANGE = ExchangeOperator(terms=())


@dataclass(frozen=True)
class ChannelSolution:
    """Eigenpairs for one l channel, plus what it cost to get them."""

    energies: np.ndarray  # shape (n_states,)
    orbitals: np.ndarray  # shape (n_states, len(r))
    iterations: int  # length of LOBPCG's residual history, so the
    # preconditioner claim in the module docstring
    # is falsifiable. See the note in solve_channel:
    # on a stagnation fallback this is the iteration
    # scipy reverted to, not the work it performed.
    residual: float  # largest achieved residual norm. Reported rather
    # than asserted away, because the attainable value
    # is set by the grid and by whether exchange is on.


class HFConvergenceError(RuntimeError):
    """The SCF loop or an inner eigensolve failed to converge.

    Raised rather than returning a result with converged=False: a plausible
    unconverged number is exactly the quiet lie the prime directive forbids.
    """


def local_hamiltonian_bands(
    v_local: np.ndarray, l: int, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal and off-diagonal of the tridiagonal local Hamiltonian.

    Same 3-point discretization and same conventions as radial_solver.py, so
    the two engines agree where they overlap. Requires a uniform grid with
    r[0] == h; see the module docstring for why that is checked and not assumed.
    """
    if r.ndim != 1 or r.size < 3:
        raise ValueError(f"radial grid must be 1-D with at least 3 points, got {r.shape}")
    h = float(r[1] - r[0])
    if h <= 0:
        raise ValueError(f"radial grid must be increasing, got h={h!r}")
    if not np.allclose(np.diff(r), h, rtol=1e-9, atol=0.0):
        raise ValueError(
            "radial grid must be uniform; the 3-point stencil in this module "
            "assumes a constant step"
        )
    if abs(float(r[0]) - h) > 1e-9 * h:
        raise ValueError(
            f"radial grid must satisfy r[0] == h, got r[0]={float(r[0])!r} with "
            f"h={h!r}. The Dirichlet condition u = 0 is imposed one step below "
            f"r[0], so any other start silently moves the origin: r[0]=1e-5 on "
            f"an h=2e-3 grid costs 4.6% on hydrogen's ground state. Use "
            f"h * np.arange(1, N + 1), as radial_solver.solve_radial does."
        )

    inv2m = 0.5
    v_eff = v_local + l * (l + 1) * inv2m / r**2
    diag = 2.0 * inv2m / h**2 + v_eff
    offdiag = np.full(r.size - 1, -inv2m / h**2)
    return diag, offdiag


def fock_operator(
    subshells: tuple[Subshell, ...],
    a_index: int,
    v_nuclear: Callable[[np.ndarray], np.ndarray],
    l: int,
    mesh: RadialMesh,
    *,
    exchange: bool = True,
) -> LinearOperator:
    """The Fock operator for subshell a as a matrix-free LinearOperator.

    Acts on S, the mesh's working variable. Exchange lives in P, so its
    argument is carried across and its result carried back; both directions are
    the same diagonal scaling, which is what keeps the operator symmetric.

    exchange=False drops the non-local term and leaves the Hartree operator:
    the counterfactual model in which electrons repel but are distinguishable.
    See the module docstring for what must move with it.
    """
    return _fock_parts(
        subshells, a_index, v_nuclear, l, mesh, exchange=exchange
    )[0]


def _fock_parts(
    subshells: tuple[Subshell, ...],
    a_index: int,
    v_nuclear: Callable[[np.ndarray], np.ndarray],
    l: int,
    mesh: RadialMesh,
    *,
    exchange: bool = True,
) -> tuple[LinearOperator, np.ndarray, np.ndarray]:
    """The Fock operator together with the local bands it was assembled from.

    solve_channel needs both - the operator to diagonalize and the bands for
    its preconditioner and cold start - and assembling them separately meant
    building the Hartree potential twice per channel, which is a sweep over
    every occupied subshell each time.
    """
    r = mesh.r
    v_local = np.asarray(v_nuclear(r), dtype=float) + direct_potential(
        subshells, a_index, r
    )
    diag, offdiag = mesh.hamiltonian_bands(v_local, l)
    scale = np.sqrt(mesh.step * mesh.jacobian)
    # Built once, applied by every LOBPCG iteration. See hf_terms.
    # The empty operator rather than a branch inside matvec: matvec runs
    # hundreds of times per channel and the question is settled before any of
    # them. An ExchangeOperator with no terms applies as an exact zero.
    exchange_op = (
        exchange_operator(subshells, a_index, r) if exchange else _NO_EXCHANGE
    )

    def matvec(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float).ravel()
        out = diag * s
        out[:-1] += offdiag * s[1:]
        out[1:] += offdiag * s[:-1]
        return out - scale * exchange_op.apply(s / scale)

    n = mesh.points
    op = LinearOperator((n, n), matvec=matvec, rmatvec=matvec, dtype=float)
    return op, diag, offdiag


def _preconditioner(
    diag: np.ndarray, offdiag: np.ndarray, lowest: float
) -> LinearOperator:
    """(H_local - sigma I)^-1 by banded Cholesky, sigma below the spectrum.

    sigma is placed one hartree below the lowest local eigenvalue so the shifted
    matrix is positive definite and the factorization needs no pivoting. The
    factorization is computed once and reused for every LOBPCG iteration.
    """
    sigma = lowest - 1.0
    ab = np.zeros((2, diag.size))
    ab[0, 1:] = offdiag
    ab[1, :] = diag - sigma
    factor = cholesky_banded(ab, lower=False)

    def apply(x: np.ndarray) -> np.ndarray:
        return cho_solve_banded((factor, False), np.asarray(x, dtype=float).ravel())

    return LinearOperator((diag.size, diag.size), matvec=apply, dtype=float)


def local_expectation(
    p: np.ndarray, v_local: np.ndarray, l: int, mesh: RadialMesh
) -> float:
    """<P| -1/2 d2/dr2 + l(l+1)/2r^2 + V |P>, from the mesh's own operator.

    Deliberately the SAME matrix solve_channel diagonalizes, wall correction
    included, rather than a finite difference assembled separately. The
    identity E = 1/2 sum q (I + eps) only holds when the one-electron integral
    and the eigenvalue come from one operator; assembling them twice makes the
    two energy routes disagree by the difference between two discretizations
    instead of by a coefficient error, which is the one thing that check is
    for.
    """
    diag, offdiag = mesh.hamiltonian_bands(v_local, l)
    s = mesh.to_s(p)
    return float(s @ (diag * s) + 2.0 * float(offdiag @ (s[:-1] * s[1:])))


def solve_channel(
    subshells: tuple[Subshell, ...],
    a_index: int,
    v_nuclear: Callable[[np.ndarray], np.ndarray],
    l: int,
    mesh: RadialMesh,
    n_states: int,
    guess: np.ndarray | None = None,
    tol: float = 1e-6,
    residual_ceiling: float = 1e-3,
    maxiter: int = 150,
    *,
    exchange: bool = True,
) -> ChannelSolution:
    """Lowest n_states eigenpairs of the Fock operator in this l channel.

    Orbitals come back shaped (n_states, len(r)), normalized to
    integral P^2 dr = 1, sign-fixed, and explicitly re-orthogonalized. The
    re-orthogonalization is not redundant: the pair
    potentials are quadratures, so the discrete operator is symmetric only to
    O(h^2) and LOBPCG's own orthogonality inherits that error.

    Why tol is 1e-6 and not something that looks more impressive: that same
    O(h^2) asymmetry floors the attainable residual. On a 20000-point grid a
    channel with exchange active stagnates around 6e-6 no matter what is asked
    of it. Requesting 1e-9 there does not improve the eigenvalue by a single
    digit in twelve - it just burns the full maxiter and then silently falls
    back, costing 8 seconds instead of 0.6. A channel with no exchange term is
    exactly symmetric and still reaches 3e-10, so nothing is given up where
    accuracy is actually available.

    Convergence is therefore gated on the achieved residual against
    residual_ceiling, NOT on the tolerance request and not on the iteration
    count. scipy reports a history whose length, on a stagnation fallback, is
    the iteration it reverted to rather than the number it ran, so a check on
    len(history) would pass happily through a solve that never converged.

    The ceiling is relative to the energy scale of the channel, not absolute.
    A residual norm ||F x - lambda x|| carries the units and the magnitude of
    F, so a fixed absolute ceiling silently demands more and more relative
    accuracy as Z grows: helium's 1s channel reaches 3e-10 while beryllium's
    2s channel, four times deeper, floors at 1.2e-4 on the same grid for the
    same reason. What actually matters is the error in the eigenvalue, which
    for a symmetric operator goes as residual^2 / gap - at 1.2e-4 with a gap of
    order 1 hartree that is 1e-8 hartree, six orders below the benchmark
    tolerance. Scaling the ceiling by |lambda| keeps the gate meaningful at
    every Z while still catching genuine failure.

    The ceiling's value is set from the observed separation, not chosen to make
    a test pass, and it is a failure detector rather than an accuracy claim.
    Healthy solves measured here span 3e-10 (helium, no exchange) to 1.3e-3
    (beryllium's 2s channel on a coarse 6000-point grid), the attainable floor
    growing as O(h^2) with the grid because that is the order of the exchange
    quadrature's asymmetry. Genuine failures - a diverged block, a collapsed
    subspace - measured 3e-2, 9e-2, 1.7e1 and 2.9e4. The default sits in the
    gap between those two populations. What establishes accuracy is not this
    gate but the grid-convergence and vendored-energy benchmarks.
    """
    r = mesh.r
    op, diag, offdiag = _fock_parts(
        subshells, a_index, v_nuclear, l, mesh, exchange=exchange
    )
    lowest = float(
        eigh_tridiagonal(
            diag, offdiag, select="i", select_range=(0, 0), eigvals_only=True
        )[0]
    )
    precond = _preconditioner(diag, offdiag, lowest)

    # Scale the request by the channel's energy magnitude, for the same reason
    # the ceiling is scaled. An absolute 1e-6 is unreachable in a channel whose
    # eigenvalues are several hartree deep, and LOBPCG answers an unreachable
    # request by running the full maxiter and then falling back - which cost
    # beryllium 248 seconds per SCF solve for no gain in any digit.
    energy_scale = max(1.0, abs(lowest))
    scaled_tol = tol * energy_scale

    # No guard vectors. Padding the block above n_states looks like cheap
    # insurance against LOBPCG being least accurate at the top of its block,
    # and measurement says it is the opposite: the extra vectors land in the
    # near-degenerate diffuse states just below zero, which never converge, so
    # scipy - whose stopping rule is over the WHOLE block - burns its entire
    # iteration budget on vectors that are then thrown away. Argon's 3s channel
    # with two guards took 302 iterations and 2.67s and reported a residual of
    # 1.3e-4; with none it took 87 iterations and 0.34s and reported 7.2e-7,
    # for an eigenvalue identical to nine digits. Every channel measured moved
    # the same way.
    block = min(n_states, r.size)

    def cold_block() -> np.ndarray:
        return eigh_tridiagonal(
            diag, offdiag, select="i", select_range=(0, block - 1)
        )[1]

    def warm_block() -> np.ndarray:
        # The guess arrives in P, the physical radial function, because that is
        # what callers hold; the solve happens in S.
        x = mesh.to_s(np.asarray(guess, dtype=float).reshape(-1, r.size)).T
        if x.shape[1] >= block:
            return x[:, :block]

        # Start from the full local spectrum, then let each guess vector
        # displace the local state it most resembles.
        #
        # The rule this replaced padded with local states from index
        # len(guess) upward, which assumed a partial guess covers the LOWEST
        # states. In the SCF it covers the highest: the 3s channel is solved
        # for three states and warm-started with the 3s alone. That put a near
        # duplicate of the guess in the block AND left no approximation to the
        # 1s at all, so LOBPCG had to discover the deepest, most contracted
        # state from nothing. Magnesium's 3s channel was 70% of its whole
        # solve at 2977 iterations.
        #
        # Matching by overlap needs no assumption about which states a guess
        # covers, so it is right for a guess of any size or position.
        local = eigh_tridiagonal(
            diag, offdiag, select="i", select_range=(0, block - 1)
        )[1]
        taken: set[int] = set()
        for column in range(x.shape[1]):
            overlaps = np.abs(local.T @ x[:, column])
            for candidate in np.argsort(overlaps)[::-1]:
                if int(candidate) not in taken:
                    taken.add(int(candidate))
                    local[:, int(candidate)] = x[:, column]
                    break
        return local

    def attempt(x: np.ndarray, budget: int):
        # QR because LOBPCG needs a well-conditioned starting block.
        w, v, history = lobpcg(
            op, np.linalg.qr(x)[0], M=precond, tol=scaled_tol, maxiter=budget,
            largest=False, retResidualNormsHistory=True,
        )
        w = np.atleast_1d(w)
        # Judge only the states this call will return. Anything else in the
        # block is scratch, and letting scratch fail the gate rejects solves
        # that are converged in every eigenvalue the caller receives.
        kept = np.argsort(w)[:n_states]
        res = float(np.max(np.atleast_1d(history[-1])[kept]))
        limit = residual_ceiling * max(1.0, float(np.max(np.abs(w[kept]))))
        return w, v, history, res, limit

    # The warm attempt is speculative and gets the plain budget; the cold
    # attempt is the one we commit to, so it gets double. Measured on
    # beryllium: a stagnating warm start allowed all 400 iterations costs 200
    # seconds, capped at 100 it costs 94, for a total energy identical to every
    # digit. Sharing one tight budget with the retry would instead turn a
    # merely slow channel into a hard failure, which is why they differ.
    starts = (
        [(warm_block, maxiter), (cold_block, 2 * maxiter)]
        if guess is not None
        else [(cold_block, 2 * maxiter)]
    )
    for attempt_index, (build, budget) in enumerate(starts):
        eigenvalues, eigenvectors, history, residual, limit = attempt(
            build(), budget
        )
        if np.isfinite(residual) and residual <= limit:
            break
        # A warm start is an optimization, not a commitment. When the previous
        # SCF step's orbital sends LOBPCG into a subspace it cannot recover
        # from - which happens intermittently rather than as a function of the
        # grid - the local spectrum is a known-good block to fall back to. The
        # retry costs nothing on the overwhelming majority of solves that
        # converge first time.
        if attempt_index == len(starts) - 1:
            raise HFConvergenceError(
                f"LOBPCG did not converge for l={l} from "
                f"{'either a warm or a cold start' if guess is not None else 'a cold start'}"
                f": achieved residual {residual:.3e} exceeds the ceiling "
                f"{limit:.3e} ({residual_ceiling:.3e} scaled by the channel's "
                f"energy magnitude); requested tolerance {scaled_tol:.3e}, "
                f"budget {budget} iterations. Returning the eigenvalue anyway "
                f"would be a plausible-looking wrong number."
            )

    order = np.argsort(eigenvalues)[:n_states]
    out = mesh.to_p(eigenvectors.T[order])

    # Modified Gram-Schmidt in the mesh's inner product, then sign-fix,
    # matching radial_solver.py's convention.
    ortho = []
    for u in out:
        for v in ortho:
            u = u - v * np.trapezoid(u * v, r)
        u = mesh.normalized(u)
        first = np.argmax(np.abs(u) > 0.01 * np.abs(u).max())
        if u[first] < 0:
            u = -u
        ortho.append(u)

    return ChannelSolution(
        energies=eigenvalues[order],
        orbitals=np.array(ortho),
        iterations=len(history),
        residual=residual,
    )


@dataclass(frozen=True)
class SCFSolution:
    subshells: tuple[Subshell, ...]
    energies: tuple[float, ...]  # eps_a, aligned with subshells
    iterations: int
    residual_history: tuple[float, ...]


def one_electron_integral(subshell: Subshell, z: float, mesh: RadialMesh) -> float:
    """I(a) = <P_a| -1/2 d2/dr2 + l(l+1)/(2r^2) - Z/r |P_a>."""
    return local_expectation(subshell.p, -z / mesh.r, subshell.l, mesh)


def orbital_energy(
    subshells: tuple[Subshell, ...],
    a_index: int,
    z: int,
    mesh: RadialMesh,
    v_nuclear: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    exchange: bool = True,
) -> float:
    """eps_a as the quadrature expectation <P_a| h + direct - exchange |P_a>.

    This is NOT redundant with the eigenvalue solve_channel returns, and the
    difference between them is worth understanding rather than hiding. The
    one-electron part here comes from the very operator that was diagonalized,
    but the direct and exchange expectations are trapezoid sums over r, while
    the operator applies those same terms through the mesh's own quadrature
    weights. Both are O(delta^2) accurate and they disagree at that order.

    So the gap is discretization, not convergence, and it is worth being
    precise about which: measured on this mesh it falls by 4.00x per halving of
    delta (He 4.28e-6 -> 1.08e-6 -> 2.70e-7; Be 1s 1.34e-5 -> 3.35e-6 ->
    8.36e-7), which is the signature of a quadrature difference and not of an
    eigensolve stopping short. Tightening the LOBPCG tolerance does not close
    it; refining the mesh does.

    That matters because the identity E = 1/2 sum q (I + eps) is exact only
    when eps and I are quadratured the same way. Fed the eigenvalue, the
    identity misses the directly assembled energy by that O(delta^2) gap; fed
    this, it agrees to machine precision (measured 2.3e-13 hartree on argon),
    which is what makes the two energy routes a real test of the angular
    coefficients rather than a test of the discretization.

    Pass v_nuclear when the nuclear potential is not the bare -Z/r Coulomb.
    """
    r = mesh.r
    a = subshells[a_index]
    v_nuc = (-z / r) if v_nuclear is None else np.asarray(v_nuclear(r), dtype=float)
    one = local_expectation(a.p, v_nuc, a.l, mesh)
    direct = float(np.trapezoid(a.p**2 * direct_potential(subshells, a_index, r), r))
    if not exchange:
        return one + direct
    k_term = float(
        np.trapezoid(a.p * exchange_apply(subshells, a_index, a.p, r), r)
    )
    return one + direct - k_term


def _interaction_energy(
    subshells: tuple[Subshell, ...], r: np.ndarray, *, exchange: bool = True
) -> float:
    """The two-electron part of the average-of-configuration functional.

    The exchange terms are the ones carrying a squared 3j symbol - the k > 0
    same-shell F_k and every cross-shell G_k - and exchange=False drops exactly
    those, leaving the k = 0 Hartree repulsion. The (q_a - 1) and q_a q_b
    counting factors are untouched: they say how many pairs there are, which
    does not depend on whether the pairs are indistinguishable.
    """
    total = 0.0
    for i, a in enumerate(subshells):
        total += (a.q * (a.q - 1) / 2.0) * slater_f(a.p, a.p, r, 0)
        if exchange:
            for k in range(2, 2 * a.l + 1, 2):
                tj = wigner_3j(a.l, k, a.l, 0, 0, 0)
                coeff = ((2 * a.l + 1) / (4 * a.l + 1)) * tj * tj
                total -= (a.q * (a.q - 1) / 2.0) * coeff * slater_f(a.p, a.p, r, k)
        for b in subshells[i + 1:]:
            total += a.q * b.q * slater_f(a.p, b.p, r, 0)
            if exchange:
                for k in range(abs(a.l - b.l), a.l + b.l + 1):
                    tj = wigner_3j(a.l, k, b.l, 0, 0, 0)
                    total -= 0.5 * a.q * b.q * tj * tj * slater_g(a.p, b.p, r, k)
    return float(total)


def total_energy_direct(
    z: int, subshells: tuple[Subshell, ...], mesh: RadialMesh, *,
    exchange: bool = True,
) -> float:
    """Route 1: assemble the energy functional term by term."""
    one = sum(a.q * one_electron_integral(a, z, mesh) for a in subshells)
    return float(one + _interaction_energy(subshells, mesh.r, exchange=exchange))


def total_energy_from_orbitals(
    subshells: tuple[Subshell, ...],
    energies: tuple[float, ...],
    z: int,
    mesh: RadialMesh,
) -> float:
    """Route 2: E = 1/2 sum_a q_a ( I(a) + eps_a ).

    Algebraically identical to route 1 but shares no code with it beyond the
    one-electron integral, so a coefficient error in _interaction_energy shows
    up as a disagreement rather than as a wrong number in both.
    """
    return float(
        0.5
        * sum(
            a.q * (one_electron_integral(a, z, mesh) + e)
            for a, e in zip(subshells, energies, strict=True)
        )
    )


def kinetic_and_potential(
    z: int, subshells: tuple[Subshell, ...], mesh: RadialMesh, *,
    exchange: bool = True,
) -> tuple[float, float]:
    """Route 3's inputs: total T and total V, for the virial ratio -V/T = 2.

    The nuclear term is taken as the difference between the full one-electron
    integral and the Z = 0 one rather than integrated separately, so that
    T + V reproduces total_energy_direct exactly. A virial ratio computed from
    a T and a V that do not add back up to the energy would be a diagnostic
    reporting on a calculation nobody ran.
    """
    zero = np.zeros_like(mesh.r)
    kinetic = 0.0
    nuclear = 0.0
    for a in subshells:
        free = local_expectation(a.p, zero, a.l, mesh)
        kinetic += a.q * free
        nuclear += a.q * (one_electron_integral(a, z, mesh) - free)
    return float(kinetic), float(
        nuclear + _interaction_energy(subshells, mesh.r, exchange=exchange)
    )


def scf(
    z: int,
    subshells: tuple[Subshell, ...],
    v_nuclear: Callable[[np.ndarray], np.ndarray],
    mesh: RadialMesh,
    alpha: float = 0.65,
    max_iterations: int = 200,
    tol: float = 1e-8,
    *,
    exchange: bool = True,
) -> SCFSolution:
    """Self-consistent field loop with damped linear mixing.

    Undamped iteration oscillates on atoms with a diffuse valence shell, so the
    new orbitals are mixed into the old at alpha rather than replacing them.
    alpha is a convergence knob, not a claim about the physics: it changes the
    path to the fixed point, never which fixed point, and the loop still exits
    only when the orbital energies stop moving.

    0.65 by measurement, chosen on WORST case rather than total. SCF iterations
    from a central-field start, on the coarse mesh, H through Ar:

        alpha   0.50  0.55  0.60  0.65  0.70  0.75
        worst     27    21    16    15    17    19

    Undamping further does not keep helping, and it fails in the direction that
    matters: neon takes 26 iterations at 0.8, 65 at 0.9, and does not converge
    at all at 1.0. So the cliff is real and 0.65 sits with margin below it,
    which is worth more than the slightly better total 0.70 posts. The previous
    0.4 cost 34 to 38 iterations on the same atoms, and since the coarse solve
    is most of the wall time, that was most of the wall time.

    Raises HFConvergenceError rather than returning an unconverged solution.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"mixing parameter must be in (0, 1], got {alpha}")

    current = tuple(subshells)
    energies = tuple(0.0 for _ in current)
    residuals: list[float] = []

    for iteration in range(1, max_iterations + 1):
        updated: list[Subshell] = []
        new_energies: list[float] = []
        for index, a in enumerate(current):
            k = a.n - a.l - 1
            channel = solve_channel(
                current, index, v_nuclear, a.l, mesh, n_states=k + 1,
                guess=a.p[None, :], exchange=exchange,
            )
            mixed = (1.0 - alpha) * a.p + alpha * channel.orbitals[k]
            mixed = mesh.normalized(mixed)
            updated.append(Subshell(n=a.n, l=a.l, q=a.q, p=mixed))
            new_energies.append(float(channel.energies[k]))

        residual = max(
            abs(new - old) for new, old in zip(new_energies, energies, strict=True)
        )
        residuals.append(residual)
        current, energies = tuple(updated), tuple(new_energies)
        if residual < tol and iteration > 1:
            return SCFSolution(current, energies, iteration, tuple(residuals))

    raise HFConvergenceError(
        f"SCF did not converge in {max_iterations} iterations for Z={z}; "
        f"last orbital-energy change {residuals[-1]:.3e} hartree"
    )
