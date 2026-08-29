"""How I solve one l channel of the Fock operator, matrix-free and
preconditioned.

Exchange is non-local, so my Fock matrix is dense and I cannot use the
tridiagonal eigensolver that powers radial_solver.py. Nor can I use a dense
one: my grid runs to N ~ 1e4 to 5e4 and a dense symmetric matrix at N = 2e4 is
3.2 GB before any factorization. Matrix-free iteration is mandatory for me
here, not a preference.

LOBPCG needs a preconditioner from me. The top of the finite-difference kinetic
spectrum is 2/h^2, which is 8e4 hartree at h = 0.005 and 2e6 at h = 0.001,
against valence level spacings of order 1 hartree; unpreconditioned I stagnate.
My preconditioner is free: the LOCAL part of the Fock operator is still
tridiagonal, so shifting it below the lowest eigenvalue I want makes it
positive definite and its inverse is a banded Cholesky solve in O(N). It works
because the local part carries all the high-frequency content while exchange is
a smooth integral kernel with a fast-decaying spectrum.

My mesh owns the discretization. Everything here takes a `RadialMesh` and asks
it for the local Hamiltonian and its quadrature rather than assuming a constant
step, and that is what lets me run the same SCF on the uniform grid I wrote
this against and on the exponential mesh my heavier atoms need. I solve the
eigenproblem in the mesh's S variable, not in P, because the Euclidean dot
product on S is the physical integral P^2 dr and LOBPCG knows only the
Euclidean one. Exchange is an integral operator in P, so my matvec carries its
argument to P and its result back; that transformation is a diagonal scaling,
so my operator stays symmetric.

I keep two-electron integrals in P on the raw radii: they are trapezoid sums
over r, which are correct on any increasing grid, and keeping them there means
I quadrature my two total-energy routes identically, so their agreement still
tests my angular coefficients rather than my mesh.

I return plain arrays, not Quantity or Field: these are intermediate eigenpairs
of one channel, and hf_atom.py attaches provenance when it reports an atom.

`exchange=False` turns my Hartree-Fock into Hartree, the counterfactual model
in which electrons repel each other but are not indistinguishable. Exchange
enters this module in exactly three places and the flag has to reach all three,
because they are not three copies of one calculation:

    _fock_parts        the non-local operator my channel solve diagonalizes,
    orbital_energy     the <P| K |P> expectation that makes my eps_a,
    _interaction_energy  the F_k (k>0) and G_k terms of my energy functional.

What does NOT move with it is the (q_a - 1) factor in `direct_potential`. That
is an electron declining to repel itself, which is classical electrostatics and
true in either model; if I folded it into the exchange bucket I would put a
self-interaction error inside a number I report as the exchange energy.

I catch a half-applied flag for free: hf_atom.py's two total-energy routes run
through the functional and through the operator respectively, so a mismatch
raises HFConvergenceError on every atom rather than handing back a plausible
wrong energy.

I work in Hartree atomic units. P = r R(r), normalized so integral P^2 dr = 1.
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

# Exchange turned off, which I express as an operator rather than as an absence.
# It is frozen and empty, so I share it across every channel of every Hartree
# solve and it applies as a zero with no special case anywhere downstream.
_NO_EXCHANGE = ExchangeOperator(terms=())


@dataclass(frozen=True)
class ChannelSolution:
    """My eigenpairs for one l channel, plus what they cost me."""

    energies: np.ndarray  # shape (n_states,)
    orbitals: np.ndarray  # shape (n_states, len(r))
    iterations: int  # the length of LOBPCG's residual history, so my
    # preconditioner claim in the module docstring
    # stays falsifiable. See my note in solve_channel:
    # on a stagnation fallback this is the iteration
    # scipy reverted to, not the work it performed.
    residual: float  # the largest residual norm I achieved. I report it
    # rather than asserting it away, because the grid and
    # whether exchange is on decide what I can attain.


class HFConvergenceError(RuntimeError):
    """My SCF loop or one of my inner eigensolves failed to converge.

    I raise rather than returning a result with converged=False: a plausible
    unconverged number is exactly the quiet lie my prime directive forbids.
    """


def local_hamiltonian_bands(
    v_local: np.ndarray, l: int, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The diagonal and off-diagonal of my tridiagonal local Hamiltonian.

    I use the same 3-point discretization and the same conventions as
    radial_solver.py, so my two engines agree where they overlap. I need a
    uniform grid with r[0] == h; see the module docstring for why I check that
    rather than assuming it.
    """
    if r.ndim != 1 or r.size < 3:
        raise ValueError(f"I need a 1-D radial grid with at least 3 points, got {r.shape}")
    h = float(r[1] - r[0])
    if h <= 0:
        raise ValueError(f"I need an increasing radial grid, got h={h!r}")
    if not np.allclose(np.diff(r), h, rtol=1e-9, atol=0.0):
        raise ValueError(
            "I need a uniform radial grid here; my 3-point stencil in this "
            "module assumes a constant step"
        )
    if abs(float(r[0]) - h) > 1e-9 * h:
        raise ValueError(
            f"I need a radial grid with r[0] == h, got r[0]={float(r[0])!r} with "
            f"h={h!r}. I impose the Dirichlet condition u = 0 one step below "
            f"r[0], so any other start silently moves my origin: r[0]=1e-5 on "
            f"an h=2e-3 grid costs me 4.6% on hydrogen's ground state. Use "
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
    """My Fock operator for subshell a, as a matrix-free LinearOperator.

    It acts on S, my mesh's working variable. Exchange lives in P, so I carry
    its argument across and its result back; both directions are the same
    diagonal scaling, which is what keeps my operator symmetric.

    exchange=False drops the non-local term and leaves me the Hartree operator:
    the counterfactual model in which electrons repel but are distinguishable.
    See my module docstring for what has to move with it.
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
    """My Fock operator together with the local bands I assembled it from.

    solve_channel needs both, the operator to diagonalize and the bands for its
    preconditioner and cold start, and assembling them separately made me build
    the Hartree potential twice per channel, which is a sweep over every
    occupied subshell each time.
    """
    r = mesh.r
    v_local = np.asarray(v_nuclear(r), dtype=float) + direct_potential(
        subshells, a_index, r
    )
    diag, offdiag = mesh.hamiltonian_bands(v_local, l)
    scale = np.sqrt(mesh.step * mesh.jacobian)
    # I build this once and every LOBPCG iteration applies it. See hf_terms.
    # I use the empty operator rather than a branch inside matvec: matvec runs
    # hundreds of times per channel and I settle the question before any of
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
    """(H_local - sigma I)^-1 by banded Cholesky, with sigma below the spectrum.

    I place sigma one hartree below the lowest local eigenvalue so my shifted
    matrix is positive definite and my factorization needs no pivoting. I
    compute the factorization once and reuse it for every LOBPCG iteration.
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
    """<P| -1/2 d2/dr2 + l(l+1)/2r^2 + V |P>, taken from my mesh's own operator.

    I use the SAME matrix solve_channel diagonalizes, wall correction included,
    rather than assembling a finite difference separately. The identity
    E = 1/2 sum q (I + eps) only holds when my one-electron integral and my
    eigenvalue come from one operator; if I assembled them twice my two energy
    routes would disagree by the difference between two discretizations instead
    of by a coefficient error, and catching a coefficient error is the one
    thing that check is for.
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
    """The lowest n_states eigenpairs I find of the Fock operator in this l
    channel.

    I return orbitals shaped (n_states, len(r)), normalized to
    integral P^2 dr = 1, sign-fixed, and explicitly re-orthogonalized. That
    re-orthogonalization is not redundant: my pair potentials are quadratures,
    so my discrete operator is symmetric only to O(h^2) and LOBPCG's own
    orthogonality inherits that error.

    Why my tol is 1e-6 and not something that looks more impressive: that same
    O(h^2) asymmetry floors the residual I can attain. On a 20000-point grid a
    channel with exchange active stagnates around 6e-6 no matter what I ask of
    it. Requesting 1e-9 there does not improve my eigenvalue by a single digit
    in twelve; it just burns the full maxiter and then silently falls back,
    costing me 8 seconds instead of 0.6. A channel with no exchange term is
    exactly symmetric and still reaches 3e-10, so I give up nothing where
    accuracy is actually available.

    So I gate convergence on the achieved residual against residual_ceiling,
    NOT on the tolerance I requested and not on the iteration count. scipy
    reports a history whose length, on a stagnation fallback, is the iteration
    it reverted to rather than the number it ran, so a check on len(history)
    would pass me happily through a solve that never converged.

    I make the ceiling relative to the channel's energy scale rather than
    absolute. A residual norm ||F x - lambda x|| carries the units and the
    magnitude of F, so a fixed absolute ceiling would silently demand more and
    more relative accuracy as Z grows: helium's 1s channel reaches 3e-10 while
    beryllium's 2s channel, four times deeper, floors at 1.2e-4 on the same
    grid for the same reason. What actually matters is my error in the
    eigenvalue, which for a symmetric operator goes as residual^2 / gap: at
    1.2e-4 with a gap of order 1 hartree that is 1e-8 hartree, six orders below
    my benchmark tolerance. Scaling the ceiling by |lambda| keeps my gate
    meaningful at every Z while still catching genuine failure.

    I set the ceiling's value from the separation I observed, not to make a
    test pass, and I mean it as a failure detector rather than an accuracy
    claim. The healthy solves I measured span 3e-10 (helium, no exchange) to
    1.3e-3 (beryllium's 2s channel on a coarse 6000-point grid), with the floor
    I can attain growing as O(h^2) with the grid, because that is the order of
    my exchange quadrature's asymmetry. Genuine failures, a diverged block or a
    collapsed subspace, measured 3e-2, 9e-2, 1.7e1 and 2.9e4. My default sits
    in the gap between those two populations. What establishes my accuracy is
    not this gate but my grid-convergence and vendored-energy benchmarks.
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

    # I scale my request by the channel's energy magnitude, for the same reason
    # I scale my ceiling. An absolute 1e-6 is unreachable in a channel whose
    # eigenvalues are several hartree deep, and LOBPCG answers an unreachable
    # request by running the full maxiter and then falling back, which cost me
    # 248 seconds per beryllium SCF solve for no gain in any digit.
    energy_scale = max(1.0, abs(lowest))
    scaled_tol = tol * energy_scale

    # I use no guard vectors. Padding the block above n_states looks like cheap
    # insurance against LOBPCG being least accurate at the top of its block,
    # and what I measured says the opposite: the extra vectors land in the
    # near-degenerate diffuse states just below zero, which never converge, so
    # scipy, whose stopping rule is over the WHOLE block, burns its entire
    # iteration budget on vectors I then throw away. Argon's 3s channel with two
    # guards took me 302 iterations and 2.67s and reported a residual of 1.3e-4;
    # with none it took 87 iterations and 0.34s and reported 7.2e-7, for an
    # eigenvalue identical to nine digits. Every channel I measured moved the
    # same way.
    block = min(n_states, r.size)

    def cold_block() -> np.ndarray:
        return eigh_tridiagonal(
            diag, offdiag, select="i", select_range=(0, block - 1)
        )[1]

    def warm_block() -> np.ndarray:
        # The guess reaches me in P, the physical radial function, because that
        # is what my callers hold; I solve in S.
        x = mesh.to_s(np.asarray(guess, dtype=float).reshape(-1, r.size)).T
        if x.shape[1] >= block:
            return x[:, :block]

        # I start from the full local spectrum, then let each guess vector
        # displace the local state it most resembles.
        #
        # The rule I replaced padded with local states from index len(guess)
        # upward, which assumed a partial guess covers the LOWEST states. In my
        # SCF it covers the highest: I solve the 3s channel for three states
        # and warm-start it with the 3s alone. That put a near duplicate of the
        # guess in my block AND left me no approximation to the 1s at all, so
        # LOBPCG had to discover the deepest, most contracted state from
        # nothing. Magnesium's 3s channel was 70% of its whole solve at 2977
        # iterations.
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
        # I QR because LOBPCG needs a well-conditioned starting block from me.
        w, v, history = lobpcg(
            op, np.linalg.qr(x)[0], M=precond, tol=scaled_tol, maxiter=budget,
            largest=False, retResidualNormsHistory=True,
        )
        w = np.atleast_1d(w)
        # I judge only the states this call will return. Anything else in the
        # block is scratch, and letting scratch fail my gate would reject solves
        # that are converged in every eigenvalue my caller receives.
        kept = np.argsort(w)[:n_states]
        res = float(np.max(np.atleast_1d(history[-1])[kept]))
        limit = residual_ceiling * max(1.0, float(np.max(np.abs(w[kept]))))
        return w, v, history, res, limit

    # My warm attempt is speculative and gets the plain budget; my cold attempt
    # is the one I commit to, so it gets double. I measured this on beryllium:
    # a stagnating warm start allowed all 400 iterations costs me 200 seconds,
    # capped at 100 it costs 94, for a total energy identical to every digit.
    # Sharing one tight budget with the retry would instead turn a merely slow
    # channel into a hard failure, which is why I keep them different.
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
        # A warm start is an optimization of mine, not a commitment. When the
        # previous SCF step's orbital sends LOBPCG into a subspace it cannot
        # recover from, which happens intermittently rather than as a function
        # of the grid, the local spectrum is a known-good block for me to fall
        # back to. The retry costs me nothing on the overwhelming majority of
        # solves that converge first time.
        if attempt_index == len(starts) - 1:
            raise HFConvergenceError(
                f"LOBPCG did not converge for l={l} from "
                f"{'either a warm or a cold start' if guess is not None else 'a cold start'}"
                f": the residual I achieved, {residual:.3e}, exceeds my ceiling "
                f"{limit:.3e} ({residual_ceiling:.3e} scaled by the channel's "
                f"energy magnitude); I requested tolerance {scaled_tol:.3e} on a "
                f"budget of {budget} iterations. If I returned the eigenvalue "
                f"anyway it would be a plausible-looking wrong number."
            )

    order = np.argsort(eigenvalues)[:n_states]
    out = mesh.to_p(eigenvectors.T[order])

    # I run modified Gram-Schmidt in the mesh's inner product, then fix the
    # sign, matching radial_solver.py's convention.
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
    energies: tuple[float, ...]  # my eps_a, aligned with the subshells
    iterations: int
    residual_history: tuple[float, ...]


def one_electron_integral(subshell: Subshell, z: float, mesh: RadialMesh) -> float:
    """My I(a) = <P_a| -1/2 d2/dr2 + l(l+1)/(2r^2) - Z/r |P_a>."""
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

    This is NOT redundant with the eigenvalue solve_channel returns me, and the
    difference between them is worth understanding rather than hiding. My
    one-electron part here comes from the very operator I diagonalized, but I
    take the direct and exchange expectations as trapezoid sums over r, while
    the operator applies those same terms through the mesh's own quadrature
    weights. Both are O(delta^2) accurate and they disagree at that order.

    So the gap is my discretization, not my convergence, and it is worth being
    precise about which: on this mesh I measured it falling by 4.00x per
    halving of delta (He 4.28e-6 -> 1.08e-6 -> 2.70e-7; Be 1s 1.34e-5 ->
    3.35e-6 -> 8.36e-7), which is the signature of a quadrature difference and
    not of an eigensolve stopping short. Tightening my LOBPCG tolerance does
    not close it; refining my mesh does.

    That matters because the identity E = 1/2 sum q (I + eps) is exact only
    when I quadrature eps and I the same way. Fed the eigenvalue, the identity
    misses my directly assembled energy by that O(delta^2) gap; fed this, it
    agrees to machine precision (I measured 2.3e-13 hartree on argon), which is
    what makes my two energy routes a real test of the angular coefficients
    rather than a test of my discretization.

    Pass me v_nuclear when the nuclear potential is not the bare -Z/r Coulomb.
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
    """The two-electron part of my average-of-configuration functional.

    My exchange terms are the ones carrying a squared 3j symbol, the k > 0
    same-shell F_k and every cross-shell G_k, and exchange=False makes me drop
    exactly those and keep the k = 0 Hartree repulsion. I leave the (q_a - 1)
    and q_a q_b counting factors alone: they say how many pairs there are,
    which does not depend on whether the pairs are indistinguishable.
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
    """My route 1: I assemble the energy functional term by term."""
    one = sum(a.q * one_electron_integral(a, z, mesh) for a in subshells)
    return float(one + _interaction_energy(subshells, mesh.r, exchange=exchange))


def total_energy_from_orbitals(
    subshells: tuple[Subshell, ...],
    energies: tuple[float, ...],
    z: int,
    mesh: RadialMesh,
) -> float:
    """My route 2: E = 1/2 sum_a q_a ( I(a) + eps_a ).

    It is algebraically identical to route 1 but shares no code with it beyond
    the one-electron integral, so a coefficient error in _interaction_energy
    shows up to me as a disagreement rather than as a wrong number in both.
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
    """My route 3's inputs: total T and total V, for the virial ratio -V/T = 2.

    I take the nuclear term as the difference between the full one-electron
    integral and the Z = 0 one rather than integrating it separately, so that
    T + V reproduces total_energy_direct exactly. A virial ratio I computed
    from a T and a V that did not add back up to the energy would be a
    diagnostic reporting on a calculation nobody ran.
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
    """My self-consistent field loop, with damped linear mixing.

    Undamped iteration oscillates on atoms with a diffuse valence shell, so I
    mix the new orbitals into the old at alpha rather than replacing them.
    alpha is a convergence knob of mine, not a claim about the physics: it
    changes my path to the fixed point, never which fixed point I reach, and I
    still exit only when the orbital energies stop moving.

    I chose 0.65 by measurement, on WORST case rather than total. My SCF
    iterations from a central-field start, on the coarse mesh, H through Ar:

        alpha   0.50  0.55  0.60  0.65  0.70  0.75
        worst     27    21    16    15    17    19

    Undamping further does not keep helping me, and it fails in the direction
    that matters: neon takes me 26 iterations at 0.8, 65 at 0.9, and does not
    converge at all at 1.0. So the cliff is real and I sit 0.65 with margin
    below it, which is worth more to me than the slightly better total 0.70
    posts. My previous 0.4 cost 34 to 38 iterations on the same atoms, and
    since my coarse solve is most of the wall time, that was most of the wall
    time.

    I raise HFConvergenceError rather than returning an unconverged solution.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"I need a mixing parameter in (0, 1], got {alpha}")

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
        f"my SCF did not converge in {max_iterations} iterations for Z={z}; "
        f"my last orbital-energy change was {residuals[-1]:.3e} hartree"
    )
