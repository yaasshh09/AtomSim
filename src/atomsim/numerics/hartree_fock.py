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

Grid convention, and it is load-bearing: the 3-point stencil imposes u = 0 one
step below r[0], so the grid must be uniform with r[0] == h for that step to
land on the origin. This matches radial_solver.solve_radial exactly. A grid
that merely starts "close to zero" is not equivalent - r[0] = 1e-5 with
h = 2e-3 puts the wall at r = -0.002 and costs 4.6% on hydrogen's ground state,
with orbitals that still look perfectly smooth. local_hamiltonian_bands
therefore refuses such a grid instead of quietly absorbing the error.

Returns plain arrays, not Quantity or Field: these are intermediate eigenpairs
of one channel, and hf_atom.py attaches provenance when it reports an atom.

Hartree atomic units. P = r R(r), normalized as integral P^2 dr = 1.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded, eigh_tridiagonal
from scipy.sparse.linalg import LinearOperator, lobpcg

from atomsim.analytic.wigner import wigner_3j
from atomsim.numerics.hf_terms import Subshell, direct_potential, exchange_apply
from atomsim.numerics.slater import slater_f, slater_g

__all__ = [
    "ChannelSolution",
    "HFConvergenceError",
    "SCFSolution",
    "fock_operator",
    "kinetic_and_potential",
    "local_hamiltonian_bands",
    "one_electron_integral",
    "orbital_energy",
    "scf",
    "solve_channel",
    "total_energy_direct",
    "total_energy_from_orbitals",
]


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
    r: np.ndarray,
) -> LinearOperator:
    """The Fock operator for subshell a as a matrix-free LinearOperator."""
    v_local = np.asarray(v_nuclear(r), dtype=float) + direct_potential(
        subshells, a_index, r
    )
    diag, offdiag = local_hamiltonian_bands(v_local, l, r)

    def matvec(psi: np.ndarray) -> np.ndarray:
        psi = np.asarray(psi, dtype=float).ravel()
        out = diag * psi
        out[:-1] += offdiag * psi[1:]
        out[1:] += offdiag * psi[:-1]
        return out - exchange_apply(subshells, a_index, psi, r)

    return LinearOperator((r.size, r.size), matvec=matvec, rmatvec=matvec,
                          dtype=float)


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


def _normalize(u: np.ndarray, r: np.ndarray) -> np.ndarray:
    return u / np.sqrt(np.trapezoid(u**2, r))


def solve_channel(
    subshells: tuple[Subshell, ...],
    a_index: int,
    v_nuclear: Callable[[np.ndarray], np.ndarray],
    l: int,
    r: np.ndarray,
    n_states: int,
    guess: np.ndarray | None = None,
    tol: float = 1e-6,
    residual_ceiling: float = 1e-3,
    maxiter: int = 150,
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
    op = fock_operator(subshells, a_index, v_nuclear, l, r)
    v_local = np.asarray(v_nuclear(r), dtype=float) + direct_potential(
        subshells, a_index, r
    )
    diag, offdiag = local_hamiltonian_bands(v_local, l, r)
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

    block = min(n_states + 2, r.size)  # guard vectors: LOBPCG is least
    # accurate at the top of its block
    def cold_block() -> np.ndarray:
        return eigh_tridiagonal(
            diag, offdiag, select="i", select_range=(0, block - 1)
        )[1]

    def warm_block() -> np.ndarray:
        x = np.asarray(guess, dtype=float).reshape(-1, r.size).T
        if x.shape[1] < block:
            # Pad from the local spectrum, skipping the states the guess
            # already covers. Padding from index 0 instead hands LOBPCG a near
            # duplicate of the guess, which measured slower on helium and
            # stopped beryllium converging at all.
            extra = eigh_tridiagonal(
                diag, offdiag, select="i", select_range=(x.shape[1], block - 1)
            )[1]
            x = np.hstack([x, extra])
        return x[:, :block]

    def attempt(x: np.ndarray, budget: int):
        # QR because LOBPCG needs a well-conditioned starting block.
        w, v, history = lobpcg(
            op, np.linalg.qr(x)[0], M=precond, tol=scaled_tol, maxiter=budget,
            largest=False, retResidualNormsHistory=True,
        )
        res = float(np.max(history[-1]))
        limit = residual_ceiling * max(1.0, float(np.max(np.abs(w))))
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
    out = eigenvectors.T[order]

    # Modified Gram-Schmidt in the trapezoid inner product, then sign-fix,
    # matching radial_solver.py's convention.
    ortho = []
    for u in out:
        for v in ortho:
            u = u - v * np.trapezoid(u * v, r)
        u = _normalize(u, r)
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


def one_electron_integral(subshell: Subshell, z: int, r: np.ndarray) -> float:
    """I(a) = <P_a| -1/2 d2/dr2 + l(l+1)/(2r^2) - Z/r |P_a>."""
    p = subshell.p
    h = float(r[1] - r[0])
    second = np.zeros_like(p)
    second[1:-1] = (p[2:] - 2.0 * p[1:-1] + p[:-2]) / h**2
    kinetic = -0.5 * np.trapezoid(p * second, r)
    centrifugal = 0.5 * subshell.l * (subshell.l + 1) * np.trapezoid(
        (p / r) ** 2, r
    )
    nuclear = -z * np.trapezoid(p**2 / r, r)
    return float(kinetic + centrifugal + nuclear)


def orbital_energy(
    subshells: tuple[Subshell, ...],
    a_index: int,
    z: int,
    r: np.ndarray,
    v_nuclear: Callable[[np.ndarray], np.ndarray] | None = None,
) -> float:
    """eps_a as the quadrature expectation <P_a| h + direct - exchange |P_a>.

    This is NOT redundant with the eigenvalue solve_channel returns, and the
    difference between them is worth understanding rather than hiding. The
    eigenvalue comes from the finite-difference operator; this comes from the
    same trapezoid quadrature that one_electron_integral and the Slater
    integrals use. They agree only to O(h^2) - about 2e-5 hartree for helium on
    a 30000-point grid.

    That matters because the identity E = 1/2 sum q (I + eps) is exact only
    when eps and I are quadratured the same way. Fed the eigenvalue, the
    identity misses the directly assembled energy by exactly that O(h^2) gap;
    fed this, it agrees to machine precision, which is what makes the two
    energy routes a real test of the angular coefficients rather than a test of
    the discretization.

    Pass v_nuclear when the nuclear potential is not the bare -Z/r Coulomb.
    """
    a = subshells[a_index]
    v_nuc = (-z / r) if v_nuclear is None else np.asarray(v_nuclear(r), dtype=float)
    one = one_electron_integral(a, 0, r) + float(np.trapezoid(a.p**2 * v_nuc, r))
    direct = float(np.trapezoid(a.p**2 * direct_potential(subshells, a_index, r), r))
    exchange = float(
        np.trapezoid(a.p * exchange_apply(subshells, a_index, a.p, r), r)
    )
    return one + direct - exchange


def _interaction_energy(subshells: tuple[Subshell, ...], r: np.ndarray) -> float:
    """The two-electron part of the average-of-configuration functional."""
    total = 0.0
    for i, a in enumerate(subshells):
        total += (a.q * (a.q - 1) / 2.0) * slater_f(a.p, a.p, r, 0)
        for k in range(2, 2 * a.l + 1, 2):
            tj = wigner_3j(a.l, k, a.l, 0, 0, 0)
            coeff = ((2 * a.l + 1) / (4 * a.l + 1)) * tj * tj
            total -= (a.q * (a.q - 1) / 2.0) * coeff * slater_f(a.p, a.p, r, k)
        for b in subshells[i + 1:]:
            total += a.q * b.q * slater_f(a.p, b.p, r, 0)
            for k in range(abs(a.l - b.l), a.l + b.l + 1):
                tj = wigner_3j(a.l, k, b.l, 0, 0, 0)
                total -= 0.5 * a.q * b.q * tj * tj * slater_g(a.p, b.p, r, k)
    return float(total)


def total_energy_direct(
    z: int, subshells: tuple[Subshell, ...], r: np.ndarray
) -> float:
    """Route 1: assemble the energy functional term by term."""
    one = sum(a.q * one_electron_integral(a, z, r) for a in subshells)
    return float(one + _interaction_energy(subshells, r))


def total_energy_from_orbitals(
    subshells: tuple[Subshell, ...],
    energies: tuple[float, ...],
    z: int,
    r: np.ndarray,
) -> float:
    """Route 2: E = 1/2 sum_a q_a ( I(a) + eps_a ).

    Algebraically identical to route 1 but shares no code with it beyond the
    one-electron integral, so a coefficient error in _interaction_energy shows
    up as a disagreement rather than as a wrong number in both.
    """
    return float(
        0.5
        * sum(
            a.q * (one_electron_integral(a, z, r) + e)
            for a, e in zip(subshells, energies, strict=True)
        )
    )


def kinetic_and_potential(
    z: int, subshells: tuple[Subshell, ...], r: np.ndarray
) -> tuple[float, float]:
    """Route 3's inputs: total T and total V, for the virial ratio -V/T = 2."""
    h = float(r[1] - r[0])
    kinetic = 0.0
    for a in subshells:
        second = np.zeros_like(a.p)
        second[1:-1] = (a.p[2:] - 2.0 * a.p[1:-1] + a.p[:-2]) / h**2
        kinetic += a.q * (
            -0.5 * np.trapezoid(a.p * second, r)
            + 0.5 * a.l * (a.l + 1) * np.trapezoid((a.p / r) ** 2, r)
        )
    nuclear = sum(-z * a.q * np.trapezoid(a.p**2 / r, r) for a in subshells)
    return float(kinetic), float(nuclear + _interaction_energy(subshells, r))


def scf(
    z: int,
    subshells: tuple[Subshell, ...],
    v_nuclear: Callable[[np.ndarray], np.ndarray],
    r: np.ndarray,
    alpha: float = 0.4,
    max_iterations: int = 200,
    tol: float = 1e-8,
) -> SCFSolution:
    """Self-consistent field loop with damped linear mixing.

    Undamped iteration oscillates on atoms with a diffuse valence shell, so the
    new orbitals are mixed into the old at alpha rather than replacing them.
    alpha = 0.4 is a starting value tuned against measured iteration counts, not
    a claim about the physics.

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
                current, index, v_nuclear, a.l, r, n_states=k + 1,
                guess=a.p[None, :],
            )
            mixed = (1.0 - alpha) * a.p + alpha * channel.orbitals[k]
            mixed = _normalize(mixed, r)
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
