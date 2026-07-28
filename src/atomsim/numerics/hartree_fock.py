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

from atomsim.numerics.hf_terms import Subshell, direct_potential, exchange_apply

__all__ = [
    "ChannelSolution",
    "HFConvergenceError",
    "fock_operator",
    "local_hamiltonian_bands",
    "solve_channel",
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


def _preconditioner(diag: np.ndarray, offdiag: np.ndarray) -> LinearOperator:
    """(H_local - sigma I)^-1 by banded Cholesky, sigma below the spectrum.

    sigma is placed one hartree below the lowest local eigenvalue so the shifted
    matrix is positive definite and the factorization needs no pivoting. The
    factorization is computed once and reused for every LOBPCG iteration.
    """
    lowest = eigh_tridiagonal(
        diag, offdiag, select="i", select_range=(0, 0), eigvals_only=True
    )[0]
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
    residual_ceiling: float = 1e-4,
    maxiter: int = 400,
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
    The ceiling sits well clear of the quadrature floor and well below a
    genuine failure, which shows up with residuals of order 1e-1 upward.
    """
    op = fock_operator(subshells, a_index, v_nuclear, l, r)
    v_local = np.asarray(v_nuclear(r), dtype=float) + direct_potential(
        subshells, a_index, r
    )
    diag, offdiag = local_hamiltonian_bands(v_local, l, r)
    precond = _preconditioner(diag, offdiag)

    block = min(n_states + 2, r.size)  # guard vectors: LOBPCG is least
    # accurate at the top of its block
    if guess is not None:
        x = np.asarray(guess, dtype=float).reshape(-1, r.size).T
        if x.shape[1] < block:
            # Top up from the local spectrum, skipping the states the guess
            # already covers: padding with the lowest local eigenvectors would
            # duplicate the guess and hand LOBPCG a rank-deficient block.
            extra = eigh_tridiagonal(
                diag, offdiag, select="i", select_range=(x.shape[1], block - 1)
            )[1]
            x = np.hstack([x, extra])
        x = x[:, :block]
    else:
        x = eigh_tridiagonal(diag, offdiag, select="i",
                             select_range=(0, block - 1))[1]

    x = np.linalg.qr(x)[0]  # LOBPCG needs a well-conditioned starting block

    eigenvalues, eigenvectors, history = lobpcg(
        op, x, M=precond, tol=tol, maxiter=maxiter, largest=False,
        retResidualNormsHistory=True,
    )
    residual = float(np.max(history[-1]))
    if not np.isfinite(residual) or residual > residual_ceiling:
        raise HFConvergenceError(
            f"LOBPCG did not converge for l={l}: achieved residual "
            f"{residual:.3e} exceeds the ceiling {residual_ceiling:.3e} "
            f"(requested tolerance {tol:.3e}, maxiter {maxiter}). Returning "
            f"the eigenvalue anyway would be a plausible-looking wrong number."
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
