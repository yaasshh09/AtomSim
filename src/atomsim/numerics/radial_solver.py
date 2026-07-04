"""Numerical radial Schrodinger solver for ARBITRARY central potentials.

Solves  -(1/2mu') u'' + [V(r) + l(l+1)/(2mu' r^2)] u = E u,  u = r R(r),
with u(0) = u(r_max) = 0, via 3-point finite differences on a uniform grid and
scipy.linalg.eigh_tridiagonal. Hartree atomic units throughout.

This one engine powers real Coulomb physics, screened multi-electron models,
and counterfactual force laws alike.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

from atomsim.provenance import Fidelity, Provenance, Quantity


@dataclass(frozen=True)
class RadialSolution:
    r: np.ndarray
    u: np.ndarray  # shape (n_states, len(r)); normalized, sign-fixed
    energies: tuple[Quantity, ...]
    l: int
    mu_ratio: float


def solve_radial(
    potential: Callable[[np.ndarray], np.ndarray],
    l: int = 0,
    mu_ratio: float = 1.0,
    r_max: float = 120.0,
    n_points: int = 24000,
    n_states: int = 3,
) -> RadialSolution:
    h = r_max / (n_points + 1)
    r = h * np.arange(1, n_points + 1)
    inv2m = 1.0 / (2.0 * mu_ratio)

    v_eff = np.asarray(potential(r), dtype=float) + l * (l + 1) * inv2m / r**2
    diag = 2.0 * inv2m / h**2 + v_eff
    offdiag = np.full(n_points - 1, -inv2m / h**2)

    eigvals, eigvecs = eigh_tridiagonal(
        diag, offdiag, select="i", select_range=(0, n_states - 1)
    )
    u = eigvecs.T.copy()

    norms = np.sqrt(np.trapezoid(u**2, r, axis=1))
    u /= norms[:, None]
    for k in range(u.shape[0]):
        first = np.argmax(np.abs(u[k]) > 0.01 * np.abs(u[k]).max())
        if u[k][first] < 0:
            u[k] = -u[k]

    provenance = Provenance(
        fidelity=Fidelity.NUMERICAL,
        method="3-point finite-difference radial Hamiltonian (u = r R), scipy eigh_tridiagonal",
        assumptions=(
            f"uniform grid: h={h:.3e} bohr, r_max={r_max:g} bohr, N={n_points}",
            "Dirichlet boundaries u(0) = u(r_max) = 0",
            "only low-lying bound states are box-converged",
        ),
        refinement=(
            "increase n_points / r_max, or use solve_radial_with_error for a quantified error"
        ),
    )
    energies = tuple(
        Quantity(
            value=float(e),
            unit="hartree",
            label=f"E[{k}] (l={l})",
            provenance=provenance,
        )
        for k, e in enumerate(eigvals)
    )
    return RadialSolution(r=r, u=u, energies=energies, l=l, mu_ratio=mu_ratio)


def solve_radial_with_error(
    potential: Callable[[np.ndarray], np.ndarray],
    l: int = 0,
    mu_ratio: float = 1.0,
    r_max: float = 120.0,
    n_points: int = 24000,
    n_states: int = 3,
) -> RadialSolution:
    """Solve at n_points and 2*n_points; attach |E_fine - E_coarse| as error estimate.

    Grid-halving is a conservative estimate for this O(h^2)-convergent scheme:
    the reported fine-grid error is smaller than the difference itself.
    """
    coarse = solve_radial(potential, l, mu_ratio, r_max, n_points, n_states)
    fine = solve_radial(potential, l, mu_ratio, r_max, 2 * n_points, n_states)

    energies = tuple(
        dataclasses.replace(
            q,
            provenance=dataclasses.replace(
                q.provenance,
                error_estimate=abs(q.value - coarse.energies[k].value),
                refinement="increase n_points further; estimate from grid-halving",
            ),
        )
        for k, q in enumerate(fine.energies)
    )
    return dataclasses.replace(fine, energies=energies)
