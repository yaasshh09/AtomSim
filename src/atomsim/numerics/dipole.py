"""Dipole matrix elements over numerically solved radial functions.

The analytic engine (`analytic/transitions.py`) integrates closed-form
hydrogenic R_nl. This does the same job for **any central potential** by reusing
the radial solver, which is what lets screened atoms, and later counterfactual
force laws, carry line strengths.

The solver returns `u = r R(r)` normalized so `integral u^2 dr = 1`, so the
dipole integral is a plain overlap with no division by r and no reconstruction
of R:

    R_dipole = integral R_b(r) r R_a(r) r^2 dr = integral u_a(r) u_b(r) r dr

**Both states must be solved on one grid.** Asking the solver twice with
per-state box sizes returns two different radial meshes, and multiplying those
sample-by-sample is meaningless. Everything here solves both l channels at the
same `r_max` and `n_points`. See
docs/superpowers/specs/2026-07-25-phase16-screened-line-strengths-design.md.
"""

from collections.abc import Callable

import numpy as np

from atomsim.numerics.radial_solver import solve_radial
from atomsim.provenance import Fidelity, Provenance, Quantity

__all__ = ["dipole_matrix_element", "dipole_box_radius"]


def dipole_box_radius(n_top: int, z_net: float = 1.0) -> float:
    """Box that comfortably holds the more extended of the two states.

    Mirrors `screened_atom._r_max`: orbital extent goes as n^2 / Z_net.
    """
    return 40.0 * (n_top + 1) ** 2 / z_net


def _overlap(
    potential: Callable[[np.ndarray], np.ndarray],
    l_a: int, k_a: int, l_b: int, k_b: int,
    r_max: float, n_points: int, mu_ratio: float,
) -> float:
    """integral u_a u_b r dr, with both states on the same grid."""
    # Same l means one eigenproblem serves both states, but it has to be solved
    # deep enough to contain the higher node count of the two.
    same_l = l_b == l_a
    states_a = max(k_a, k_b) + 1 if same_l else k_a + 1
    sol_a = solve_radial(
        potential, l=l_a, mu_ratio=mu_ratio, r_max=r_max,
        n_points=n_points, n_states=states_a,
    )
    sol_b = sol_a if same_l else solve_radial(
        potential, l=l_b, mu_ratio=mu_ratio, r_max=r_max,
        n_points=n_points, n_states=k_b + 1,
    )
    r = sol_a.r
    return float(np.trapezoid(sol_a.u[k_a] * sol_b.u[k_b] * r, r))


def dipole_matrix_element(
    potential: Callable[[np.ndarray], np.ndarray],
    l_a: int, k_a: int, l_b: int, k_b: int,
    n_top: int,
    z_net: float = 1.0,
    n_points: int = 8000,
    mu_ratio: float = 1.0,
) -> Quantity:
    """Radial dipole matrix element <b|r|a> in bohr, for a central potential.

    States are named by (l, k) with k the radial node count, so k = n - l - 1
    for a hydrogen-like labelling. `n_top` sizes the box: pass the larger n of
    the pair. The error estimate comes from grid-halving, the same convention
    the radial solver itself uses.

    The sign is the solver's: `solve_radial` fixes each u to start positive, so
    the element is reproducible but its overall sign carries no physics. Only
    |R|^2 enters a rate.
    """
    if k_a < 0 or k_b < 0:
        raise ValueError(f"node indices must be >= 0, got k_a={k_a}, k_b={k_b}")
    if l_a < 0 or l_b < 0:
        raise ValueError(f"l must be >= 0, got l_a={l_a}, l_b={l_b}")
    r_max = dipole_box_radius(n_top, z_net)
    coarse = _overlap(potential, l_a, k_a, l_b, k_b, r_max, n_points, mu_ratio)
    fine = _overlap(potential, l_a, k_a, l_b, k_b, r_max, 2 * n_points, mu_ratio)
    return Quantity(
        value=fine,
        unit="bohr",
        label=f"<l={l_b},k={k_b}|r|l={l_a},k={k_a}>",
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method=(
                "overlap integral of u = rR from the finite-difference radial "
                "solver: <b|r|a> = integral u_a u_b r dr (both states on one grid)"
            ),
            assumptions=(
                f"shared uniform grid: r_max={r_max:g} bohr, N={2 * n_points}",
                "trapezoid quadrature on the solver grid",
                "only box-converged bound states are meaningful",
            ),
            error_estimate=abs(fine - coarse),
            refinement="increase n_points or r_max; estimate from grid-halving",
        ),
    )
