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
docs/specs/2026-07-25-phase16-screened-line-strengths-design.md.
"""

from collections.abc import Callable

import numpy as np

from atomsim.numerics.radial_solver import RadialSolution, solve_radial
from atomsim.provenance import Fidelity, Provenance, Quantity

__all__ = [
    "dipole_matrix_element",
    "dipole_box_radius",
    "dipole_from_solutions",
    "grid_points_for",
]

#: Target grid spacing in bohr. The dipole overlap converges in h, not in r_max
#: (past the point where the box holds both states), and the finite-difference
#: scheme is O(h^2): at h = 0.01 a screened valence element is good to ~0.1%,
#: which is well under the GSZ model error it will be combined with.
_H_TARGET = 0.01


def dipole_box_radius(n_top: int, z_net: float = 1.0) -> float:
    """Box that comfortably holds the more extended of the two states.

    Mirrors `screened_atom._r_max`: orbital extent goes as n^2 / Z_net.

    The coefficient is 10, not the 40 this started at, because the box now sets
    the cost: `grid_points_for` holds h fixed, so points scale with r_max. At 10
    the hydrogenic <6p|r|5s> and <4p|r|1s> integrals, and the screened Na
    3s->3p, agree with the 40 box to six significant digits; the value only
    starts to move at a coefficient of 2.5, so this keeps a 4x margin in box
    size and costs a quarter as much.
    """
    return 10.0 * (n_top + 1) ** 2 / z_net


def grid_points_for(r_max: float, h_target: float = _H_TARGET) -> int:
    """Point count that keeps the spacing at or below `h_target`.

    Sizing the grid by point count alone is a trap: a generous box with a fixed
    N silently coarsens h. Before this existed, a 640-bohr box at N = 8000 gave
    h = 0.08 and a 6.7% error on the Na 3s->3p element, with nothing in the
    returned number to say so.
    """
    return int(np.ceil(r_max / h_target))


def dipole_from_solutions(
    sol_a: RadialSolution, k_a: int, sol_b: RadialSolution, k_b: int
) -> float:
    """integral u_a u_b r dr for two states already solved on the same grid.

    Separated out so a caller with many lines can solve each l channel once and
    reuse it, instead of re-running the eigenproblem per line.
    """
    if sol_a.r.shape != sol_b.r.shape or not np.array_equal(sol_a.r, sol_b.r):
        raise ValueError(
            "the two states must be solved on one grid; got "
            f"{sol_a.r.size} and {sol_b.r.size} points"
        )
    r = sol_a.r
    return float(np.trapezoid(sol_a.u[k_a] * sol_b.u[k_b] * r, r))


def _overlap(
    potential: Callable[[np.ndarray], np.ndarray],
    l_a: int, k_a: int, l_b: int, k_b: int,
    r_max: float, n_points: int, mu_ratio: float,
) -> float:
    """integral u_a u_b r dr, solving both states on one grid."""
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
    return dipole_from_solutions(sol_a, k_a, sol_b, k_b)


def dipole_matrix_element(
    potential: Callable[[np.ndarray], np.ndarray],
    l_a: int, k_a: int, l_b: int, k_b: int,
    n_top: int,
    z_net: float = 1.0,
    n_points: int | None = None,
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
    if n_points is None:
        n_points = grid_points_for(r_max)
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
