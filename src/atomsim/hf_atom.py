"""Solve a many-electron atom by restricted Hartree-Fock (APPROXIMATION).

The counterpart of `screened_atom.py`, one model up. GSZ hands every electron
the same fitted central field; here each subshell gets its own Fock operator,
built from the other orbitals and solved to self-consistency, so the total
energy is variational and can be compared against vendored Hartree-Fock
references rather than only against spectra.

What is still missing is correlation: Hartree-Fock is a single-determinant
ansatz, so the energy sits ABOVE the exact non-relativistic energy by the
correlation energy (~0.04 hartree for helium, ~0.7 for argon). That gap is the
model error, and it is the reason `total_energy` is APPROXIMATION with the
grid error carried as a numerical sub-scale rather than as the headline number.

See docs/superpowers/specs/2026-07-27-phase21-hartree-fock-design.md.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline

from atomsim.atoms import Configuration, is_ground, total_electrons, validate_config
from atomsim.numerics.hartree_fock import (
    HFConvergenceError,
    SCFSolution,
    kinetic_and_potential,
    orbital_energy,
    scf,
    total_energy_direct,
    total_energy_from_orbitals,
)
from atomsim.numerics.hf_terms import Subshell
from atomsim.numerics.radial_solver import solve_radial
from atomsim.numerics.screening import gsz_parameters, screened_potential
from atomsim.provenance import Fidelity, Field, Provenance, Quantity

__all__ = [
    "HFOrbital",
    "HFResult",
    "hf_grid",
    "solve_hartree_fock",
]

# The two energy routes are algebraically identical, so a disagreement above
# this is a coefficient bug, not a discretization error. See Task 6.
_ROUTE_AGREEMENT = 1e-6

_TOTAL_ENERGY_METHOD = (
    "self-consistent restricted Hartree-Fock, average of configuration; "
    "matrix-free preconditioned LOBPCG on a uniform radial grid; "
    "Richardson-extrapolated in h^2 from a grid-halving pair"
)
_TOTAL_ENERGY_ASSUMPTIONS = (
    "no electron correlation; variational, so E_HF >= E_exact "
    "(non-relativistic, infinite nuclear mass)",
    "average of configuration: one energy per configuration, not per term",
    "infinite nuclear mass (mu_ratio = 1)",
)
_TOTAL_ENERGY_REFINEMENT = (
    "configuration interaction or many-body perturbation theory would "
    "recover the correlation energy"
)
_DIAGNOSTIC_METHOD = "property of the converged solution, not a claim about the atom"


@dataclass(frozen=True)
class HFOrbital:
    n: int
    l: int
    occupancy: int
    energy: Quantity  # APPROXIMATION, hartree
    P: Field  # r R_nl(r), on the solver grid


@dataclass(frozen=True)
class HFResult:
    key: str
    z: int
    n_electrons: int
    config: Configuration
    is_ground: bool
    orbitals: tuple[HFOrbital, ...]
    total_energy: Quantity  # APPROXIMATION
    kinetic: Quantity  # NUMERICAL
    potential: Quantity  # NUMERICAL
    virial_ratio: Quantity  # NUMERICAL, target 2
    iterations: int
    residual_history: tuple[float, ...]
    converged: bool
    provenance: Provenance


def hf_grid(z: int, n_electrons: int, n_top: int) -> tuple[float, int]:
    """Box radius and point count for a Hartree-Fock solve.

    Two competing scales. The 1s core contracts as 1/Z, so the step must scale
    as 1/Z to hold the relative discretization error fixed: the error in the
    core eigenvalue goes as (h Z)^2. The valence extends as n^2 / Z_net, which
    sets the box.

    This is deliberately much tighter than screened_atom._r_max, whose
    40 (n_top+1)^2 box would spend most of a Hartree-Fock grid on vacuum. It is
    also the reason this phase stops at Z = 18: holding h Z fixed on a UNIFORM
    grid becomes unaffordable well before Z = 54, and a logarithmic mesh is the
    real fix (Phase 22).
    """
    z_net = max(z - n_electrons + 1, 1)
    r_max = min(40.0, 8.0 * (n_top + 1) ** 2 / z_net)
    h = 0.01 / z
    return r_max, int(r_max / h)


def _grid(r_max: float, points: int) -> np.ndarray:
    """The solver grid, r[0] == h. Matches radial_solver.solve_radial exactly,
    so a GSZ warm start lands on the same points without interpolation."""
    h = r_max / (points + 1)
    return h * np.arange(1, points + 1)


def _normalized(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    return p / np.sqrt(np.trapezoid(p**2, r))


def _richardson(fine: float, coarse: float) -> float:
    """Cancel the leading h^2 error term from a grid-halving pair.

    The discretization converges as a clean h^2 - measured ratios of 3.78,
    3.89, 3.95, 3.97, 3.99 against the exact hydrogen energy as the grid
    halves - so (4 E_h - E_2h) / 3 removes that term and leaves O(h^4). On
    hydrogen it turns a 1.8e-4 hartree error into 6.6e-6 at exactly the cost
    already being paid for the error estimate.

    Note the direction: finite differences are NOT variational here, and the
    energy approaches its limit from below. Extrapolating is therefore what
    keeps E_HF above the exact energy, not what threatens it.
    """
    return (4.0 * fine - coarse) / 3.0


def _start_potential(z: int, n_electrons: int):
    """The central field the first guess is drawn from.

    GSZ where Szydlik and Green fitted it, and the bare nucleus otherwise. The
    bare-nucleus fallback is the choice that lets sulfur and chlorine run at
    all: their GSZ parameters were never published, and a hydrogenic guess at
    Z_eff = Z - N + 1 = 1 would start every core orbital an order of magnitude
    too diffuse. Starting from -Z/r errs the other way, too contracted, which
    the SCF screens outward; it also invents no parameter.
    """
    if n_electrons == 1:
        return screened_potential(z, n_electrons)  # exactly -Z/r, no screening term
    try:
        gsz_parameters(z, n_electrons)
    except ValueError:
        return lambda rr: -float(z) / np.asarray(rr, dtype=float)
    return screened_potential(z, n_electrons)


def _guess_from_central_field(
    z: int, n_electrons: int, config: Configuration, r_max: float, points: int
) -> tuple[Subshell, ...]:
    """One central-field solve per l channel, reused across its subshells."""
    potential = _start_potential(z, n_electrons)
    by_l: dict[int, np.ndarray] = {}
    for (n, l), _ in config:
        needed = n - l  # radial states 0..n-l-1
        if l not in by_l or by_l[l].shape[0] < needed:
            by_l[l] = solve_radial(
                potential, l=l, mu_ratio=1.0, r_max=r_max,
                n_points=points, n_states=needed,
            ).u
    return tuple(
        Subshell(n=n, l=l, q=q, p=by_l[l][n - l - 1].copy())
        for (n, l), q in config
    )


def _refine(
    coarse: SCFSolution, coarse_r: np.ndarray, fine_r: np.ndarray, r_max: float
) -> tuple[Subshell, ...]:
    """Interpolate a converged coarse solution onto the fine grid.

    The coarse solve is needed anyway for the grid-halving error estimate, so
    seeding the fine solve with it is free: the fine SCF starts a step or two
    from its own fixed point instead of from a central-field guess.

    Two details that are not decoration.

    The endpoints are anchored at P(0) = P(r_max) = 0. The fine grid begins
    inside the coarse grid's first point and ends outside its last, and a
    linear interpolant clamps rather than extrapolating: without the inner
    anchor every P is flattened across the innermost interval, putting a kink
    exactly where -Z/r and the kinetic term are largest. That alone cost
    hydrogen 2.8% of its total energy.

    The interpolant is cubic rather than linear because the SCF then runs the
    result through a second difference. A piecewise-linear P has zero curvature
    between coarse nodes and all of it concentrated at them, and the kinetic
    operator amplifies that by 1/h^2 - an O(h_coarse^2) amplitude error becomes
    an O(h_coarse^2 / h_fine^2) energy error, which is not small at all. It also
    made LOBPCG stagnate on every fine-grid step, so this is a speed fix as
    much as an accuracy one.
    """
    knots = np.concatenate(([0.0], coarse_r, [r_max]))
    return tuple(
        Subshell(
            n=a.n, l=a.l, q=a.q,
            p=_normalized(
                CubicSpline(knots, np.concatenate(([0.0], a.p, [0.0])))(fine_r),
                fine_r,
            ),
        )
        for a in coarse.subshells
    )


def _solve_on_grid(
    z: int,
    n_electrons: int,
    config: Configuration,
    r_max: float,
    points: int,
    start: tuple[Subshell, ...] | None,
) -> tuple[np.ndarray, SCFSolution, tuple[float, ...], float]:
    """Run the SCF on one grid; return the grid, the solution, the quadrature
    orbital energies and the directly assembled total energy."""
    r = _grid(r_max, points)
    if start is None:
        start = _guess_from_central_field(z, n_electrons, config, r_max, points)

    # The SCF residual is a change in orbital energies, and the deepest of
    # those scales as Z^2/2, so a fixed absolute tolerance silently demands
    # more significant figures as Z grows. Scale it to keep the demand fixed at
    # roughly ten digits on the 1s level.
    solution = scf(
        z, start, lambda rr: -z / rr, r, tol=1e-9 * max(1, z**2)
    )
    energies = tuple(
        orbital_energy(solution.subshells, i, z, r)
        for i in range(len(solution.subshells))
    )
    return r, solution, energies, total_energy_direct(z, solution.subshells, r)


@lru_cache(maxsize=8)
def solve_hartree_fock(
    z: int, n_electrons: int, config: Configuration
) -> HFResult:
    """Converge the restricted Hartree-Fock equations for one atom or ion.

    Raises HFConvergenceError rather than returning an unconverged result: a
    HFResult with converged=False would be a quiet lie in object form.
    """
    if z < 1:
        raise ValueError(f"Z must be >= 1, got {z}")
    if not 1 <= n_electrons <= z + 1:
        raise ValueError(f"N must be in [1, Z+1], got {n_electrons} (Z={z})")
    validate_config(config)
    if total_electrons(config) != n_electrons:
        raise ValueError(
            f"configuration holds {total_electrons(config)} electrons, "
            f"not the {n_electrons} requested"
        )

    n_top = max(n for (n, _), _ in config)
    r_max, points = hf_grid(z, n_electrons, n_top)

    coarse_r, coarse, coarse_energies, e_coarse = _solve_on_grid(
        z, n_electrons, config, r_max, points // 2, start=None
    )
    r, solution, energies, e_direct = _solve_on_grid(
        z, n_electrons, config, r_max, points,
        start=_refine(coarse, coarse_r, _grid(r_max, points), r_max),
    )

    # Route 2 shares no code with route 1 beyond the one-electron integral, so
    # a disagreement is an angular-coefficient bug rather than a coarse grid.
    e_identity = total_energy_from_orbitals(solution.subshells, energies, z, r)
    if abs(e_direct - e_identity) > _ROUTE_AGREEMENT:
        raise HFConvergenceError(
            f"the two total-energy routes disagree by "
            f"{abs(e_direct - e_identity):.3e} hartree for Z={z}, N={n_electrons}; "
            f"that is a coding error, not a discretization one"
        )

    kinetic, potential = kinetic_and_potential(z, solution.subshells, r)
    e_total = _richardson(e_direct, e_coarse)

    energy_prov = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=_TOTAL_ENERGY_METHOD,
        assumptions=_TOTAL_ENERGY_ASSUMPTIONS,
        # The size of the correction Richardson just applied. The residual
        # error is O(h^4) and measures ~30x smaller than this, so quoting the
        # correction itself is the conservative choice.
        error_estimate=abs(e_total - e_direct),
        refinement=_TOTAL_ENERGY_REFINEMENT,
    )
    diagnostic_prov = Provenance(
        fidelity=Fidelity.NUMERICAL,
        method=_DIAGNOSTIC_METHOD,
        assumptions=(f"converged in {solution.iterations} SCF iterations",),
    )

    orbitals = tuple(
        HFOrbital(
            n=a.n, l=a.l, occupancy=a.q,
            energy=Quantity(
                _richardson(fine, crude), "hartree", f"eps_{a.n}{a.l}", energy_prov
            ),
            P=Field(
                values=a.p, grid=r, unit="bohr^-1/2", grid_unit="bohr",
                label=f"P_{a.n}{a.l}", provenance=energy_prov,
            ),
        )
        for a, fine, crude in zip(
            solution.subshells, energies, coarse_energies, strict=True
        )
    )

    return HFResult(
        key=f"z{z}n{n_electrons}",
        z=z,
        n_electrons=n_electrons,
        config=config,
        is_ground=is_ground(config),
        orbitals=orbitals,
        total_energy=Quantity(e_total, "hartree", "E_total", energy_prov),
        kinetic=Quantity(kinetic, "hartree", "T", diagnostic_prov),
        potential=Quantity(potential, "hartree", "V", diagnostic_prov),
        virial_ratio=Quantity(
            -potential / kinetic, "dimensionless", "-V/T", diagnostic_prov
        ),
        iterations=solution.iterations,
        residual_history=solution.residual_history,
        converged=True,
        provenance=energy_prov,
    )
