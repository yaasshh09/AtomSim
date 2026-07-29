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

import dataclasses
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh_tridiagonal

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
from atomsim.numerics.mesh import RadialMesh, mesh_for_atom_at_step
from atomsim.numerics.screening import gsz_parameters, screened_potential
from atomsim.provenance import Fidelity, Field, Provenance, Quantity

__all__ = [
    "HFOrbital",
    "HFResult",
    "hf_mesh",
    "solve_hartree_fock",
]

# The two energy routes are algebraically identical, so a disagreement above
# this is a coefficient bug, not a discretization error. See Task 6.
_ROUTE_AGREEMENT = 1e-6

_TOTAL_ENERGY_METHOD = (
    "self-consistent restricted Hartree-Fock, average of configuration; "
    "matrix-free preconditioned LOBPCG on an exponential radial mesh"
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

# The mesh's own optimum; see numerics/mesh.py for the measurement behind it.
# The matching inner radius deliberately is NOT duplicated here - mesh.py owns
# it, and mesh_for_atom_at_step is what keeps the point count consistent with
# it.
_MESH_STEP = 0.01

# The part of the error a refinement pair is structurally blind to.
#
# Both meshes in the pair share r_min, so halving the step cancels out of the
# inner-wall truncation and the eigensolver conditioning noise entirely. Those
# do not vanish as delta -> 0; they are the floor numerics/mesh.py derives and
# pins, and past it the spread keeps shrinking while the answer stops moving.
# Quoting the spread alone would therefore claim an accuracy that tightens
# without bound while the real error sits still - the exact shape of a number
# that lies about itself.
#
# Measured here on the total energy rather than inherited from the mesh's
# single-eigenvalue figure, because a total energy sums several orbital
# contributions and their floors accumulate. Extrapolating the delta^2 term
# away from a refinement pair leaves, relative to |E|: He 3.9e-6, Be 3.8e-6,
# Ne 3.0e-6, Mg 2.9e-6, Ar 2.5e-6 - flat in Z and about 1.6x the mesh's own
# 2.4e-6, as summing contributions predicts.
_MESH_FLOOR_RELATIVE = 4.0e-6


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


def hf_mesh(z: int, n_electrons: int, n_top: int, refinement: int = 1) -> RadialMesh:
    """The mesh a Hartree-Fock solve runs on.

    Exponential, because a uniform grid has to resolve the 1s core (which
    contracts as 1/Z) and reach the valence tail at the same time, and pays for
    the finer of the two everywhere. Argon needed 72000 uniform points and
    about an hour; it needs roughly 1400 here.

    The box is set by where the valence actually reaches, n^2 / Z_net, but
    generously: the cost of a larger box is only logarithmic on this mesh, so
    there is no reason to crowd the tail. The step is chosen to land near
    delta = 0.01, which numerics/mesh.py measures as the sweet spot - past it
    the eigensolver's conditioning noise grows faster than the discretization
    error falls, so more points make the answer worse.

    `refinement` halves the step, and exists so the caller can run the same
    physics on two meshes and quote the difference as an error estimate. Both
    meshes share r_min and r_max exactly and differ only in point count, which
    is what makes their difference a clean statement about the step.
    """
    z_net = max(z - n_electrons + 1, 1)
    r_max = min(60.0, 12.0 * (n_top + 1) ** 2 / z_net)
    return mesh_for_atom_at_step(z, r_max, _MESH_STEP / refinement)


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
    z: int, n_electrons: int, config: Configuration, mesh: RadialMesh
) -> tuple[Subshell, ...]:
    """One central-field solve per l channel, reused across its subshells.

    Diagonalized on this mesh rather than through radial_solver.solve_radial,
    which builds a uniform grid of its own: a guess sampled somewhere else and
    interpolated across is exactly the kind of piecewise-linear kink that cost
    hydrogen 2.8% of its energy earlier in this module's history.
    """
    v = np.asarray(_start_potential(z, n_electrons)(mesh.r), dtype=float)
    by_l: dict[int, np.ndarray] = {}
    for (n, l), _ in config:
        needed = n - l  # radial states 0..n-l-1
        if l not in by_l or by_l[l].shape[0] < needed:
            diag, offdiag = mesh.hamiltonian_bands(v, l)
            vectors = eigh_tridiagonal(
                diag, offdiag, select="i", select_range=(0, needed - 1)
            )[1]
            by_l[l] = mesh.to_p(vectors.T)
    return tuple(
        Subshell(n=n, l=l, q=q, p=mesh.normalized(by_l[l][n - l - 1]))
        for (n, l), q in config
    )


def _refine(
    coarse: SCFSolution, coarse_mesh: RadialMesh, fine: RadialMesh
) -> tuple[Subshell, ...]:
    """Interpolate a converged coarse solution onto the finer mesh.

    The coarse solve is needed anyway for the error estimate, so seeding the
    fine solve with it is free: the fine SCF starts a step or two from its own
    fixed point instead of from a central-field guess.

    Two details that are not decoration.

    The endpoints are anchored at P = 0 on both walls. On this mesh that is no
    longer load-bearing the way it was on uniform grids, where the two grids
    had different first points and an interpolant that clamps rather than
    extrapolating flattened every P across the innermost interval, putting a
    kink exactly where -Z/r and the kinetic term are largest, at a cost of 2.8%
    of hydrogen's total energy. Both meshes here share r_min and r_max exactly,
    so nothing is ever evaluated outside the coarse span. The anchors stay
    because P(0) = 0 is true and giving the spline that knot shapes it
    correctly approaching r_min, which is where the amplitude is changing
    fastest.

    The interpolant is cubic rather than linear because the SCF then runs the
    result through a second difference. A piecewise-linear P has zero curvature
    between knots and all of it concentrated at them, and the kinetic operator
    amplifies that by 1/h^2, so an O(h_coarse^2) amplitude error becomes an
    O(h_coarse^2 / h_fine^2) energy error, which is not small at all. It also
    made LOBPCG stagnate on every fine step, so this is a speed fix as much as
    an accuracy one.
    """
    knots = np.concatenate(([0.0], coarse_mesh.r, [coarse_mesh.outer_wall]))
    return tuple(
        Subshell(
            n=a.n, l=a.l, q=a.q,
            p=fine.normalized(
                CubicSpline(knots, np.concatenate(([0.0], a.p, [0.0])))(fine.r)
            ),
        )
        for a in coarse.subshells
    )


def _solve_on_grid(
    z: int,
    n_electrons: int,
    config: Configuration,
    mesh: RadialMesh,
    start: tuple[Subshell, ...] | None,
) -> tuple[SCFSolution, tuple[float, ...], float]:
    """Run the SCF on one mesh; return the solution, the quadrature orbital
    energies and the directly assembled total energy."""
    if start is None:
        start = _guess_from_central_field(z, n_electrons, config, mesh)

    # The SCF residual is a change in orbital energies, and the deepest of
    # those scales as Z^2/2, so a fixed absolute tolerance silently demands
    # more significant figures as Z grows. Scale it to keep the demand fixed at
    # roughly ten digits on the 1s level.
    solution = scf(
        z, start, lambda rr: -z / rr, mesh, tol=1e-9 * max(1, z**2)
    )
    energies = tuple(
        orbital_energy(solution.subshells, i, z, mesh)
        for i in range(len(solution.subshells))
    )
    return solution, energies, total_energy_direct(z, solution.subshells, mesh)


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
    coarse_mesh = hf_mesh(z, n_electrons, n_top, refinement=1)
    mesh = hf_mesh(z, n_electrons, n_top, refinement=2)
    coarse, coarse_energies, e_coarse = _solve_on_grid(
        z, n_electrons, config, coarse_mesh, start=None
    )
    solution, energies, e_direct = _solve_on_grid(
        z, n_electrons, config, mesh,
        start=_refine(coarse, coarse_mesh, mesh),
    )

    # Route 2 shares no code with route 1 beyond the one-electron integral, so
    # a disagreement is an angular-coefficient bug rather than a coarse grid.
    e_identity = total_energy_from_orbitals(solution.subshells, energies, z, mesh)
    if abs(e_direct - e_identity) > _ROUTE_AGREEMENT:
        raise HFConvergenceError(
            f"the two total-energy routes disagree by "
            f"{abs(e_direct - e_identity):.3e} hartree for Z={z}, N={n_electrons}; "
            f"that is a coding error, not a discretization one"
        )

    kinetic, potential = kinetic_and_potential(z, solution.subshells, mesh)

    energy_prov = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=_TOTAL_ENERGY_METHOD,
        assumptions=_TOTAL_ENERGY_ASSUMPTIONS,
        # Two independent error sources, added rather than maxed because they
        # are independent: the spread between the two meshes, which measures
        # the step discretization, plus the mesh floor the spread cannot see
        # (see _MESH_FLOOR_RELATIVE). Deliberately NOT Richardson-extrapolated
        # away - the residual past the delta^2 term is conditioning noise, not
        # a smooth power of delta, and extrapolating noise sharpens nothing.
        #
        # Checked against the vendored energies, this brackets the true
        # deviation for every atom solved here, by 1.6x (Be) to 2.2x (Ar).
        error_estimate=(
            abs(e_direct - e_coarse) + _MESH_FLOOR_RELATIVE * abs(e_direct)
        ),
        refinement=_TOTAL_ENERGY_REFINEMENT,
    )
    diagnostic_prov = Provenance(
        fidelity=Fidelity.NUMERICAL,
        method=_DIAGNOSTIC_METHOD,
        assumptions=(f"converged in {solution.iterations} SCF iterations",),
    )
    # The orbital amplitude gets its own provenance carrying NO error estimate,
    # rather than borrowing the energy's. Provenance.error_estimate is
    # documented as being in the unit of the quantity it describes, and P is in
    # bohr^-1/2: an error bar in hartree attached to it would not be a loose
    # error bar, it would be a number in the wrong dimension. The mesh spread
    # for the orbital SHAPE is not something this solve estimates, and saying
    # nothing is the honest form of not knowing.
    shape_prov = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=f"{_TOTAL_ENERGY_METHOD}; radial amplitude sampled on the solver mesh",
        assumptions=_TOTAL_ENERGY_ASSUMPTIONS,
        refinement=_TOTAL_ENERGY_REFINEMENT,
    )

    orbitals = tuple(
        HFOrbital(
            n=a.n, l=a.l, occupancy=a.q,
            # Each orbital energy gets ITS OWN spread, not the total energy's.
            # eps_3p is about -0.6 hartree for argon while E_total is -527, so
            # handing every orbital the total's error bar would overstate the
            # valence uncertainty by three orders of magnitude and understate
            # nothing usefully. Same two terms as the total: the coarse-to-fine
            # spread plus the mesh floor the spread cannot see.
            energy=Quantity(
                fine, "hartree", f"eps_{a.n}{a.l}",
                dataclasses.replace(
                    energy_prov,
                    error_estimate=(
                        abs(fine - crude) + _MESH_FLOOR_RELATIVE * abs(fine)
                    ),
                ),
            ),
            P=Field(
                values=a.p, grid=mesh.r, unit="bohr^-1/2", grid_unit="bohr",
                label=f"P_{a.n}{a.l}", provenance=shape_prov,
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
        total_energy=Quantity(e_direct, "hartree", "E_total", energy_prov),
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
