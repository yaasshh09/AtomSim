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

from atomsim.analytic.angular import spherical_harmonic
from atomsim.analytic.dirac import dirac_energy
from atomsim.analytic.hydrogen import energy as hydrogen_energy
from atomsim.analytic.wavefunction import WavefunctionValues
from atomsim.atoms import (
    Configuration,
    aufbau_configuration,
    is_ground,
    is_single_term,
    open_subshells,
    total_electrons,
    validate_config,
)
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
    "evaluate_hf_state",
    "hf_exchange_energy",
    "hf_mesh",
    "hf_radial",
    "hf_valence_ionization_energy",
    "solve_hartree_fock",
]

#: Resampling density for evaluate_hf_state, matching screened_atom.py. The
#: solve itself runs on a few thousand mesh points; this is the grid the
#: interpolation reads, and it is deliberately finer than the 400 a plot needs.
_HF_EVAL_POINTS = 4096

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
    "infinite nuclear mass (mu_ratio = 1)",
)
# Added only when a subshell is partially filled. Restricted Hartree-Fock gives
# both spins the same radial function, which for a closed shell is no
# constraint at all (the two spin populations are identical anyway) and for an
# open shell forbids the core from polarizing around the unpaired electrons.
_OPEN_SHELL_ASSUMPTION = (
    "restricted: one radial function per subshell shared by both spins, so the "
    "core cannot spin-polarize around an unpaired electron; that omission is "
    "far smaller than the missing correlation energy above"
)
# Added only when the configuration spans more than one term.
_MULTI_TERM_ASSUMPTION = (
    "average of configuration: one energy per configuration, not per term, so "
    "this energy lies among the terms the configuration splits into rather "
    "than on the lowest of them"
)

# Everything below is the exchange=False branch: the Hartree model.
_HARTREE_METHOD = (
    "self-consistent Hartree, average of configuration; the same solve with "
    "the exchange term removed from the Fock operator and from the energy "
    "functional"
)
# Leads the assumption list, and says which counterfactual this is. A badge
# reading COUNTERFACTUAL without naming the altered rule is decoration.
_HARTREE_ALTERATION = (
    "COUNTERFACTUAL: electrons are treated as distinguishable, so the "
    "wavefunction is a product rather than an antisymmetrized determinant and "
    "there is no exchange term at all"
)
# The disclosure that stops the badge from being read as the stronger claim.
_HARTREE_PAULI_INTACT = (
    "the Pauli principle is NOT switched off: subshell occupancies are still "
    "capped at 2(2l+1) and the configuration is unchanged, so this is 'the "
    "wavefunction is not antisymmetric', not 'the electrons may all fall into 1s'"
)
_HARTREE_SELF_INTERACTION = (
    "an electron still does not repel itself: the (q-1) pair count is "
    "electrostatics, true in either model, and is not part of what was removed"
)
_HARTREE_REFINEMENT = (
    "turn exchange back on; this model is not an approximation to the real "
    "atom that a better calculation would improve on"
)
_TOTAL_ENERGY_REFINEMENT = (
    "configuration interaction or many-body perturbation theory would "
    "recover the correlation energy"
)
#: Below this Z the neglected relativity is under a tenth of a percent of the
#: deepest orbital, which is far under the correlation energy already disclosed
#: above it, so quantifying it separately would be noise. Set from the formula
#: in _relativistic_scale, not chosen: (Z*alpha)^2/4 reaches 1e-3 near Z = 9.
_RELATIVITY_WORTH_STATING_Z = 9
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
    # False means the Hartree model: distinguishable electrons, no exchange.
    # Carried on the result rather than left implicit in the provenance text so
    # a caller can branch on it without parsing prose.
    exchange: bool
    orbitals: tuple[HFOrbital, ...]
    total_energy: Quantity  # APPROXIMATION, or COUNTERFACTUAL if exchange=False
    kinetic: Quantity  # NUMERICAL
    potential: Quantity  # NUMERICAL
    virial_ratio: Quantity  # NUMERICAL, target 2
    # SCF iterations on the fine mesh, which is warm-started from the coarse
    # one and so converges in a handful whatever the mixing does.
    iterations: int
    # SCF iterations on the coarse mesh, which starts from a central field and
    # is where nearly all the wall time goes. Reported separately because the
    # two respond to completely different things: a mixing parameter that has
    # regressed roughly triples this one and barely moves `iterations`. A
    # performance guard wants this number.
    coarse_iterations: int
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


def _relativistic_scale(z: int) -> float:
    """How large the neglected relativity is, as a fraction of the 1s energy.

    "Non-relativistic" is already in the assumption list, but as a word it says
    nothing about whether the reader should care, and the answer changes by two
    orders of magnitude across the atoms this module solves: 0.003% for helium,
    0.4% for argon, 1.7% at Z = 36. A phrase that reads identically in all three
    cases is not a disclosure.

    Measured, not modelled: the exact hydrogenic Dirac 1s energy against the
    Schrodinger one at the same Z, using the dirac_energy this repo already
    ships. The 1s is the right orbital to ask because relativity is a
    core effect - it lives where the electron moves fastest - and the 1s pair
    dominates the total energy at every Z here.

    This is an order-of-magnitude scale for what is missing, not a correction
    to apply: a real atom's screening puts its 1s at slightly less than the
    hydrogenic value, so this reads a little high, which is the safe direction
    for an honesty estimate. Note it stays well under the correlation energy
    the assumption list leads with, which is why it is stated second.
    """
    schrodinger = hydrogen_energy(1, Z=z).value
    relativistic = dirac_energy(1, 0.5, Z=z).value
    return abs(relativistic - schrodinger) / abs(schrodinger)


def _energy_assumptions(
    config: Configuration, z: int, exchange: bool = True
) -> tuple[str, ...]:
    """What this configuration actually costs the reader, and nothing more.

    Two of the four claims are conditional, because disclosing a limitation the
    solve does not have misleads exactly as much as hiding one it does.

    Neon fills every subshell it touches: there is no spin to polarize and no
    second term to average over, so both extra lines would be noise. Lithium
    has an open 2s, so it pays the restriction, but its configuration spans one
    term (2S) and the configuration average is the degeneracy-weighted mean of
    the term energies - with one term in the sum, that mean is exactly that
    term. Claiming otherwise would hand the reader an error bar that is not
    there. Carbon's 2p2 spans 3P, 1D and 1S, and there the average really is
    none of them.
    """
    out = list(_TOTAL_ENERGY_ASSUMPTIONS)
    if not exchange:
        # Ahead of the rest, because these change what the number IS rather
        # than how close it lands to the truth. The correlation line keeps its
        # place behind them and stays true: Hartree is missing correlation as
        # well as exchange, and the reader is owed both.
        out[:0] = [
            _HARTREE_ALTERATION,
            _HARTREE_PAULI_INTACT,
            _HARTREE_SELF_INTERACTION,
        ]
    if z >= _RELATIVITY_WORTH_STATING_Z:
        out.append(
            f"neglects relativity, which at Z = {z} shifts the hydrogenic 1s "
            f"by {100 * _relativistic_scale(z):.2f}% of its energy; that is the "
            f"scale of what is missing here, not a correction to apply"
        )
    if open_subshells(config):
        out.append(_OPEN_SHELL_ASSUMPTION)
    if not is_single_term(config):
        out.append(_MULTI_TERM_ASSUMPTION)
    return tuple(out)


def _solve_on_grid(
    z: int,
    n_electrons: int,
    config: Configuration,
    mesh: RadialMesh,
    start: tuple[Subshell, ...] | None,
    exchange: bool = True,
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
        z, start, lambda rr: -z / rr, mesh, tol=1e-9 * max(1, z**2),
        exchange=exchange,
    )
    energies = tuple(
        orbital_energy(solution.subshells, i, z, mesh, exchange=exchange)
        for i in range(len(solution.subshells))
    )
    return solution, energies, total_energy_direct(
        z, solution.subshells, mesh, exchange=exchange
    )


@lru_cache(maxsize=8)
def solve_hartree_fock(
    z: int, n_electrons: int, config: Configuration, exchange: bool = True
) -> HFResult:
    """Converge the restricted Hartree-Fock equations for one atom or ion.

    Raises HFConvergenceError rather than returning an unconverged result: a
    HFResult with converged=False would be a quiet lie in object form.

    exchange=False solves the Hartree model instead - electrons that repel but
    are distinguishable - and the result comes back COUNTERFACTUAL rather than
    APPROXIMATION. It is a positional argument and part of the cache key, so
    the two models never share a cached solve.
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
        z, n_electrons, config, coarse_mesh, start=None, exchange=exchange
    )
    solution, energies, e_direct = _solve_on_grid(
        z, n_electrons, config, mesh,
        start=_refine(coarse, coarse_mesh, mesh),
        exchange=exchange,
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

    kinetic, potential = kinetic_and_potential(
        z, solution.subshells, mesh, exchange=exchange
    )

    assumptions = _energy_assumptions(config, z, exchange)
    energy_prov = Provenance(
        # COUNTERFACTUAL rather than APPROXIMATION when exchange is off, and
        # the distinction is not cosmetic. The truth-distance tiers say how far
        # a number is from the real atom; this number is not trying to be the
        # real atom at all. Calling it APPROXIMATION would invite the reader to
        # treat the gap to the reference energy as an error, when the gap IS
        # the physics the toggle exists to show.
        fidelity=Fidelity.APPROXIMATION if exchange else Fidelity.COUNTERFACTUAL,
        method=_TOTAL_ENERGY_METHOD if exchange else _HARTREE_METHOD,
        assumptions=assumptions,
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
        refinement=_TOTAL_ENERGY_REFINEMENT if exchange else _HARTREE_REFINEMENT,
    )
    diagnostic_prov = Provenance(
        fidelity=Fidelity.NUMERICAL,
        method=_DIAGNOSTIC_METHOD,
        assumptions=(
            f"converged in {coarse.iterations} SCF iterations on the coarse "
            f"mesh and {solution.iterations} on the fine one, which starts "
            f"from the coarse solution rather than from a central field",
        ),
    )
    # The orbital amplitude gets its own provenance carrying NO error estimate,
    # rather than borrowing the energy's. Provenance.error_estimate is
    # documented as being in the unit of the quantity it describes, and P is in
    # bohr^-1/2: an error bar in hartree attached to it would not be a loose
    # error bar, it would be a number in the wrong dimension. The mesh spread
    # for the orbital SHAPE is not something this solve estimates, and saying
    # nothing is the honest form of not knowing.
    shape_prov = Provenance(
        # The orbital SHAPE is as counterfactual as the energy. Exchange changes
        # the operator these came out of, so a Hartree 2s is a different curve,
        # not the same curve at a different accuracy.
        fidelity=Fidelity.APPROXIMATION if exchange else Fidelity.COUNTERFACTUAL,
        method=(
            f"{_TOTAL_ENERGY_METHOD if exchange else _HARTREE_METHOD}; "
            f"radial amplitude sampled on the solver mesh"
        ),
        assumptions=assumptions,
        refinement=_TOTAL_ENERGY_REFINEMENT if exchange else _HARTREE_REFINEMENT,
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
        # The key names the calculation, not just the atom, so a Hartree solve
        # and a Hartree-Fock solve of the same atom cannot be mistaken for each
        # other anywhere downstream that caches or labels by key.
        key=f"z{z}n{n_electrons}" + ("" if exchange else "-nox"),
        z=z,
        n_electrons=n_electrons,
        config=config,
        is_ground=is_ground(config),
        exchange=exchange,
        orbitals=orbitals,
        total_energy=Quantity(e_direct, "hartree", "E_total", energy_prov),
        kinetic=Quantity(kinetic, "hartree", "T", diagnostic_prov),
        potential=Quantity(potential, "hartree", "V", diagnostic_prov),
        virial_ratio=Quantity(
            -potential / kinetic, "dimensionless", "-V/T", diagnostic_prov
        ),
        iterations=solution.iterations,
        coarse_iterations=coarse.iterations,
        residual_history=solution.residual_history,
        converged=True,
        provenance=energy_prov,
    )


def hf_valence_ionization_energy(result: HFResult) -> Quantity:
    """IE = -epsilon_valence, by Koopmans' theorem.

    Mirrors screened_atom.valence_ionization_energy so the two models are
    swappable, but the approximation being made here is NOT the same one, and
    the provenance says so rather than inheriting the solve's.

    Koopmans equates the ionization energy with minus the orbital energy, which
    assumes the remaining N-1 electrons do not relax when one leaves. They do.
    That error does not shrink with a finer mesh or a tighter SCF, so it is a
    model error stacked on top of Hartree-Fock's own, and it is not small:
    helium's Koopmans IE is 24.98 eV against 24.587 measured, an overestimate
    of 0.39 eV, because ionizing a two-electron atom contracts what is left of
    it hard. Alkalis are far better, since removing the lone valence electron
    barely disturbs the closed core: lithium lands 0.05 eV out.

    The two neglected effects push opposite ways, which is worth stating
    because it explains why the error is not simply "HF is missing
    correlation". Frozen orbitals overestimate the IE; missing correlation
    underestimates it. Relaxing the ion properly (Delta-SCF) gives helium
    23.45 eV, 1.14 eV LOW, which is its correlation energy almost exactly.
    """
    occupied = [o for o in result.orbitals if o.occupancy > 0]
    if not occupied:
        raise ValueError("no occupied orbitals")
    valence = max(occupied, key=lambda o: o.energy.value)
    base = valence.energy.provenance
    prov = dataclasses.replace(
        base,
        method=f"{base.method}; ionization energy = -epsilon_valence (Koopmans)",
        assumptions=base.assumptions
        + (
            "Koopmans: the N-1 remaining electrons do not relax, which "
            "overestimates the ionization energy (0.39 eV for helium, less for "
            "atoms whose valence electron sits outside a closed core)",
        ),
        refinement=(
            "a Delta-SCF ionization energy, E(ion) - E(atom), relaxes the ion "
            "and removes the Koopmans error, leaving only the correlation one"
        ),
    )
    return Quantity(-valence.energy.value, "hartree", "IE_valence", prov)


def hf_exchange_energy(
    z: int, n_electrons: int, config: Configuration
) -> Quantity:
    """E_HF - E_Hartree: what antisymmetry is worth to this atom, in hartree.

    Negative, because exchange stabilizes. Both models are solved here rather
    than subtracted by the caller, and that is the point of the function
    existing: the two energies must come off the same mesh at the same
    refinement, and a UI that fetched them separately would be free to difference
    a fine solve against a coarse one and report the mesh spread as physics.

    Zero, exactly, for helium and for every other atom whose configuration is a
    single closed s shell - and that is the model being right, not the toggle
    failing. Exchange couples same-spin pairs only; 1s2 holds one spin up and one
    spin down, so there is no such pair and `exchange_operator` builds no terms.
    The k = 0 same-shell integral that a reader might expect to see here is
    already carried by the (q - 1) factor in the direct potential, where it
    belongs, in both models.

    The difference of two variational energies is not itself variational, so
    this carries no error bar against any exact quantity. It gets the loosest of
    the two solves' mesh estimates, which is an honest statement about the
    arithmetic and not a claim about the physics.
    """
    with_exchange = solve_hartree_fock(z, n_electrons, config, True)
    without = solve_hartree_fock(z, n_electrons, config, False)
    delta = with_exchange.total_energy.value - without.total_energy.value

    return Quantity(
        delta,
        "hartree",
        "E_exchange",
        Provenance(
            # COUNTERFACTUAL, because half of what produced it is: this number
            # cannot be measured, only computed by running an experiment on a
            # universe that does not exist and differencing.
            fidelity=Fidelity.COUNTERFACTUAL,
            method=(
                "E(Hartree-Fock) - E(Hartree): the same atom on the same mesh, "
                "solved once with the exchange term and once without"
            ),
            assumptions=(
                "the stabilization an antisymmetric wavefunction buys, at the "
                "average-of-configuration level; not an observable",
                "exactly zero whenever no two electrons share a spin, which "
                "includes helium and every closed single-s-shell configuration",
            )
            + _TOTAL_ENERGY_ASSUMPTIONS,
            error_estimate=max(
                with_exchange.total_energy.provenance.error_estimate or 0.0,
                without.total_energy.provenance.error_estimate or 0.0,
            ),
            refinement=(
                "correlation energy is the other half of what a single "
                "determinant misses, and is not included in this difference"
            ),
        ),
    )


def _occupied_orbital(z: int, n_electrons: int, n: int, l: int) -> HFOrbital:
    """The converged orbital for one subshell, or a refusal.

    Hartree-Fock cannot hand back an arbitrary channel the way a central-field
    model can. There is no single potential here: each occupied subshell has
    its own Fock operator, built from the others, so an unoccupied subshell has
    no operator to be an eigenfunction of. Asking for one is a question this
    model cannot answer, and inventing a channel by borrowing another
    subshell's operator would answer a different question silently.
    """
    if n <= l:
        raise ValueError(f"n must be > l, got n={n}, l={l}")
    # The ground configuration for however many electrons are present, which
    # for a neutral atom is just the element's own.
    result = solve_hartree_fock(z, n_electrons, aufbau_configuration(n_electrons))
    for orbital in result.orbitals:
        if (orbital.n, orbital.l) == (n, l):
            return orbital
    held = ", ".join(f"{o.n}{'spdf'[o.l]}" for o in result.orbitals)
    raise ValueError(
        f"subshell {n}{'spdf'[l]} is not occupied in Z={z}, N={n_electrons} "
        f"(which holds {held}); Hartree-Fock builds one Fock operator per "
        f"occupied subshell, so there is no operator for an empty one"
    )


def hf_radial(
    z: int, n_electrons: int, n: int, l: int, points: int = 400,
) -> tuple[Field, Field]:
    """R_nl(r) and the radial density r^2 R^2, on a uniform display grid.

    Mirrors screened_atom.screened_radial, including its convention that the
    second field is the probability density and not the amplitude. Note the
    solver's own HFOrbital.P is the amplitude P = r R, a different quantity
    with a different unit; the naming follows screened_atom because these are
    what a caller plots.
    """
    orbital = _occupied_orbital(z, n_electrons, n, l)
    solver_r = orbital.P.grid
    grid = np.linspace(solver_r[0], solver_r[-1], points)
    # R = P / r. The mesh never reaches r = 0, so this needs no special case,
    # which is exactly why the exponential mesh starts where it does.
    values = np.interp(grid, solver_r, orbital.P.values / solver_r)
    prov = dataclasses.replace(
        orbital.P.provenance,
        method=f"{orbital.P.provenance.method}; R_nl = P/r resampled uniformly",
    )
    r_field = Field(
        values=values, grid=grid, unit="bohr^-3/2", grid_unit="bohr",
        label=f"R_{n},{l}(r)", provenance=prov,
    )
    p_field = Field(
        values=grid**2 * values**2, grid=grid, unit="bohr^-1", grid_unit="bohr",
        label=f"P_{n},{l}(r) = r^2 R^2", provenance=prov,
    )
    return r_field, p_field


def evaluate_hf_state(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    positions: np.ndarray,
    *,
    basis: str = "complex",
) -> WavefunctionValues:
    """psi_nlm = Hartree-Fock R_nl(|r|) x hydrogenic Y_lm, at (N, 3) positions.

    Mirrors evaluate_screened_state. The angular factor is still the hydrogenic
    harmonic: restricted Hartree-Fock on a spherically averaged configuration
    leaves the angular dependence exactly Y_lm, so this is the model's own
    shape rather than a convenience.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")

    r = np.linalg.norm(pos, axis=1)
    safe_r = np.where(r > 0.0, r, 1.0)
    theta = np.arccos(np.clip(pos[:, 2] / safe_r, -1.0, 1.0))
    theta = np.where(r > 0.0, theta, 0.0)
    phi = np.arctan2(pos[:, 1], pos[:, 0])

    r_field, _ = hf_radial(z, n_electrons, n, l, points=_HF_EVAL_POINTS)
    # Inside the first mesh point, hold R flat rather than extrapolating; past
    # the box, zero. Both match evaluate_screened_state.
    R = np.interp(r, r_field.grid, r_field.values, left=r_field.values[0], right=0.0)
    angular = spherical_harmonic(l, m, theta, phi, basis=basis)

    base = r_field.provenance
    prov = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=(
            f"psi_nlm = Hartree-Fock R_nl (P/r) x {angular.provenance.method}; "
            f"{base.method}"
        ),
        assumptions=base.assumptions
        + angular.provenance.assumptions
        + ("values in bohr^-3/2 at Cartesian positions in bohr",),
        error_estimate=base.error_estimate,
    )
    return WavefunctionValues(
        values=R * angular.values, positions=pos, n=n, l=l, m=m, Z=z, mu_ratio=1.0,
        basis=basis, provenance=prov,
    )
