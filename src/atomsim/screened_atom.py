"""Solve a screened multi-electron atom in the GSZ central field (APPROXIMATION).

Each angular momentum l is solved once in V_eff(r); radial state k is principal
number n = k + l + 1. The configuration decides occupancy and thus the summed
energy -- the field itself depends only on (Z, N). Orbital energies are
APPROXIMATION (model error dominates) carrying the numerical solve error as a
quantified sub-scale. See docs/superpowers/specs/
2026-07-18-phase6-screened-atoms-design.md.
"""

import dataclasses
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from atomsim.analytic.angular import spherical_harmonic
from atomsim.analytic.wavefunction import WavefunctionValues
from atomsim.atoms import Configuration, aufbau_configuration, is_ground
from atomsim.numerics.dipole import (
    dipole_box_radius,
    dipole_from_solutions,
    grid_points_for,
)
from atomsim.numerics.radial_solver import RadialSolution, solve_radial, solve_radial_with_error
from atomsim.numerics.screening import screened_potential, screening_provenance
from atomsim.provenance import Fidelity, Field, Provenance, Quantity

_SCREENED_EVAL_POINTS = 4096


@dataclass(frozen=True)
class Orbital:
    n: int
    l: int
    occupancy: int
    energy: Quantity  # APPROXIMATION, hartree


@dataclass(frozen=True)
class ScreenedAtomResult:
    key: str
    z: int
    n_electrons: int
    config: Configuration
    is_ground: bool
    orbitals: tuple[Orbital, ...]
    total_energy: Quantity
    provenance: Provenance


def _r_max(z: int, n_electrons: int, n_top: int) -> float:
    # Orbital extent scales as n^2 / Z_net, where Z_net = Z - N + 1 is the charge
    # the outermost electron feels asymptotically. Neutral atoms (Z_net = 1) keep
    # the 40*(n+1)^2 box; more highly charged ions get a proportionally tighter
    # box, so the compact orbitals stay well resolved on the uniform grid.
    z_net = z - n_electrons + 1
    return 40.0 * (n_top + 1) ** 2 / z_net


def _solve_energies(z: int, n_electrons: int, l: int, n_states: int) -> tuple[Quantity, ...]:
    potential = screened_potential(z, n_electrons)
    r_max = _r_max(z, n_electrons, n_states + l)
    sol = solve_radial_with_error(
        potential, l=l, mu_ratio=1.0, r_max=r_max, n_states=n_states
    )
    prov_model = screening_provenance(z, n_electrons)
    out = []
    for e in sol.energies:
        merged = Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=f"{prov_model.method}; radial Schrodinger solved numerically",
            assumptions=prov_model.assumptions + e.provenance.assumptions,
            error_estimate=e.provenance.error_estimate,  # numerical sub-scale
            refinement=prov_model.refinement,
        )
        out.append(dataclasses.replace(e, provenance=merged))
    return tuple(out)


def solve_screened_atom(
    z: int, n_electrons: int, config: Configuration,
    l_max: int = 2, n_states_per_l: int = 4,
) -> ScreenedAtomResult:
    occ = {nl: c for nl, c in config}
    l_top = max((l for (_, l), _ in config), default=0)
    n_top = max((n for (n, _), _ in config), default=1)
    l_max = max(l_max, l_top)
    n_states = max(n_states_per_l, n_top)  # enough radial states to reach n_top

    orbitals: list[Orbital] = []
    for l in range(l_max + 1):
        energies = _solve_energies(z, n_electrons, l, n_states)
        for k, e in enumerate(energies):
            n = k + l + 1
            orbitals.append(Orbital(n=n, l=l, occupancy=occ.get((n, l), 0), energy=e))
    orbitals.sort(key=lambda o: (o.energy.value, o.n, o.l))

    total = sum(o.occupancy * o.energy.value for o in orbitals)
    total_prov = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method="sum of occupancy-weighted independent-particle orbital energies",
        assumptions=(
            "not a variational total energy; ignores e-e double counting",
            f"configuration {'ground' if is_ground(config) else 'non-ground'}",
        ),
    )
    return ScreenedAtomResult(
        key=f"z{z}n{n_electrons}",
        z=z, n_electrons=n_electrons, config=config, is_ground=is_ground(config),
        orbitals=tuple(orbitals),
        total_energy=Quantity(total, "hartree", "E_total", total_prov),
        provenance=screening_provenance(z, n_electrons),
    )


def valence_ionization_energy(result: ScreenedAtomResult) -> Quantity:
    occupied = [o for o in result.orbitals if o.occupancy > 0]
    if not occupied:
        raise ValueError("no occupied orbitals")
    valence = max(occupied, key=lambda o: o.energy.value)
    prov = dataclasses.replace(
        valence.energy.provenance,
        method=valence.energy.provenance.method + "; ionization energy = -epsilon_valence",
    )
    return Quantity(-valence.energy.value, "hartree", "IE_valence", prov)


def screened_radial(
    z: int, n_electrons: int, n: int, l: int, points: int = 400,
) -> tuple[Field, Field]:
    if n <= l:
        raise ValueError(f"n must be > l, got n={n}, l={l}")
    k = n - l - 1
    potential = screened_potential(z, n_electrons)
    r_max = _r_max(z, n_electrons, n)
    sol = solve_radial_with_error(
        potential, l=l, mu_ratio=1.0, r_max=r_max, n_states=k + 1
    )
    r_solver = sol.r
    R = sol.u[k] / r_solver  # R = u / r
    grid = np.linspace(r_solver[0], r_solver[-1], points)
    R_i = np.interp(grid, r_solver, R)
    # These two fields are shapes, not energies, so they carry NO error
    # estimate. They used to borrow the eigenvalue's, which is in hartree,
    # while R is in bohr^-3/2 and r^2 R^2 in bohr^-1. Provenance.error_estimate
    # is documented as being in the unit of the quantity it describes, so that
    # was not a loose bar on the shape, it was a number in the wrong dimension
    # presented as an uncertainty on R. What this solve does estimate is the
    # error in the ENERGY, and that bar is still on the energy where it belongs
    # (see ScreenedAtomResult.orbitals). The grid error in the shape itself is
    # not something this routine measures, and saying nothing is the honest
    # form of not knowing. Same fix, same reasoning, as hf_atom.shape_prov.
    prov = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=f"{screening_provenance(z, n_electrons).method}; numerical R_nl = u/r",
        assumptions=screening_provenance(z, n_electrons).assumptions,
        error_estimate=None,
    )
    r_field = Field(values=R_i, grid=grid, unit="bohr^-3/2", grid_unit="bohr",
                    label=f"R_{n},{l}(r)", provenance=prov)
    p_field = Field(values=grid**2 * R_i**2, grid=grid, unit="bohr^-1",
                    grid_unit="bohr", label=f"P_{n},{l}(r) = r^2 R^2", provenance=prov)
    return r_field, p_field


def _density_grid(z: int, n_electrons: int, n_top: int) -> tuple[float, int]:
    """Box and point count for a whole atom's density, not for one orbital.

    `_r_max` above sizes a box around the orbital being asked for, and the
    solver's default point count then sets the spacing to whatever falls out.
    For one orbital that is fine, because the box and the orbital scale
    together. For a density it is not: the box has to hold the outermost
    occupied shell while the spacing has to resolve the innermost, and those
    two are Z apart.

    Measured, on neutral argon: the `_r_max` box is 640 bohr, the default
    48000 points put h at 0.013, and the 1s peaks at 0.055 with about four
    points across it. The density then loses 0.13 of an electron and splits
    the K shell into two maxima at 0.054 and 0.066 bohr, a fourth shell argon
    does not have. Both survive any refinement of the display grid, because
    neither is a display problem.

    So the box is sized to the valence and the spacing to the core. 4(n+1)^2 /
    Z_net still clears the outermost orbital by tens of decay lengths (argon's
    3p is bound at 0.6 hartree, so 64 bohr is e^-140 out), and h = 1/(40 Z)
    puts about fifty points across a 1s of scale 1/Z. Against the wider box
    this MOVES the deep orbitals: argon's 1s goes from -111.88 to -114.05
    hartree, and shrinking the box further to 25 bohr moves it only another
    0.005, so the narrow answer is the converged one. The valence energies,
    which are what the screened model is judged on, move by under 0.002
    hartree either way.
    """
    r_max = 4.0 * (n_top + 1) ** 2 / (z - n_electrons + 1)
    return r_max, int(np.ceil(r_max * 40.0 * z))


def screened_total_radial_density(
    z: int, n_electrons: int, config: Configuration | None = None,
    points: int = 400,
) -> Field:
    """D(r) = sum_a q_a u_a(r)^2, the whole atom's radial density in electrons/bohr.

    The observable one. A single orbital is a basis choice, and this sum is not:
    integrate it over any shell and you get how many electrons are in it. The
    GSZ counterpart of `hf_atom.hf_total_radial_density`, and the two disagree
    in a way worth looking at, which is the point of having both.

    The error estimate is the closure residual |integral D dr - N| in electrons.
    Every u_a is normalized to one on the solver's mesh, so N is exact there and
    what is left after resampling is a real error in the right unit. It settles
    at a floor set by the solve rather than falling to zero, because
    interpolating u and then squaring sits just under the true u^2 between
    nodes.

    APPROXIMATION, and of a different thing than the energies: GSZ was fitted to
    reproduce a potential, so a density read off its orbitals is further from
    the data the model was built on than any energy this module returns.
    """
    cfg = aufbau_configuration(n_electrons) if config is None else config
    occupied = [(nl, q) for nl, q in cfg if q > 0]
    if not occupied:
        raise ValueError("no occupied subshells, so no density")
    n_top = max(n for (n, _), _ in occupied)
    l_max = max(l for (_, l), _ in occupied)
    r_max, n_points = _density_grid(z, n_electrons, n_top)

    potential = screened_potential(z, n_electrons)
    channels = {
        l: solve_radial(
            potential, l=l, mu_ratio=1.0, r_max=r_max,
            n_points=n_points, n_states=n_top - l,
        )
        for l in range(l_max + 1)
    }

    solver_r = channels[0].r
    grid = np.geomspace(solver_r[0], solver_r[-1], points)
    values = np.zeros_like(grid)
    for (n, l), q in occupied:
        if n <= l:
            raise ValueError(f"n must be > l for a real subshell, got n={n}, l={l}")
        u = channels[l].u[n - l - 1]
        values += q * np.interp(grid, channels[l].r, u) ** 2

    n_total = sum(q for _, q in occupied)
    residual = abs(float(np.trapezoid(values, grid)) - float(n_total))
    model = screening_provenance(z, n_electrons)
    return Field(
        values=values, grid=grid, unit="electrons/bohr", grid_unit="bohr",
        label=f"D(r) = sum_a q_a u_a(r)^2 (N = {n_total})",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                f"{model.method}; occupancy-weighted sum of the squared radial "
                f"functions, resampled onto a logarithmic grid"
            ),
            assumptions=model.assumptions + (
                "GSZ was fitted to a potential, not to a density, so this shape "
                "is further from the model's own data than its energies are",
                "one central field for every electron, so the shells share a "
                "potential rather than each seeing the others",
                f"solved in a {r_max:.0f} bohr box at h = {r_max / n_points:.1e} "
                f"bohr, sized to hold the valence and resolve the core",
            ),
            error_estimate=residual,
            refinement=(
                "an orbital-dependent potential, i.e. Hartree-Fock, which this "
                "engine also solves and which needs no fitted parameters"
            ),
        ),
    )


#: Deliberately small. These entries are not cheap objects: a solved channel on
#: the fine dipole grid carries n_states * ~1e5 floats, so one atom's worth of
#: them is ~20 MB. A whole line list needs only about six, so 16 covers any
#: single spectrum with room for the previous one and still caps what a
#: long-running server retains at a few tens of MB, instead of growing with
#: every element the user visits.
_DIPOLE_CHANNEL_CACHE = 16


@lru_cache(maxsize=_DIPOLE_CHANNEL_CACHE)
def _dipole_channel(
    z: int, n_electrons: int, l: int, r_max: float, n_points: int, n_states: int
) -> RadialSolution:
    """One l channel solved for the dipole grid, cached.

    A spectrum asks for many lines but they run over only a handful of l
    channels, so without this the same eigenproblem is re-solved per line.
    """
    return solve_radial(
        screened_potential(z, n_electrons), l=l, mu_ratio=1.0,
        r_max=r_max, n_points=n_points, n_states=n_states,
    )


def screened_dipole_integral(
    z: int, n_electrons: int, n_a: int, l_a: int, n_b: int, l_b: int,
    n_box: int | None = None,
) -> Quantity:
    """Radial dipole matrix element <b|r|a> in bohr for a screened atom.

    APPROXIMATION, not NUMERICAL: the GSZ model error dominates the grid error,
    and labelling this by its discretization alone would understate it. The
    grid-halving figure is still reported as the numerical sub-scale.

    `n_box` sizes the shared grid for that principal quantum number instead of
    the pair's own. A caller working through a whole line list should pass the
    largest n in it, so every line lands on one grid: the box only has to be
    big enough, and one box per atom means each l channel is solved once rather
    than once per distinct pair, which is both faster and bounded in memory.
    Accuracy is unaffected because the spacing, not the box, sets the error.
    """
    for n, l, name in ((n_a, l_a, "a"), (n_b, l_b, "b")):
        if n <= l:
            raise ValueError(f"state {name}: n must be > l, got n={n}, l={l}")
    n_top = max(n_a, n_b, n_box or 0)
    z_net = z - n_electrons + 1
    r_max = dipole_box_radius(n_top, z_net)
    n_points = grid_points_for(r_max)

    def overlap(points: int) -> float:
        # Solve each channel deep enough for every k this atom can ask of it,
        # so the cache key stays the same across the whole line list.
        sol_a = _dipole_channel(z, n_electrons, l_a, r_max, points, n_top - l_a)
        sol_b = _dipole_channel(z, n_electrons, l_b, r_max, points, n_top - l_b)
        return dipole_from_solutions(sol_a, n_a - l_a - 1, sol_b, n_b - l_b - 1)

    coarse = overlap(n_points)
    fine = overlap(2 * n_points)
    model = screening_provenance(z, n_electrons)
    return Quantity(
        value=fine,
        unit="bohr",
        label=f"<{n_b},{l_b}|r|{n_a},{l_a}> (Z={z}, N={n_electrons})",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                f"{model.method}; dipole = integral u_a u_b r dr over the "
                "numerically solved radials (both states on one grid)"
            ),
            assumptions=model.assumptions
            + (
                f"shared uniform grid: r_max={r_max:g} bohr, N={2 * n_points}",
                "GSZ model error dominates the quoted grid-halving figure",
                "independent-particle: no correlation or core polarization",
            ),
            error_estimate=abs(fine - coarse),
            refinement=model.refinement,
        ),
    )


def evaluate_screened_state(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    positions: np.ndarray,
    *,
    basis: str = "complex",
) -> WavefunctionValues:
    """psi_nlm = numerical screened R_nl(|r|) x hydrogenic Y_lm, at (N, 3) positions."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")

    r = np.linalg.norm(pos, axis=1)
    safe_r = np.where(r > 0.0, r, 1.0)
    theta = np.arccos(np.clip(pos[:, 2] / safe_r, -1.0, 1.0))
    theta = np.where(r > 0.0, theta, 0.0)
    phi = np.arctan2(pos[:, 1], pos[:, 0])

    r_field, _ = screened_radial(z, n_electrons, n, l, points=_SCREENED_EVAL_POINTS)
    R = np.interp(r, r_field.grid, r_field.values, left=r_field.values[0], right=0.0)
    angular = spherical_harmonic(l, m, theta, phi, basis=basis)
    values = R * angular.values

    base = screening_provenance(z, n_electrons)
    prov = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=(
            f"psi_nlm = numerical screened R_nl (u/r) x {angular.provenance.method}; "
            f"{base.method}"
        ),
        assumptions=base.assumptions
        + angular.provenance.assumptions
        + ("values in bohr^-3/2 at Cartesian positions in bohr",),
        error_estimate=r_field.provenance.error_estimate,
    )
    return WavefunctionValues(
        values=values, positions=pos, n=n, l=l, m=m, Z=z, mu_ratio=1.0,
        basis=basis, provenance=prov,
    )
