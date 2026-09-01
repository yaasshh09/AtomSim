"""My spectral lines, from level differences, with selection rules and
provenance.

On gross structure (fine_structure=False) I use EXACT Bohr levels, mu-scaled,
and my lines are (n_u, l_u) -> (n_l, l_l) with Delta l = +/-1. With fine
structure on I use APPROXIMATION levels from the alpha^2 Pauli shifts, with
Delta j in {0, +/-1}. My wavelengths are vacuum, in nm, and my energies are in
eV. For the NIST comparison see my compare_lines API, which reads vendored
reference data and never queries anything live.
"""

import itertools
import json
from dataclasses import dataclass
from importlib import resources

from scipy import constants as _sc

from atomsim.analytic.fine_structure import level_energy
from atomsim.analytic.hydrogen import energy
from atomsim.analytic.transitions import (
    A_from_radial_dipole,
    einstein_A,
    einstein_A_fine,
    f_from_radial_dipole,
    oscillator_strength,
    oscillator_strength_fine,
)
from atomsim.constants import HARTREE_EV
from atomsim.populations import (
    Level,
    ThermalConditions,
    ThermalState,
    boltzmann_fractions,
    level_column_fraction,
    level_degeneracy,
    line_emissivity,
    partition_function,
    saha_ionization_fraction,
)
from atomsim.provenance import Fidelity, Provenance, Quantity
from atomsim.systems import System

_EV_NM = _sc.h * _sc.c / _sc.e * 1e9  # my photon wavelength(nm) = _EV_NM / E(eV)

_REFERENCE_FILES = {
    "h": "nist_h_i.json", "d": "nist_d_i.json", "he+": "nist_he_ii.json",
    "he": "nist_he_i.json", "li": "nist_li_i.json", "na": "nist_na_i.json",
}

_DEFAULT_TOL = {False: 3e-5, True: 1e-5}  # relative, one per fidelity tier of mine


@dataclass(frozen=True)
class SpectralLine:
    n_upper: int
    l_upper: int
    j_upper: float | None
    n_lower: int
    l_lower: int
    j_lower: float | None
    energy: Quantity      # eV
    wavelength: Quantity  # nm, vacuum
    einstein_a: Quantity | None = None           # s^-1, spontaneous emission rate
    oscillator_strength: Quantity | None = None  # dimensionless, absorption f
    #: eV/s per atom of the element. I set this only when you gave me thermal
    #: conditions: it is an emission rate I modelled, not a measured brightness.
    emissivity: Quantity | None = None
    #: The fraction of all atoms of the element I put in this line's **lower**
    #: level. I set it alongside the emissivity, out of the same Boltzmann pass.
    #: It is what turns a single column density for the gas into a per-line
    #: optical depth, and without it I cannot build an absorption spectrum at
    #: all.
    lower_fraction: Quantity | None = None


_L_LETTERS = "spdfghi"


def orbital_label(n: int, l: int) -> str:
    """A level of mine in spectroscopic notation: (3, 2) -> "3d"."""
    return f"{n}{_L_LETTERS[l]}" if l < len(_L_LETTERS) else f"{n}(l={l})"


def subshell_label(line: "SpectralLine") -> str:
    """A line of mine named by the levels it joins: "3d->2p".

    I need this wherever lines have to be told apart rather than merely counted.
    A gross-structure series label like "3->2" names three transitions at one
    wavelength with oscillator strengths of 0.014, 0.435 and 0.696, which is
    fine for a series and useless for anything that has to pick one.
    """
    return (
        f"{orbital_label(line.n_upper, line.l_upper)}"
        f"->{orbital_label(line.n_lower, line.l_lower)}"
    )


@dataclass(frozen=True)
class LineList:
    system_key: str
    n_max: int
    fine_structure: bool
    lines: tuple[SpectralLine, ...]
    provenance: Provenance
    #: I set this only when strengths were wanted and I cannot give them
    #: honestly. It says which case applies and what I am missing, so the view
    #: never has to guess why every bar is the same height.
    intensity_note: str | None = None
    #: Present exactly when my lines carry an emissivity. It holds the conditions
    #: that produced them plus how much of the gas I ionized, so the view can say
    #: what it is showing rather than just showing it.
    thermal: ThermalState | None = None


def _strength_provenance(dipole: Quantity, formula: str, value: float) -> Provenance:
    """I carry the dipole integral's own provenance up to the strength I build
    on it.

    Both f and A go as |R|^2, so R's *relative* error doubles. I store the
    absolute number, matching every other error_estimate of mine.
    """
    rel = (
        (dipole.provenance.error_estimate or 0.0) / abs(dipole.value)
        if dipole.value else 0.0
    )
    return Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=f"{formula}, with R from [{dipole.provenance.method}]",
        assumptions=dipole.provenance.assumptions,
        error_estimate=2.0 * rel * abs(value),
        refinement=dipole.provenance.refinement,
    )


def _thermal_state(
    levels: tuple[Level, ...],
    keys: tuple[tuple, ...],
    chi_ev: float,
    thermal: ThermalConditions,
    chi_assumptions: tuple[str, ...] = (),
) -> tuple[ThermalState, dict[tuple, float]]:
    """I run the LTE chain over a level list, indexed by the caller's own keys.

    I build my levels from whatever produced the line list rather than from a
    second source of energies, so my populations and my transition energies
    cannot disagree about where the levels are.

    You have to give me `chi_ev` rather than letting me read it off the list:
    the highest level in a list I truncated at n_max is nowhere near the
    continuum, so taking it as the ionization energy would understate chi badly
    and ionize my gas far too easily.

    `chi_assumptions` lets a caller whose chi is itself an estimate say so on
    the quantity that used it, rather than leaving it looking exact.

    I return the state to hang on the LineList, plus a key -> occupation lookup
    for my per-line loop.
    """
    u = partition_function(levels, thermal.temperature_k)
    ionized = saha_ionization_fraction(
        thermal.temperature_k,
        thermal.electron_density_cm3,
        chi_ev,
        u_neutral=u.value,
    )
    # Saha divides by U, so my ionization fraction inherits the truncation U
    # discloses. saha_ionization_fraction takes U as a bare float and cannot know
    # where it came from, so I attach the disclosure here, where I do know it.
    # Without this I would state the cutoff on my partition function and then
    # drop it silently from the number I built on it.
    extra = tuple(a for a in u.provenance.assumptions if "truncat" in a) + chi_assumptions
    if extra:
        ionized = Quantity(
            ionized.value, ionized.unit, ionized.label,
            Provenance(
                fidelity=ionized.provenance.fidelity,
                method=ionized.provenance.method,
                assumptions=ionized.provenance.assumptions + extra,
                error_estimate=ionized.provenance.error_estimate,
                refinement=ionized.provenance.refinement,
            ),
        )
    fractions = boltzmann_fractions(levels, thermal.temperature_k)
    return (
        ThermalState(conditions=thermal, ionized_fraction=ionized, partition_function=u),
        {k: f.value for k, f in zip(keys, fractions, strict=True)},
    )


def _hydrogenic_thermal(levels: list, thermal: ThermalConditions):
    """I build the population levels for a hydrogen-like system's own level list.

    My energies come straight from the level Quantities I built the line list
    from, shifted so the ground level sits at zero, and my chi is that ground
    level's binding energy.
    """
    ground_h = min(e.value for _, _, _, e in levels)
    specs = tuple(
        Level(
            n=n,
            label=f"{n},{l},{j}",
            energy_ev=(e.value - ground_h) * HARTREE_EV,
            degeneracy=level_degeneracy(l, j),
        )
        for n, l, j, e in levels
    )
    keys = tuple((n, l, j) for n, l, j, _ in levels)
    return _thermal_state(specs, keys, -ground_h * HARTREE_EV, thermal)


def _levels(system: System, n_max: int, fine_structure: bool):
    """I yield (n, l, j, E_hartree Quantity) for every level up to n_max."""
    for n in range(1, n_max + 1):
        for l in range(n):
            if fine_structure:
                js = [l - 0.5, l + 0.5] if l > 0 else [0.5]
                for j in js:
                    yield n, l, j, level_energy(
                        n, l, j, Z=system.Z,
                        mu_ratio=system.mu_ratio.value, m_over_M=system.m_over_M,
                    )
            else:
                yield n, l, None, energy(n, Z=system.Z, mu_ratio=system.mu_ratio.value)


def transition_lines(
    system: System,
    n_max: int,
    fine_structure: bool = False,
    intensities: bool = False,
    thermal: ThermalConditions | None = None,
) -> LineList:
    """Every dipole-allowed emission line I find among levels with n <= n_max.

    With `intensities`, each line also carries its Einstein A (s^-1) and its
    absorption oscillator strength, out of my closed-form dipole engine. With
    `fine_structure` I resolve those rates by j through the 6j branching factor,
    so the components of a multiplet add back up to my gross rate.

    With `thermal`, each line additionally carries an LTE emissivity, and my
    list carries the ionization fraction those conditions produced. Thermal
    implies intensities, since I build emissivity on A.
    """
    if n_max < 2:
        raise ValueError(f"n_max must be >= 2 to have any transition, got {n_max}")
    levels = list(_levels(system, n_max, fine_structure))
    if thermal is not None:
        intensities = True
    state, occupation = _hydrogenic_thermal(levels, thermal) if thermal else (None, {})
    lines: list[SpectralLine] = []
    for (nu, lu, ju, eu), (nl, ll_, jl, el) in itertools.permutations(levels, 2):
        if eu.value <= el.value:
            continue
        if abs(lu - ll_) != 1:
            continue
        if fine_structure and abs(ju - jl) > 1.0 + 1e-12:
            continue
        de_ev = (eu.value - el.value) * HARTREE_EV
        tier = Fidelity.APPROXIMATION if fine_structure else Fidelity.EXACT
        prov = Provenance(
            fidelity=tier,
            method=(
                f"I took a level difference: [{eu.provenance.method}] minus "
                f"[{el.provenance.method}]; photon lambda = hc/dE (vacuum)"
            ),
            assumptions=eu.provenance.assumptions
            + ("I apply the electric-dipole selection rules (Delta l = +/-1"
               + (", Delta j in {0, +/-1})" if fine_structure else ")"),),
            error_estimate=(
                None if eu.provenance.error_estimate is None
                else (eu.provenance.error_estimate
                      + (el.provenance.error_estimate or 0.0)) * HARTREE_EV
            ),
            refinement=eu.provenance.refinement,
        )
        label = f"{nu}->{nl}"
        a_coeff = f_value = None
        if intensities:
            kw = {"Z": system.Z, "mu_ratio": system.mu_ratio.value}
            if fine_structure:
                # I pass the real fine-structure energy: a within-n component
                # like 2p_3/2 -> 2s_1/2 has no gross difference, and A goes as
                # dE^3.
                dE_h = eu.value - el.value
                a_coeff = einstein_A_fine(nu, lu, ju, nl, ll_, jl, dE_hartree=dE_h, **kw)
                f_value = oscillator_strength_fine(
                    nl, ll_, jl, nu, lu, ju, dE_hartree=dE_h, **kw
                )
            else:
                a_coeff = einstein_A(nu, lu, nl, ll_, **kw)
                f_value = oscillator_strength(nl, ll_, nu, lu, **kw)
        eps = None
        lower_frac = None
        if state is not None:
            neutral = 1.0 - state.ionized_fraction.value
            # This is a population and not a strength: I have it as soon as I
            # have conditions, whether or not anyone wanted my dipole integrals.
            lower_frac = level_column_fraction(occupation[(nl, ll_, jl)], neutral)
            if a_coeff is not None:
                eps = line_emissivity(
                    upper_fraction=occupation[(nu, lu, ju)],
                    neutral_fraction=neutral,
                    einstein_a=a_coeff.value,
                    photon_energy_ev=de_ev,
                )
        lines.append(
            SpectralLine(
                n_upper=nu, l_upper=lu, j_upper=ju,
                n_lower=nl, l_lower=ll_, j_lower=jl,
                energy=Quantity(de_ev, "eV", f"dE {label}", prov),
                wavelength=Quantity(_EV_NM / de_ev, "nm (vacuum)", f"lambda {label}", prov),
                einstein_a=a_coeff,
                oscillator_strength=f_value,
                emissivity=eps,
                lower_fraction=lower_frac,
            )
        )
    lines.sort(key=lambda ln: ln.wavelength.value)
    return LineList(
        system_key=system.key,
        n_max=n_max,
        fine_structure=fine_structure,
        lines=tuple(lines),
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION if fine_structure else Fidelity.EXACT,
            method="I took the dipole-allowed level differences (see my per-line provenance)",
            assumptions=("I list emission lines only (E_upper > E_lower)",
                         "I report vacuum wavelengths in nm and energies in eV"),
        ),
        thermal=state,
    )


def _screened_thermal(result, levels: list, thermal: ThermalConditions):
    """My population levels for a screened atom, with a Koopmans ionization
    energy.

    I take chi as the binding energy of the outermost *occupied* orbital, which
    is Koopmans' theorem: it assumes the remaining orbitals do not relax when
    the electron leaves. That is a further approximation of mine stacked on a
    GSZ model already only good to a few percent on valence energies, so I name
    it in my assumptions rather than presenting it as the ionization energy.
    """
    ground_h = min(e.value for _, _, e in levels)
    specs = tuple(
        Level(
            n=n, label=f"{n},{l}",
            energy_ev=(e.value - ground_h) * HARTREE_EV,
            degeneracy=level_degeneracy(l, None),
        )
        for n, l, e in levels
    )
    keys = tuple((n, l) for n, l, _ in levels)
    occupied = [o for o in result.orbitals if o.occupancy > 0]
    # My outermost occupied is the least bound, so the highest energy occupied.
    chi_ev = -max(o.energy.value for o in occupied) * HARTREE_EV
    return _thermal_state(
        specs, keys, chi_ev, thermal,
        chi_assumptions=(
            f"I took the ionization energy chi = {chi_ev:.4g} eV from Koopmans' "
            "theorem (the binding energy of my outermost occupied GSZ orbital, "
            "with no relaxation of the remaining orbitals), stacked on a "
            "screening model already only good to a few percent on valence "
            "energies",
        ),
    )


def screened_transition_lines(
    result, intensities: bool = False, thermal: ThermalConditions | None = None
) -> LineList:
    """The dipole-allowed emission lines I find among a screened atom's orbital
    energies.

    `result` is a screened_atom.ScreenedAtomResult. I leave it untyped here to
    avoid a circular import. My lines are (n_u, l_u) -> (n_l, l_l) with
    Delta l = +/-1 and E_upper > E_lower; energies in eV, vacuum wavelengths in
    nm, all APPROXIMATION.

    With `intensities`, each line also carries an Einstein A and an oscillator
    strength I built from a dipole integral over the radials I solved
    numerically. With `thermal`, it also carries an LTE emissivity, whose
    ionization step leans on a Koopmans estimate of chi (see
    `_screened_thermal`).
    """
    from atomsim.screened_atom import screened_dipole_integral  # circular at module scope

    levels = [(o.n, o.l, o.energy) for o in result.orbitals]
    # I use one box for the whole list, sized by its most extended state, so
    # every line reuses the same handful of l channels I solved.
    n_box = max((n for n, _, _ in levels), default=1)
    if thermal is not None:
        intensities = True
    state, occupation = (
        _screened_thermal(result, levels, thermal) if thermal else (None, {})
    )
    lines: list[SpectralLine] = []
    for (nu, lu, eu), (nl, ll_, el) in itertools.permutations(levels, 2):
        if eu.value <= el.value or abs(lu - ll_) != 1:
            continue
        de_ev = (eu.value - el.value) * HARTREE_EV
        prov = Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                f"I took a screened orbital difference: [{eu.provenance.method}] "
                f"minus [{el.provenance.method}]; photon lambda = hc/dE (vacuum)"
            ),
            assumptions=eu.provenance.assumptions
            + ("I apply the electric-dipole selection rule (Delta l = +/-1)",),
            error_estimate=(
                None if eu.provenance.error_estimate is None
                else (eu.provenance.error_estimate + (el.provenance.error_estimate or 0.0))
                * HARTREE_EV
            ),
        )
        label = f"{orbital_label(nu, lu)}->{orbital_label(nl, ll_)}"
        a_coeff = f_value = None
        if intensities:
            dE_h = eu.value - el.value
            R = screened_dipole_integral(
                result.z, result.n_electrons, nl, ll_, nu, lu, n_box=n_box
            )
            a_val = A_from_radial_dipole(dE_h, lu, ll_, R.value)
            f_val = f_from_radial_dipole(dE_h, ll_, lu, R.value)
            a_coeff = Quantity(
                a_val, "s^-1", f"A {label}",
                _strength_provenance(
                    R, "A = (4/3) alpha^3 dE^3 (l_max/(2l'+1)) |R|^2 / t_au", a_val
                ),
            )
            f_value = Quantity(
                f_val, "dimensionless", f"f {label}",
                _strength_provenance(R, "f = (2/3) dE (l_max/(2l+1)) |R|^2", f_val),
            )
        eps = None
        lower_frac = None
        if state is not None:
            neutral = 1.0 - state.ionized_fraction.value
            lower_frac = level_column_fraction(occupation[(nl, ll_)], neutral)
            if a_coeff is not None:
                eps = line_emissivity(
                    upper_fraction=occupation[(nu, lu)],
                    neutral_fraction=neutral,
                    einstein_a=a_coeff.value,
                    photon_energy_ev=de_ev,
                )
        lines.append(SpectralLine(
            n_upper=nu, l_upper=lu, j_upper=None, n_lower=nl, l_lower=ll_, j_lower=None,
            energy=Quantity(de_ev, "eV", f"dE {label}", prov),
            wavelength=Quantity(_EV_NM / de_ev, "nm (vacuum)", f"lambda {label}", prov),
            einstein_a=a_coeff,
            oscillator_strength=f_value,
            emissivity=eps,
            lower_fraction=lower_frac,
        ))
    lines.sort(key=lambda ln: ln.wavelength.value)
    return LineList(
        system_key=result.key,
        n_max=max((o.n for o in result.orbitals), default=1),
        fine_structure=False, lines=tuple(lines),
        thermal=state,
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                "I took the dipole-allowed screened orbital differences "
                "(see my per-line provenance)"
            ),
            assumptions=("I list emission lines only (E_upper > E_lower)",
                         "my transition energies are independent-particle"),
        ),
    )


@dataclass(frozen=True)
class ReferenceLine:
    wavelength_nm: float
    uncertainty_nm: float | None
    label: str


@dataclass(frozen=True)
class ReferenceData:
    species: str
    citation: str
    retrieved: str
    medium: str
    lines: tuple[ReferenceLine, ...]


@dataclass(frozen=True)
class LineComparison:
    line: SpectralLine
    reference_nm: float
    reference_uncertainty_nm: float | None
    delta_nm: float
    relative_error: float
    within_tolerance: bool


def load_reference(system_key: str) -> ReferenceData | None:
    """My vendored NIST reference for a preset, or None. I never query live."""
    filename = _REFERENCE_FILES.get(system_key)
    if filename is None:
        return None
    ref = resources.files("atomsim.data").joinpath(filename)
    if not ref.is_file():
        return None
    raw = json.loads(ref.read_text(encoding="utf-8"))
    return ReferenceData(
        species=raw["species"],
        citation=raw["citation"],
        retrieved=raw["retrieved"],
        medium=raw["medium"],
        lines=tuple(
            ReferenceLine(
                wavelength_nm=ln["wavelength_nm"],
                uncertainty_nm=ln.get("uncertainty_nm"),
                label=ln.get("label", ""),
            )
            for ln in raw["lines"]
        ),
    )


def compare_lines(
    line_list: LineList,
    reference: ReferenceData,
    tolerance_relative: float | None = None,
    window_relative: float = 0.01,
) -> tuple[LineComparison, ...]:
    """I match each reference line to my nearest computed line and report the
    residuals.

    I keep two separate scales. `window_relative` decides whether a reference
    line's transition is present in my computed set at all, which is a coarse
    association cut; `tolerance_relative` decides whether the matched pair
    passes, which is my disclosed accuracy bar and feeds only the
    within_tolerance flag. They differ for my approximate models: a GSZ valence
    line may sit several percent off the real wavelength and still be the
    correct transition, so I keep it and report the residual, while a reference
    line with no computed transition near it I drop. The 0.01 default preserves
    my exact hydrogenic behavior.
    """
    tol = tolerance_relative if tolerance_relative is not None else _DEFAULT_TOL[
        line_list.fine_structure
    ]
    out: list[LineComparison] = []
    for ref in reference.lines:
        if not line_list.lines:
            break
        nearest = min(
            line_list.lines, key=lambda ln: abs(ln.wavelength.value - ref.wavelength_nm)
        )
        delta = nearest.wavelength.value - ref.wavelength_nm
        rel = abs(delta) / ref.wavelength_nm
        if rel > window_relative:
            continue  # I have no transition near this reference line, so it is not in my set
        out.append(
            LineComparison(
                line=nearest,
                reference_nm=ref.wavelength_nm,
                reference_uncertainty_nm=ref.uncertainty_nm,
                delta_nm=delta,
                relative_error=rel,
                within_tolerance=rel <= tol,
            )
        )
    return tuple(out)
