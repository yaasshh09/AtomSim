"""How I turn an emission rate into a line intensity: Boltzmann, Saha, emissivity.

This is the layer where I stop talking about one atom and start talking about a
gas. An Einstein A tells me how fast an atom in the upper level decays; it tells
me nothing about how many atoms are up there, which is what decides whether you
actually see the line. Boltzmann answers that for a gas at one temperature, and
Saha answers the prior question of whether I have any neutral atom left to do
the emitting.

Everything here is APPROXIMATION and my assumptions are the error bar. Local
thermodynamic equilibrium is a strong claim: real nebulae are not in LTE, and
neither is a discharge lamp. My arithmetic given the model is exact, so I invent
no error_estimate for it, a tight number on a model this schematic would be
worse than none.

See docs/specs/2026-07-26-phase17-population-modelling-design.md.
"""

import math
from dataclasses import dataclass

from scipy import constants as _sc

from atomsim.provenance import Fidelity, Provenance, Quantity

__all__ = [
    "Level",
    "ThermalConditions",
    "ThermalState",
    "boltzmann_fractions",
    "hydrogen_levels",
    "level_column_fraction",
    "level_degeneracy",
    "line_emissivity",
    "partition_function",
    "saha_ionization_fraction",
]

#: Boltzmann constant in eV/K, the unit I keep every energy in here.
K_EV: float = _sc.physical_constants["Boltzmann constant in eV/K"][0]

#: (2 pi m_e k / h^2) in m^-2 K^-1. The Saha right-hand side is this times T,
#: all to the 3/2. I keep it as one constant so I apply the exponent once, in
#: one place, to a quantity whose units are written down.
_SAHA_COEFF: float = 2.0 * math.pi * _sc.m_e * _sc.k / (_sc.h**2)

#: Spectroscopic letters, skipping j by convention. Long enough for any n_max
#: I accept; past it my label falls back to the number, because a wrong letter
#: would be worse than an honest "l=20".
_L_SYMBOLS = "spdfghiklmnoqrtuv"


def _l_symbol(l: int) -> str:
    return _L_SYMBOLS[l] if l < len(_L_SYMBOLS) else f"l={l}"


#: Shared by everything I do here: the model, not my arithmetic, is the error.
_LTE = (
    "LTE (local thermodynamic equilibrium): one temperature sets both the "
    "level populations and the ionization",
    "optically thin: no radiative transfer, no self-absorption, no escape "
    "probability",
)


@dataclass(frozen=True)
class Level:
    """A level I can put atoms in.

    I measure `energy_ev` up from the ground level, so it is >= 0 and the
    Boltzmann factor never overflows. I carry `n` only so I can name the
    partition function's truncation in the provenance.
    """

    n: int
    label: str
    energy_ev: float
    degeneracy: int


@dataclass(frozen=True)
class ThermalConditions:
    """My two knobs. I keep them together so you cannot hand me half of them."""

    temperature_k: float
    electron_density_cm3: float


@dataclass(frozen=True)
class ThermalState:
    """What your conditions produced, which I carry alongside a line list.

    The view needs all of this to be honest about what it draws: how much of the
    gas is even neutral, and what I got for the partition function with its
    cutoff attached.
    """

    conditions: ThermalConditions
    ionized_fraction: Quantity
    partition_function: Quantity


def level_degeneracy(l: int, j: float | None) -> int:
    """Statistical weight of a level: 2j+1 when j is resolved, else 2(2l+1).

    One definition, which I share between the hydrogen constructor here and
    whatever builds levels from a system's own energies, so the two schemes
    cannot drift apart. Both sum to 2n^2 per shell.
    """
    return int(round(2.0 * j + 1.0)) if j is not None else 2 * (2 * l + 1)


def hydrogen_levels(n_max: int, fine_structure: bool = False) -> tuple[Level, ...]:
    """The hydrogen levels I model up to `n_max`, with their statistical weights.

    I use 2(2l+1) for a gross (n, l) sublevel and 2j+1 for a fine-structure
    (n, l, j) one. Both sum to 2n^2 per shell, which is my arithmetic check and
    the tests assert it.

    I use the Bohr energies relative to the ground state. Fine structure shifts
    them by O(alpha^2), which is far below kT at any temperature where the
    populations differ from 0 or 1, so I use the same energies for both
    schemes; what fine structure changes here is how I split the weight.
    """
    if n_max < 1:
        raise ValueError(f"n_max must be >= 1, got {n_max}")
    ionization_ev = _sc.physical_constants["Rydberg constant times hc in eV"][0]
    levels: list[Level] = []
    for n in range(1, n_max + 1):
        energy = ionization_ev * (1.0 - 1.0 / (n * n))
        for l in range(n):
            if fine_structure:
                for j in ([0.5] if l == 0 else [l - 0.5, l + 0.5]):
                    levels.append(Level(
                        n=n, label=f"{n}{_l_symbol(l)}{j:g}",
                        energy_ev=energy, degeneracy=level_degeneracy(l, j),
                    ))
            else:
                levels.append(Level(
                    n=n, label=f"{n}{_l_symbol(l)}",
                    energy_ev=energy, degeneracy=level_degeneracy(l, None),
                ))
    return tuple(levels)


def _check_temperature(temperature_k: float) -> None:
    if temperature_k <= 0.0:
        raise ValueError(f"temperature must be > 0 K, got {temperature_k}")


def _truncation_note(levels: tuple[Level, ...]) -> tuple[str, ...]:
    n_max = max(x.n for x in levels)
    return (
        f"I truncated the partition function at n_max={n_max}: the exact sum "
        "diverges, since infinitely many bound states crowd the ionization "
        "limit carrying unbounded statistical weight",
    )


def partition_function(levels: tuple[Level, ...], temperature_k: float) -> Quantity:
    """My U(T) = sum g_i exp(-E_i / kT), cut off at the level list's own n_max.

    The exact sum does not converge. Its terms tend to g_n exp(-chi/kT), which
    grows as 2n^2, so my answer depends on where I stop, and the cutoff is part
    of the number rather than an implementation detail of mine.

    The physical resolution is that a plasma at finite density has no n = 100
    states to occupy: neighbouring ions blur them into the continuum. My n_max
    is a crude stand-in for that cutoff, and my refinement says what would
    replace it.
    """
    _check_temperature(temperature_k)
    kt = K_EV * temperature_k
    value = sum(x.degeneracy * math.exp(-x.energy_ev / kt) for x in levels)
    return Quantity(
        value=value,
        unit="dimensionless",
        label=f"U(T={temperature_k:g} K)",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="I summed U = sum_i g_i exp(-E_i / kT) over the levels you gave me",
            assumptions=_LTE + _truncation_note(levels),
            refinement=(
                "an occupation-probability treatment (pressure ionization / "
                "lowering of the ionization potential) would hand me a physical "
                "cutoff instead of n_max"
            ),
        ),
    )


def boltzmann_fractions(
    levels: tuple[Level, ...], temperature_k: float
) -> tuple[Quantity, ...]:
    """The share of the *neutral* population I put in each level, one per level.

    These sum to exactly 1 by construction: they are shares of the neutrals, not
    of the whole gas. What fraction of the gas is neutral is Saha's job, and I
    combine the two in `line_emissivity`.
    """
    _check_temperature(temperature_k)
    kt = K_EV * temperature_k
    weights = [x.degeneracy * math.exp(-x.energy_ev / kt) for x in levels]
    total = sum(weights)
    truncation = _truncation_note(levels)
    return tuple(
        Quantity(
            value=w / total,
            unit="dimensionless",
            label=f"N({x.label})/N_neutral at {temperature_k:g} K",
            provenance=Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method="I used Boltzmann: N_i/N = g_i exp(-E_i / kT) / U(T)",
                assumptions=_LTE + truncation,
                refinement=(
                    "non-LTE level kinetics (collisional-radiative balance) "
                    "would replace my single temperature"
                ),
            ),
        )
        for x, w in zip(levels, weights, strict=True)
    )


def saha_ionization_fraction(
    temperature_k: float,
    electron_density_cm3: float,
    chi_ev: float,
    u_neutral: float = 2.0,
    u_ion: float = 1.0,
) -> Quantity:
    """The fraction of the element I find ionized, in [0, 1].

        n_II / n_I = (2 U_II / U_I) (2 pi m_e k T / h^2)^(3/2) exp(-chi/kT) / n_e

    I default `u_ion` to 1 because removing the one electron from a
    hydrogen-like atom leaves a bare nucleus, which has a single state.

    I take the electron density as an independent control here, and I do not
    solve it self-consistently with the ionization that produces it. That is a
    real departure from an equilibrium gas: you can dial in a (T, n_e) pair that
    no single gas would hold. It is a deliberate lab knob and I put it on the
    record in the provenance.
    """
    _check_temperature(temperature_k)
    if electron_density_cm3 <= 0.0:
        raise ValueError(
            f"electron density must be > 0 cm^-3, got {electron_density_cm3}"
        )
    kt = K_EV * temperature_k
    n_e_m3 = electron_density_cm3 * 1e6
    # exp() underflows to 0.0 for a cold gas, which is the right answer here
    # (nothing ionized) rather than something I need to guard against.
    ratio = (
        2.0 * (u_ion / u_neutral)
        * (_SAHA_COEFF * temperature_k) ** 1.5
        * math.exp(-chi_ev / kt)
        / n_e_m3
    )
    return Quantity(
        value=ratio / (1.0 + ratio),
        unit="dimensionless",
        label=f"ionized fraction at {temperature_k:g} K, n_e={electron_density_cm3:g} cm^-3",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                "I used Saha: n_II n_e / n_I = 2 (U_II/U_I) "
                "(2 pi m_e k T / h^2)^(3/2) exp(-chi/kT)"
            ),
            assumptions=_LTE + (
                "you control the electron density yourself: I do NOT solve it "
                "self-consistently with the ionization it drives",
                f"I used chi = {chi_ev:g} eV, U_ion = {u_ion:g}, "
                f"U_neutral = {u_neutral:g}",
                "ideal gas: no Coulomb interaction between the charged species, "
                "no lowering of the ionization potential",
                "I model a single ionization stage only",
            ),
            refinement=(
                "solving n_e together with the ionization, and lowering chi for "
                "the plasma environment, would remove both of my idealizations"
            ),
        ),
    )


def level_column_fraction(
    lower_fraction: float,
    neutral_fraction: float,
) -> Quantity:
    """The share of all atoms of the element I put in a transition's lower level.

        N_l / N_element = (1 - x) * (N_l / N_neutral)

    This is the absorption twin of my `line_emissivity`, and the one number I
    cannot build an absorption spectrum without. A line only absorbs out of the
    level it starts in, so a total column density says nothing on its own: the
    same cloud is opaque in Lyman-alpha and transparent in Balmer-alpha purely
    because this fraction differs by orders of magnitude between 1s and 2s.

    I normalize per atom of the *element* for the same reason I do it in the
    emissivity: it makes a column density a property of the gas rather than of
    how ionized it happens to be, so you can hand one N to every line I list.
    """
    value = neutral_fraction * lower_fraction
    return Quantity(
        value=value,
        unit="dimensionless",
        label="lower-level fraction",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="I took N_l/N_element = (1 - x) (N_l/N_neutral)",
            assumptions=_LTE + (
                f"I took the neutral fraction 1 - x = {neutral_fraction:.4g} "
                "from Saha; a fully ionized gas has no bound electrons left to "
                "absorb",
                "I count per atom of the element (neutral + ionized), so a "
                "column density counts the whole element and this fraction "
                "selects the absorbing level",
            ),
            refinement=(
                "departures from LTE would set the lower level's population "
                "independently of the temperature I use for the profile width"
            ),
        ),
    )


def line_emissivity(
    upper_fraction: float,
    neutral_fraction: float,
    einstein_a: float,
    photon_energy_ev: float,
) -> Quantity:
    """The energy I radiate in one line, per second, per atom of the element.

        eps = (1 - x) * (N_u / N_neutral) * A * h nu

    I give this a dimension on purpose. A 0-to-1 "relative intensity" would hide
    that my whole spectrum dims as the gas ionizes, which is half of what the
    density control exists to show.

    I count per atom of the *element*, neutral and ionized together, so my
    number is comparable across temperatures: the denominator does not move when
    the gas ionizes.
    """
    value = neutral_fraction * upper_fraction * einstein_a * photon_energy_ev
    return Quantity(
        value=value,
        unit="eV/s per atom",
        label="line emissivity",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="I took eps = (1 - x) (N_u/N_neutral) A h nu",
            assumptions=_LTE + (
                f"I took the neutral fraction 1 - x = {neutral_fraction:.4g} "
                "from Saha; a fully ionized gas emits no bound-bound lines at "
                "all",
                "I count per atom of the element (neutral + ionized), so my "
                "denominator is fixed as the gas ionizes",
                "I count spontaneous emission only: no stimulated emission, no "
                "absorption back out of the beam",
            ),
            refinement=(
                "radiative transfer through a finite optical depth would turn "
                "this number of mine into a predicted observed brightness"
            ),
        ),
    )
