"""Validation for LTE level populations and ionization.

There is no closed-form ground truth to check this against the way the analytic
hydrogen engine anchors the numerical solver, because Boltzmann and Saha *are*
the model. What can be checked is that the model behaves the way statistical
mechanics says it must: sums that have to be exactly one, limits at high and low
temperature that have known values, and monotonic directions that cannot come
out backwards without something being wrong.

The one external anchor is deliberately coarse. Hydrogen at photospheric
densities is about half ionized somewhere near 10^4 K, and that is asserted as
an order of magnitude rather than a value, because the exact crossover moves
with the electron density that is a free control here.
"""

import math

import pytest

from atomsim.populations import (
    Level,
    boltzmann_fractions,
    hydrogen_levels,
    line_emissivity,
    partition_function,
    saha_ionization_fraction,
)
from atomsim.provenance import Fidelity

#: eV/K, written out by hand so this file checks the formula against an
#: independent constant rather than against the module's own. It is a truncated
#: CODATA value, good to about 2e-11 relative, which sets the tolerance the
#: by-hand test below can honestly ask for. The truncation itself is checked
#: against scipy at the bottom of the file.
K_EV = 8.617333262e-5


def two_level(gap_ev=1.0, g0=1, g1=3):
    return (
        Level(n=1, label="ground", energy_ev=0.0, degeneracy=g0),
        Level(n=2, label="excited", energy_ev=gap_ev, degeneracy=g1),
    )


# --- Boltzmann ------------------------------------------------------------


@pytest.mark.parametrize("t", [300.0, 5000.0, 20000.0, 1e6])
def test_fractions_sum_to_one_at_every_temperature(t):
    fractions = boltzmann_fractions(hydrogen_levels(n_max=6), t)
    assert sum(f.value for f in fractions) == pytest.approx(1.0, rel=1e-12)


def test_a_two_level_system_matches_the_hand_computed_ratio():
    """The one place the arithmetic is simple enough to check by hand."""
    t = 10000.0
    levels = two_level(gap_ev=1.0, g0=1, g1=3)
    f = boltzmann_fractions(levels, t)
    expected = 3.0 * math.exp(-1.0 / (K_EV * t))
    assert f[1].value / f[0].value == pytest.approx(expected, rel=1e-9)


def test_low_temperature_puts_everything_in_the_ground_level():
    f = boltzmann_fractions(hydrogen_levels(n_max=6), 300.0)
    assert f[0].value == pytest.approx(1.0, abs=1e-12)
    assert all(x.value < 1e-100 for x in f[1:])


def test_high_temperature_approaches_the_degeneracy_ratio():
    """With kT far above every gap the exponentials all go to 1, so the
    populations are pure statistical weight."""
    levels = hydrogen_levels(n_max=4)
    f = boltzmann_fractions(levels, 1e9)
    total_g = sum(x.degeneracy for x in levels)
    for level, frac in zip(levels, f, strict=True):
        assert frac.value == pytest.approx(level.degeneracy / total_g, rel=1e-3)


def test_excited_fraction_rises_monotonically_with_temperature():
    levels = hydrogen_levels(n_max=6)
    n2 = [boltzmann_fractions(levels, t)[1].value for t in (3000, 6000, 10000, 20000)]
    assert all(a < b for a, b in zip(n2, n2[1:], strict=False)), n2


def test_rejects_a_non_positive_temperature():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="temperature"):
            boltzmann_fractions(hydrogen_levels(n_max=3), bad)


# --- degeneracies ---------------------------------------------------------


@pytest.mark.parametrize("n_max", [1, 3, 6])
def test_gross_degeneracies_sum_to_two_n_squared_per_shell(n_max):
    levels = hydrogen_levels(n_max=n_max)
    for n in range(1, n_max + 1):
        g = sum(x.degeneracy for x in levels if x.n == n)
        assert g == 2 * n * n, f"shell n={n}"


@pytest.mark.parametrize("n_max", [1, 3, 6])
def test_fine_structure_degeneracies_also_sum_to_two_n_squared(n_max):
    levels = hydrogen_levels(n_max=n_max, fine_structure=True)
    for n in range(1, n_max + 1):
        g = sum(x.degeneracy for x in levels if x.n == n)
        assert g == 2 * n * n, f"shell n={n}"


# --- partition function ---------------------------------------------------


def test_partition_function_is_the_ground_degeneracy_when_cold():
    u = partition_function(hydrogen_levels(n_max=6), 300.0)
    assert u.value == pytest.approx(2.0, rel=1e-9)


def test_partition_function_discloses_its_truncation():
    """The sum diverges. A number quoted without its cutoff would be a lie by
    omission, so the cutoff rides in the assumptions.
    """
    u = partition_function(hydrogen_levels(n_max=6), 10000.0)
    assert u.provenance.fidelity is Fidelity.APPROXIMATION
    assert any("truncat" in a.lower() for a in u.provenance.assumptions)
    assert any("n_max=6" in a for a in u.provenance.assumptions)
    assert u.provenance.refinement


def test_truncation_matters_more_at_high_temperature():
    """This is why the cutoff is disclosed rather than chosen quietly: at low T
    the tail is irrelevant, at high T it dominates.
    """
    def spread(t):
        small = partition_function(hydrogen_levels(n_max=3), t).value
        large = partition_function(hydrogen_levels(n_max=12), t).value
        return large / small

    assert spread(3000.0) == pytest.approx(1.0, rel=1e-6)
    assert spread(50000.0) > 1.5


# --- Saha -----------------------------------------------------------------


def test_ionization_is_a_fraction():
    for t in (3000.0, 10000.0, 50000.0):
        x = saha_ionization_fraction(t, electron_density_cm3=1e13, chi_ev=13.6)
        assert 0.0 <= x.value <= 1.0


def test_ionization_rises_with_temperature():
    xs = [
        saha_ionization_fraction(t, 1e13, 13.6).value
        for t in (5000, 8000, 12000, 20000)
    ]
    assert all(a < b for a, b in zip(xs, xs[1:], strict=False)), xs


def test_ionization_falls_with_electron_density():
    """More free electrons drive recombination, so a denser gas stays neutral
    to a higher temperature."""
    xs = [
        saha_ionization_fraction(10000.0, ne, 13.6).value
        for ne in (1e10, 1e13, 1e16, 1e19)
    ]
    assert all(a > b for a, b in zip(xs, xs[1:], strict=False)), xs


def test_hydrogen_is_about_half_ionized_near_ten_thousand_kelvin():
    """The one external anchor, asserted as an order of magnitude.

    At a photospheric electron density the crossover sits in the thousands of
    kelvin, not the hundreds and not the hundreds of thousands. The exact value
    moves with n_e, which is a free control here, so it is not pinned.
    """
    def x(t):
        return saha_ionization_fraction(t, 1e13, 13.6).value

    assert x(3000.0) < 0.01
    assert x(30000.0) > 0.99
    lo, hi = 3000.0, 30000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if x(mid) < 0.5:
            lo = mid
        else:
            hi = mid
    assert 5e3 < lo < 2e4, f"half-ionization at {lo:.0f} K"


def test_the_half_ionization_temperature_rises_with_density():
    def t_half(ne):
        lo, hi = 1000.0, 200000.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if saha_ionization_fraction(mid, ne, 13.6).value < 0.5:
                lo = mid
            else:
                hi = mid
        return lo

    temps = [t_half(ne) for ne in (1e10, 1e13, 1e16)]
    assert all(a < b for a, b in zip(temps, temps[1:], strict=False)), temps


def test_saha_says_it_is_not_self_consistent():
    """n_e is an independent knob here, not solved with the ionization it
    drives. That is a real departure from an equilibrium gas and it has to be
    on the record.
    """
    x = saha_ionization_fraction(10000.0, 1e13, 13.6)
    assert x.provenance.fidelity is Fidelity.APPROXIMATION
    assert any("self-consist" in a.lower() for a in x.provenance.assumptions)
    assert any("LTE" in a for a in x.provenance.assumptions)


def test_rejects_a_non_positive_density():
    with pytest.raises(ValueError, match="density"):
        saha_ionization_fraction(10000.0, 0.0, 13.6)


# --- emissivity -----------------------------------------------------------


def test_emissivity_is_the_product_of_its_three_factors():
    eps = line_emissivity(
        upper_fraction=0.01, neutral_fraction=0.5, einstein_a=1e8, photon_energy_ev=10.2
    )
    assert eps.value == pytest.approx(0.01 * 0.5 * 1e8 * 10.2)
    assert "eV" in eps.unit and "s" in eps.unit


def test_emissivity_vanishes_in_a_fully_ionized_gas():
    """No neutrals left, no bound-bound emission. A zero here is physics, and
    it is reported with the reason attached rather than as a bare 0.0.
    """
    eps = line_emissivity(0.01, 0.0, 1e8, 10.2)
    assert eps.value == 0.0
    assert any("ioniz" in a.lower() for a in eps.provenance.assumptions)


def test_emissivity_carries_the_optically_thin_assumption():
    eps = line_emissivity(0.01, 0.5, 1e8, 10.2)
    assert eps.provenance.fidelity is Fidelity.APPROXIMATION
    assert any("optically thin" in a.lower() for a in eps.provenance.assumptions)


# --- the constant ---------------------------------------------------------


def test_boltzmann_constant_matches_scipy():
    """K_EV above is hand-written for the by-hand ratio test; if it drifts from
    CODATA that test silently checks the wrong thing."""
    from scipy import constants as sc

    assert K_EV == pytest.approx(
        sc.physical_constants["Boltzmann constant in eV/K"][0], rel=1e-9
    )
