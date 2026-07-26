"""Validation for thermal emissivity attached to spectra.

The population physics is validated in test_populations.py. What is checked
here is the wiring: that the levels the populations run over are the same
levels the lines are built from, that the temperature moves the spectrum in the
directions it must, and that the ionization ceiling shows up as the whole
spectrum dimming rather than as a silently rescaled picture.
"""

import pytest

from atomsim.atoms import aufbau_configuration
from atomsim.populations import ThermalConditions
from atomsim.provenance import Fidelity
from atomsim.screened_atom import solve_screened_atom
from atomsim.spectra import screened_transition_lines, transition_lines
from atomsim.systems import get_system

H = get_system("h")
PHOTOSPHERE = 1e13  # cm^-3, roughly solar; the tests only rely on the order


def _find(lines, n_up, n_low):
    for ln in lines:
        if (ln.n_upper, ln.n_lower) == (n_up, n_low):
            return ln
    raise AssertionError(f"no line {n_up} -> {n_low}")


def _lines(t, ne=PHOTOSPHERE, n_max=6, **kw):
    return transition_lines(H, n_max=n_max, thermal=ThermalConditions(t, ne), **kw)


def test_emissivity_is_off_unless_conditions_are_given():
    ll = transition_lines(H, n_max=4, intensities=True)
    assert all(ln.emissivity is None for ln in ll.lines)
    assert ll.thermal is None


def test_thermal_implies_intensities():
    """Emissivity is built on A, so asking for one without the other would be
    incoherent rather than merely unsupported."""
    ll = _lines(10000.0, n_max=4)
    assert all(ln.einstein_a is not None for ln in ll.lines)
    assert all(ln.emissivity is not None for ln in ll.lines)


def test_the_line_list_carries_the_conditions_that_produced_it():
    ll = _lines(9000.0, ne=1e12)
    assert ll.thermal.conditions.temperature_k == 9000.0
    assert ll.thermal.conditions.electron_density_cm3 == 1e12
    assert 0.0 <= ll.thermal.ionized_fraction.value <= 1.0
    assert ll.thermal.partition_function.value >= 2.0


def test_balmer_gains_on_lyman_as_the_gas_heats():
    """The n=3 population rises faster than n=2, so the ratio climbs with T.

    Balmer never actually overtakes Lyman-alpha here, and should not: in an
    optically thin LTE gas Lyman-alpha has both the larger A and the more
    populated upper level. Real nebulae show Balmer because Lyman-alpha is
    optically thick, which is exactly the assumption this model discloses it
    does not make.
    """
    ratios = []
    for t in (4000.0, 8000.0, 15000.0, 30000.0):
        ll = _lines(t)
        ratios.append(
            _find(ll.lines, 3, 2).emissivity.value
            / _find(ll.lines, 2, 1).emissivity.value
        )
    assert all(a < b for a, b in zip(ratios, ratios[1:], strict=False)), ratios


def test_the_spectrum_dims_once_the_gas_is_ionized():
    """Past the ionization knee there are fewer neutrals to emit, so the total
    emissivity falls even though every level is more excited. A model that kept
    getting brighter would have dropped the Saha factor.
    """
    def total(t):
        return sum(ln.emissivity.value for ln in _lines(t).lines)

    assert total(9000.0) > total(4000.0), "should brighten before the knee"
    assert total(50000.0) < total(9000.0), "should dim after it"


def test_a_denser_gas_stays_neutral_and_therefore_brighter():
    hot = 12000.0
    thin = _lines(hot, ne=1e10)
    thick = _lines(hot, ne=1e16)
    assert thick.thermal.ionized_fraction.value < thin.thermal.ionized_fraction.value
    assert (
        _find(thick.lines, 2, 1).emissivity.value
        > _find(thin.lines, 2, 1).emissivity.value
    )


def test_emissivity_ordering_is_not_just_the_A_ordering():
    """If it were, the thermal layer would be doing nothing. The population
    weighting has to actually reorder some pair of lines.
    """
    ll = _lines(6000.0)
    by_a = [id(x) for x in sorted(ll.lines, key=lambda x: -x.einstein_a.value)]
    by_eps = [id(x) for x in sorted(ll.lines, key=lambda x: -x.emissivity.value)]
    assert by_a != by_eps


def test_wavelengths_and_rates_do_not_move_when_conditions_are_added():
    plain = transition_lines(H, n_max=5, intensities=True)
    warm = _lines(10000.0, n_max=5)
    assert [x.wavelength.value for x in plain.lines] == [
        x.wavelength.value for x in warm.lines
    ]
    assert [x.einstein_a.value for x in plain.lines] == [
        x.einstein_a.value for x in warm.lines
    ]


def test_emissivity_provenance_names_LTE_and_optical_thinness():
    eps = _find(_lines(10000.0, n_max=4).lines, 2, 1).emissivity
    assert eps.provenance.fidelity is Fidelity.APPROXIMATION
    assert any("LTE" in a for a in eps.provenance.assumptions)
    assert any("optically thin" in a.lower() for a in eps.provenance.assumptions)
    assert eps.unit == "eV/s per atom"


def test_fine_structure_components_carry_their_own_emissivity():
    ll = transition_lines(
        H, n_max=3, fine_structure=True,
        thermal=ThermalConditions(10000.0, PHOTOSPHERE),
    )
    assert ll.lines and all(x.emissivity is not None for x in ll.lines)
    assert all(x.emissivity.value >= 0.0 for x in ll.lines)


def test_the_within_n_microwave_lines_are_negligible_when_weighted():
    """The two-regime problem, seen from the physics side: those lines have
    A ~ 1e-12, so thermally weighted they are nothing, which is why the bar
    half of the display problem dissolves on its own.
    """
    ll = transition_lines(
        H, n_max=4, fine_structure=True,
        thermal=ThermalConditions(10000.0, PHOTOSPHERE),
    )
    within = [x for x in ll.lines if x.n_upper == x.n_lower]
    across = [x for x in ll.lines if x.n_upper != x.n_lower]
    assert within and across
    assert max(x.emissivity.value for x in within) < 1e-6 * max(
        x.emissivity.value for x in across
    )


# --- screened atoms -------------------------------------------------------


def _screened(key_z, t, ne=PHOTOSPHERE):
    config = aufbau_configuration(key_z)
    from atomsim.atoms import total_electrons

    result = solve_screened_atom(key_z, total_electrons(config), config)
    return screened_transition_lines(result, thermal=ThermalConditions(t, ne))


def test_screened_atoms_get_emissivity_too():
    ll = _screened(3, 8000.0)
    assert ll.lines and all(x.emissivity is not None for x in ll.lines)
    assert ll.thermal is not None


def test_screened_ionization_discloses_the_koopmans_estimate():
    """chi comes from Koopmans on a GSZ orbital, which is an approximation on
    an approximation. It must not look like a measured ionization energy.
    """
    ll = _screened(3, 8000.0)
    assumptions = ll.thermal.ionized_fraction.provenance.assumptions
    assert any("Koopmans" in a for a in assumptions)
    assert any("relaxation" in a for a in assumptions)


def test_lithium_ionizes_more_easily_than_hydrogen():
    """Li's 2s valence electron is bound by about 5.4 eV against hydrogen's
    13.6, so at the same conditions far more of it is ionized. This is the
    check that chi is actually being read from the atom rather than hardcoded.
    """
    t, ne = 6000.0, PHOTOSPHERE
    li = _screened(3, t, ne).thermal.ionized_fraction.value
    h = transition_lines(
        H, n_max=6, thermal=ThermalConditions(t, ne)
    ).thermal.ionized_fraction.value
    assert li > h
    assert li > 0.5 > h


def test_rejects_impossible_conditions():
    for bad in (ThermalConditions(0.0, 1e13), ThermalConditions(1e4, 0.0)):
        with pytest.raises(ValueError):
            transition_lines(H, n_max=3, thermal=bad)
