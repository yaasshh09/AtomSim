"""Validation for line strengths on screened (GSZ) atoms.

The tight anchor for the numerical dipole engine is the hydrogenic limit, and
it lives in test_numeric_dipole.py: fed a bare Coulomb potential the same code
reproduces the exact closed-form integrals. What is checked here is the layer
above it, where a real screened atom is involved and GSZ model error dominates.

So these assertions are deliberately loose. GSZ is a one-parameter fit to
Hartree-Fock potentials, not a correlated calculation, and a tight bound on an
alkali oscillator strength would be a claim the model cannot support. What can
be asserted is structure: every allowed line gets a positive finite rate, the
alkali resonance lines come out at the order-unity strength that makes them
resonance lines, a Rydberg series falls off monotonically, and the provenance
names both error sources instead of just the grid.
"""

import numpy as np
import pytest

from atomsim.atoms import atom_for_key, aufbau_configuration, total_electrons
from atomsim.provenance import Fidelity
from atomsim.screened_atom import screened_dipole_integral, solve_screened_atom
from atomsim.spectra import screened_transition_lines


def _lines(key: str, intensities: bool = True):
    element = atom_for_key(key)
    config = aufbau_configuration(element.z)
    result = solve_screened_atom(element.z, total_electrons(config), config)
    return screened_transition_lines(result, intensities=intensities)


@pytest.fixture(scope="module")
def helium():
    return _lines("he")


@pytest.fixture(scope="module")
def lithium():
    return _lines("li")


@pytest.fixture(scope="module")
def sodium():
    return _lines("na")


def _find(lines, n_up, l_up, n_low, l_low):
    for ln in lines:
        if (ln.n_upper, ln.l_upper, ln.n_lower, ln.l_lower) == (n_up, l_up, n_low, l_low):
            return ln
    raise AssertionError(f"no line {n_up}{l_up} -> {n_low}{l_low} in list")


def test_strengths_are_off_by_default():
    ll = _lines("he", intensities=False)
    assert ll.lines
    assert all(ln.einstein_a is None for ln in ll.lines)
    assert all(ln.oscillator_strength is None for ln in ll.lines)


def test_no_apology_note_once_the_strengths_are_real(helium):
    """Phase 16 closed the gap the note used to describe, so the note has to go.

    A stale note claiming the dipole integral is hydrogenic-only would now be
    the model lying about itself, which is the one thing it may not do.
    """
    assert helium.intensity_note is None


@pytest.mark.parametrize("key", ["he", "li", "na"])
def test_every_allowed_line_gets_a_positive_finite_rate(key, request):
    ll = request.getfixturevalue({"he": "helium", "li": "lithium", "na": "sodium"}[key])
    assert ll.lines
    for ln in ll.lines:
        assert ln.einstein_a is not None, f"{ln.n_upper}{ln.l_upper} has no A"
        assert np.isfinite(ln.einstein_a.value) and ln.einstein_a.value > 0.0
        assert np.isfinite(ln.oscillator_strength.value)
        assert ln.oscillator_strength.value > 0.0


def test_alkali_resonance_lines_have_order_unity_oscillator_strength(lithium, sodium):
    """Li 2p->2s and Na 3p->3s are one-electron resonance transitions, so f is
    order unity rather than order 0.01.

    The band is wide on purpose. Measured here: Li 0.725, Na 0.956, against
    literature values near 0.75 and 0.96. Those literature numbers are not
    asserted, because a few-percent agreement from a one-parameter screening
    model is partly luck and a tightened bound would be a false claim about
    what GSZ can deliver.
    """
    li = _find(lithium.lines, 2, 1, 2, 0)
    na = _find(sodium.lines, 3, 1, 3, 0)
    assert 0.4 < li.oscillator_strength.value < 1.5
    assert 0.4 < na.oscillator_strength.value < 1.5


def test_the_resonance_line_is_the_strongest_valence_line(sodium):
    """Not the strongest line outright: A goes as dE^3, so in an
    independent-particle orbital spectrum the core transitions (np -> 1s, keV
    scale) beat every valence line by orders of magnitude. Among lines that end
    above the closed core, the resonance line does win.
    """
    valence = [ln for ln in sodium.lines if ln.n_lower > 2]
    strongest = max(valence, key=lambda ln: ln.einstein_a.value)
    assert (strongest.n_upper, strongest.l_upper) == (3, 1)
    assert (strongest.n_lower, strongest.l_lower) == (3, 0)


@pytest.mark.parametrize("key", ["he", "li", "na"])
def test_oscillator_strength_falls_along_a_rydberg_series(key, request):
    """np -> 1s: the upper states get more diffuse with n, the overlap with the
    compact 1s shrinks, so f falls monotonically. A model that got this
    backwards would be wrong about the wavefunctions, not just their scale.
    """
    ll = request.getfixturevalue({"he": "helium", "li": "lithium", "na": "sodium"}[key])
    series = sorted(
        (ln for ln in ll.lines if (ln.n_lower, ln.l_lower, ln.l_upper) == (1, 0, 1)),
        key=lambda ln: ln.n_upper,
    )
    assert len(series) >= 3, "need a series to test a trend"
    f = [ln.oscillator_strength.value for ln in series]
    assert all(a > b for a, b in zip(f, f[1:], strict=False)), f


def test_provenance_names_both_error_sources_not_just_the_grid(sodium):
    """APPROXIMATION, not NUMERICAL. Tagging it by its discretization alone
    would advertise the smaller of the two errors as the only one.
    """
    a = _find(sodium.lines, 3, 1, 3, 0).einstein_a
    assert a.provenance.fidelity is Fidelity.APPROXIMATION
    assert "Green-Sellin-Zachor" in a.provenance.method
    assert any("GSZ model error dominates" in s for s in a.provenance.assumptions)
    assert any("grid" in s for s in a.provenance.assumptions)
    assert a.provenance.error_estimate > 0.0


def test_the_strength_error_is_absolute_and_doubles_the_dipole_relative_error(sodium):
    """f and A both go as |R|^2, so R's relative error enters twice. The stored
    figure is absolute, matching every other error_estimate in the codebase.
    """
    line = _find(sodium.lines, 3, 1, 3, 0)
    dipole = screened_dipole_integral(11, 11, 3, 0, 3, 1)
    rel = dipole.provenance.error_estimate / abs(dipole.value)
    for q in (line.einstein_a, line.oscillator_strength):
        assert q.provenance.error_estimate == pytest.approx(2.0 * rel * abs(q.value))
        assert q.provenance.error_estimate < abs(q.value), "error should not swamp the value"


def test_the_dipole_integral_is_symmetric_in_the_two_states():
    forward = screened_dipole_integral(11, 11, 3, 0, 3, 1).value
    backward = screened_dipole_integral(11, 11, 3, 1, 3, 0).value
    assert forward == pytest.approx(backward, rel=1e-12)


def test_a_widely_separated_pair_still_shares_one_grid():
    """3s and 6p would be handed very different natural boxes. Solved on the
    common grid the overlap is small but finite and correctly signed, not the
    garbage two mismatched grids would produce.
    """
    v = screened_dipole_integral(11, 11, 3, 0, 6, 1)
    near = screened_dipole_integral(11, 11, 3, 0, 3, 1)
    assert np.isfinite(v.value) and v.value != 0.0
    assert abs(v.value) < abs(near.value), "the distant pair must overlap less"


def test_rejects_a_state_with_n_not_greater_than_l():
    with pytest.raises(ValueError, match="n must be > l"):
        screened_dipole_integral(11, 11, 1, 1, 2, 0)
