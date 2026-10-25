"""Finite nuclear size: the leading-order shift against published coefficients."""

import pytest
from scipy import constants as sc

from atomsim.constants import HARTREE_EV
from atomsim.nuclear_size import finite_size_shift, load_lamb_reference, muonic_lamb_shift
from atomsim.provenance import Fidelity, Provenance, Quantity
from atomsim.systems import get_system

_A0_FM = sc.physical_constants["Bohr radius"][0] * 1e15
_HARTREE_HZ = sc.physical_constants["hartree-hertz relationship"][0]


def _radius_fm(r_fm: float) -> Quantity:
    return Quantity(r_fm / _A0_FM, "bohr", "r", Provenance(Fidelity.EXACT, "test radius"))


def test_hydrogen_1s_shift_is_about_a_megahertz():
    h = get_system("h")
    q = finite_size_shift(1, 0, h.nuclear_radius, mu_ratio=h.mu_ratio.value)
    # The textbook size of the proton-size shift of hydrogen 1S: ~1.1 MHz.
    assert q.value * _HARTREE_HZ / 1e6 == pytest.approx(1.10, rel=0.03)
    assert q.provenance.fidelity is Fidelity.APPROXIMATION


def test_muonic_2s_coefficient_matches_the_published_one():
    # Leading order should land on the full theory's -5.2275 meV/fm^2 to within
    # the next-order terms this module states as its own error.
    mu = get_system("mu-h").mu_ratio.value
    q = finite_size_shift(2, 0, _radius_fm(1.0), mu_ratio=mu)
    ours = q.value * HARTREE_EV * 1e3
    published = load_lamb_reference().r2_coefficient[0]
    assert abs(ours - published) < q.provenance.error_estimate * HARTREE_EV * 1e3
    assert ours == pytest.approx(published, rel=0.01)


def test_scaling_laws():
    r = _radius_fm(1.0)
    base = finite_size_shift(1, 0, r).value
    assert finite_size_shift(2, 0, r).value == pytest.approx(base / 8, rel=1e-12)
    assert finite_size_shift(1, 0, r, Z=3).value == pytest.approx(base * 81, rel=1e-12)
    assert finite_size_shift(1, 0, r, mu_ratio=2).value == pytest.approx(base * 8, rel=1e-12)


def test_p_states_get_a_disclosed_zero():
    q = finite_size_shift(2, 1, _radius_fm(0.84))
    assert q.value == 0.0
    assert 0 < q.provenance.error_estimate < 1e-9 * finite_size_shift(2, 0, _radius_fm(0.84)).value


def test_the_two_radii_predict_lamb_shifts_a_third_of_a_mev_apart():
    radii = load_lamb_reference().radii_fm
    small = muonic_lamb_shift(*radii["muonic"][:2])
    large = muonic_lamb_shift(*radii["codata_2010"][:2])
    assert small.value - large.value == pytest.approx(0.329, abs=0.002)
    # The CODATA 2010 radius' own uncertainty cannot close that gap.
    assert small.value - large.value > 5 * large.provenance.error_estimate
