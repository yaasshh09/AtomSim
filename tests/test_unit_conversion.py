"""Display conversions must carry the error bar into the new unit, not just the value."""

import pytest
from scipy import constants as sc

from atomsim.constants import BOHR_RADIUS_PM, HARTREE_EV
from atomsim.provenance import Fidelity, Provenance, Quantity
from atomsim.server.app import _to_ev, _to_pm


def _q(err):
    return Quantity(2.0, "x", "q", Provenance(Fidelity.NUMERICAL, "m", error_estimate=err))


def test_ev_conversion_scales_the_error():
    q = _to_ev(_q(1e-3))
    assert q.value == pytest.approx(2.0 * HARTREE_EV)
    assert q.provenance.error_estimate == pytest.approx(1e-3 * HARTREE_EV)


def test_pm_conversion_scales_the_error():
    q = _to_pm(_q(1e-3))
    assert q.provenance.error_estimate == pytest.approx(1e-3 * BOHR_RADIUS_PM)


def test_an_exact_value_converts_with_the_constants_own_uncertainty():
    """No error of its own, but the conversion factor is a measured constant."""
    value, _, unc = sc.physical_constants["Hartree energy in eV"]
    q = _to_ev(_q(None))
    assert q.provenance.error_estimate == pytest.approx(2.0 * unc, rel=1e-12)
    assert q.provenance.error_estimate / q.value < 1e-11
