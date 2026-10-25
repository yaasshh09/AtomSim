"""Rabi oscillations: the master equation against the closed forms it must reduce to."""

import math

import numpy as np
import pytest

from atomsim.provenance import Fidelity
from atomsim.rabi import angular_factor_z, rabi_drive


def test_1s_2p0_dipole_is_the_textbook_value():
    # <2p0|z|1s> = 128 sqrt(2) / 243 bohr.
    d = rabi_drive((1, 0, 0), (2, 1, 0), 1e6).dipole.value
    assert abs(d) == pytest.approx(128 * math.sqrt(2) / 243, rel=1e-12)


def test_angular_factor_closed_forms():
    assert angular_factor_z(0, 1, 0) == pytest.approx(1 / math.sqrt(3))
    assert angular_factor_z(1, 2, 1) == pytest.approx(math.sqrt(3 / 15))
    assert angular_factor_z(2, 1, 0) == angular_factor_z(1, 2, 0)
    assert angular_factor_z(0, 2, 0) == 0.0


@pytest.mark.parametrize("detuning", [0.0, 300.0, -1500.0])
def test_no_decay_reduces_to_the_rabi_formula(detuning):
    r = rabi_drive((1, 0, 0), (2, 1, 0), 1e8, detuning_mhz=detuning, decay=False)
    assert np.max(np.abs(r.excited.values - r.excited_undamped.values)) < 1e-9
    peak = (r.rabi_frequency.value / r.effective_frequency.value) ** 2
    assert r.excited.values.max() == pytest.approx(peak, rel=1e-3)
    assert r.lost.values.max() < 1e-15


def test_rabi_frequency_goes_as_root_intensity():
    a = rabi_drive((1, 0, 0), (2, 1, 0), 1e6).rabi_frequency.value
    b = rabi_drive((1, 0, 0), (2, 1, 0), 4e6).rabi_frequency.value
    assert b == pytest.approx(2 * a, rel=1e-12)


@pytest.mark.parametrize("intensity,detuning", [(1e5, 0.0), (1e7, 50.0), (3e3, -20.0)])
def test_closed_pair_settles_at_the_steady_state(intensity, detuning):
    # 2p0 decays only back to 1s0, so the pair is closed and rho_ee has a formula.
    r = rabi_drive((1, 0, 0), (2, 1, 0), intensity, detuning_mhz=detuning)
    assert r.return_fraction.value == pytest.approx(1.0, abs=1e-12)
    t_end = r.excited.grid[-1]
    if t_end * r.decay_rate.value >= 5.0:
        assert r.excited.values[-1] == pytest.approx(r.steady_excited.value, abs=0.01)


def test_leaky_pair_pumps_population_away():
    # 3p0 -> 1s0 is driven, but 3p also decays to 2s, which never comes back.
    r = rabi_drive((1, 0, 0), (3, 1, 0), 1e6)
    assert r.return_fraction.value < 0.9
    assert r.steady_excited is None
    assert np.all(np.diff(r.lost.values) >= -1e-14)
    assert r.lost.values[-1] > 0.0


def test_refusals():
    with pytest.raises(ValueError, match="same m"):
        rabi_drive((2, 1, 0), (3, 2, 1), 1e6)
    with pytest.raises(ValueError, match="no pi coupling"):
        rabi_drive((1, 0, 0), (3, 2, 0), 1e6)
    with pytest.raises(ValueError, match="rotating-wave"):
        rabi_drive((1, 0, 0), (2, 1, 0), 1e20)


def test_tier_and_disclosure():
    r = rabi_drive((1, 0, 0), (2, 1, 0), 1e6)
    assert r.excited.provenance.fidelity is Fidelity.APPROXIMATION
    assert any("rotating-wave" in a for a in r.excited.provenance.assumptions)
