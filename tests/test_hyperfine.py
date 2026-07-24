"""Physics validation for the magnetic-dipole hyperfine engine (s-states).

Five independent experimental anchors (both nuclear-spin values, Z = 1 and 2,
positive and negative moments) lock the non-relativistic Fermi-contact formula.
A regression in the mass factor, the g-factor convention, or the Z/n scaling
moves one of these outside tolerance.
"""

import pytest
from scipy import constants as sc

from atomsim.analytic.hyperfine import (
    HyperfineLevel,
    hyperfine_constant,
    hyperfine_levels,
    hyperfine_report,
)
from atomsim.provenance import Fidelity
from atomsim.systems import get_system, hydrogen_like

# Hartree -> MHz (frequency), for comparison with spectroscopy.
_HA_MHZ = sc.physical_constants["hartree-hertz relationship"][0] / 1e6


def _gI(moment_ratio_name: str, spin: float) -> float:
    """Nuclear g-factor g_I = (mu/mu_N) / I from a CODATA moment ratio."""
    return sc.physical_constants[moment_ratio_name][0] / spin


def _mu_ratio(mass_ratio_name: str) -> float:
    """Reduced-mass ratio mu/m_e for an electron orbiting a nucleus of mass M."""
    big = sc.physical_constants[mass_ratio_name][0]
    return big / (big + 1.0)


def _A_mhz(n, Z, mu_ratio, g_I) -> float:
    return hyperfine_constant(n, Z=Z, mu_ratio=mu_ratio, g_I=g_I).value * _HA_MHZ


# --- the five anchors -------------------------------------------------------

def test_hydrogen_1s_is_the_21cm_line():
    gI = _gI("proton mag. mom. to nuclear magneton ratio", 0.5)
    A = _A_mhz(1, 1, _mu_ratio("proton-electron mass ratio"), gI)
    # I = J = 1/2 => splitting F=1<->0 equals A. Experiment: 1420.405751 MHz.
    assert A == pytest.approx(1420.405751, rel=1e-3)


def test_hydrogen_2s_scales_as_n_cubed():
    gI = _gI("proton mag. mom. to nuclear magneton ratio", 0.5)
    mu = _mu_ratio("proton-electron mass ratio")
    A1 = _A_mhz(1, 1, mu, gI)
    A2 = _A_mhz(2, 1, mu, gI)
    assert A2 == pytest.approx(177.556, rel=1e-3)          # experiment
    assert A1 / A2 == pytest.approx(8.0, rel=1e-9)          # exact 1/n^3


def test_deuterium_1s_splitting():
    gI = _gI("deuteron mag. mom. to nuclear magneton ratio", 1.0)
    A = _A_mhz(1, 1, _mu_ratio("deuteron-electron mass ratio"), gI)
    # I=1, J=1/2 => F=1/2,3/2; splitting = 1.5 A. Experiment: 327.384 MHz.
    assert 1.5 * A == pytest.approx(327.384, rel=1e-3)


def test_tritium_1s_splitting():
    gI = _gI("triton mag. mom. to nuclear magneton ratio", 0.5)
    A = _A_mhz(1, 1, _mu_ratio("triton-electron mass ratio"), gI)
    assert A == pytest.approx(1516.701, rel=1e-3)


def test_helium3_ion_1s_splitting_is_negative():
    gI = _gI("helion mag. mom. to nuclear magneton ratio", 0.5)  # negative moment
    A = _A_mhz(1, 2, _mu_ratio("helion-electron mass ratio"), gI)
    assert A < 0.0
    assert A == pytest.approx(-8665.65, rel=1e-3)


# --- structure --------------------------------------------------------------

def test_F_range_for_spin_half_nucleus():
    gI = _gI("proton mag. mom. to nuclear magneton ratio", 0.5)
    levels = hyperfine_levels(1, I=0.5, Z=1, mu_ratio=1.0, g_I=gI)
    assert sorted(lv.F for lv in levels) == [0.0, 1.0]


def test_F_range_for_spin_one_nucleus():
    gI = _gI("deuteron mag. mom. to nuclear magneton ratio", 1.0)
    levels = hyperfine_levels(1, I=1.0, Z=1, mu_ratio=1.0, g_I=gI)
    assert sorted(lv.F for lv in levels) == [0.5, 1.5]


def test_centroid_theorem_weighted_shifts_sum_to_zero():
    # sum over F of (2F+1) * shift(F) = 0: hyperfine only redistributes, the
    # gross-level centroid is preserved.
    gI = _gI("proton mag. mom. to nuclear magneton ratio", 0.5)
    levels = hyperfine_levels(1, I=0.5, Z=1, mu_ratio=1.0, g_I=gI)
    weighted = sum((2 * lv.F + 1) * lv.shift.value for lv in levels)
    assert weighted == pytest.approx(0.0, abs=1e-18)


def test_negative_moment_inverts_level_order():
    pos = hyperfine_levels(1, I=0.5, Z=1, mu_ratio=1.0, g_I=+5.5857)
    neg = hyperfine_levels(1, I=0.5, Z=1, mu_ratio=1.0, g_I=-4.2553)
    hi_pos = max(pos, key=lambda lv: lv.F)  # F=1
    hi_neg = max(neg, key=lambda lv: lv.F)
    assert hi_pos.shift.value > 0.0   # F=1 above centroid for positive moment
    assert hi_neg.shift.value < 0.0   # inverted for negative moment


def test_energy_is_gross_plus_shift():
    from atomsim.analytic.hydrogen import energy
    gI = _gI("proton mag. mom. to nuclear magneton ratio", 0.5)
    levels = hyperfine_levels(1, I=0.5, Z=1, mu_ratio=1.0, g_I=gI)
    e_gross = energy(1, Z=1, mu_ratio=1.0).value
    for lv in levels:
        assert lv.energy.value == pytest.approx(e_gross + lv.shift.value)


def test_provenance_is_approximation_with_error_and_neglected_scales():
    gI = _gI("proton mag. mom. to nuclear magneton ratio", 0.5)
    q = hyperfine_constant(1, Z=1, mu_ratio=1.0, g_I=gI)
    assert q.provenance.fidelity is Fidelity.APPROXIMATION
    assert q.provenance.error_estimate is not None and q.provenance.error_estimate > 0
    joined = " ".join(q.provenance.assumptions).lower()
    assert "contact" in joined
    assert "l > 0" in joined or "l>0" in joined  # deferred channel disclosed


# --- preset wrapper ---------------------------------------------------------

def test_report_hydrogen_splits_and_is_available():
    rep = hyperfine_report(1, get_system("h"))
    assert rep.available
    assert rep.I == 0.5
    assert len(rep.levels) == 2
    split = (max(lv.energy.value for lv in rep.levels)
             - min(lv.energy.value for lv in rep.levels)) * _HA_MHZ
    assert split == pytest.approx(1420.405751, rel=1e-3)


def test_report_helium4_ion_has_no_hyperfine_spin_zero():
    # the he+ preset is He-4 (alpha, I=0): no magnetic moment, no splitting.
    rep = hyperfine_report(1, get_system("he+"))
    assert rep.available
    assert rep.I == 0.0
    assert len(rep.levels) == 1
    assert rep.levels[0].shift.value == 0.0
    assert "spin" in (rep.note or "").lower()


@pytest.mark.parametrize("key", ["ps", "mu-h"])
def test_report_special_systems_unavailable_with_reason(key):
    rep = hyperfine_report(1, get_system(key))
    assert not rep.available
    assert rep.reason and len(rep.reason) > 0


def test_report_generic_Z_has_no_identified_nucleus():
    rep = hyperfine_report(1, hydrogen_like(3))
    assert not rep.available
    assert "nucleus" in rep.reason.lower()


def test_level_type_and_only_s_states():
    gI = _gI("proton mag. mom. to nuclear magneton ratio", 0.5)
    levels = hyperfine_levels(2, I=0.5, Z=1, mu_ratio=1.0, g_I=gI)
    assert all(isinstance(lv, HyperfineLevel) for lv in levels)
    # n must be >= 1
    with pytest.raises(ValueError):
        hyperfine_constant(0, Z=1, mu_ratio=1.0, g_I=gI)
