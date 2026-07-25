"""Validation for hydrogen electric-dipole transition strengths.

The radial functions are exact; the dipole integral is adaptive quadrature,
checked against analytic anchors (the 1s->2p oscillator strength 0.4162, the
exact <2p|r|1s>), against NIST lifetimes (2p, 3p, 3d, 4f), against the
Thomas-Reiche-Kuhn partial sum over the discrete 1s->np series, and against the
exact Z^4 / reduced-mass scaling laws. Selection-rule zeros are exact.
"""

import math

import pytest

from atomsim.analytic.transitions import (
    _dipole_quadrature,
    _gauss_laguerre_nodes,
    dipole_radial_integral,
    einstein_A,
    lifetime,
    oscillator_strength,
)
from atomsim.provenance import Fidelity


def test_dipole_integral_1s_2p_matches_the_closed_form():
    # R_10 R_21 r^3 = (1/sqrt6) r^4 exp(-3r/2), so the integral is exactly
    # 4! / ((3/2)^5 sqrt 6) = 1.290266201959863... bohr (30-digit Decimal check).
    # The rule is exact for this integrand, so demand near-machine agreement.
    assert dipole_radial_integral(1, 0, 2, 1).value == pytest.approx(
        1.2902662019598634, rel=1e-12
    )


def test_dipole_integral_is_symmetric_under_swapping_the_states():
    """Checked below the cache: dipole_radial_integral canonicalizes its key, so
    calling it both ways would hit one entry and assert nothing."""
    for n, l, n2, l2 in [(1, 0, 2, 1), (3, 2, 5, 1), (4, 3, 6, 2)]:
        nodes = _gauss_laguerre_nodes(n, n2)
        forward = _dipole_quadrature(n, l, n2, l2, 1.0, nodes)
        backward = _dipole_quadrature(n2, l2, n, l, 1.0, nodes)
        assert forward == pytest.approx(backward, rel=1e-12)
        assert forward != 0.0


def test_oscillator_strength_1s_2p():
    f = oscillator_strength(1, 0, 2, 1).value
    assert f == pytest.approx(0.41620, rel=1e-3)


def test_oscillator_strength_1s_np_decreases_with_n():
    f2 = oscillator_strength(1, 0, 2, 1).value
    f3 = oscillator_strength(1, 0, 3, 1).value
    f4 = oscillator_strength(1, 0, 4, 1).value
    assert f2 > f3 > f4 > 0.0


def test_einstein_A_2p_to_1s():
    a = einstein_A(2, 1, 1, 0).value
    assert a == pytest.approx(6.27e8, rel=3e-3)  # NIST ASD 6.27e8 s^-1


def test_lifetime_2p_is_1p6_ns():
    tau = lifetime(2, 1).value
    assert tau == pytest.approx(1.596e-9, rel=5e-3)


def test_ground_state_is_radiatively_stable():
    assert math.isinf(lifetime(1, 0).value)


def test_selection_rule_forbids_delta_l_zero_and_two():
    assert oscillator_strength(1, 0, 2, 0).value == 0.0   # dl = 0
    assert oscillator_strength(2, 0, 3, 2).value == 0.0   # dl = 2
    assert einstein_A(2, 0, 1, 0).value == 0.0            # dl = 0 emission


def test_absorption_requires_upper_above_lower():
    with pytest.raises(ValueError):
        oscillator_strength(2, 1, 1, 0)   # 2p is above 1s: not an absorption


def test_provenance_is_numerical_with_error():
    q = dipole_radial_integral(1, 0, 2, 1)
    assert q.provenance.fidelity is Fidelity.NUMERICAL
    assert q.provenance.error_estimate is not None and q.provenance.error_estimate >= 0
    f = oscillator_strength(1, 0, 2, 1)
    assert f.provenance.fidelity is Fidelity.NUMERICAL


def test_higher_series_2p_3d_is_strong():
    # 2p->3d is a strong absorption (f ~ 0.7); sanity on l up-transitions.
    f = oscillator_strength(2, 1, 3, 2).value
    assert 0.5 < f < 0.9


def test_lyman_series_matches_bethe_salpeter_table():
    """f(1s->np) against the standard tabulated values (Bethe-Salpeter)."""
    reference = {2: 0.4162, 3: 0.07910, 4: 0.02900, 5: 0.013940, 6: 0.0078000}
    for n, f_ref in reference.items():
        assert oscillator_strength(1, 0, n, 1).value == pytest.approx(f_ref, rel=1e-3)


def test_thomas_reiche_kuhn_partial_sum_over_lyman_series():
    """Sum of f over the whole discrete 1s->np series is 0.5650 (rest is continuum).

    This is the strongest single check on the quadrature: it exercises every n
    at once, so a normalization slip or a quietly bad integral cannot hide.
    """
    discrete = sum(oscillator_strength(1, 0, n, 1).value for n in range(2, 40))
    tail = 1.6 / (2 * 39**2)  # f ~ 1.6/n^3 tail beyond n = 39, ~5e-4
    assert discrete + tail == pytest.approx(0.5650, abs=5e-4)


def test_oscillator_strength_is_independent_of_Z():
    """f is a pure number: the Z^6 in dE^3 and the Z^-2 in |R|^2 cancel exactly."""
    f_h = oscillator_strength(1, 0, 2, 1, Z=1).value
    f_he = oscillator_strength(1, 0, 2, 1, Z=2).value
    assert f_he == pytest.approx(f_h, rel=1e-10)


def test_einstein_A_scales_as_Z_to_the_fourth():
    a_h = einstein_A(2, 1, 1, 0, Z=1).value
    a_he = einstein_A(2, 1, 1, 0, Z=2).value
    assert a_he / a_h == pytest.approx(16.0, rel=1e-9)


def test_lifetime_scales_inversely_with_reduced_mass():
    """A ~ mu (dE^3 gives mu^3, |R|^2 gives mu^-2), so tau ~ 1/mu."""
    tau_h = lifetime(2, 1, mu_ratio=1.0).value
    tau_half = lifetime(2, 1, mu_ratio=0.5).value
    assert tau_half / tau_h == pytest.approx(2.0, rel=1e-9)


@pytest.mark.parametrize(
    ("n", "l", "tau_ns"),
    [(2, 1, 1.596), (3, 1, 5.273), (3, 2, 15.46), (4, 3, 72.49)],
)
def test_radiative_lifetimes_match_nist_sums(n, l, tau_ns):
    """tau = 1/sum(A) against lifetimes implied by the NIST ASD A-values."""
    assert lifetime(n, l).value == pytest.approx(tau_ns * 1e-9, rel=5e-3)


def test_quadrature_error_estimate_is_tracked_and_at_roundoff():
    """Node-doubling must agree to roundoff: the rule is exact for this integrand."""
    for n, l, n2, l2 in [(1, 0, 2, 1), (3, 2, 5, 1), (6, 5, 7, 6)]:
        q = dipole_radial_integral(n, l, n2, l2)
        assert 0.0 <= q.provenance.error_estimate < 1e-10 * abs(q.value)
    tau = lifetime(2, 1)
    assert 0.0 <= tau.provenance.error_estimate < 1e-9 * tau.value


def test_rejects_unphysical_Z_and_mass_ratio():
    with pytest.raises(ValueError):
        oscillator_strength(1, 0, 2, 1, Z=0)
    with pytest.raises(ValueError):
        oscillator_strength(1, 0, 2, 1, mu_ratio=-1.0)
