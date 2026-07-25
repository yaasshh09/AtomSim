"""Validation for j-resolved (fine-structure) electric-dipole rates.

The strongest checks here are the two that cannot be satisfied by a plausible
but wrong formula: summing the components must reproduce the gross rate from the
Phase 13 engine exactly, and the 2:1 doublet intensity ratio is a real
spectroscopic fact rather than a self-consistency identity.
"""

import pytest

from atomsim.analytic.transitions import (
    einstein_A,
    einstein_A_fine,
    lifetime,
    lifetime_fine,
    oscillator_strength_fine,
)
from atomsim.provenance import Fidelity


def test_sums_to_the_gross_rate():
    """Resolving the lower level into j components cannot change the total rate."""
    for n_up, l_up, n_low, l_low in [
        (2, 1, 1, 0), (3, 1, 1, 0), (3, 2, 2, 1), (4, 3, 3, 2), (5, 1, 3, 2),
    ]:
        gross = einstein_A(n_up, l_up, n_low, l_low).value
        for j_up in ([l_up - 0.5, l_up + 0.5] if l_up > 0 else [0.5]):
            js = [l_low - 0.5, l_low + 0.5] if l_low > 0 else [0.5]
            total = sum(
                einstein_A_fine(n_up, l_up, j_up, n_low, l_low, j).value for j in js
            )
            assert total == pytest.approx(gross, rel=1e-9)


def test_the_two_p_components_have_equal_rates_not_a_two_to_one_ratio():
    """A(p_3/2 -> s_1/2) == A(p_1/2 -> s_1/2).

    The famous 2:1 doublet ratio lives in the line *strength* and in observed
    intensity, where the upper level's 4 sublevels outweigh the other's 2. The
    Einstein A is a per-atom rate with that degeneracy already divided out, so
    the two components decay equally fast. NIST bears this out for the Na D
    lines: 6.16e7 and 6.14e7 s^-1.
    """
    upper_3_2 = einstein_A_fine(2, 1, 1.5, 1, 0, 0.5).value
    upper_1_2 = einstein_A_fine(2, 1, 0.5, 1, 0, 0.5).value
    assert upper_3_2 == pytest.approx(upper_1_2, rel=1e-9)


def test_degeneracy_weighted_rates_do_show_the_two_to_one_ratio():
    """(2j'+1) A is where the 2:1 appears, since that is the line strength."""
    strong = 4 * einstein_A_fine(2, 1, 1.5, 1, 0, 0.5).value   # j' = 3/2, g = 4
    weak = 2 * einstein_A_fine(2, 1, 0.5, 1, 0, 0.5).value     # j' = 1/2, g = 2
    assert strong / weak == pytest.approx(2.0, rel=1e-9)


# Multiplet branching ratios, derived independently of the Racah code below.
#
# These constrain how a rate *divides*, not just its total, so unlike the sum
# rules they cannot be satisfied by a 6j that is uniformly wrong. Each ratio
# follows from three facts only: the two sum rules, and that Delta j = 2 is
# forbidden. Writing S(j', j) = (2j+1)(2j'+1) {j 1 j'; l' 1/2 l}^2 for the
# symmetric line strength, the sum rules read
#
#     sum over lower j :  sum_j  S(j', j) = (2j'+1) / (2l' + 1)
#     sum over upper j':  sum_j' S(j', j) = (2j+1)  / (2l  + 1)
#
# and A is proportional to S / (2j' + 1).
#
# d -> p (l' = 2, l = 1). S(5/2, 1/2) = 0, so the j' = 5/2 row gives
# S(5/2, 3/2) = 6/5; the j = 3/2 column gives S(3/2, 3/2) = 4/3 - 6/5 = 2/15;
# the j' = 3/2 row gives S(3/2, 1/2) = 4/5 - 2/15 = 2/3. Both channels share
# j' = 3/2, so the ratio is (2/3) : (2/15) = 5 : 1.
#
# f -> d (l' = 3, l = 2). S(7/2, 3/2) = 0, so S(7/2, 5/2) = 8/7; the j = 5/2
# column gives S(5/2, 5/2) = 6/5 - 8/7 = 2/35; the j' = 5/2 row gives
# S(5/2, 3/2) = 6/7 - 2/35 = 4/5. Ratio (4/5) : (2/35) = 14 : 1.


def test_d_to_p_branching_ratio_is_five_to_one():
    to_p_half = einstein_A_fine(3, 2, 1.5, 2, 1, 0.5).value
    to_p_three_halves = einstein_A_fine(3, 2, 1.5, 2, 1, 1.5).value
    assert to_p_half / to_p_three_halves == pytest.approx(5.0, rel=1e-9)


def test_f_to_d_branching_ratio_is_fourteen_to_one():
    favoured = einstein_A_fine(4, 3, 2.5, 3, 2, 1.5).value
    other = einstein_A_fine(4, 3, 2.5, 3, 2, 2.5).value
    assert favoured / other == pytest.approx(14.0, rel=1e-9)


def test_both_2p_fine_levels_keep_the_gross_lifetime():
    """2p decays only to 1s, so resolving j must not move the lifetime."""
    gross = lifetime(2, 1).value
    for j in (0.5, 1.5):
        assert lifetime_fine(2, 1, j).value == pytest.approx(gross, rel=1e-9)


def test_delta_j_selection_rule_gives_exact_zeros():
    # d_5/2 -> s_1/2 is |dj| = 2: forbidden, and the 6j makes it structurally 0.
    assert einstein_A_fine(3, 2, 2.5, 2, 0, 0.5).value == 0.0
    # dl = 0 stays forbidden too.
    assert einstein_A_fine(3, 0, 0.5, 2, 0, 0.5).value == 0.0


def test_delta_j_zero_is_allowed_when_l_changes():
    """p_1/2 -> s_1/2 is dj = 0 and perfectly allowed."""
    assert einstein_A_fine(2, 1, 0.5, 1, 0, 0.5).value > 0.0


def test_rejects_a_j_that_does_not_belong_to_its_l():
    with pytest.raises(ValueError):
        einstein_A_fine(2, 1, 2.5, 1, 0, 0.5)   # j must be l +/- 1/2
    with pytest.raises(ValueError):
        einstein_A_fine(2, 1, 1.5, 1, 0, 1.5)   # l = 0 admits only j = 1/2


def test_fine_oscillator_strengths_sum_over_upper_j_to_the_gross_value():
    """Absorption f out of a fixed lower j, summed over the upper j components,
    returns the gross f: the 6j sum rule again, with the columns swapped."""
    from atomsim.analytic.transitions import oscillator_strength

    gross = oscillator_strength(1, 0, 2, 1).value
    parts = [oscillator_strength_fine(1, 0, 0.5, 2, 1, j).value for j in (0.5, 1.5)]
    assert all(p > 0 for p in parts)
    assert sum(parts) == pytest.approx(gross, rel=1e-9)


def test_provenance_names_the_6j_and_stays_numerical():
    a = einstein_A_fine(2, 1, 1.5, 1, 0, 0.5)
    assert a.provenance.fidelity is Fidelity.NUMERICAL
    assert "6j" in a.provenance.method
    assert a.unit == "s^-1"


def test_every_allowed_component_is_positive_and_finite():
    import math

    for n_up, l_up, n_low, l_low in [(3, 1, 2, 0), (4, 2, 2, 1), (5, 3, 4, 2)]:
        for j_up in (l_up - 0.5, l_up + 0.5):
            for j_low in ([l_low - 0.5, l_low + 0.5] if l_low > 0 else [0.5]):
                v = einstein_A_fine(n_up, l_up, j_up, n_low, l_low, j_low).value
                assert v >= 0.0 and math.isfinite(v)
