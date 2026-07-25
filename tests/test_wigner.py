"""Validation for the Wigner 6j symbol.

Angular-momentum algebra is easy to get subtly wrong and hard to notice, so this
checks it four independent ways: a value derived by hand from the closed form
for a zero argument, the symmetries the symbol must obey under permuting its
columns, the triangle conditions that produce the selection rules, and the sum
rule that the fine-structure line strengths in transitions.py actually depend on.
"""

import math

import pytest

from atomsim.analytic.wigner import triangular, wigner_6j


def test_zero_argument_closed_form():
    """{a b c; d e 0} = delta(a,e) delta(b,d) (-1)^(a+b+c) / sqrt((2a+1)(2b+1)).

    For {1/2 1 j'; 1 1/2 0} that is +/- 1/sqrt(6), so the square is 1/6. This is
    the value derived by hand in the Phase 15 spec, and it anchors everything.
    """
    for j2 in (0.5, 1.5):
        assert wigner_6j(0.5, 1, j2, 1, 0.5, 0) ** 2 == pytest.approx(1 / 6, rel=1e-12)


def test_zero_argument_closed_form_over_a_range():
    for a in (0.5, 1.0, 1.5, 2.0, 2.5):
        for b in (1.0, 2.0):
            for c in (abs(a - b), a + b):
                expected = ((-1) ** (a + b + c)) / math.sqrt((2 * a + 1) * (2 * b + 1))
                assert wigner_6j(a, b, c, b, a, 0) == pytest.approx(expected, rel=1e-10)


def test_symmetric_under_permuting_columns():
    """The 6j is invariant under any permutation of its three columns."""
    args = (1.0, 2.0, 2.0, 1.5, 1.5, 2.5)
    j1, j2, j3, j4, j5, j6 = args
    base = wigner_6j(*args)
    assert base != 0.0
    assert wigner_6j(j2, j1, j3, j5, j4, j6) == pytest.approx(base, rel=1e-12)
    assert wigner_6j(j3, j2, j1, j6, j5, j4) == pytest.approx(base, rel=1e-12)
    assert wigner_6j(j1, j3, j2, j4, j6, j5) == pytest.approx(base, rel=1e-12)


def test_symmetric_under_swapping_upper_and_lower_in_two_columns():
    j1, j2, j3, j4, j5, j6 = 1.0, 2.0, 2.0, 1.5, 1.5, 2.5
    base = wigner_6j(j1, j2, j3, j4, j5, j6)
    assert wigner_6j(j4, j5, j3, j1, j2, j6) == pytest.approx(base, rel=1e-12)
    assert wigner_6j(j1, j5, j6, j4, j2, j3) == pytest.approx(base, rel=1e-12)


def test_triangle_violations_are_exactly_zero():
    assert triangular(1, 1, 5) is False
    assert wigner_6j(1, 1, 5, 1, 1, 1) == 0.0     # (j1 j2 j3) fails
    assert wigner_6j(1, 1, 1, 1, 1, 9) == 0.0     # (j1 j5 j6) fails
    # A half-integer sum that is not an integer is not a valid triad either.
    assert wigner_6j(0.5, 0.5, 0.5, 1, 1, 1) == 0.0


def test_rejects_negative_and_non_half_integer_arguments():
    with pytest.raises(ValueError):
        wigner_6j(-1, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        wigner_6j(0.3, 1, 1, 1, 1, 1)


@pytest.mark.parametrize("l_up", [1, 2, 3, 4])
def test_the_sum_rule_the_line_strengths_rest_on(l_up):
    """sum_j (2j+1) {j 1 j'; l' 1/2 l}^2 == 1/(2l'+1), for every upper j'.

    Physically: resolving the lower level into its j components cannot change
    the total rate out of the upper one. If the 6j is wrong, this fails.
    """
    for l_low in (l_up - 1, l_up + 1):
        if l_low < 0:
            continue
        for j_up in ([l_up - 0.5, l_up + 0.5] if l_up > 0 else [0.5]):
            js = [l_low - 0.5, l_low + 0.5] if l_low > 0 else [0.5]
            total = sum(
                (2 * j + 1) * wigner_6j(j, 1, j_up, l_up, 0.5, l_low) ** 2 for j in js
            )
            assert total == pytest.approx(1.0 / (2 * l_up + 1), rel=1e-10)


def test_known_tabulated_value():
    """{1 1 1; 1 1 1} = 1/6, a standard table entry."""
    assert wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1 / 6, rel=1e-12)


def _sum_over_x(a, b, c, d, y, yp):
    """sum_x (2x+1) {a b x; c d y} {a b x; c d y'} over the allowed x range."""
    total = 0.0
    x = abs(a - b)
    while x <= a + b + 1e-9:
        total += (
            (2 * x + 1) * wigner_6j(a, b, x, c, d, y) * wigner_6j(a, b, x, c, d, yp)
        )
        x += 1
    return total


@pytest.mark.parametrize(
    ("a", "b", "c", "d", "y"),
    [(1, 1, 1, 1, 1), (2, 1, 2, 1, 1), (1.5, 1.5, 0.5, 0.5, 1)],
)
def test_orthogonality_diagonal_term_is_one(a, b, c, d, y):
    """sum_x (2x+1)(2y+1) {a b x; c d y}^2 == 1.

    Arguments must make every triad valid: (1, 1.5, 1) for instance sums to 3.5
    and is not a triad at all, so that combination is identically zero and
    proves nothing.
    """
    assert (2 * y + 1) * _sum_over_x(a, b, c, d, y, y) == pytest.approx(1.0, rel=1e-9)


def test_orthogonality_off_diagonal_term_vanishes():
    """The same sum with y != y' must cancel to zero, which squares cannot fake."""
    assert _sum_over_x(1, 1, 1, 1, 1, 2) == pytest.approx(0.0, abs=1e-12)
    assert _sum_over_x(2, 1, 2, 1, 1, 2) == pytest.approx(0.0, abs=1e-12)
