import math

import pytest

from atomsim.analytic.wigner import wigner_3j


def test_trivial_symbol_is_one():
    assert wigner_3j(0, 0, 0, 0, 0, 0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "args, expected",
    [
        ((1, 1, 0, 0, 0, 0), -1.0 / math.sqrt(3.0)),
        ((1, 0, 1, 0, 0, 0), -1.0 / math.sqrt(3.0)),
        ((2, 2, 0, 0, 0, 0), 1.0 / math.sqrt(5.0)),
        ((1, 2, 1, 0, 0, 0), 2.0 / math.sqrt(30.0)),
        ((2, 2, 2, 0, 0, 0), -math.sqrt(2.0 / 35.0)),
    ],
)
def test_closed_form_values(args, expected):
    """Cross-checked against the closed form for m1=m2=m3=0:

    3j = (-1)^g sqrt( (J-2j1)!(J-2j2)!(J-2j3)! / (J+1)! )
         * g! / [ (g-j1)!(g-j2)!(g-j3)! ],  J = j1+j2+j3 even, g = J/2.
    """
    assert wigner_3j(*args) == pytest.approx(expected, rel=1e-12)


def test_general_m_value():
    # 3j(1,1,0;1,-1,0) = (-1)^(j1-j2-m3)/sqrt(2j3+1) * <1,1;1,-1|0,0>
    #                  = 1 * 1 * (1/sqrt(3))
    assert wigner_3j(1, 1, 0, 1, -1, 0) == pytest.approx(1.0 / math.sqrt(3.0))


def test_zero_when_m_do_not_sum_to_zero():
    assert wigner_3j(1, 1, 0, 1, 0, 0) == 0.0


def test_zero_when_parity_forbids():
    # l1 + k + l2 odd: the (0,0,0) symbol must vanish identically.
    assert wigner_3j(1, 1, 1, 0, 0, 0) == 0.0


def test_zero_when_triangle_fails():
    assert wigner_3j(1, 1, 5, 0, 0, 0) == 0.0


def test_even_permutation_invariance():
    a = wigner_3j(2, 1, 1, 0, 0, 0)
    b = wigner_3j(1, 1, 2, 0, 0, 0)
    assert a == pytest.approx(b)


def test_odd_permutation_sign():
    # Swapping two columns multiplies by (-1)^(j1+j2+j3).
    a = wigner_3j(1, 2, 1, 0, 0, 0)
    b = wigner_3j(2, 1, 1, 0, 0, 0)
    assert a == pytest.approx((-1.0) ** (1 + 2 + 1) * b)


@pytest.mark.parametrize("m3", [-1, 0, 1, 2])
def test_orthogonality_sum_rule(m3):
    """For FIXED m3, summing over m1 (with m2 = -m3 - m1 determined):

        sum_m1 (2 j3 + 1) |3j(j1 j2 j3; m1 m2 m3)|^2 = 1

    m3 is an external label of the relation, not a free summation index.
    Summing over m1 and m2 independently would add up 2*j3+1 separate
    m3-slices and give 2*j3+1, not 1.
    """
    j1, j2, j3 = 2, 1, 2
    total = 0.0
    for m1 in range(-j1, j1 + 1):
        m2 = -m3 - m1
        if abs(m2) > j2:
            continue
        total += (2 * j3 + 1) * wigner_3j(j1, j2, j3, m1, m2, m3) ** 2
    assert total == pytest.approx(1.0)


def test_rejects_non_half_integer():
    with pytest.raises(ValueError):
        wigner_3j(0.3, 1, 1, 0, 0, 0)
