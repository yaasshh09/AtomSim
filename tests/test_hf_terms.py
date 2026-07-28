"""Pins the average-of-configuration Fock terms.

These coefficients fail silently when wrong: bad angular factors still produce
converged, smooth, believable orbitals, just at the wrong energy. So each one
is checked against something derived independently of the implementation -
a closed-shell limit, a closed-form 3j value, or a one-electron atom that must
feel nothing at all.
"""

import numpy as np
import pytest

from atomsim.analytic.wigner import wigner_3j
from atomsim.numerics.hf_terms import (
    Subshell,
    direct_potential,
    exchange_apply,
    exchange_coefficient,
    same_shell_coefficient,
)
from atomsim.numerics.slater import pair_potential


def hydrogenic_1s(r, z):
    return 2.0 * z**1.5 * r * np.exp(-z * r)


@pytest.fixture
def grid():
    return np.linspace(1e-6, 40.0, 40000)


def test_hydrogen_feels_no_direct_potential(grid):
    """One electron: (q - 1) = 0, so there is no self-interaction. This is the
    check that rejected the first candidate convention."""
    shells = (Subshell(n=1, l=0, q=1, p=hydrogenic_1s(grid, 1.0)),)
    assert np.allclose(direct_potential(shells, 0, grid), 0.0)


def test_hydrogen_feels_no_exchange(grid):
    shells = (Subshell(n=1, l=0, q=1, p=hydrogenic_1s(grid, 1.0)),)
    psi = hydrogenic_1s(grid, 1.0)
    assert np.allclose(exchange_apply(shells, 0, psi, grid), 0.0)


def test_helium_direct_is_exactly_one_hartree_potential(grid):
    """q = 2, l = 0: the electron sees exactly one unit of U_0, no more."""
    p = hydrogenic_1s(grid, 2.0)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    assert np.allclose(
        direct_potential(shells, 0, grid), pair_potential(p, p, grid, 0)
    )


def test_helium_has_no_exchange_term(grid):
    """An s shell admits no k > 0, and the k = 0 self-exchange is already
    accounted for by the (q - 1) factor in the direct term."""
    p = hydrogenic_1s(grid, 2.0)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    assert np.allclose(exchange_apply(shells, 0, p, grid), 0.0)


def test_beryllium_cross_shell_exchange_is_one_unit_of_u0(grid):
    """For 1s2 2s2, the exchange on 1s is (q_b/2) * tj^2 * U_0[1s,2s] P_2s
    with q_b = 2 and tj(0,0,0)^2 = 1, so exactly U_0[1s,2s] P_2s."""
    p1 = hydrogenic_1s(grid, 4.0)
    p2 = np.sqrt(2.0) * grid * (1.0 - grid) * np.exp(-grid)  # a 2s-like trial
    p2 /= np.sqrt(np.trapezoid(p2**2, grid))
    shells = (Subshell(n=1, l=0, q=2, p=p1), Subshell(n=2, l=0, q=2, p=p2))
    got = exchange_apply(shells, 0, p1, grid)
    want = pair_potential(p1, p2, grid, 0) * p2
    assert np.allclose(got, want, rtol=1e-10)


def test_beryllium_direct_is_one_unit_of_its_own_plus_two_of_the_other(grid):
    """The closed-shell Fock operator on 1s is h + J_1s + 2 J_2s - K_2s. This
    is the direct half of that: 2J_1s + 2J_2s minus the self-interaction J_1s.
    """
    p1 = hydrogenic_1s(grid, 4.0)
    p2 = np.sqrt(2.0) * grid * (1.0 - grid) * np.exp(-grid)
    p2 /= np.sqrt(np.trapezoid(p2**2, grid))
    shells = (Subshell(n=1, l=0, q=2, p=p1), Subshell(n=2, l=0, q=2, p=p2))
    want = pair_potential(p1, p1, grid, 0) + 2.0 * pair_potential(p2, p2, grid, 0)
    assert np.allclose(direct_potential(shells, 0, grid), want, rtol=1e-10)


def test_closed_shell_coefficients_match_the_independent_derivation():
    """Averaged coefficients must equal the closed-shell ones when q is full."""
    for l_a in (0, 1, 2):
        q_full = 2 * (2 * l_a + 1)
        for k in range(2, 2 * l_a + 1, 2):
            averaged = same_shell_coefficient(l_a, k, q_full)
            closed = (2 * l_a + 1) * wigner_3j(l_a, k, l_a, 0, 0, 0) ** 2
            assert averaged == pytest.approx(closed)


def test_cross_shell_coefficient_matches_closed_shell_form():
    for l_a, l_b in ((0, 1), (1, 1), (1, 2)):
        q_full_b = 2 * (2 * l_b + 1)
        for k in range(abs(l_a - l_b), l_a + l_b + 1):
            averaged = exchange_coefficient(l_a, k, l_b, q_full_b)
            closed = (2 * l_b + 1) * wigner_3j(l_a, k, l_b, 0, 0, 0) ** 2
            assert averaged == pytest.approx(closed)


def test_p_shell_f2_coefficient_is_the_slater_value():
    """For p^6, ((2l+1)/(4l+1)) * tj(1,2,1)^2 = (3/5) * (2/15) = 2/25 per pair,
    so the Fock-equation coefficient is (q - 1) * 2/25 = 5 * 2/25.

    tj(1,2,1) = 2/sqrt(30) from the closed form, cross-checked in
    tests/test_wigner_3j.py, so this number is derived and not recalled.
    """
    assert same_shell_coefficient(1, 2, 6) == pytest.approx(5.0 * 2.0 / 25.0)


def test_parity_forbidden_multipole_has_zero_coefficient():
    assert exchange_coefficient(0, 0, 1, 6) == 0.0  # l_a + k + l_b odd


def test_same_shell_coefficient_rejects_k_zero():
    """k = 0 is carried by the (q_a - 1) factor in direct_potential. Asking for
    it here would double-count it, so the contract refuses rather than obliges.
    """
    with pytest.raises(ValueError, match="k > 0"):
        same_shell_coefficient(1, 0, 6)


def test_exchange_is_linear_in_the_trial_function(grid):
    """LOBPCG applies this operator to search directions, not just to occupied
    orbitals, so it has to be a genuine linear operator in psi."""
    p1 = hydrogenic_1s(grid, 4.0)
    p2 = np.sqrt(2.0) * grid * (1.0 - grid) * np.exp(-grid)
    p2 /= np.sqrt(np.trapezoid(p2**2, grid))
    shells = (Subshell(n=1, l=0, q=2, p=p1), Subshell(n=2, l=0, q=2, p=p2))

    u, v = hydrogenic_1s(grid, 3.0), hydrogenic_1s(grid, 1.5)
    combined = exchange_apply(shells, 0, 2.0 * u - 3.0 * v, grid)
    separate = 2.0 * exchange_apply(shells, 0, u, grid) - 3.0 * exchange_apply(
        shells, 0, v, grid
    )
    assert np.allclose(combined, separate, rtol=1e-10)


def test_exchange_operator_is_symmetric(grid):
    """<u|K|v> = <v|K|u>. A non-symmetric Fock operator would give complex
    eigenvalues and is the classic sign that a pair potential got its two
    orbital arguments the wrong way round."""
    p1 = hydrogenic_1s(grid, 4.0)
    p2 = np.sqrt(2.0) * grid * (1.0 - grid) * np.exp(-grid)
    p2 /= np.sqrt(np.trapezoid(p2**2, grid))
    shells = (Subshell(n=1, l=0, q=2, p=p1), Subshell(n=2, l=0, q=2, p=p2))

    u, v = hydrogenic_1s(grid, 3.0), hydrogenic_1s(grid, 1.5)
    ukv = np.trapezoid(u * exchange_apply(shells, 0, v, grid), grid)
    vku = np.trapezoid(v * exchange_apply(shells, 0, u, grid), grid)
    assert ukv == pytest.approx(vku, rel=1e-10)


def test_p_shell_exchange_includes_the_k_two_multipole(grid):
    """A p^6 subshell must pick up its own k = 2 term. If the same-shell loop
    were skipped entirely, this would be zero."""
    p = hydrogenic_1s(grid, 6.0)
    shells = (Subshell(n=2, l=1, q=6, p=p),)
    got = exchange_apply(shells, 0, p, grid)
    want = same_shell_coefficient(1, 2, 6) * pair_potential(p, p, grid, 2) * p
    assert np.allclose(got, want, rtol=1e-10)
    assert not np.allclose(got, 0.0)
