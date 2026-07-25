"""Validation for the dipole matrix element over numerically solved radials.

The tight anchor is the hydrogenic limit: fed a pure Coulomb potential, the
numerical engine has to reproduce the exact closed-form dipole integrals from
the analytic engine. That single check validates the grid, the u = rR
normalization convention, the overlap integral and the error estimate together,
without leaning on any remembered literature number.
"""

import numpy as np
import pytest

from atomsim.analytic.transitions import A_from_radial_dipole, dipole_radial_integral
from atomsim.numerics.dipole import dipole_matrix_element
from atomsim.provenance import Fidelity


def coulomb(z: float = 1.0):
    return lambda r: -z / r


@pytest.mark.parametrize(
    ("n_a", "l_a", "n_b", "l_b"),
    [(1, 0, 2, 1), (1, 0, 3, 1), (2, 1, 3, 2), (2, 0, 3, 1), (3, 1, 4, 2)],
)
def test_reproduces_the_exact_hydrogenic_dipole_integral(n_a, l_a, n_b, l_b):
    exact = dipole_radial_integral(n_a, l_a, n_b, l_b).value
    got = dipole_matrix_element(
        coulomb(), l_a=l_a, k_a=n_a - l_a - 1, l_b=l_b, k_b=n_b - l_b - 1,
        n_top=max(n_a, n_b),
    )
    assert got.value == pytest.approx(exact, rel=2e-3)
    assert got.unit == "bohr"


def test_common_grid_holds_when_the_two_states_want_different_boxes():
    """1s is tiny and 4p is broad: solved on separate natural grids the overlap
    would be meaningless, so this is the check that the common grid is real."""
    exact = dipole_radial_integral(1, 0, 4, 1).value
    got = dipole_matrix_element(coulomb(), l_a=0, k_a=0, l_b=1, k_b=2, n_top=4)
    assert got.value == pytest.approx(exact, rel=5e-3)


def test_symmetric_in_the_two_states():
    forward = dipole_matrix_element(coulomb(), 0, 0, 1, 0, n_top=2).value
    backward = dipole_matrix_element(coulomb(), 1, 0, 0, 0, n_top=2).value
    assert forward == pytest.approx(backward, rel=1e-12)


def test_scales_with_Z_like_the_analytic_engine():
    """<r> ~ 1/Z, so the dipole integral halves when Z doubles."""
    one = dipole_matrix_element(coulomb(1.0), 0, 0, 1, 0, n_top=2).value
    two = dipole_matrix_element(coulomb(2.0), 0, 0, 1, 0, n_top=2).value
    assert one / two == pytest.approx(2.0, rel=5e-3)


def test_error_estimate_is_present_and_shrinks_with_a_finer_grid():
    coarse = dipole_matrix_element(coulomb(), 0, 0, 1, 0, n_top=2, n_points=2000)
    fine = dipole_matrix_element(coulomb(), 0, 0, 1, 0, n_top=2, n_points=16000)
    assert coarse.provenance.error_estimate > 0.0
    assert fine.provenance.error_estimate < coarse.provenance.error_estimate
    exact = dipole_radial_integral(1, 0, 2, 1).value
    assert abs(fine.value - exact) < abs(coarse.value - exact)


def test_an_einstein_A_built_from_the_numerical_dipole_matches_nist():
    """End to end: numerical R plus the shared formula gives A(2p->1s)."""
    from atomsim.analytic.hydrogen import energy

    R = dipole_matrix_element(coulomb(), 0, 0, 1, 0, n_top=2).value
    dE = energy(2).value - energy(1).value
    assert A_from_radial_dipole(dE, 1, 0, R) == pytest.approx(6.27e8, rel=1e-2)


def test_provenance_is_numerical_for_a_bare_potential():
    q = dipole_matrix_element(coulomb(), 0, 0, 1, 0, n_top=2)
    assert q.provenance.fidelity is Fidelity.NUMERICAL
    assert "grid-halving" in q.provenance.refinement


def test_orthogonality_of_same_l_states_is_not_claimed_as_a_dipole():
    """Same l is not an E1 channel; the caller decides, but the overlap itself
    must still be a finite number rather than a NaN."""
    v = dipole_matrix_element(coulomb(), 0, 0, 0, 1, n_top=2).value
    assert np.isfinite(v)


def test_rejects_a_negative_node_index():
    with pytest.raises(ValueError):
        dipole_matrix_element(coulomb(), 0, -1, 1, 0, n_top=2)
