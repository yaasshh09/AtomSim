import numpy as np
import pytest

from atomsim.numerics.slater import pair_potential, slater_f, slater_g


def hydrogenic_1s(r: np.ndarray, z: float) -> np.ndarray:
    """P_1s = r R_1s, normalized so that integral P^2 dr = 1."""
    return 2.0 * z**1.5 * r * np.exp(-z * r)


@pytest.fixture
def grid():
    return np.linspace(1e-6, 60.0, 60000)


def test_u0_goes_as_one_over_r_outside_the_charge(grid):
    p = hydrogenic_1s(grid, 1.0)
    u0 = pair_potential(p, p, grid, 0)
    far = grid > 30.0
    assert np.allclose(u0[far], 1.0 / grid[far], rtol=1e-6)


def test_u0_is_finite_at_the_origin(grid):
    p = hydrogenic_1s(grid, 1.0)
    u0 = pair_potential(p, p, grid, 0)
    assert np.isfinite(u0).all()
    assert u0[0] > 0.0


def test_pair_potential_is_symmetric_in_its_orbitals(grid):
    a = hydrogenic_1s(grid, 1.0)
    b = hydrogenic_1s(grid, 2.0)
    assert np.allclose(pair_potential(a, b, grid, 1), pair_potential(b, a, grid, 1))


def test_f0_of_hydrogenic_1s_matches_the_analytic_value(grid):
    # F0(1s,1s) = 5Z/8 hartree for a hydrogenic 1s pair.
    for z in (1.0, 2.0):
        p = hydrogenic_1s(grid, z)
        assert slater_f(p, p, grid, 0) == pytest.approx(5.0 * z / 8.0, rel=1e-5)


def test_g_equals_f_when_both_orbitals_are_the_same(grid):
    p = hydrogenic_1s(grid, 1.5)
    assert slater_g(p, p, grid, 0) == pytest.approx(slater_f(p, p, grid, 0), rel=1e-10)


def test_higher_k_pair_potential_decays_faster(grid):
    p = hydrogenic_1s(grid, 1.0)
    u0 = pair_potential(p, p, grid, 0)
    u2 = pair_potential(p, p, grid, 2)
    far = grid > 20.0
    assert np.all(u2[far] < u0[far])


def test_rejects_negative_k(grid):
    p = hydrogenic_1s(grid, 1.0)
    with pytest.raises(ValueError):
        pair_potential(p, p, grid, -1)
