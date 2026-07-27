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


def test_direct_integral_is_symmetric_under_orbital_exchange(grid):
    """F^k(ab) = F^k(ba): the direct integral cannot tell which electron is
    which. Tests the quadrature, not a tautology."""
    a = hydrogenic_1s(grid, 1.0)
    b = hydrogenic_1s(grid, 2.5)
    assert slater_f(a, b, grid, 0) == pytest.approx(slater_f(b, a, grid, 0), rel=1e-8)


def test_exchange_integral_never_exceeds_the_direct_one(grid):
    """G^0(ab) <= F^0(ab) by Cauchy-Schwarz on a positive-definite kernel.
    Holds for distinct orbitals, where the two integrals genuinely differ."""
    a = hydrogenic_1s(grid, 1.0)
    b = hydrogenic_1s(grid, 2.5)
    direct = slater_f(a, b, grid, 0)
    exchange = slater_g(a, b, grid, 0)
    assert exchange <= direct
    assert exchange != pytest.approx(direct)


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


def test_rejects_grid_starting_at_zero():
    grid = np.linspace(0.0, 40.0, 4000)
    p = hydrogenic_1s(grid, 1.0)
    with pytest.raises(ValueError):
        pair_potential(p, p, grid, 0)
