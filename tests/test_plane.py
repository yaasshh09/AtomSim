import numpy as np
import pytest

from atomsim.analytic.wavefunction import evaluate_state
from atomsim.plane import default_half_extent, plane_grid, screened_plane_grid
from atomsim.provenance import Fidelity


def test_density_matches_direct_evaluation():
    pg = plane_grid(2, 1, 0, quantity="density", resolution=33)
    i, j = 5, 20
    pos = np.array([[pg.axis[j], 0.0, pg.axis[i]]])
    psi = evaluate_state(2, 1, 0, pos).values[0]
    assert pg.values[i, j] == pytest.approx(abs(psi) ** 2, rel=1e-12)


def test_psi_sign_structure_2pz():
    # 2p_z: psi > 0 for z > 0, psi < 0 for z < 0 (rows are z, ascending)
    pg = plane_grid(2, 1, 0, quantity="psi", resolution=33)
    mid = 16  # axis[16] == 0.0 for a 33-point symmetric grid
    assert pg.values[mid + 5, mid] > 0.0
    assert pg.values[mid - 5, mid] < 0.0


def test_density_nonnegative_shape_dtype():
    pg = plane_grid(3, 2, 1, quantity="density", basis="real", resolution=17)
    assert pg.values.shape == (17, 17)
    assert pg.values.dtype == np.float64
    assert (pg.values >= 0.0).all()
    assert pg.basis == "real"


def test_default_extent_and_axis():
    assert default_half_extent(2) == pytest.approx(10.0)
    pg = plane_grid(2, 0, 0, resolution=9)
    assert pg.axis[0] == pytest.approx(-10.0)
    assert pg.axis[-1] == pytest.approx(10.0)
    # mu-scaling shrinks the frame like the orbital itself
    assert default_half_extent(1, Z=1, mu_ratio=100.0) == pytest.approx(0.025)


def test_validation_errors():
    with pytest.raises(ValueError):
        plane_grid(1, 0, 0, quantity="colour")
    with pytest.raises(ValueError):
        plane_grid(1, 0, 0, resolution=1)
    with pytest.raises(ValueError):
        plane_grid(1, 0, 0, half_extent=-1.0)
    with pytest.raises(ValueError):
        plane_grid(1, 1, 0)  # l >= n


def test_provenance_and_units_are_honest():
    dens = plane_grid(1, 0, 0, resolution=8)
    assert dens.provenance.fidelity is Fidelity.EXACT
    assert "y=0" in dens.provenance.method
    assert dens.unit == "bohr^-3"
    psi = plane_grid(1, 0, 0, quantity="psi", resolution=8)
    assert psi.unit == "bohr^-3/2"
    assert any("real" in a for a in psi.provenance.assumptions)


def test_progress_monotone_to_one():
    seen: list[float] = []
    plane_grid(1, 0, 0, resolution=8, progress=seen.append)
    assert seen[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(seen, seen[1:]))


def test_screened_plane_is_approximation_and_real_signed():
    pg = screened_plane_grid(11, 11, 3, 1, 0, quantity="psi", resolution=64)
    assert pg.provenance.fidelity is Fidelity.APPROXIMATION
    assert pg.values.shape == (64, 64)
    assert np.isrealobj(pg.values)
    assert np.isfinite(pg.values).all()
    assert pg.Z == 11


def test_screened_plane_node_count_along_axis():
    # Na 3s density: 2 radial nodes -> the signed psi along +z axis changes sign twice.
    pg = screened_plane_grid(11, 11, 3, 0, 0, quantity="psi", resolution=401)
    mid = pg.values.shape[1] // 2           # x = 0 column
    col = pg.values[pg.values.shape[0] // 2 :, mid]  # z >= 0 half-ray
    nz = col[np.abs(col) > np.max(np.abs(col)) * 1e-3]
    sign_changes = int(np.sum(np.diff(np.sign(nz)) != 0))
    assert sign_changes == 2


def test_hydrogen_plane_unchanged_exact():
    pg = plane_grid(2, 1, 0, quantity="psi", resolution=32)
    assert pg.provenance.fidelity is Fidelity.EXACT
    assert pg.values.shape == (32, 32)


def _exact_j_y(n, l, m, x, z, Z=1, mu_ratio=1.0, h=1e-5):
    """Im(psi* d psi/dy) / mu at (x, 0, z), straight from the complex psi."""
    pos = np.array([[x, -h, z], [x, 0.0, z], [x, h, z]])
    psi = evaluate_state(n, l, m, pos, Z=Z, mu_ratio=mu_ratio).values
    dpsi = (psi[2] - psi[0]) / (2 * h)
    return float(np.imag(np.conj(psi[1]) * dpsi)) / mu_ratio


@pytest.mark.parametrize(
    "n,l,m,Z,mu",
    [(2, 1, 1, 1, 1.0), (3, 2, -2, 1, 1.0), (4, 3, 3, 2, 0.99986), (2, 1, -1, 1, 0.5)],
)
def test_current_through_plane_is_im_psi_star_grad_psi(n, l, m, Z, mu):
    from atomsim.plane import current_through_plane

    pg = current_through_plane(
        plane_grid(n, l, m, quantity="density", Z=Z, mu_ratio=mu, resolution=21)
    )
    assert pg.quantity == "current"
    assert pg.provenance.fidelity is Fidelity.EXACT
    for i, j in [(3, 4), (10, 15), (17, 2), (12, 19)]:
        exact = _exact_j_y(n, l, m, pg.axis[j], pg.axis[i], Z, mu)
        assert pg.values[i, j] == pytest.approx(exact, rel=1e-6, abs=1e-14)


def test_real_orbitals_and_m0_carry_no_current():
    from atomsim.plane import current_through_plane

    real = current_through_plane(plane_grid(2, 1, 1, basis="real", resolution=17))
    m0 = current_through_plane(plane_grid(3, 1, 0, resolution=17))
    assert not real.values.any() and not m0.values.any()
    assert any("real function" in a for a in real.provenance.assumptions)


def test_current_flows_in_one_side_and_out_the_other():
    from atomsim.plane import current_through_plane

    pg = current_through_plane(plane_grid(2, 1, 1, resolution=33))
    mid = 16
    # Circulation about z: through the plane one way at x > 0, back at x < 0.
    assert pg.values[mid + 3, mid + 5] > 0 > pg.values[mid + 3, mid - 5]
    assert pg.values[mid + 3, mid + 5] == pytest.approx(-pg.values[mid + 3, mid - 5])
    assert pg.values[:, mid].tolist() == [0.0] * 33  # the axis itself
