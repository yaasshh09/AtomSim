import numpy as np
import pytest
from scipy.stats import kstest

from atomsim.provenance import Fidelity
from atomsim.sampling import SampleCloud, sample_density

COUNT = 100_000


def _radii(cloud: SampleCloud) -> np.ndarray:
    return np.linalg.norm(cloud.positions.astype(float), axis=1)


def test_positions_shape_dtype_and_metadata():
    cloud = sample_density(2, 1, 0, count=5_000, seed=1)
    assert cloud.positions.shape == (5_000, 3)
    assert cloud.positions.dtype == np.float32
    assert (cloud.n, cloud.l, cloud.m) == (2, 1, 0)
    assert np.isfinite(cloud.positions).all()


def test_provenance_is_numerical_and_states_seed_and_count():
    cloud = sample_density(1, 0, 0, count=2_000, seed=7)
    assert cloud.provenance.fidelity is Fidelity.NUMERICAL
    joined = " ".join(cloud.provenance.assumptions)
    assert "seed=7" in joined
    assert "2000" in joined.replace(",", "").replace("_", "")


def test_1s_radial_distribution_ks_against_analytic_cdf():
    # 1s: F(r) = 1 - exp(-2r) (1 + 2r + 2r^2)
    cloud = sample_density(1, 0, 0, count=COUNT, seed=42)
    r = _radii(cloud)
    ks = kstest(r, lambda x: 1.0 - np.exp(-2.0 * x) * (1.0 + 2.0 * x + 2.0 * x**2))
    assert ks.statistic < 0.01, ks


def test_1s_mean_radius():
    r = _radii(sample_density(1, 0, 0, count=COUNT, seed=42))
    assert r.mean() == pytest.approx(1.5, abs=0.02)  # <r>_1s = 1.5 bohr


def test_2p_mean_radius():
    r = _radii(sample_density(2, 1, 0, count=COUNT, seed=3))
    assert r.mean() == pytest.approx(5.0, abs=0.05)  # <r>_2,1 = 5 bohr


def test_1s_angular_isotropy():
    cloud = sample_density(1, 0, 0, count=COUNT, seed=11)
    r = _radii(cloud)
    cos_theta = cloud.positions[:, 2].astype(float) / r
    assert cos_theta.mean() == pytest.approx(0.0, abs=0.01)
    assert (cos_theta**2).mean() == pytest.approx(1.0 / 3.0, abs=0.01)


def test_2p_m0_angular_distribution():
    # |Y_10|^2 ~ cos^2(theta): pdf over x=cos(theta) is (3/2) x^2 -> E[x^2] = 3/5
    cloud = sample_density(2, 1, 0, count=COUNT, seed=5)
    r = _radii(cloud)
    cos_theta = cloud.positions[:, 2].astype(float) / r
    assert (cos_theta**2).mean() == pytest.approx(0.6, abs=0.01)


def test_seed_reproducibility():
    a = sample_density(3, 2, 1, count=1_000, seed=99)
    b = sample_density(3, 2, 1, count=1_000, seed=99)
    assert np.array_equal(a.positions, b.positions)


def test_progress_callback_monotonic_and_complete():
    calls: list[float] = []
    sample_density(1, 0, 0, count=10_000, seed=0, progress=calls.append, n_chunks=10)
    assert len(calls) == 10
    assert calls[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(calls, calls[1:]))


def test_rejects_invalid_quantum_numbers():
    with pytest.raises(ValueError):
        sample_density(1, 1, 0, count=100)   # l == n
    with pytest.raises(ValueError):
        sample_density(2, 1, 2, count=100)   # |m| > l
    with pytest.raises(ValueError):
        sample_density(1, 0, 0, count=0)     # count must be positive
