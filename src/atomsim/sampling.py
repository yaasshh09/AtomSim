"""Monte-Carlo sampling of |psi_nlm|^2 — sampling IS physics and carries provenance.

Factorized inverse-CDF sampling in the complex spherical-harmonic basis:
r from P(r) = r^2 R_nl^2, cos(theta) from the normalized |Theta_lm|^2, and
phi uniform (|Y_lm|^2 is phi-independent for complex Y_lm). Real-orbital
sampling (phi-dependent) arrives with the M2 angular module.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import lpmv

from atomsim.analytic.hydrogen import radial_wavefunction, validate_quantum_numbers
from atomsim.provenance import Fidelity, Provenance

_R_GRID_POINTS = 8192
_X_GRID_POINTS = 4096


@dataclass(frozen=True)
class SampleCloud:
    """Positions sampled from |psi_nlm|^2, in bohr. Container carries provenance."""

    positions: np.ndarray  # (count, 3) float32
    n: int
    l: int
    m: int
    Z: int
    mu_ratio: float
    provenance: Provenance


def _radial_inverse_cdf(n: int, l: int, Z: int, mu_ratio: float):
    """Grid r and CDF of P(r) = r^2 R_nl^2 for inverse-CDF sampling."""
    r_max = 20.0 * n * n / (Z * mu_ratio)  # P(r_max)/P_peak < 1e-15 for all l < n
    r = np.linspace(0.0, r_max, _R_GRID_POINTS)
    R = radial_wavefunction(n, l, r, Z=Z, mu_ratio=mu_ratio).values
    p = r * r * R * R
    cdf = cumulative_trapezoid(p, r, initial=0.0)
    cdf /= cdf[-1]
    return r, cdf, r_max


def _costheta_inverse_cdf(l: int, m: int):
    """Grid x = cos(theta) and CDF of |Theta_lm|^2 (normalization cancels)."""
    x = np.linspace(-1.0, 1.0, _X_GRID_POINTS)
    p = lpmv(abs(m), l, x) ** 2
    cdf = cumulative_trapezoid(p, x, initial=0.0)
    cdf /= cdf[-1]
    return x, cdf


def sample_density(
    n: int,
    l: int,
    m: int,
    count: int,
    Z: int = 1,
    mu_ratio: float = 1.0,
    seed: int = 0,
    progress: Callable[[float], None] | None = None,
    n_chunks: int = 10,
) -> SampleCloud:
    """Draw `count` positions from |psi_nlm|^2 (complex Y_lm basis)."""
    validate_quantum_numbers(n, l)
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")

    rng = np.random.default_rng(seed)
    r_grid, r_cdf, r_max = _radial_inverse_cdf(n, l, Z, mu_ratio)
    x_grid, x_cdf = _costheta_inverse_cdf(l, m)

    sizes = np.full(n_chunks, count // n_chunks)
    sizes[: count % n_chunks] += 1
    chunks: list[np.ndarray] = []
    done = 0
    for size in sizes:
        if size == 0:
            if progress is not None:
                progress(done / count if count else 1.0)
            continue
        r = np.interp(rng.random(size), r_cdf, r_grid)
        cos_t = np.interp(rng.random(size), x_cdf, x_grid)
        sin_t = np.sqrt(np.clip(1.0 - cos_t**2, 0.0, 1.0))
        phi = rng.uniform(0.0, 2.0 * np.pi, size)
        xyz = np.stack(
            [r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t], axis=1
        )
        chunks.append(xyz.astype(np.float32))
        done += int(size)
        if progress is not None:
            progress(done / count)

    positions = np.concatenate(chunks)
    provenance = Provenance(
        fidelity=Fidelity.NUMERICAL,
        method=(
            "factorized inverse-CDF Monte-Carlo of |psi_nlm|^2: "
            f"r from P(r)=r^2 R^2 (grid N={_R_GRID_POINTS}, r_max={r_max:g} bohr), "
            f"cos(theta) from |Theta_lm|^2 (grid N={_X_GRID_POINTS}), phi uniform"
        ),
        assumptions=(
            "complex spherical-harmonic basis (|Y_lm|^2 is phi-independent)",
            f"RNG PCG64 seed={seed}, count={count}",
            "positions in bohr",
        ),
        refinement="increase CDF grid resolution or sample count",
    )
    return SampleCloud(
        positions=positions, n=n, l=l, m=m, Z=Z, mu_ratio=mu_ratio, provenance=provenance
    )
