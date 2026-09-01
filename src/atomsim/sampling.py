"""How I sample |psi_nlm|^2 by Monte-Carlo. My sampling IS physics, so it
carries provenance.

I factorize the inverse-CDF draw: r from P(r) = r^2 R_nl^2 and cos(theta) from
the normalized |Theta_lm|^2 in both bases. I draw phi uniformly in the complex
basis, because |Y_lm|^2 does not depend on it, and from the analytic
cos^2/sin^2(m phi) marginal for real orbitals, because |S_lm|^2 stays separable
in theta and phi.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import lpmv

from atomsim.analytic.hydrogen import radial_wavefunction, validate_quantum_numbers
from atomsim.atoms import Configuration
from atomsim.hf_atom import hf_radial
from atomsim.numerics.screening import screening_provenance
from atomsim.provenance import Fidelity, Provenance
from atomsim.screened_atom import screened_radial

_R_GRID_POINTS = 8192
_X_GRID_POINTS = 4096


@dataclass(frozen=True)
class SampleCloud:
    """The positions I sampled from |psi_nlm|^2, in bohr, with my provenance."""

    positions: np.ndarray  # (count, 3) float32
    n: int
    l: int
    m: int
    Z: int
    mu_ratio: float
    basis: str
    provenance: Provenance


def _radial_inverse_cdf_tabulated(r_grid: np.ndarray, R_values: np.ndarray):
    """My r grid and the CDF of P(r) = r^2 R^2, from a tabulated R_nl."""
    p = r_grid * r_grid * R_values * R_values
    cdf = cumulative_trapezoid(p, r_grid, initial=0.0)
    cdf /= cdf[-1]
    return r_grid, cdf, float(r_grid[-1])


def _radial_inverse_cdf(n: int, l: int, Z: int, mu_ratio: float):
    """My r grid and the CDF of P(r) = r^2 R_nl^2, from the analytic R_nl."""
    r_max = 20.0 * n * n / (Z * mu_ratio)  # P(r_max)/P_peak < 1e-15 for every l < n
    r = np.linspace(0.0, r_max, _R_GRID_POINTS)
    R = radial_wavefunction(n, l, r, Z=Z, mu_ratio=mu_ratio).values
    return _radial_inverse_cdf_tabulated(r, R)


def _costheta_inverse_cdf(l: int, m: int):
    """My x = cos(theta) grid and the CDF of |Theta_lm|^2. The normalization cancels."""
    x = np.linspace(-1.0, 1.0, _X_GRID_POINTS)
    p = lpmv(abs(m), l, x) ** 2
    cdf = cumulative_trapezoid(p, x, initial=0.0)
    cdf /= cdf[-1]
    return x, cdf


def _phi_inverse_cdf(m: int):
    """My phi grid and the CDF of the real-basis phi marginal (cos^2/sin^2 type)."""
    phi = np.linspace(0.0, 2.0 * np.pi, _X_GRID_POINTS)
    am = abs(m)
    if m > 0:
        cdf = (phi / 2.0 + np.sin(2.0 * am * phi) / (4.0 * am)) / np.pi
    else:
        cdf = (phi / 2.0 - np.sin(2.0 * am * phi) / (4.0 * am)) / np.pi
    cdf /= cdf[-1]
    return phi, cdf


def _draw_positions(count, r_grid, r_cdf, x_grid, x_cdf, phi_sampler, seed, n_chunks, progress):
    """I draw `count` Cartesian positions (bohr) from my factorized CDFs."""
    rng = np.random.default_rng(seed)
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
        if phi_sampler is None:
            phi = rng.uniform(0.0, 2.0 * np.pi, size)
        else:
            phi = np.interp(rng.random(size), phi_sampler[1], phi_sampler[0])
        xyz = np.stack(
            [r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t], axis=1
        )
        chunks.append(xyz.astype(np.float32))
        done += int(size)
        if progress is not None:
            progress(done / count)
    return np.concatenate(chunks)


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
    basis: str = "complex",
) -> SampleCloud:
    """I draw `count` positions from |psi_nlm|^2 in the angular basis you name."""
    validate_quantum_numbers(n, l)
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if basis not in ("complex", "real"):
        raise ValueError(f"basis must be 'complex' or 'real', got {basis!r}")

    phi_sampler = _phi_inverse_cdf(m) if (basis == "real" and m != 0) else None
    r_grid, r_cdf, r_max = _radial_inverse_cdf(n, l, Z, mu_ratio)
    x_grid, x_cdf = _costheta_inverse_cdf(l, m)
    positions = _draw_positions(
        count, r_grid, r_cdf, x_grid, x_cdf, phi_sampler, seed, n_chunks, progress
    )
    phi_desc = (
        "phi uniform (|Y_lm|^2 is phi-independent)"
        if phi_sampler is None
        else "phi from analytic real-basis marginal (cos^2/sin^2 m phi)"
    )
    provenance = Provenance(
        fidelity=Fidelity.NUMERICAL,
        method=(
            f"I sampled |psi_nlm|^2 by factorized inverse-CDF Monte-Carlo "
            f"({basis} basis): r from P(r)=r^2 R^2 (grid N={_R_GRID_POINTS}, "
            f"r_max={r_max:g} bohr), cos(theta) from |Theta_lm|^2 "
            f"(grid N={_X_GRID_POINTS}), {phi_desc}"
        ),
        assumptions=(
            f"I used the {basis} angular basis",
            f"I drew from RNG PCG64 seed={seed}, count={count}",
            "I report positions in bohr",
        ),
        refinement="I could raise my CDF grid resolution or my sample count",
    )
    return SampleCloud(
        positions=positions, n=n, l=l, m=m, Z=Z, mu_ratio=mu_ratio,
        basis=basis, provenance=provenance,
    )


def sample_screened_density(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    count: int,
    *,
    seed: int = 0,
    progress: Callable[[float], None] | None = None,
    n_chunks: int = 10,
    basis: str = "complex",
) -> SampleCloud:
    """I draw `count` positions from |psi_nlm|^2 for a screened GSZ/GJG atom.

    My radial source is the numerical screened R_nl, and my angular part is the
    same central-field Y_lm I use for hydrogen. I call this APPROXIMATION,
    because the model error dominates.
    """
    validate_quantum_numbers(n, l)
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if basis not in ("complex", "real"):
        raise ValueError(f"basis must be 'complex' or 'real', got {basis!r}")

    r_field, _ = screened_radial(z, n_electrons, n, l, points=_R_GRID_POINTS)
    r_grid, r_cdf, r_max = _radial_inverse_cdf_tabulated(r_field.grid, r_field.values)
    x_grid, x_cdf = _costheta_inverse_cdf(l, m)
    phi_sampler = _phi_inverse_cdf(m) if (basis == "real" and m != 0) else None
    positions = _draw_positions(
        count, r_grid, r_cdf, x_grid, x_cdf, phi_sampler, seed, n_chunks, progress
    )

    base = screening_provenance(z, n_electrons)
    phi_desc = (
        "phi uniform (|Y_lm|^2 is phi-independent)"
        if phi_sampler is None
        else "phi from analytic real-basis marginal (cos^2/sin^2 m phi)"
    )
    provenance = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=(
            f"I sampled |psi_nlm|^2 by factorized inverse-CDF Monte-Carlo over a "
            f"numerical screened R_nl ({basis} basis): r from P(r)=r^2 R^2 "
            f"(grid N={r_grid.size}, r_max={r_max:g} bohr), cos(theta) from "
            f"|Theta_lm|^2, {phi_desc}; {base.method}"
        ),
        assumptions=base.assumptions
        + (
            f"I used the {basis} angular basis",
            f"I drew from RNG PCG64 seed={seed}, count={count}",
            "I report positions in bohr",
        ),
        error_estimate=r_field.provenance.error_estimate,
        refinement=(
            "I could raise my CDF grid resolution, my sample count, "
            "or my radial solver resolution"
        ),
    )
    return SampleCloud(
        positions=positions, n=n, l=l, m=m, Z=z, mu_ratio=1.0,
        basis=basis, provenance=provenance,
    )


def sample_hf_density(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    count: int,
    *,
    seed: int = 0,
    progress: Callable[[float], None] | None = None,
    n_chunks: int = 10,
    basis: str = "complex",
    config: Configuration | None = None,
    exchange: bool = True,
    pauli: bool = True,
) -> SampleCloud:
    """I draw `count` positions from |psi_nlm|^2 for a Hartree-Fock orbital.

    I keep the same shape as sample_screened_density, and for the same reason:
    my radial source is a tabulated numerical R_nl and my angular part is the
    central-field Y_lm, so only the first line differs between the two models.
    Keeping them identical below that line is what lets me put the two models
    in the same camera.

    I inherit my fidelity from the solve rather than asserting it here, so the
    two counterfactual switches carry through to the badge on my cloud.
    """
    validate_quantum_numbers(n, l)
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if basis not in ("complex", "real"):
        raise ValueError(f"basis must be 'complex' or 'real', got {basis!r}")

    r_field, _ = hf_radial(
        z, n_electrons, n, l, points=_R_GRID_POINTS,
        config=config, exchange=exchange, pauli=pauli,
    )
    r_grid, r_cdf, r_max = _radial_inverse_cdf_tabulated(r_field.grid, r_field.values)
    x_grid, x_cdf = _costheta_inverse_cdf(l, m)
    phi_sampler = _phi_inverse_cdf(m) if (basis == "real" and m != 0) else None
    positions = _draw_positions(
        count, r_grid, r_cdf, x_grid, x_cdf, phi_sampler, seed, n_chunks, progress
    )

    base = r_field.provenance
    phi_desc = (
        "phi uniform (|Y_lm|^2 is phi-independent)"
        if phi_sampler is None
        else "phi from analytic real-basis marginal (cos^2/sin^2 m phi)"
    )
    provenance = Provenance(
        fidelity=base.fidelity,
        method=(
            f"I sampled |psi_nlm|^2 by factorized inverse-CDF Monte-Carlo over a "
            f"Hartree-Fock R_nl ({basis} basis): r from P(r)=r^2 R^2 "
            f"(grid N={r_grid.size}, r_max={r_max:g} bohr), cos(theta) from "
            f"|Theta_lm|^2, {phi_desc}; {base.method}"
        ),
        assumptions=base.assumptions
        + (
            f"I used the {basis} angular basis",
            f"I drew from RNG PCG64 seed={seed}, count={count}",
            "I report positions in bohr",
        ),
        error_estimate=base.error_estimate,
        refinement=base.refinement,
    )
    return SampleCloud(
        positions=positions, n=n, l=l, m=m, Z=z, mu_ratio=1.0,
        basis=basis, provenance=provenance,
    )
