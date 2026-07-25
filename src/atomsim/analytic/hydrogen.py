"""Exact analytic solutions for hydrogen-like (one-electron) atoms.

Hartree atomic units (hbar = m_e = e = 1/(4 pi eps0) = 1). The reduced-mass
ratio mu_ratio = mu/m_e makes isotopes, muonic atoms, and positronium exact
within the same formulas. This module is also the ground truth that validates
the numerical radial solver.
"""

import math

import numpy as np
from scipy.special import eval_genlaguerre

from atomsim.provenance import Fidelity, Field, Provenance, Quantity

_EXACT_ASSUMPTIONS = (
    "non-relativistic Schrodinger equation",
    "point nucleus (no finite-size or QED effects)",
    "nuclear motion included only via reduced-mass ratio",
    "no external fields",
)


def validate_quantum_numbers(n: int, l: int = 0) -> None:
    if n < 1:
        raise ValueError(f"principal quantum number n must be >= 1, got {n}")
    if not 0 <= l < n:
        raise ValueError(f"orbital quantum number l must satisfy 0 <= l < n, got l={l}, n={n}")


def _validate_physical(Z: int, mu_ratio: float) -> None:
    if Z < 1:
        raise ValueError(f"nuclear charge Z must be >= 1, got {Z}")
    if not mu_ratio > 0:
        raise ValueError(f"reduced-mass ratio must be positive, got {mu_ratio}")


def energy(n: int, Z: int = 1, mu_ratio: float = 1.0) -> Quantity:
    """Exact bound-state energy E_n = -mu_ratio * Z^2 / (2 n^2), in hartree."""
    validate_quantum_numbers(n)
    _validate_physical(Z, mu_ratio)
    value = -mu_ratio * Z**2 / (2.0 * n**2)
    return Quantity(
        value=value,
        unit="hartree",
        label=f"E_{n} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method="closed-form Bohr formula E_n = -mu' Z^2 / (2 n^2)",
            assumptions=_EXACT_ASSUMPTIONS,
        ),
    )


def _radial_eval(n: int, l: int, r: np.ndarray, kappa: float) -> np.ndarray:
    """Bare R_nl(r) values, no Field wrapper -- the one copy of the closed form.

    Callers must have validated (n, l, Z, mu_ratio) already. `transitions.py`
    uses this directly inside its quadrature inner loop, so the normalization
    lives here once rather than being transcribed per module.
    """
    rho = 2.0 * kappa * r / n
    norm = math.sqrt(
        (2.0 * kappa / n) ** 3
        * math.factorial(n - l - 1)
        / (2.0 * n * math.factorial(n + l))
    )
    return norm * np.exp(-rho / 2.0) * rho**l * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)


def radial_wavefunction(
    n: int, l: int, r: np.ndarray, Z: int = 1, mu_ratio: float = 1.0
) -> Field:
    """Normalized radial wavefunction R_nl(r) in atomic units (r in bohr).

    R_nl = N * exp(-rho/2) * rho^l * L_{n-l-1}^{2l+1}(rho),  rho = 2 Z mu' r / n.
    """
    validate_quantum_numbers(n, l)
    _validate_physical(Z, mu_ratio)
    grid = np.asarray(r, dtype=float)
    values = _radial_eval(n, l, grid, Z * mu_ratio)
    return Field(
        values=values,
        grid=grid,
        unit="bohr^-3/2",
        grid_unit="bohr",
        label=f"R_{n},{l} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method="closed-form R_nl (normalized generalized-Laguerre form)",
            assumptions=_EXACT_ASSUMPTIONS
            + ("float64 Laguerre evaluation reliable for n <= 20",),
        ),
    )


def mean_radius(n: int, l: int, Z: int = 1, mu_ratio: float = 1.0) -> Quantity:
    """Exact <r> = (3 n^2 - l(l+1)) / (2 Z mu'), in bohr."""
    validate_quantum_numbers(n, l)
    _validate_physical(Z, mu_ratio)
    value = (3.0 * n**2 - l * (l + 1)) / (2.0 * Z * mu_ratio)
    return Quantity(
        value=value,
        unit="bohr",
        label=f"<r>_{n},{l} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method="closed-form <r> = (3 n^2 - l(l+1)) / (2 Z mu')",
            assumptions=_EXACT_ASSUMPTIONS,
        ),
    )


def angular_momentum_magnitude(l: int) -> Quantity:
    """|L| = sqrt(l(l+1)) hbar, the exact magnitude from the L^2 eigenvalue."""
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l}")
    return Quantity(
        value=float(np.sqrt(l * (l + 1))),
        unit="hbar",
        label=f"|L| (l={l})",
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method="sqrt(l(l+1)) hbar from the L^2 eigenvalue of the (n,l,m) eigenstate",
            assumptions=_EXACT_ASSUMPTIONS,
        ),
    )
