"""Exact analytic solutions for hydrogen-like (one-electron) atoms.

Hartree atomic units (hbar = m_e = e = 1/(4 pi eps0) = 1). The reduced-mass
ratio mu_ratio = mu/m_e makes isotopes, muonic atoms, and positronium exact
within the same formulas. This module is also the ground truth that validates
the numerical radial solver.
"""

import math

import numpy as np
from scipy.special import eval_genlaguerre

from atomsim.provenance import Fidelity, Provenance, Quantity

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


def energy(n: int, Z: int = 1, mu_ratio: float = 1.0) -> Quantity:
    """Exact bound-state energy E_n = -mu_ratio * Z^2 / (2 n^2), in hartree."""
    validate_quantum_numbers(n)
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


def radial_wavefunction(
    n: int, l: int, r: np.ndarray, Z: int = 1, mu_ratio: float = 1.0
) -> np.ndarray:
    """Normalized radial wavefunction R_nl(r) in atomic units (r in bohr).

    R_nl = N * exp(-rho/2) * rho^l * L_{n-l-1}^{2l+1}(rho),  rho = 2 Z mu' r / n.
    Reliable for n <= 20 (float64 generalized-Laguerre evaluation).
    """
    validate_quantum_numbers(n, l)
    kappa = Z * mu_ratio
    rho = 2.0 * kappa * np.asarray(r, dtype=float) / n
    norm = math.sqrt(
        (2.0 * kappa / n) ** 3
        * math.factorial(n - l - 1)
        / (2.0 * n * math.factorial(n + l))
    )
    return norm * np.exp(-rho / 2.0) * rho**l * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)


def mean_radius(n: int, l: int, Z: int = 1, mu_ratio: float = 1.0) -> float:
    """Exact <r> = (3 n^2 - l(l+1)) / (2 Z mu'), in bohr."""
    validate_quantum_numbers(n, l)
    return (3.0 * n**2 - l * (l + 1)) / (2.0 * Z * mu_ratio)
