"""The full hydrogen-like wavefunctions, psi_nlm(r) = R_nl(r) Y_lm(theta, phi).

Both angular bases are first-class (spec 5.1). The complex basis returns
complex128 values, whose phase feeds the phase-as-hue rendering; the real basis
returns float64. Positions are Cartesian, in bohr.
"""

from dataclasses import dataclass

import numpy as np

from atomsim.analytic.angular import spherical_harmonic, validate_angular
from atomsim.analytic.hydrogen import radial_wavefunction, validate_quantum_numbers
from atomsim.provenance import Fidelity, Provenance


@dataclass(frozen=True)
class WavefunctionValues:
    """psi_nlm evaluated at Cartesian points (bohr). The container carries its provenance."""

    values: np.ndarray      # (N,) complex128 or float64, unit bohr^-3/2
    positions: np.ndarray   # (N, 3) float, bohr
    n: int
    l: int
    m: int
    Z: int
    mu_ratio: float
    basis: str
    provenance: Provenance


def evaluate_state(
    n: int,
    l: int,
    m: int,
    positions: np.ndarray,
    Z: int = 1,
    mu_ratio: float = 1.0,
    basis: str = "complex",
) -> WavefunctionValues:
    """Evaluate psi_nlm at (N, 3) Cartesian positions in bohr."""
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")

    r = np.linalg.norm(pos, axis=1)
    safe_r = np.where(r > 0.0, r, 1.0)
    theta = np.arccos(np.clip(pos[:, 2] / safe_r, -1.0, 1.0))
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    theta = np.where(r > 0.0, theta, 0.0)

    radial = radial_wavefunction(n, l, r, Z=Z, mu_ratio=mu_ratio)
    angular = spherical_harmonic(l, m, theta, phi, basis=basis)
    values = radial.values * angular.values

    return WavefunctionValues(
        values=values,
        positions=pos,
        n=n,
        l=l,
        m=m,
        Z=Z,
        mu_ratio=mu_ratio,
        basis=basis,
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method=(
                "psi_nlm = R_nl (closed-form Laguerre) x "
                f"{angular.provenance.method}"
            ),
            assumptions=radial.provenance.assumptions + angular.provenance.assumptions
            + ("values in bohr^-3/2 at Cartesian positions in bohr",),
        ),
    )
