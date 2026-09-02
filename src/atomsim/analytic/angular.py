"""Angular structure: complex spherical harmonics Y_lm and real orbitals S_lm.

Both bases are first-class outputs (spec 5.1). The complex Y_lm carry the
Condon-Shortley phase (scipy's convention). The real orbitals use the standard
chemistry combinations: m > 0 gives the cos(m phi) type, m < 0 the sin(|m| phi)
type. Which basis is in play stays visible, since it is the chemistry versus
physics teaching moment, so `basis` rides in the provenance-carrying result and
is never left a hidden default.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import sph_harm_y

from atomsim.provenance import Fidelity, Provenance

_L_LETTERS = "spdfghik"

_CHEMISTRY_LABELS = {
    (0, 0): "s",
    (1, 0): "p_z", (1, 1): "p_x", (1, -1): "p_y",
    (2, 0): "d_z2", (2, 1): "d_xz", (2, -1): "d_yz",
    (2, 2): "d_x2-y2", (2, -2): "d_xy",
    (3, 0): "f_z3", (3, 1): "f_xz2", (3, -1): "f_yz2",
    (3, 2): "f_z(x2-y2)", (3, -2): "f_xyz",
    (3, 3): "f_x(x2-3y2)", (3, -3): "f_y(3x2-y2)",
}


@dataclass(frozen=True)
class AngularValues:
    """Y_lm or S_lm evaluated on (theta, phi) points, with their provenance."""

    values: np.ndarray  # complex128 for basis="complex", float64 for "real"
    theta: np.ndarray
    phi: np.ndarray
    l: int
    m: int
    basis: str
    provenance: Provenance


def validate_angular(l: int, m: int) -> None:
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l}")
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")


def real_orbital_label(l: int, m: int) -> str:
    validate_angular(l, m)
    if (l, m) in _CHEMISTRY_LABELS:
        return _CHEMISTRY_LABELS[(l, m)]
    letter = _L_LETTERS[l] if l < len(_L_LETTERS) else f"(l={l})"
    if m == 0:
        return f"{letter}(m=0)"
    kind = "cos" if m > 0 else "sin"
    return f"{letter}(m={m:+d}, {kind})"


def spherical_harmonic(
    l: int, m: int, theta: np.ndarray, phi: np.ndarray, basis: str = "complex"
) -> AngularValues:
    """Evaluate Y_lm (complex, Condon-Shortley) or S_lm (real) on the angles given."""
    validate_angular(l, m)
    if basis not in ("complex", "real"):
        raise ValueError(f"basis must be 'complex' or 'real', got {basis!r}")
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    if basis == "complex":
        values: np.ndarray = sph_harm_y(l, m, theta, phi)
        method = "complex spherical harmonic Y_lm (Condon-Shortley phase, scipy sph_harm_y)"
        assumptions = ("physics convention: eigenstates of L_z",)
    else:
        y_abs = sph_harm_y(l, abs(m), theta, phi)
        sign = (-1.0) ** abs(m)
        if m == 0:
            values = np.real(y_abs)
        elif m > 0:
            values = sign * np.sqrt(2.0) * np.real(y_abs)
        else:
            values = sign * np.sqrt(2.0) * np.imag(y_abs)
        method = (
            "real spherical harmonic S_lm = sqrt(2) (-1)^m Re/Im Y_l|m| "
            "(chemistry convention: m>0 cos-type, m<0 sin-type)"
        )
        assumptions = (
            "chemistry convention: NOT eigenstates of L_z for m != 0",
            f"orbital label: {real_orbital_label(l, m)}",
        )

    return AngularValues(
        values=values,
        theta=theta,
        phi=phi,
        l=l,
        m=m,
        basis=basis,
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method=method,
            assumptions=assumptions,
        ),
    )
