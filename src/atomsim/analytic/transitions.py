"""Electric-dipole transition strengths for hydrogen-like atoms.

The spectrum has energies and wavelengths; this adds intensities. From the exact
radial functions R_nl we integrate the radial dipole matrix element

    R = integral_0^inf  R_{n'l'}(r) * r * R_{nl}(r) * r^2 dr   [bohr]

by adaptive quadrature (NUMERICAL: the wavefunctions are exact, the integral
carries its quadrature error), then form the absorption oscillator strength, the
Einstein A (spontaneous emission rate), and the radiative lifetime:

    f_abs(nl -> n'l') = (2/3) dE (l_max / (2l+1)) |R|^2
    A_emit(n'l' -> nl) = (4/3) alpha^3 dE^3 (l_max / (2l'+1)) |R|^2   [1/t_au]
    tau(n'l') = 1 / sum A_emit

l is the lower level's l, l_max = max(l, l'), dE > 0 in hartree. Electric-dipole
selection rule l' = l +/- 1 is exact: other pairs return a disclosed zero.
One-electron hydrogenic; no fine structure or QED in the rates. See
docs/superpowers/specs/2026-07-24-phase13-transition-strengths-design.md.
"""

import math

import numpy as np
from scipy import constants as _sc
from scipy.integrate import quad

from atomsim.analytic.hydrogen import (
    _radial_eval,
    _validate_physical,
    energy,
    validate_quantum_numbers,
)
from atomsim.constants import ALPHA
from atomsim.provenance import Fidelity, Provenance, Quantity

_T_AU = _sc.physical_constants["atomic unit of time"][0]  # seconds per atomic time unit

_ONE_ELECTRON = (
    "one-electron hydrogenic wavefunctions (exact)",
    "electric-dipole (E1) approximation; higher multipoles neglected",
    "no fine-structure/relativistic or QED correction to the rate",
)


def _radial_func(n: int, l: int, Z: int, mu_ratio: float):
    """Closure R_nl(r) over hydrogen's own closed form (identical normalization)."""
    validate_quantum_numbers(n, l)
    _validate_physical(Z, mu_ratio)
    kappa = Z * mu_ratio
    return lambda r: _radial_eval(n, l, r, kappa)


def dipole_radial_integral(
    n: int, l: int, n2: int, l2: int, Z: int = 1, mu_ratio: float = 1.0,
) -> Quantity:
    """Radial dipole matrix element <n2 l2 | r | n l> in bohr (symmetric in the pair)."""
    r1 = _radial_func(n, l, Z, mu_ratio)
    r2 = _radial_func(n2, l2, Z, mu_ratio)
    value, abserr = quad(lambda r: r1(r) * r2(r) * r**3, 0.0, np.inf, limit=200)
    return Quantity(
        value=value,
        unit="bohr",
        label=f"<{n2},{l2}|r|{n},{l}> (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method="adaptive quadrature of the exact R_nl dipole integral (scipy quad)",
            assumptions=_ONE_ELECTRON,
            error_estimate=abserr,
            refinement="closed-form Gordon hypergeometric dipole integral",
        ),
    )


def _forbidden(kind: str, label: str, unit: str) -> Quantity:
    """An exact zero from the E1 selection rule (disclosed, never silent)."""
    return Quantity(
        value=0.0,
        unit=unit,
        label=label,
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method=f"electric-dipole selection rule: {kind} (Delta l = +/-1 required)",
            assumptions=("selection-rule zero, not a numerical underflow",),
            error_estimate=0.0,
        ),
    )


def oscillator_strength(
    n_low: int, l_low: int, n_up: int, l_up: int, Z: int = 1, mu_ratio: float = 1.0,
) -> Quantity:
    """Absorption oscillator strength f for nl -> n'l' (dimensionless)."""
    validate_quantum_numbers(n_low, l_low)
    validate_quantum_numbers(n_up, l_up)
    dE = energy(n_up, Z=Z, mu_ratio=mu_ratio).value - energy(n_low, Z=Z, mu_ratio=mu_ratio).value
    if dE <= 0.0:
        raise ValueError(
            "absorption requires the upper level above the lower "
            f"(got E({n_up}) <= E({n_low}))"
        )
    label = f"f {n_low}{l_low}->{n_up}{l_up}"
    if abs(l_up - l_low) != 1:
        return _forbidden("Delta l != +/-1", label, "dimensionless")
    R = dipole_radial_integral(n_low, l_low, n_up, l_up, Z=Z, mu_ratio=mu_ratio)
    l_max = max(l_low, l_up)
    f = (2.0 / 3.0) * dE * (l_max / (2.0 * l_low + 1.0)) * R.value**2
    rerr = (R.provenance.error_estimate or 0.0)
    f_err = 2.0 * abs(f) * (rerr / abs(R.value)) if R.value != 0.0 else 0.0
    return Quantity(
        value=f,
        unit="dimensionless",
        label=label,
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method="f = (2/3) dE (l_max/(2l+1)) |R|^2, R from dipole_radial_integral",
            assumptions=_ONE_ELECTRON,
            error_estimate=f_err,
            refinement=R.provenance.refinement,
        ),
    )


def einstein_A(
    n_up: int, l_up: int, n_low: int, l_low: int, Z: int = 1, mu_ratio: float = 1.0,
) -> Quantity:
    """Spontaneous emission rate A for n'l' -> nl, in s^-1 (0 if not a decay channel)."""
    validate_quantum_numbers(n_up, l_up)
    validate_quantum_numbers(n_low, l_low)
    dE = energy(n_up, Z=Z, mu_ratio=mu_ratio).value - energy(n_low, Z=Z, mu_ratio=mu_ratio).value
    label = f"A {n_up}{l_up}->{n_low}{l_low}"
    if abs(l_up - l_low) != 1 or dE <= 0.0:
        return _forbidden("no E1 decay channel", label, "s^-1")
    R = dipole_radial_integral(n_up, l_up, n_low, l_low, Z=Z, mu_ratio=mu_ratio)
    l_max = max(l_up, l_low)
    a_au = (4.0 / 3.0) * ALPHA**3 * dE**3 * (l_max / (2.0 * l_up + 1.0)) * R.value**2
    a_s = a_au / _T_AU
    rerr = (R.provenance.error_estimate or 0.0)
    a_err = 2.0 * abs(a_s) * (rerr / abs(R.value)) if R.value != 0.0 else 0.0
    return Quantity(
        value=a_s,
        unit="s^-1",
        label=label,
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method="A = (4/3) alpha^3 dE^3 (l_max/(2l'+1)) |R|^2 / t_au (atomic-time unit)",
            assumptions=_ONE_ELECTRON,
            error_estimate=a_err,
            refinement=R.provenance.refinement,
        ),
    )


def lifetime(n: int, l: int, Z: int = 1, mu_ratio: float = 1.0) -> Quantity:
    """Radiative lifetime tau = 1 / sum(A) of level (n, l), in seconds.

    Infinite for a level with no E1 decay channel (e.g. the 1s ground state).
    """
    validate_quantum_numbers(n, l)
    total = 0.0
    var = 0.0
    for n2 in range(1, n):                 # hydrogen energy depends on n only
        for l2 in (l - 1, l + 1):
            if 0 <= l2 < n2:
                a = einstein_A(n, l, n2, l2, Z=Z, mu_ratio=mu_ratio)
                total += a.value
                var += (a.provenance.error_estimate or 0.0) ** 2
    if total <= 0.0:
        value, err = math.inf, 0.0
    else:
        value, err = 1.0 / total, math.sqrt(var) / total**2
    return Quantity(
        value=value,
        unit="s",
        label=f"tau {n}{l} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method="tau = 1 / sum_{lower} A(n l -> n' l'), E1 channels only",
            assumptions=_ONE_ELECTRON
            + ("sum over all lower dipole-allowed levels (n' < n)",),
            error_estimate=err,
            refinement="include fine-structure-resolved rates and higher multipoles",
        ),
    )
