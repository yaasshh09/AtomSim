"""Electric-dipole transition strengths for hydrogen-like atoms.

The spectrum has energies and wavelengths; this adds intensities. From the exact
radial functions R_nl we integrate the radial dipole matrix element

    R = integral_0^inf  R_{n'l'}(r) * r * R_{nl}(r) * r^2 dr   [bohr]

by Gauss-Laguerre quadrature (NUMERICAL: the wavefunctions are exact and the
rule is exact for this integrand, but the residual roundoff is measured by
node-doubling and reported), then form the absorption oscillator strength, the
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
from functools import lru_cache

import numpy as np
from scipy import constants as _sc
from scipy.special import roots_laguerre

from atomsim.analytic.hydrogen import (
    _radial_eval,
    _validate_physical,
    energy,
    validate_quantum_numbers,
)
from atomsim.analytic.wigner import wigner_6j
from atomsim.constants import ALPHA
from atomsim.provenance import Fidelity, Provenance, Quantity

_T_AU = _sc.physical_constants["atomic unit of time"][0]  # seconds per atomic time unit

_ONE_ELECTRON = (
    "one-electron hydrogenic wavefunctions (exact)",
    "electric-dipole (E1) approximation; higher multipoles neglected",
    "no fine-structure/relativistic or QED correction to the rate",
)


def _gauss_laguerre_nodes(n: int, n2: int) -> int:
    """Node count that makes the dipole integral exact up to float64 roundoff.

    R_{n'l'}(r) r^3 R_nl(r) = exp(-a r) * P(r) with a = kappa (1/n + 1/n') and P
    a polynomial of degree (n-l-1) + (n'-l'-1) + l + l' + 3 = n + n' + 1. That is
    exactly the Gauss-Laguerre weight, and N nodes integrate degree 2N-1 exactly,
    so N = ceil((n + n' + 2) / 2) leaves no truncation error -- only roundoff.
    """
    return (n + n2 + 3) // 2


@lru_cache(maxsize=4096)
def _dipole_value_and_error(
    n: int, l: int, n2: int, l2: int, kappa: float
) -> tuple[float, float, int]:
    """Cached (value, roundoff estimate, node count). Keyed on the ordered pair.

    The integrand is symmetric under swapping the two states, so the caller
    canonicalizes the key and one entry serves both directions. A spectrum asks
    for the same handful of integrals for f and for A, and again on every
    redraw, so this turns the second and later asks into a dict lookup.
    """
    nodes = _gauss_laguerre_nodes(n, n2)
    coarse = _dipole_quadrature(n, l, n2, l2, kappa, nodes)
    fine = _dipole_quadrature(n, l, n2, l2, kappa, 2 * nodes)
    return fine, abs(fine - coarse), 2 * nodes


def _dipole_quadrature(n: int, l: int, n2: int, l2: int, kappa: float, nodes: int) -> float:
    """Gauss-Laguerre evaluation of int R_{n2 l2}(r) r^3 R_{n l}(r) dr, in bohr."""
    a = kappa * (1.0 / n + 1.0 / n2)
    x, w = roots_laguerre(nodes)
    r = x / a
    # The rule carries an exp(-x) weight; the wavefunctions already supply
    # exp(-a r) = exp(-x), so undo it once to leave the polynomial part.
    integrand = _radial_eval(n, l, r, kappa) * _radial_eval(n2, l2, r, kappa) * r**3 * np.exp(x)
    return float(np.dot(w, integrand) / a)


def dipole_radial_integral(
    n: int, l: int, n2: int, l2: int, Z: int = 1, mu_ratio: float = 1.0,
) -> Quantity:
    """Radial dipole matrix element <n2 l2 | r | n l> in bohr (symmetric in the pair)."""
    validate_quantum_numbers(n, l)
    validate_quantum_numbers(n2, l2)
    _validate_physical(Z, mu_ratio)
    kappa = Z * mu_ratio
    # Symmetric in the two states: canonicalize so both directions share a cache entry.
    a, b = sorted(((n, l), (n2, l2)))
    value, err, nodes = _dipole_value_and_error(a[0], a[1], b[0], b[1], kappa)
    return Quantity(
        value=value,
        unit="bohr",
        label=f"<{n2},{l2}|r|{n},{l}> (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method=(
                f"Gauss-Laguerre quadrature of the exact R_nl dipole integral "
                f"({nodes} nodes; the integrand is exp(-a r) times a degree-"
                f"{n + n2 + 1} polynomial, so the rule is exact bar roundoff)"
            ),
            assumptions=_ONE_ELECTRON,
            error_estimate=err,
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


def _validate_j(l: int, j: float, name: str) -> None:
    """A one-electron level has j = l +/- 1/2 only (and j = 1/2 when l = 0)."""
    allowed = [l - 0.5, l + 0.5] if l > 0 else [0.5]
    if not any(abs(j - a) < 1e-9 for a in allowed):
        raise ValueError(
            f"{name}: j must be l +/- 1/2 for l = {l} (allowed {allowed}), got {j}"
        )


def _fine_branching(l_up: int, j_up: float, l_low: int, j_low: float) -> float:
    """(2 j_low + 1) {j_low 1 j_up; l_up 1/2 l_low}^2, the j-branching factor.

    Summed over j_low this equals 1 / (2 l_up + 1), which is exactly the factor
    the gross-structure rate carries -- so the components always add back up to
    the unresolved rate. Forbidden combinations give 0 through the 6j's triangle
    conditions, with no separate Delta j test.
    """
    return (2.0 * j_low + 1.0) * wigner_6j(j_low, 1, j_up, l_up, 0.5, l_low) ** 2


_FINE_METHOD = (
    "A = (4/3) alpha^3 dE^3 (2j+1) {j 1 j'; l' 1/2 l}^2 l_max |R|^2 / t_au; "
    "the 6j symbol splits the multiplet rate across j (spin is a spectator)"
)


def einstein_A_fine(
    n_up: int, l_up: int, j_up: float,
    n_low: int, l_low: int, j_low: float,
    Z: int = 1, mu_ratio: float = 1.0,
    dE_hartree: float | None = None,
) -> Quantity:
    """Spontaneous emission rate for one fine-structure component, in s^-1.

    `dE_hartree` is the true transition energy. It matters: A scales as dE^3,
    and a within-n component such as 2p_3/2 -> 2s_1/2 has *no* gross energy
    difference at all, so falling back to the n-only value would divide a real
    microwave transition down to zero. Callers that know the fine-structure
    energies (the spectrum builder does) should pass them. Omitting it uses the
    gross difference, which is right to order alpha^2 for a genuine n -> n' line.
    """
    validate_quantum_numbers(n_up, l_up)
    validate_quantum_numbers(n_low, l_low)
    _validate_j(l_up, j_up, "upper level")
    _validate_j(l_low, j_low, "lower level")
    dE = (
        dE_hartree if dE_hartree is not None
        else energy(n_up, Z=Z, mu_ratio=mu_ratio).value
        - energy(n_low, Z=Z, mu_ratio=mu_ratio).value
    )
    label = f"A {n_up}{l_up}(j={j_up})->{n_low}{l_low}(j={j_low})"
    if abs(l_up - l_low) != 1 or dE <= 0.0:
        return _forbidden("no E1 decay channel", label, "s^-1")
    branch = _fine_branching(l_up, j_up, l_low, j_low)
    if branch == 0.0:
        return _forbidden("6j triangle rule (Delta j = 0, +/-1)", label, "s^-1")
    R = dipole_radial_integral(n_up, l_up, n_low, l_low, Z=Z, mu_ratio=mu_ratio)
    l_max = max(l_up, l_low)
    a_s = (4.0 / 3.0) * ALPHA**3 * dE**3 * branch * l_max * R.value**2 / _T_AU
    rerr = R.provenance.error_estimate or 0.0
    return Quantity(
        value=a_s,
        unit="s^-1",
        label=label,
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method=_FINE_METHOD,
            assumptions=_ONE_ELECTRON
            + ("rate resolved by j; transition energy is the gross (n-only) value",),
            error_estimate=2.0 * abs(a_s) * (rerr / abs(R.value)) if R.value else 0.0,
            refinement=R.provenance.refinement,
        ),
    )


def oscillator_strength_fine(
    n_low: int, l_low: int, j_low: float,
    n_up: int, l_up: int, j_up: float,
    Z: int = 1, mu_ratio: float = 1.0,
    dE_hartree: float | None = None,
) -> Quantity:
    """Absorption oscillator strength for one fine-structure component.

    `dE_hartree` is the true transition energy; see `einstein_A_fine` for why
    the gross value will not do for a within-n component.
    """
    validate_quantum_numbers(n_low, l_low)
    validate_quantum_numbers(n_up, l_up)
    _validate_j(l_low, j_low, "lower level")
    _validate_j(l_up, j_up, "upper level")
    dE = (
        dE_hartree if dE_hartree is not None
        else energy(n_up, Z=Z, mu_ratio=mu_ratio).value
        - energy(n_low, Z=Z, mu_ratio=mu_ratio).value
    )
    if dE <= 0.0:
        raise ValueError(
            "absorption requires the upper level above the lower "
            f"(got E({n_up}) <= E({n_low}))"
        )
    label = f"f {n_low}{l_low}(j={j_low})->{n_up}{l_up}(j={j_up})"
    if abs(l_up - l_low) != 1:
        return _forbidden("Delta l != +/-1", label, "dimensionless")
    # Same 6j, columns swapped: here the upper level's degeneracy is the weight.
    branch = _fine_branching(l_low, j_low, l_up, j_up)
    if branch == 0.0:
        return _forbidden("6j triangle rule (Delta j = 0, +/-1)", label, "dimensionless")
    R = dipole_radial_integral(n_low, l_low, n_up, l_up, Z=Z, mu_ratio=mu_ratio)
    l_max = max(l_low, l_up)
    f = (2.0 / 3.0) * dE * branch * l_max * R.value**2
    rerr = R.provenance.error_estimate or 0.0
    return Quantity(
        value=f,
        unit="dimensionless",
        label=label,
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method=(
                "f = (2/3) dE (2j'+1) {j' 1 j; l 1/2 l'}^2 l_max |R|^2; "
                "the 6j symbol splits the multiplet strength across j"
            ),
            assumptions=_ONE_ELECTRON
            + ("strength resolved by j; transition energy is the gross (n-only) value",),
            error_estimate=2.0 * abs(f) * (rerr / abs(R.value)) if R.value else 0.0,
            refinement=R.provenance.refinement,
        ),
    )


def lifetime_fine(
    n: int, l: int, j: float, Z: int = 1, mu_ratio: float = 1.0
) -> Quantity:
    """Radiative lifetime of the fine-structure level (n, l, j), in seconds."""
    validate_quantum_numbers(n, l)
    _validate_j(l, j, "level")
    total = 0.0
    var = 0.0
    for n2 in range(1, n):
        for l2 in (l - 1, l + 1):
            if not 0 <= l2 < n2:
                continue
            for j2 in ([l2 - 0.5, l2 + 0.5] if l2 > 0 else [0.5]):
                a = einstein_A_fine(n, l, j, n2, l2, j2, Z=Z, mu_ratio=mu_ratio)
                total += a.value
                var += (a.provenance.error_estimate or 0.0) ** 2
    if total <= 0.0:
        value, err = math.inf, 0.0
    else:
        value, err = 1.0 / total, math.sqrt(var) / total**2
    return Quantity(
        value=value,
        unit="s",
        label=f"tau {n}{l}(j={j}) (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.NUMERICAL,
            method="tau = 1 / sum A(n l j -> n' l' j'), E1 channels only",
            assumptions=_ONE_ELECTRON
            + ("sum over all lower dipole-allowed fine levels (n' < n)",),
            error_estimate=err,
            refinement="include higher multipoles and QED corrections to the rate",
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
