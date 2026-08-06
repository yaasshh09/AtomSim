"""Zeeman effect: fine structure + linear Zeeman, the Breit-Rabi crossover.

For a shell n, each (l, m_j) with both j = l +/- 1/2 present forms a 2x2 block
(fine-structure diagonal + linear-Zeeman coupling via <S_z>); stretched states
(|m_j| = l+1/2) and all of l=0 are 1x1 blocks, exactly linear in B. The 2x2
eigenvalues are the closed-form Breit-Rabi roots, so no numerical eigensolver is
needed and the result is exact-of-the-model with zero numerical error.

APPROXIMATION by construction: the linear-Zeeman model omits the diamagnetic B^2
term and uses g_s = 2. COUNTERFACTUAL when alpha is altered. See docs/
specs/2026-07-23-phase10-zeeman-field-design.md.
"""

import math
from dataclasses import dataclass

from atomsim.analytic.dirac import dirac_energy
from atomsim.analytic.fine_structure import level_energy
from atomsim.analytic.hydrogen import validate_quantum_numbers
from atomsim.constants import ALPHA, B0_TESLA
from atomsim.provenance import Fidelity, Provenance, Quantity

_MU_B_AU = 0.5  # Bohr magneton in atomic units (e = hbar = m_e = 1)
MU_B_PER_TESLA = _MU_B_AU / B0_TESLA  # hartree per tesla, prefactor on (J_z + S_z)
_G2 = 2.0 * 0.00116  # anomalous-moment scale on the spin Zeeman part

_Z_ASSUMPTIONS = (
    "linear (paramagnetic) Zeeman only; diamagnetic B^2 term neglected",
    "electron g_s = 2 exactly (anomalous moment ~0.1% of the spin part neglected)",
    "coupling to other n manifolds neglected",
    "diagonal from the selected level model (alpha^2 fine structure or exact Dirac)",
)


@dataclass(frozen=True)
class ZeemanSublevel:
    m_j: float
    branch: str            # "upper" | "lower" | "single"
    j_label: float         # low-field good quantum number (the j at B=0)
    high_field_label: str  # (m_l, m_s) the state approaches at large B
    energy: Quantity


def lande_g(l: int, j: float) -> float:
    """Lande g-factor for a one-electron (s = 1/2) state."""
    s = 0.5
    return 1.0 + (j * (j + 1.0) + s * (s + 1.0) - l * (l + 1.0)) / (2.0 * j * (j + 1.0))


def _high_field_label(m_j: float, m_s: float) -> str:
    return f"m_l={m_j - m_s:g}, m_s={m_s:+g}"


def _mean_sq_radius(n: int, l: int, Z: int) -> float:
    """<r^2> in bohr^2 for a hydrogenic (n,l) state (diamagnetic scale)."""
    return (n * n / (2.0 * Z * Z)) * (5.0 * n * n + 1.0 - 3.0 * l * (l + 1.0))


def zeeman_sublevels(
    n: int, l: int, Z: int = 1, mu_ratio: float = 1.0, m_over_M: float = 0.0,
    alpha: float = ALPHA, b_tesla: float = 0.0, dirac: bool = False,
) -> list[ZeemanSublevel]:
    """Breit-Rabi sublevels of the (n, l) shell in a field B (tesla)."""
    validate_quantum_numbers(n, l)
    if Z < 1:
        raise ValueError(f"Z must be >= 1, got {Z}")
    if b_tesla < 0:
        raise ValueError(f"b_tesla must be >= 0, got {b_tesla}")

    def diag(j: float) -> Quantity:
        if dirac:
            return dirac_energy(n, j, Z=Z, mu_ratio=mu_ratio, alpha=alpha)
        return level_energy(
            n, l, j, Z=Z, mu_ratio=mu_ratio, m_over_M=m_over_M, alpha=alpha
        )

    muB_b = MU_B_PER_TESLA * b_tesla  # hartree
    altered = not math.isclose(alpha, ALPHA, rel_tol=1e-12)
    fidelity = Fidelity.COUNTERFACTUAL if altered else Fidelity.APPROXIMATION
    diamag = 0.125 * (b_tesla / B0_TESLA) ** 2 * _mean_sq_radius(n, l, Z)
    denom = 2 * l + 1
    method = (
        "Breit-Rabi (fine structure + linear Zeeman, g_s=2) eigenvalue; "
        f"mu_B*B = {muB_b:.3e} hartree at B = {b_tesla:g} T"
        + ("; exact Dirac diagonal split by a perturbative linear-Zeeman model" if dirac else "")
        + (f"; altered alpha = {alpha:g} (real {ALPHA:g})" if altered else "")
    )

    def make(value: float, m_j: float, branch: str, j_label: float, m_s: float,
             underlying_err: float | None) -> ZeemanSublevel:
        err = (underlying_err or 0.0) + diamag + _G2 * abs(muB_b * m_j)
        return ZeemanSublevel(
            m_j=m_j, branch=branch, j_label=j_label,
            high_field_label=_high_field_label(m_j, m_s),
            energy=Quantity(
                value=value, unit="hartree",
                label=f"E_Zeeman {n},{l},m_j={m_j:g},{branch} (B={b_tesla:g}T)",
                provenance=Provenance(
                    fidelity=fidelity, method=method, assumptions=_Z_ASSUMPTIONS,
                    error_estimate=err,
                    refinement=(
                        "diamagnetic (B^2) term, then Paschen-Back beyond the "
                        "two-effect model"
                    ),
                ),
            ),
        )

    j_up = l + 0.5
    e_up = diag(j_up)
    m_values = [(-(l + 0.5) + k) for k in range(2 * l + 2)]  # -(l+1/2) .. +(l+1/2)
    out: list[ZeemanSublevel] = []
    for m_j in m_values:
        stretched = abs(m_j) > l  # |m_j| == l+1/2
        if l == 0 or stretched:
            # 1x1 block, only j = l+1/2, exactly linear in B.
            zeeman_diag = muB_b * m_j * (2 * l + 2) / denom
            m_s = math.copysign(0.5, m_j)
            out.append(make(
                e_up.value + zeeman_diag, m_j, "single", j_up, m_s,
                e_up.provenance.error_estimate,
            ))
            continue
        # 2x2 block for j = l+1/2 (upper) and j = l-1/2 (lower).
        e_dn = diag(l - 0.5)
        h00 = e_up.value + muB_b * m_j * (2 * l + 2) / denom
        h11 = e_dn.value + muB_b * m_j * (2 * l) / denom
        h01 = muB_b * math.sqrt((l + 0.5) ** 2 - m_j * m_j) / denom
        mean = 0.5 * (h00 + h11)
        disc = math.hypot(0.5 * (h00 - h11), h01)
        out.append(make(  # upper -> state A: (m_l=m_j-1/2, m_s=+1/2)
            mean + disc, m_j, "upper", j_up, 0.5, e_up.provenance.error_estimate,
        ))
        out.append(make(  # lower -> state B: (m_l=m_j+1/2, m_s=-1/2)
            mean - disc, m_j, "lower", l - 0.5, -0.5, e_dn.provenance.error_estimate,
        ))
    return out
