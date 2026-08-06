"""Magnetic-dipole hyperfine structure: the nuclear spin talks to the electron.

The nuclear spin I couples to the electron's angular momentum J; each level
splits into total-angular-momentum states F = I + J, |I-J| .. I+J. For an
s-electron (l = 0) the coupling is the Fermi contact interaction, driven by the
electron density at the nucleus. The flagship case is hydrogen 1s: I = J = 1/2,
F = 0 or 1, and the F=1 -> F=0 transition is the 21 cm line, 1420.4 MHz.

The hyperfine coupling constant of an ns level (energy, hartree):

    A(ns) = (2/3) g_e g_I (m_e/m_p) alpha^2 (mu/m_e)^3 (Z^3 / n^3)

g_e is the measured electron moment, g_I = (mu_nuc/mu_N)/I the nuclear g-factor,
and m_e/m_p is FIXED (the nuclear magneton mu_N = e hbar / 2 m_p is defined with
the proton mass for every nucleus; using the nucleus's own mass is a factor-of-2
bug for deuterium, locked out by tests/test_hyperfine.py).

APPROXIMATION tier: non-relativistic Fermi contact, s-states only. Neglected
scales are quantified in the error estimate. See
docs/specs/2026-07-24-phase12-hyperfine-structure-design.md.
"""

from dataclasses import dataclass

from scipy import constants as _sc

from atomsim.analytic.hydrogen import energy
from atomsim.constants import ALPHA
from atomsim.provenance import Fidelity, Provenance, Quantity
from atomsim.systems import System

# Fixed real-universe inputs (measured moments / masses, EXACT source).
_G_E = abs(_sc.physical_constants["electron g factor"][0])          # 2.0023193...
_M_E_OVER_M_P = 1.0 / _sc.physical_constants["proton-electron mass ratio"][0]

# s-state electron total angular momentum (l = 0, s = 1/2).
_J_S = 0.5

_HF_ASSUMPTIONS = (
    "Fermi contact interaction, s-states (l = 0) only; the electron density at "
    "the nucleus drives the coupling",
    "non-relativistic: electron g from the measured free-electron moment and "
    "exact reduced mass; bound-state QED beyond that neglected",
    "relativistic (Breit) correction ~ (Z alpha)^2 neglected (grows with Z)",
    "nuclear structure (finite size / Zemach radius, the hyperfine anomaly) neglected",
    "the l > 0 orbital + spin-dipolar channel is deferred, not included here",
)

_HF_METHOD = (
    "magnetic-dipole hyperfine, Fermi contact: "
    "A = (2/3) g_e g_I (m_e/m_p) alpha^2 (mu/m_e)^3 Z^3/n^3"
)


def hyperfine_constant(
    n: int, Z: int = 1, mu_ratio: float = 1.0, g_I: float = 0.0,
) -> Quantity:
    """Hyperfine coupling constant A of the ns level, in hartree.

    A enters the level energies as E(F) = (A/2)[F(F+1) - I(I+1) - J(J+1)].
    g_I is the nuclear g-factor (mu_nuc / mu_N) / I; g_I = 0 for a spin-0 nucleus.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if Z < 1:
        raise ValueError(f"Z must be >= 1, got {Z}")

    value = (
        (2.0 / 3.0) * _G_E * g_I * _M_E_OVER_M_P
        * ALPHA**2 * mu_ratio**3 * Z**3 / n**3
    )
    # Neglected physics: relativistic (Z alpha)^2, plus a bound-QED + nuclear-
    # structure floor (~1e-4). This upper-bounds the true residual (~6e-5 for H,
    # ~1e-4 for He-3) so the engine never claims more precision than it has.
    error = abs(value) * ((Z * ALPHA) ** 2 + 1e-4)
    return Quantity(
        value=value,
        unit="hartree",
        label=f"A_hf {n}s (Z={Z}, mu/m_e={mu_ratio:g}, g_I={g_I:g})",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=_HF_METHOD,
            assumptions=_HF_ASSUMPTIONS,
            error_estimate=error,
            refinement=(
                "relativistic (Dirac) hyperfine, then bound-state QED and the "
                "nuclear Zemach correction; and the l > 0 dipolar channel"
            ),
        ),
    )


@dataclass(frozen=True)
class HyperfineLevel:
    F: float
    shift: Quantity     # hyperfine shift from the gross ns level, hartree
    energy: Quantity    # gross energy + shift, hartree


def _f_values(I: float, J: float) -> list[float]:
    """Total angular momentum F = |I-J| .. I+J in integer steps."""
    lo = abs(I - J)
    n_steps = round(I + J - lo)
    return [lo + k for k in range(n_steps + 1)]


def hyperfine_levels(
    n: int, I: float, Z: int = 1, mu_ratio: float = 1.0, g_I: float = 0.0,
) -> list[HyperfineLevel]:
    """F sublevels of the ns level (J = 1/2), each carrying provenance."""
    A = hyperfine_constant(n, Z=Z, mu_ratio=mu_ratio, g_I=g_I)
    e_gross = energy(n, Z=Z, mu_ratio=mu_ratio).value
    J = _J_S
    ij = I * (I + 1.0)
    jj = J * (J + 1.0)

    out: list[HyperfineLevel] = []
    for F in _f_values(I, J):
        shift_val = 0.5 * A.value * (F * (F + 1.0) - ij - jj)
        shift = Quantity(
            value=shift_val,
            unit="hartree",
            label=f"dE_hf {n}s F={F:g}",
            provenance=Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method=f"{_HF_METHOD}; dE(F) = (A/2)[F(F+1) - I(I+1) - J(J+1)]",
                assumptions=_HF_ASSUMPTIONS,
                error_estimate=abs(shift_val) * ((Z * ALPHA) ** 2 + 1e-4),
                refinement=A.provenance.refinement,
            ),
        )
        out.append(HyperfineLevel(
            F=F,
            shift=shift,
            energy=Quantity(
                value=e_gross + shift_val,
                unit="hartree",
                label=f"E {n}s F={F:g} (Z={Z}, mu/m_e={mu_ratio:g})",
                provenance=shift.provenance,
            ),
        ))
    return out


# --- nuclei, keyed by system preset ----------------------------------------

@dataclass(frozen=True)
class Nucleus:
    name: str
    I: float
    g_I: float          # nuclear g-factor; 0.0 for a spin-0 nucleus
    note: str = ""


def _gI_from_codata(moment_ratio_name: str, spin: float) -> float:
    return _sc.physical_constants[moment_ratio_name][0] / spin


_NUCLEI: dict[str, Nucleus] = {
    "h": Nucleus("proton", 0.5,
                 _gI_from_codata("proton mag. mom. to nuclear magneton ratio", 0.5)),
    "d": Nucleus("deuteron", 1.0,
                 _gI_from_codata("deuteron mag. mom. to nuclear magneton ratio", 1.0)),
    "t": Nucleus("triton", 0.5,
                 _gI_from_codata("triton mag. mom. to nuclear magneton ratio", 0.5)),
    "he+": Nucleus("alpha particle (He-4)", 0.0, 0.0,
                   note="I = 0: a spin-0 nucleus has no magnetic moment, "
                        "so there is no hyperfine splitting (He-3 would split)"),
}

_UNAVAILABLE: dict[str, str] = {
    "ps": "positronium: the partner is a positron with a Bohr-magneton moment and "
          "annihilation contributes; the nuclear-magneton contact formula does not apply",
    "mu-h": "muonic hydrogen: the orbiter is a muon whose own g-factor and mass "
            "enter the coupling; this electron-contact formula does not apply",
}


@dataclass(frozen=True)
class HyperfineReport:
    available: bool
    n: int
    system_key: str
    nucleus_name: str | None = None
    I: float | None = None
    A: Quantity | None = None
    levels: tuple[HyperfineLevel, ...] = ()
    note: str | None = None      # e.g. the spin-0 explanation (available but no split)
    reason: str | None = None    # why hyperfine is unavailable for this system


def hyperfine_report(n: int, system: System) -> HyperfineReport:
    """Hyperfine F-levels of the ns shell for a preset system, or an honest reason.

    Returns the F sublevels for a nucleus with a defined moment (H, D, T), a
    single unsplit level for a spin-0 nucleus (He-4), or available=False with a
    reason for systems the contact formula does not describe (positronium,
    muonic hydrogen, generic Z with no identified nucleus).
    """
    key = system.key
    if key in _UNAVAILABLE:
        return HyperfineReport(
            available=False, n=n, system_key=key, reason=_UNAVAILABLE[key]
        )
    nucleus = _NUCLEI.get(key)
    if nucleus is None:
        return HyperfineReport(
            available=False, n=n, system_key=key,
            reason="no identified nucleus with a measured magnetic moment",
        )

    mu = system.mu_ratio.value
    A = hyperfine_constant(n, Z=system.Z, mu_ratio=mu, g_I=nucleus.g_I)
    levels = hyperfine_levels(
        n, I=nucleus.I, Z=system.Z, mu_ratio=mu, g_I=nucleus.g_I
    )
    return HyperfineReport(
        available=True, n=n, system_key=key,
        nucleus_name=nucleus.name, I=nucleus.I, A=A,
        levels=tuple(levels), note=nucleus.note or None,
    )
