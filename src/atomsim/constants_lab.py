"""My What-If Lab: I derive the observable consequences of altering the raw
constants.

You give me the five fundamental constants (hbar, e, m_e, eps0, c) as
multipliers on their real CODATA values. From an altered FundamentalConstants I
derive the three quantities that are actually observable for a one-electron atom
against fixed SI rulers: the fine-structure constant alpha, the Bohr radius
(size), and the Hartree energy (binding). I return each as a Quantity whose
fidelity is COUNTERFACTUAL when any multiplier departs from 1 and EXACT
otherwise. My per-observable `changed` flag IS the degeneracy lesson: many
distinct multiplier tuples leave all three observables identical.
"""

import math
from dataclasses import dataclass

from atomsim.constants import FundamentalConstants
from atomsim.provenance import Fidelity, Provenance, Quantity

_MULT_NAMES = ("hbar", "e", "m_e", "eps0", "c")


@dataclass(frozen=True)
class DerivedObservable:
    quantity: Quantity
    ratio: float   # altered / real
    changed: bool  # true when I find ratio not close to 1


@dataclass(frozen=True)
class ConstantsReport:
    alpha: DerivedObservable
    bohr_radius_pm: DerivedObservable
    hartree_ev: DerivedObservable
    altered: bool


def _observable(
    label: str, unit: str, alt_value: float, real_value: float,
    formula: str, altered: bool, applied: str,
) -> DerivedObservable:
    ratio = alt_value / real_value
    changed = not math.isclose(ratio, 1.0, rel_tol=1e-9)
    method = formula
    if altered:
        method += f"; I altered the raw constants ({applied})"
    return DerivedObservable(
        quantity=Quantity(
            value=alt_value,
            unit=unit,
            label=label,
            provenance=Provenance(
                fidelity=Fidelity.COUNTERFACTUAL if altered else Fidelity.EXACT,
                method=method,
                assumptions=(
                    "I measure this observable against fixed real-universe SI rulers",
                ),
            ),
        ),
        ratio=ratio,
        changed=changed,
    )


def analyze_constants(
    hbar: float = 1.0, e: float = 1.0, m_e: float = 1.0,
    eps0: float = 1.0, c: float = 1.0,
) -> ConstantsReport:
    """I derive alpha, the Bohr radius (pm) and the Hartree energy (eV) from your multipliers."""
    mults = (hbar, e, m_e, eps0, c)
    altered = any(not math.isclose(v, 1.0, rel_tol=1e-12) for v in mults)
    applied = ", ".join(
        f"{n}x{v:g}" for n, v in zip(_MULT_NAMES, mults, strict=True)
        if not math.isclose(v, 1.0, rel_tol=1e-12)
    )

    real = FundamentalConstants.codata()
    alt = FundamentalConstants(
        hbar=real.hbar * hbar, e=real.e * e, m_e=real.m_e * m_e,
        eps0=real.eps0 * eps0, c=real.c * c,
    )
    # 1 eV = (the real elementary charge) joules, a fixed SI ruler I never alter.
    joule_per_ev = real.e

    return ConstantsReport(
        alpha=_observable(
            "alpha", "", alt.alpha, real.alpha,
            "alpha = e^2 / (4 pi eps0 hbar c)", altered, applied,
        ),
        bohr_radius_pm=_observable(
            "Bohr radius", "pm", alt.bohr_radius * 1e12, real.bohr_radius * 1e12,
            "a0 = 4 pi eps0 hbar^2 / (m_e e^2)", altered, applied,
        ),
        hartree_ev=_observable(
            "Hartree energy", "eV",
            alt.hartree_energy / joule_per_ev, real.hartree_energy / joule_per_ev,
            "E_h = hbar^2 / (m_e a0^2)", altered, applied,
        ),
        altered=altered,
    )
