"""Exact Dirac-Coulomb energy for hydrogen-like atoms (the fine-structure refinement).

Closed form of the one-body Dirac equation in a point Coulomb field. EXACT for that
model, which still omits the Lamb shift/QED, hyperfine structure, finite nuclear size,
and two-body recoil beyond reduced-mass scaling. See docs/specs/
2026-07-23-phase9-dirac-hydrogen-design.md.
"""

import math

from atomsim.analytic.hydrogen import energy
from atomsim.constants import ALPHA
from atomsim.provenance import Fidelity, Provenance, Quantity

_DIRAC_ASSUMPTIONS = (
    "exact eigenvalue of the one-body Dirac-Coulomb equation (point nucleus)",
    "no Lamb shift / QED radiative corrections (this splits 2s1/2 from 2p1/2 in reality)",
    "no hyperfine structure, no finite-nuclear-size correction",
    "reduced mass by mu-scaling the rest energy; two-body relativistic recoil neglected",
)


def _validate(n: int, j: float, Z: int, alpha: float) -> None:
    if n < 1:
        raise ValueError(f"principal quantum number n must be >= 1, got {n}")
    if j < 0.5 or abs((j - 0.5) - round(j - 0.5)) > 1e-12:
        raise ValueError(f"j must be a half-integer >= 1/2, got {j}")
    if (j + 0.5) > n:
        raise ValueError(f"j={j} is not allowed for n={n} (need j <= n-1/2)")
    if Z < 1:
        raise ValueError(f"nuclear charge Z must be >= 1, got {Z}")
    if Z * alpha >= j + 0.5:
        raise ValueError(
            f"supercritical: Z*alpha = {Z * alpha:g} >= j+1/2 = {j + 0.5:g}; "
            "the point-Coulomb Dirac solution is not real here"
        )


def dirac_energy(
    n: int, j: float, Z: int = 1, mu_ratio: float = 1.0, alpha: float = ALPHA
) -> Quantity:
    """Exact Dirac-Coulomb binding energy E(n, j) in hartree (rest energy subtracted)."""
    _validate(n, j, Z, alpha)
    gamma = math.sqrt((j + 0.5) ** 2 - (Z * alpha) ** 2)
    d = n - (j + 0.5) + gamma
    # E_bind = mu*c^2 ((1+x)^(-1/2) - 1) with x = (Za/D)^2. The bracket subtracts two
    # near-1 numbers, so evaluate it cancellation-free: (1+x)^(-1/2) - 1 = -x/(s(1+s)),
    # s = sqrt(1+x). Scaled by 1/alpha^2 (~1.9e4) the naive form loses ~1e-11.
    x = (Z * alpha / d) ** 2
    s = math.sqrt(1.0 + x)
    e_bind = (mu_ratio / alpha**2) * (-x / (s * (1.0 + s)))

    altered = not math.isclose(alpha, ALPHA, rel_tol=1e-12)
    bohr = energy(n, Z=Z, mu_ratio=mu_ratio).value
    # Omitted-physics scale (Lamb-dominated), an honesty order-of-magnitude, not a bound.
    omitted = abs(bohr) * (Z * alpha) ** 3
    method = "exact Dirac-Coulomb energy E(n,j) = mu*c^2([1+(Za/D)^2]^(-1/2) - 1)"
    if altered:
        method += f"; altered fine-structure constant alpha = {alpha:g} (real {ALPHA:g})"
    return Quantity(
        value=e_bind,
        unit="hartree",
        label=f"E_Dirac {n},j={j:g} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.COUNTERFACTUAL if altered else Fidelity.EXACT,
            method=method,
            assumptions=_DIRAC_ASSUMPTIONS,
            error_estimate=omitted,
            refinement="QED / Lamb shift (2s-2p splitting), then hyperfine structure",
        ),
    )


def dirac_fine_splitting(
    n: int, l: int, Z: int = 1, mu_ratio: float = 1.0, alpha: float = ALPHA
) -> float:
    """E(n, j=l+1/2) - E(n, j=l-1/2) in hartree; requires l >= 1."""
    if l < 1:
        raise ValueError(f"fine splitting needs l >= 1, got {l}")
    hi = dirac_energy(n, l + 0.5, Z=Z, mu_ratio=mu_ratio, alpha=alpha).value
    lo = dirac_energy(n, l - 0.5, Z=Z, mu_ratio=mu_ratio, alpha=alpha).value
    return hi - lo
