"""The exact Dirac-Coulomb energy and radial functions for hydrogen-like atoms, the
refinement that the perturbative fine structure points at.

This is the closed form of the one-body Dirac equation in a point Coulomb
field. It is EXACT for that model, and that model still omits the Lamb shift and
QED, hyperfine structure, finite nuclear size, and two-body recoil beyond
reduced-mass scaling. See docs/specs/2026-07-23-phase9-dirac-hydrogen-design.md.
"""

import math

import numpy as np
from scipy.special import gammaln, hyp1f1

from atomsim.analytic.hydrogen import energy
from atomsim.constants import ALPHA
from atomsim.provenance import Fidelity, Field, Provenance, Quantity

_DIRAC_ASSUMPTIONS = (
    "the exact eigenvalue of the one-body Dirac-Coulomb equation (point nucleus)",
    "no Lamb shift or QED radiative corrections, and in reality "
    "those split 2s1/2 from 2p1/2",
    "no hyperfine structure and no finite-nuclear-size correction",
    "the reduced mass enters by mu-scaling the rest energy, and two-body "
    "relativistic recoil is neglected",
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
    """The exact Dirac-Coulomb energy E(n, j), in hartree, rest energy subtracted."""
    _validate(n, j, Z, alpha)
    gamma = math.sqrt((j + 0.5) ** 2 - (Z * alpha) ** 2)
    d = n - (j + 0.5) + gamma
    # E_bind = mu*c^2 ((1+x)^(-1/2) - 1) with x = (Za/D)^2. The bracket subtracts
    # two near-1 numbers, so evaluate it cancellation-free:
    # (1+x)^(-1/2) - 1 = -x/(s(1+s)), s = sqrt(1+x). Scaled by 1/alpha^2
    # (~1.9e4) the naive form would lose ~1e-11.
    x = (Z * alpha / d) ** 2
    s = math.sqrt(1.0 + x)
    e_bind = (mu_ratio / alpha**2) * (-x / (s * (1.0 + s)))

    altered = not math.isclose(alpha, ALPHA, rel_tol=1e-12)
    bohr = energy(n, Z=Z, mu_ratio=mu_ratio).value
    # The scale of the physics omitted here, dominated by the Lamb shift. It is
    # an honest order of magnitude, not a bound.
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
            refinement=(
                "adding QED and the Lamb shift (the 2s-2p splitting), "
                "then hyperfine structure"
            ),
        ),
    )


def dirac_fine_splitting(
    n: int, l: int, Z: int = 1, mu_ratio: float = 1.0, alpha: float = ALPHA
) -> float:
    """E(n, j=l+1/2) - E(n, j=l-1/2) in hartree. Needs l >= 1."""
    if l < 1:
        raise ValueError(f"fine splitting needs l >= 1, got {l}")
    hi = dirac_energy(n, l + 0.5, Z=Z, mu_ratio=mu_ratio, alpha=alpha).value
    lo = dirac_energy(n, l - 0.5, Z=Z, mu_ratio=mu_ratio, alpha=alpha).value
    return hi - lo


def dirac_kappa(l: int, j: float) -> int:
    """The Dirac quantum number: -(j+1/2) for j = l+1/2, +(j+1/2) for j = l-1/2."""
    if abs(j - (l + 0.5)) < 1e-12:
        return -int(round(j + 0.5))
    if l > 0 and abs(j - (l - 0.5)) < 1e-12:
        return int(round(j + 0.5))
    raise ValueError(f"j must be l +/- 1/2 (and 1/2 for l = 0), got l={l}, j={j}")


def dirac_radial(
    n: int, l: int, j: float, r: np.ndarray,
    Z: int = 1, mu_ratio: float = 1.0, alpha: float = ALPHA,
) -> tuple[Field, Field]:
    """The exact Dirac-Coulomb radial functions (g, f), large and small, in bohr^-3/2.

    The bound spinor is (g(r) Omega_kappa,m, i f(r) Omega_-kappa,m), normalized
    so that the integral of (g^2 + f^2) r^2 dr is 1. With gamma = sqrt(kappa^2 -
    (Z alpha)^2), n_r = n - |kappa|, N = sqrt(n_r^2 + 2 n_r gamma + kappa^2) and
    rho = 2 lambda r (lambda = sqrt(mu^2 c^4 - E^2) / c):

        g, f = +/- C sqrt(mu c^2 +/- E) rho^(gamma-1) e^(-rho/2)
               [(N - kappa) M(-n_r, 2gamma+1, rho) -/+ n_r M(1-n_r, 2gamma+1, rho)]

    (Bethe and Salpeter section 14). The overall sign is chosen so that g has
    a positive overlap with the Schrodinger R_nl, which it becomes as alpha -> 0.
    For j = 1/2 gamma < 1, so g and f diverge (integrably) at r = 0: that is the
    point nucleus, and r must be positive there.
    """
    _validate(n, j, Z, alpha)
    if not 0 <= l < n:
        raise ValueError(f"need 0 <= l < n, got n={n}, l={l}")
    kappa = dirac_kappa(l, j)
    grid = np.asarray(r, dtype=float)
    c = 1.0 / alpha
    za = Z * alpha
    gam = math.sqrt(kappa * kappa - za * za)
    n_r = n - abs(kappa)
    rest = mu_ratio * c * c
    # E = rest / s with s = sqrt(1 + x). Both rest^2 - E^2 and rest - E subtract
    # near-equal numbers (at alpha -> 0 that lost every digit), so they are
    # written out: rest^2 - E^2 = rest^2 x / s^2, rest - E = rest x / (s (1 + s)).
    x = (za / (n_r + gam)) ** 2
    s = math.sqrt(1.0 + x)
    e_tot = rest / s
    e_minus = rest * x / (s * (1.0 + s))
    lam = rest * math.sqrt(x) / (s * c)
    big_n = math.sqrt(n_r * n_r + 2.0 * n_r * gam + kappa * kappa)
    rho = 2.0 * lam * grid
    a = (big_n - kappa) * hyp1f1(-n_r, 2.0 * gam + 1.0, rho)
    b = n_r * hyp1f1(1 - n_r, 2.0 * gam + 1.0, rho) if n_r > 0 else np.zeros_like(rho)
    with np.errstate(divide="ignore"):
        envelope = rho ** (gam - 1.0) * np.exp(-rho / 2.0)
    log_c = (
        1.5 * math.log(2.0 * lam) - gammaln(2.0 * gam + 1.0)
        + 0.5 * (gammaln(2.0 * gam + n_r + 1.0) - math.log(4.0 * big_n * (big_n - kappa))
                 - gammaln(n_r + 1.0) - math.log(rest))
    )
    sign = -1.0 if kappa > 0 else 1.0
    norm = sign * math.exp(log_c)
    g = norm * math.sqrt(rest + e_tot) * envelope * (a - b)
    f = -norm * math.sqrt(e_minus) * envelope * (a + b)

    altered = not math.isclose(alpha, ALPHA, rel_tol=1e-12)
    prov = Provenance(
        fidelity=Fidelity.COUNTERFACTUAL if altered else Fidelity.EXACT,
        method=(
            "closed-form Dirac-Coulomb radial functions (confluent hypergeometric form, "
            f"kappa = {kappa}, gamma = {gam:.6g})"
            + (f"; altered alpha = {alpha:g}" if altered else "")
        ),
        assumptions=_DIRAC_ASSUMPTIONS,
        refinement="a finite nuclear charge radius, which removes the r = 0 divergence",
    )
    label = f"{n},l={l},j={j:g} (Z={Z})"
    return (
        Field(g, grid, "bohr^-3/2", "bohr", f"g_{label}", prov),
        Field(f, grid, "bohr^-3/2", "bohr", f"f_{label}", prov),
    )
