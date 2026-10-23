"""Expectation values of a bound state, and the virial theorem as a live check.

For a hydrogen-like (n, l) the radial moments are closed-form (Kramers):

    <r>      = a (3n^2 - l(l+1)) / 2           <1/r>   = 1 / (a n^2)
    <r^2>    = a^2 n^2 (5n^2 + 1 - 3l(l+1)) / 2 <1/r^2> = 1 / (a^2 n^3 (l + 1/2))
    <1/r^3>  = 1 / (a^3 n^3 l (l + 1/2)(l + 1))   (l > 0)

with a = 1/(Z mu') bohr, and <T> = -E, <V> = 2E. Those are EXACT.

The virial theorem says more than that, and says it for any central potential:
every bound eigenstate has 2<T> = <r dV/dr> (for V ~ r^k, 2<T> = k<V>). So
integrating <T> and <r dV/dr> straight from a wavefunction, with no formula in
between, is a test of the wavefunction itself: a closed-form one passes to
quadrature accuracy, a converged numerical one passes to its grid accuracy, and
a state the solver's box is squeezing fails, because the box wall is a force the
theorem does not know about.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from atomsim.analytic.hydrogen import (
    _EXACT_ASSUMPTIONS,
    _radial_eval,
    validate_quantum_numbers,
)
from atomsim.provenance import Fidelity, Provenance, Quantity

#: Relative step for r dV/dr = [V(r(1+e)) - V(r(1-e))] / 2e, which needs only V itself
#: and so works for a typed free-form potential as well as a preset.
_LOG_STEP = 1e-5


def hydrogen_expectations(n: int, l: int, Z: int = 1, mu_ratio: float = 1.0) -> list[Quantity]:
    """The closed-form radial moments and energy split of a hydrogen-like (n, l)."""
    validate_quantum_numbers(n, l)
    a = 1.0 / (Z * mu_ratio)
    L = l * (l + 1)
    r1 = a * (3 * n**2 - L) / 2
    r2 = a**2 * n**2 * (5 * n**2 + 1 - 3 * L) / 2
    e = -Z**2 * mu_ratio / (2 * n**2)

    def q(value, unit, label, formula):
        return Quantity(
            value=float(value), unit=unit, label=label,
            provenance=Provenance(
                fidelity=Fidelity.EXACT,
                method=f"closed form {formula}, a = a0/(Z mu')",
                assumptions=_EXACT_ASSUMPTIONS,
            ),
        )

    out = [
        q(r1, "bohr", "<r>", "<r> = a(3n^2 - l(l+1))/2"),
        q(math.sqrt(r2 - r1**2), "bohr", "Delta r",
          "Delta r = sqrt(<r^2> - <r>^2), from the two Kramers moments"),
        q(r2, "bohr^2", "<r^2>", "<r^2> = a^2 n^2 (5n^2 + 1 - 3l(l+1))/2"),
        q(1 / (a * n**2), "bohr^-1", "<1/r>", "<1/r> = 1/(a n^2)"),
        q(1 / (a**2 * n**3 * (l + 0.5)), "bohr^-2", "<1/r^2>", "<1/r^2> = 1/(a^2 n^3 (l+1/2))"),
    ]
    if l > 0:
        out.append(q(1 / (a**3 * n**3 * l * (l + 0.5) * (l + 1)), "bohr^-3", "<1/r^3>",
                     "<1/r^3> = 1/(a^3 n^3 l(l+1/2)(l+1))"))
    out += [
        q(-e, "hartree", "<T>", "<T> = -E (virial theorem, Coulomb)"),
        q(2 * e, "hartree", "<V>", "<V> = 2E (virial theorem, Coulomb)"),
    ]
    return out


@dataclass(frozen=True)
class VirialCheck:
    """<T>, <V> and <r dV/dr> integrated from a wavefunction, and how far 2<T> misses."""

    kinetic: Quantity
    potential: Quantity
    r_dv_dr: Quantity
    residual: Quantity  # (2<T> - <r dV/dr>) / |2<T>|, dimensionless


def virial_check(
    r: np.ndarray,
    u: np.ndarray,
    potential: Callable[[np.ndarray], np.ndarray],
    l: int,
    mu_ratio: float,
    source: str,
    fidelity: Fidelity = Fidelity.NUMERICAL,
) -> VirialCheck:
    """Integrate <T>, <V>, <r dV/dr> from u = rR on a uniform grid with u(0) = 0.

    <T> = (1/2mu) integral [u'^2 + l(l+1) u^2 / r^2] dr, with u' by central
    differences on the grid padded by u(0) = 0, so it uses the wavefunction's
    shape and never its eigenvalue: the check cannot pass by construction.
    """
    r = np.asarray(r, dtype=float)
    u = np.asarray(u, dtype=float)
    h = float(r[1] - r[0])
    norm = np.trapezoid(u**2, r)
    u = u / math.sqrt(norm)
    rp = np.concatenate(([0.0], r))
    up = np.concatenate(([0.0], u))
    du = np.gradient(up, h)
    density = up**2
    with np.errstate(divide="ignore", invalid="ignore"):
        centrifugal = np.where(rp > 0, l * (l + 1) * density / rp**2, 0.0)
    t = float(np.trapezoid(du**2 + centrifugal, rp)) / (2 * mu_ratio)
    v_vals = np.asarray(potential(r), dtype=float)
    v = float(np.trapezoid(v_vals * u**2, r))
    rdv = (np.asarray(potential(r * (1 + _LOG_STEP)), dtype=float)
           - np.asarray(potential(r * (1 - _LOG_STEP)), dtype=float)) / (2 * _LOG_STEP)
    w = float(np.trapezoid(rdv * u**2, r))
    residual = (2 * t - w) / abs(2 * t) if t != 0 else float("nan")

    prov = Provenance(
        fidelity=fidelity,
        method=(
            f"integrated from {source} on a uniform {len(r)}-point grid "
            f"(h = {h:.3g} bohr) by the trapezoid rule; u' by central differences"
        ),
        assumptions=(
            "u(0) = 0 and u vanishes at the far edge of the grid",
            "r dV/dr by a symmetric difference in log r, so any V(r) will do",
        ),
        refinement="refine the grid; for a solver state, also enlarge its box",
    )

    def q(value, unit, label):
        return Quantity(value=value, unit=unit, label=label, provenance=prov)

    return VirialCheck(
        kinetic=q(t, "hartree", "<T>"),
        potential=q(v, "hartree", "<V>"),
        r_dv_dr=q(w, "hartree", "<r dV/dr>"),
        residual=Quantity(
            value=residual, unit="", label="(2<T> - <r dV/dr>) / |2<T>|",
            provenance=Provenance(
                fidelity=fidelity,
                method=prov.method + "; the virial theorem says this is 0 for any "
                "bound eigenstate of a central potential",
                assumptions=prov.assumptions + (
                    "a large value means the state is not a converged eigenstate: "
                    "unresolved by the grid, or pushed on by the solver's box",
                ),
                refinement=prov.refinement,
            ),
        ),
    )


#: Grid for the closed-form check: the density is negligible past this many <r>.
_HYDROGEN_REACH = 12.0
_HYDROGEN_POINTS = 60000


def hydrogen_virial(n: int, l: int, Z: int = 1, mu_ratio: float = 1.0) -> VirialCheck:
    """The live check on the closed-form R_nl: it must come out at quadrature accuracy."""
    validate_quantum_numbers(n, l)
    a = 1.0 / (Z * mu_ratio)
    r_max = _HYDROGEN_REACH * a * (3 * n**2 - l * (l + 1)) / 2 + 10 * a
    r = np.linspace(r_max / _HYDROGEN_POINTS, r_max, _HYDROGEN_POINTS)
    u = r * _radial_eval(n, l, r, Z * mu_ratio)
    return virial_check(
        r, u, lambda x: -Z / x, l, mu_ratio,
        source=f"the closed-form R_{n},{l}", fidelity=Fidelity.NUMERICAL,
    )
