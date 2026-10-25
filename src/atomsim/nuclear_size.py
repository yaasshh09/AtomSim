"""The finite size of the nucleus, and the proton radius puzzle it led to.

The engine's Coulomb potential is a point charge. A real nucleus is a charge
cloud of rms radius r_N, and inside it the potential is softer than -Z/r. An
s electron has a nonzero density at the origin, so it feels that softening; to
first order, for any spherical charge distribution,

    Delta E = (2 pi / 3) Z |psi(0)|^2 r_N^2 = (2/3) Z^4 mu^3 r_N^2 / n^3   [hartree]

(|psi_n00(0)|^2 = (Z mu)^3 / (pi n^3)). Only r_N^2 appears: the shape of the
nucleus enters at the next order, the Zemach moment, a relative Z mu r_N
smaller. States with l > 0 vanish at the origin like r^l, so their shift is
smaller again by (Z mu r_N)^2 and is reported as zero with that as its error.

The mu^3 is the whole story of the puzzle. A muon is about 186 times heavier
than an electron in this formula, so muonic hydrogen's shift is 186^3 (six
million) times larger, and its 2S-2P Lamb shift measures r_p squared to a
precision electronic hydrogen cannot reach. In 2010 that measurement gave a
radius 4% smaller than everything before it.

The published Lamb shift theory (QED terms this project does not compute) is
vendored in data/muonic_hydrogen_lamb.json with its citation, and used only to
turn a radius into a prediction; the r_p^2 coefficient it quotes is also a
check on the leading-order formula here.
"""

import json
import math
from dataclasses import dataclass
from importlib import resources

from atomsim.analytic.hydrogen import validate_quantum_numbers
from atomsim.constants import ALPHA
from atomsim.provenance import Fidelity, Provenance, Quantity

_ASSUMPTIONS = (
    "first-order perturbation theory in the softened potential inside the nucleus",
    "non-relativistic |psi(0)|^2; the relativistic and radiative corrections to the "
    "finite-size term are of relative order (Z alpha)^2 and alpha",
    "only the rms radius enters; the charge distribution's shape (Zemach moment) is "
    "a relative Z mu r_N correction",
)


def finite_size_shift(
    n: int, l: int, radius: Quantity, Z: int = 1, mu_ratio: float = 1.0
) -> Quantity:
    """The energy shift of level (n, l) from a nucleus of rms radius `radius` (bohr).

    Positive: the softened potential binds less.
    """
    validate_quantum_numbers(n, l)
    r = radius.value
    s_shift = (2.0 / 3.0) * Z**4 * mu_ratio**3 * r * r / n**3
    zmr = Z * mu_ratio * r
    if l > 0:
        return Quantity(0.0, "hartree", f"finite-size shift {n},l={l}", Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="l > 0 vanishes at the nucleus like r^l: no first-order r_N^2 term",
            assumptions=_ASSUMPTIONS,
            error_estimate=s_shift * zmr**2,
        ))
    # Next-order terms (Zemach, relativistic, radiative) as a relative size, and
    # the radius' own uncertainty, which enters squared.
    rel_model = 2.0 * zmr + (Z * ALPHA) ** 2 + ALPHA
    rel_radius = 2.0 * (radius.provenance.error_estimate or 0.0) / r
    return Quantity(s_shift, "hartree", f"finite-size shift {n}s", Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method="Delta E = (2/3) Z^4 mu^3 r_N^2 / n^3 (leading order, any spherical nucleus)",
        assumptions=_ASSUMPTIONS + (f"r_N from {radius.provenance.method}",),
        error_estimate=s_shift * math.hypot(rel_model, rel_radius),
        refinement="the Zemach third moment and the relativistic finite-size correction",
    ))


@dataclass(frozen=True)
class LambReference:
    citation: str
    retrieved: str
    constant: tuple[float, float]
    r2_coefficient: tuple[float, float]
    two_photon_exchange: tuple[float, float]
    radii_fm: dict[str, tuple[float, float, str]]


def load_lamb_reference() -> LambReference:
    """The vendored muonic hydrogen Lamb shift theory. Never a live query."""
    raw = json.loads(
        resources.files("atomsim.data").joinpath("muonic_hydrogen_lamb.json")
        .read_text(encoding="utf-8")
    )
    ls = raw["lamb_shift"]

    def pair(d):
        return (d["value"], d["uncertainty"])

    return LambReference(
        citation=raw["citation"],
        retrieved=raw["retrieved"],
        constant=pair(ls["constant"]),
        r2_coefficient=pair(ls["r2_coefficient"]),
        two_photon_exchange=pair(ls["two_photon_exchange"]),
        radii_fm={
            k: (v["value"], v["uncertainty"], v["source"]) for k, v in raw["radii_fm"].items()
        },
    )


def muonic_lamb_shift(r_fm: float, r_unc_fm: float = 0.0) -> Quantity:
    """The predicted 2S-2P Lamb shift of muonic hydrogen for a proton radius r_fm, in meV."""
    ref = load_lamb_reference()
    (c, dc), (k, dk), (t, dt) = ref.constant, ref.r2_coefficient, ref.two_photon_exchange
    value = c - k * r_fm**2 + t
    err = math.sqrt(dc**2 + (dk * r_fm**2) ** 2 + dt**2 + (2 * k * r_fm * r_unc_fm) ** 2)
    return Quantity(value, "meV", f"muonic H 2S-2P Lamb shift at r_p = {r_fm:g} fm", Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method="published QED prediction Delta E = C - k r_p^2 + E_TPE, evaluated at r_p",
        assumptions=(
            f"the QED terms are transcribed from {ref.citation} (retrieved {ref.retrieved}), "
            "not computed by this project",
        ),
        error_estimate=err,
    ))
