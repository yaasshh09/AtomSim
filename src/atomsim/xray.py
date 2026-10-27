"""K-alpha X-ray lines and Moseley's law.

Knock out a 1s electron and a 2p electron falls into the hole, emitting a
K-alpha photon. In 1913 Moseley found the square root of its frequency rising
in a straight line with the element's position, sqrt(E) ~ (Z - 1), and that
straight line is what put the periodic table in order of nuclear charge rather
than atomic weight. The "- 1" is the other 1s electron, screening the nucleus
from the 2p one; Bohr's formula with that screening gives

    E_Moseley = (3/4) (Z - 1)^2 Ry

This module computes the line properly as well, by Delta-SCF Hartree-Fock: the
energy of the ion with a 1s hole minus the energy of the ion with a 2p hole,
each solved self-consistently, so the remaining electrons relax around each
hole. That relaxation is worth tens of eV here, which is why Koopmans' frozen
orbital energies are not used. What Hartree-Fock still leaves out is mostly
relativity: the 1s electron of krypton moves at a quarter of the speed of
light. It is estimated separately from the Dirac formula for a hydrogenic ion
of charge Z - 1, not folded in, so the Hartree-Fock line stays a clean
non-relativistic number and the estimate stays visibly an estimate.

The measured lines are vendored in data/nist_xray_kalpha.json, never queried
live. Non-relativistic Hartree-Fock gives one K-alpha line; the measured pair
K-alpha1 and K-alpha2 is split by spin-orbit, and the comparison is made to
their statistical average, (2 E1 + E2) / 3, which is what a calculation
without spin-orbit is a calculation of.
"""

import dataclasses
import json
from dataclasses import dataclass
from importlib import resources

from atomsim.atoms import Configuration, aufbau_configuration
from atomsim.constants import ALPHA
from atomsim.hf_atom import HFResult, solve_hartree_fock
from atomsim.provenance import Fidelity, Provenance, Quantity

NIST_XRAY = json.loads(
    resources.files("atomsim.data")
    .joinpath("nist_xray_kalpha.json")
    .read_text(encoding="utf-8")
)

#: The elements with a vendored K-alpha line, which is also as far as the
#: Hartree-Fock solver has been exercised.
XRAY_Z = tuple(a["z"] for a in NIST_XRAY["atoms"])

# Leading-order Dirac shifts of hydrogenic levels, per Z^4 alpha^2 (deeper is
# positive): 1s by 1/8, 2p3/2 by 1/128, 2p1/2 by 5/128. The 2p pair is
# averaged with its 2:1 statistical weights.
_REL_1S = 1 / 8
_REL_2P = (2 * (1 / 128) + 5 / 128) / 3

_DSCF_METHOD = (
    "Delta-SCF Hartree-Fock: E(1s hole) - E(2p hole), both ions solved "
    "self-consistently so the other electrons relax around each hole"
)


@dataclass(frozen=True)
class KalphaResult:
    z: int
    symbol: str
    hole_1s: HFResult
    hole_2p: HFResult
    delta_scf: Quantity  # hartree, APPROXIMATION
    relativity: Quantity  # hartree, APPROXIMATION: what Dirac would add
    moseley: Quantity  # hartree, APPROXIMATION: (3/4)(Z-1)^2 Ry


def _with_hole(config: Configuration, shell: tuple[int, int]) -> Configuration:
    return tuple((nl, q - 1 if nl == shell else q) for nl, q in config)


def hole_configurations(z: int) -> tuple[Configuration, Configuration]:
    """The Aufbau ground configuration with one 1s, then one 2p, electron removed."""
    ground = aufbau_configuration(z)
    return _with_hole(ground, (1, 0)), _with_hole(ground, (2, 1))


def moseley_energy(z: int) -> Quantity:
    """(3/4) (Z - 1)^2 Ry: Bohr's 2 -> 1 line with one electron's worth of screening."""
    return Quantity(0.375 * (z - 1) ** 2, "hartree", "E_Moseley", Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method="Moseley's law: a hydrogenic n = 2 -> 1 line at nuclear charge Z - 1",
        assumptions=(
            "the one remaining 1s electron screens exactly one unit of charge",
            "everything else in the atom is ignored",
        ),
        refinement="the Delta-SCF Hartree-Fock line beside it",
    ))


def relativity_estimate(z: int) -> Quantity:
    """The Dirac shift of a hydrogenic 2p -> 1s line at charge Z - 1, j-averaged.

    The screening charge is the dominant uncertainty. Z - 1 is right for the 1s
    and generous for the 2p, which sits outside both 1s electrons and the other
    2s and 2p ones; the error bar is how far the estimate moves if the 2p is
    taken at Z - 4 instead.
    """
    a2 = ALPHA**2
    value = (_REL_1S - _REL_2P) * (z - 1) ** 4 * a2
    spread = _REL_2P * ((z - 1) ** 4 - max(z - 4, 0) ** 4) * a2
    return Quantity(value, "hartree", "Delta E_rel", Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=(
            "leading-order Dirac energies of a hydrogenic ion of charge Z - 1, "
            "2p averaged over j with weights 2:1"
        ),
        assumptions=(
            "screening as in Moseley's law: Z - 1 for both the 1s hole and the 2p electron",
            "order (Z alpha)^2 only; QED and higher relativistic orders are left out",
        ),
        error_estimate=spread,
        refinement="a Dirac-Fock calculation, which is what NIST's theory column is",
    ))


def kalpha(z: int) -> KalphaResult:
    """The K-alpha line of element Z by Delta-SCF, beside Moseley and a relativity estimate."""
    if z not in XRAY_Z:
        raise ValueError(f"K-alpha is computed for Z in {XRAY_Z[0]}..{XRAY_Z[-1]}, got {z}")
    hole_1s_config, hole_2p_config = hole_configurations(z)
    hole_1s = solve_hartree_fock(z, z - 1, hole_1s_config)
    hole_2p = solve_hartree_fock(z, z - 1, hole_2p_config)
    e1, e2 = hole_1s.total_energy, hole_2p.total_energy
    prov = dataclasses.replace(
        e1.provenance,
        method=_DSCF_METHOD,
        # Two mesh errors on two large energies, added. They are largely the
        # same error and largely cancel, so this overstates the line's error.
        error_estimate=(e1.provenance.error_estimate or 0) + (e2.provenance.error_estimate or 0),
        refinement="a relativistic (Dirac-Fock) solve; see the relativity estimate",
    )
    symbol = next(a["symbol"] for a in NIST_XRAY["atoms"] if a["z"] == z)
    return KalphaResult(
        z=z,
        symbol=symbol,
        hole_1s=hole_1s,
        hole_2p=hole_2p,
        delta_scf=Quantity(e1.value - e2.value, "hartree", "E_Kalpha", prov),
        relativity=relativity_estimate(z),
        moseley=moseley_energy(z),
    )


def measured_kalpha(z: int) -> dict:
    """The vendored entry for Z: ``ka1`` and ``ka2``, theory and experiment."""
    for a in NIST_XRAY["atoms"]:
        if a["z"] == z:
            return a
    raise KeyError(f"no vendored K-alpha line for Z = {z}")
