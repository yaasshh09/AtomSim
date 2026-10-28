"""The periodic table under a changed fine-structure constant.

In atomic units, non-relativistic atomic structure does not contain alpha at
all: every Hartree-Fock orbital, energy and radius in this package is the same
number whatever alpha is. Alpha enters in two places only. It sets the
conversion to everyday units (a hartree is alpha^2 m c^2, a bohr is
hbar / (alpha m c)), which changes every atom by the same factor and leaves
chemistry's pattern alone. And it sets how big relativity is, through Z alpha,
which grows along the table and changes it from the bottom up.

This module measures the second, element by element: the share of the
hydrogen-like 1s binding energy that relativity accounts for, exactly, from
the Dirac formula at the scaled alpha. Where Z alpha reaches 1 the point-
nucleus Dirac 1s stops existing at all, and that is the edge of the periodic
table: near Z = 137 in this world, near 68 if alpha were doubled. (A finite
nucleus moves the real edge up somewhat; the point-nucleus edge is the clean
statement and is labelled as such.)
"""

from dataclasses import dataclass

from atomsim.analytic.dirac import dirac_energy
from atomsim.analytic.hydrogen import energy as schrodinger_energy
from atomsim.constants import ALPHA
from atomsim.provenance import Fidelity, Provenance, Quantity

Z_RANGE = range(1, 119)


@dataclass(frozen=True)
class AlphaTable:
    alpha_multiplier: float
    alpha: float
    #: The largest Z up to 118 with a point-nucleus Dirac 1s: Z alpha < 1.
    z_edge: int
    #: Relativity's share of the 1s binding, per Z; None past the edge.
    relativistic_share: tuple[Quantity | None, ...]


def alpha_table(alpha_multiplier: float = 1.0) -> AlphaTable:
    """Relativity's share of the 1s binding for Z = 1..118 at alpha times the multiplier."""
    if not 0.1 <= alpha_multiplier <= 10.0:
        raise ValueError(f"alpha multiplier must be in [0.1, 10], got {alpha_multiplier}")
    alpha = ALPHA * alpha_multiplier
    fidelity = Fidelity.EXACT if alpha_multiplier == 1.0 else Fidelity.COUNTERFACTUAL
    shares: list[Quantity | None] = []
    for z in Z_RANGE:
        if z * alpha >= 1.0:
            shares.append(None)
            continue
        dirac = dirac_energy(1, 0.5, Z=z, alpha=alpha).value
        plain = schrodinger_energy(1, Z=z).value
        shares.append(Quantity((dirac - plain) / dirac, "dimensionless", f"rel. share, Z = {z}",
            Provenance(
                fidelity=fidelity,
                method="(E_Dirac - E_Schrodinger) / E_Dirac for a hydrogen-like 1s at charge Z",
                assumptions=(
                    "point nucleus, no QED, one electron: the scale of relativity for "
                    "the innermost electrons, not a many-electron calculation",
                    f"alpha = {alpha_multiplier:g} times its measured value",
                ),
            )))
    z_edge = max(z for z, s in zip(Z_RANGE, shares, strict=True) if s is not None)
    return AlphaTable(alpha_multiplier, alpha, z_edge, tuple(shares))
