"""My perturbative fine structure: spin-orbit + relativistic KE + Darwin, to
order alpha^2.

I use the standard combined Pauli result, a function of n and j only:
    Delta E = -(mu' Z^4 alpha^2 / (2 n^4)) (n/(j+1/2) - 3/4)   [hartree]
This is APPROXIMATION tier by construction, and I put the three known scales I
neglect (alpha^4 terms, nuclear recoil O(m/M), electron g-2) into my error
estimate.
"""

import math

from atomsim.analytic.hydrogen import energy, validate_quantum_numbers
from atomsim.constants import ALPHA
from atomsim.provenance import Fidelity, Provenance, Quantity

_G2 = 2.0 * 0.00116  # the anomalous-moment scale on the spin-orbit piece

_FS_ASSUMPTIONS = (
    "I use the Pauli approximation to order alpha^2: spin-orbit + relativistic kinetic + Darwin",
    "I take electron g = 2 exactly, neglecting an anomalous moment worth ~0.1% of the splitting",
    "I scale the reduced mass to leading order and neglect nuclear recoil O(m/M)",
    "I model no Lamb shift or QED, and no hyperfine structure",
)


def validate_j(l: int, j: float) -> None:
    if j < 0.5 or abs(abs(j - l) - 0.5) > 1e-12:
        raise ValueError(f"j must be l +/- 1/2 (and >= 1/2), got l={l}, j={j}")


def fine_structure_shift(
    n: int, l: int, j: float, Z: int = 1, mu_ratio: float = 1.0,
    m_over_M: float = 0.0, alpha: float = ALPHA,
) -> Quantity:
    """The fine-structure energy shift Delta E(n, l, j) I compute, in hartree.

    I report APPROXIMATION at the real alpha and COUNTERFACTUAL when alpha has
    been altered, which is what my What-If constants lab does. The `alpha`
    argument is the seam a future FundamentalConstants.alpha will supply, so I
    will need no signature change then.
    """
    validate_quantum_numbers(n, l)
    validate_j(l, j)
    value = -(mu_ratio * Z**4 * alpha**2 / (2.0 * n**4)) * (n / (j + 0.5) - 0.75)
    error = abs(value) * ((Z * alpha) ** 2 + m_over_M + _G2)
    altered = not math.isclose(alpha, ALPHA, rel_tol=1e-12)
    method = (
        "combined Pauli fine structure "
        "dE = -(mu' Z^4 alpha^2 / 2 n^4)(n/(j+1/2) - 3/4)"
    )
    if altered:
        method += f"; altered fine-structure constant alpha = {alpha:g} (real {ALPHA:g})"
    return Quantity(
        value=value,
        unit="hartree",
        label=f"dE_fs {n},{l},j={j:g} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.COUNTERFACTUAL if altered else Fidelity.APPROXIMATION,
            method=method,
            assumptions=_FS_ASSUMPTIONS,
            error_estimate=error,
            refinement=(
                "I could solve Dirac hydrogen exactly instead "
                "(the planned Phase 3 flagship)"
            ),
        ),
    )


def level_energy(
    n: int, l: int, j: float, Z: int = 1, mu_ratio: float = 1.0,
    m_over_M: float = 0.0, alpha: float = ALPHA,
) -> Quantity:
    """The Bohr energy plus my fine-structure shift, in hartree.

    I let the fidelity follow the shift: APPROXIMATION at real alpha,
    COUNTERFACTUAL once it is altered.
    """
    bohr = energy(n, Z=Z, mu_ratio=mu_ratio)
    shift = fine_structure_shift(
        n, l, j, Z=Z, mu_ratio=mu_ratio, m_over_M=m_over_M, alpha=alpha
    )
    return Quantity(
        value=bohr.value + shift.value,
        unit="hartree",
        label=f"E {n},{l},j={j:g} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=shift.provenance.fidelity,
            method=f"{bohr.provenance.method} + {shift.provenance.method}",
            assumptions=_FS_ASSUMPTIONS,
            error_estimate=shift.provenance.error_estimate,
            refinement=shift.provenance.refinement,
        ),
    )
