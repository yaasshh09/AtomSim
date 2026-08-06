"""Stark effect: hydrogen in a static electric field, the parabolic manifold.

A uniform field along z keeps axial symmetry (m good) but breaks rotational
symmetry (l not good); the hydrogen + (-F z) Hamiltonian separates exactly in
parabolic coordinates with non-negative integers n1, n2 and n = n1 + n2 + |m| + 1.
Second-order degenerate perturbation theory is closed form: each shell n splits
into n^2 sublevels labelled by the electric quantum number k = n1 - n2, linear in
the field (the l-degeneracy signature) with a quadratic correction.

APPROXIMATION by construction: second order only, and the Stark manifold is not
truly bound (a static field ionizes; the series is asymptotic and diverges near
F_ion ~ Z^3 mu^2 / (16 n^4) a.u.). No alpha dependence (non-relativistic), so no
COUNTERFACTUAL branch. See docs/specs/2026-07-24-phase11-stark-effect-design.md.
"""

from dataclasses import dataclass

from atomsim.analytic.hydrogen import energy
from atomsim.constants import E0_V_PER_M
from atomsim.provenance import Fidelity, Provenance, Quantity

_S_ASSUMPTIONS = (
    "second-order perturbation theory (linear + quadratic); third and higher orders neglected",
    "static field: the manifold is a resonance, not a true bound state "
    "(field ionization neglected)",
    "gross-structure only: fine structure and its low-field crossover neglected",
    "non-relativistic: independent of alpha (altering alpha does not change this shift)",
)


@dataclass(frozen=True)
class StarkSublevel:
    n1: int
    n2: int
    m: int
    k: int            # n1 - n2, the electric quantum number
    energy: Quantity


def stark_sublevels(
    n: int, Z: int = 1, mu_ratio: float = 1.0, field_mv_per_m: float = 0.0,
) -> list[StarkSublevel]:
    """Parabolic Stark sublevels of the shell n in a field F (MV/m)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if Z < 1:
        raise ValueError(f"Z must be >= 1, got {Z}")
    if field_mv_per_m < 0:
        raise ValueError(f"field_mv_per_m must be >= 0, got {field_mv_per_m}")

    f_au = field_mv_per_m * 1e6 / E0_V_PER_M  # atomic units of field
    e_bohr = energy(n, Z=Z, mu_ratio=mu_ratio).value
    zm = Z * mu_ratio
    # classical field-ionization scale (a.u.); guards the F/F_ion error ratio.
    f_ion = (Z ** 3) * (mu_ratio ** 2) / (16.0 * n ** 4)

    method = (
        "parabolic Stark, 2nd-order perturbation theory (linear + quadratic); "
        f"F = {field_mv_per_m:g} MV/m = {f_au:.3e} a.u. "
        f"(F/F_ion = {f_au / f_ion:.2e})"
    )

    out: list[StarkSublevel] = []
    for m in range(-(n - 1), n):
        am = abs(m)
        for n1 in range(0, n - am):
            n2 = n - am - 1 - n1
            k = n1 - n2
            # Hydrogenic scaling (energy unit mu*Z^2, length 1/(mu*Z), field mu^2*Z^3):
            # linear shift ~ F/(Z*mu); quadratic ~ F^2/(Z^4 * mu^3). The reduced-mass
            # powers differ (1 vs 3) and are NOT both (Z*mu): see the polarizability
            # E2(n=1) = -(9/4) F^2 / (Z^4 mu^3), i.e. alpha = 9/(2 Z^4 mu^3) a.u.
            lin = 1.5 * n * k * f_au / zm
            quad = (
                -(1.0 / 16.0) * n ** 4
                * (17 * n * n - 3 * k * k - 9 * m * m + 19)
                * f_au * f_au / (Z ** 4 * mu_ratio ** 3)
            )
            value = e_bohr + lin + quad
            err = abs(quad) * (f_au / f_ion) if f_au > 0.0 else 0.0
            out.append(StarkSublevel(
                n1=n1, n2=n2, m=m, k=k,
                energy=Quantity(
                    value=value, unit="hartree",
                    label=f"E_Stark {n},n1={n1},n2={n2},m={m} (F={field_mv_per_m:g}MV/m)",
                    provenance=Provenance(
                        fidelity=Fidelity.APPROXIMATION, method=method,
                        assumptions=_S_ASSUMPTIONS, error_estimate=err,
                        refinement=(
                            "third-order Stark, then the full non-perturbative "
                            "(field-ionization) resonance treatment"
                        ),
                    ),
                ),
            ))
    return out
