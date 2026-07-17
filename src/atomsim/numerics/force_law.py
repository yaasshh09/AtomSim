"""What-If force law: energy levels under a counterfactual V(r) = -Z / r^p.

Only the 1/r Coulomb shape makes hydrogen's energy depend on n alone; bending the
exponent p away from 1 lifts the accidental l-degeneracy. This module drives the
numerical radial solver with the power-law potential (NUMERICAL) and pairs each
level with the EXACT closed-form hydrogen level it maps to, so the contrast is an
honest EXACT-vs-COUNTERFACTUAL one. p is clamped to [0.5, 1.5], comfortably inside
the fall-to-center threshold (p -> 2), so every returned state is box-converged.
"""

from dataclasses import dataclass, replace

import numpy as np

from atomsim.analytic.hydrogen import energy as hydrogen_energy
from atomsim.numerics.radial_solver import solve_radial_with_error
from atomsim.provenance import Quantity
from atomsim.systems import System, get_system

P_MIN = 0.5
P_MAX = 1.5


@dataclass(frozen=True)
class ForceLawLevel:
    radial_index: int   # 0-based node count k
    energy: Quantity    # NUMERICAL, hartree, carries a grid-halving error estimate


@dataclass(frozen=True)
class ReferenceLevel:
    n: int
    energy: Quantity    # EXACT closed-form hydrogen level, hartree


@dataclass(frozen=True)
class ForceLawResult:
    p: float
    l: int
    z: int
    system_key: str
    counterfactual: tuple[ForceLawLevel, ...]
    reference: tuple[ReferenceLevel, ...]


def force_law_levels(
    p: float, l: int, system: str | System = "h", n_states: int = 4
) -> ForceLawResult:
    if not P_MIN <= p <= P_MAX:
        raise ValueError(f"p must be in [{P_MIN}, {P_MAX}], got {p}")
    if l < 0:
        raise ValueError(f"orbital quantum number l must be >= 0, got {l}")
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")

    # Accept a bare key (registered systems) or an already-resolved System, so the
    # server can hand us generic hydrogen-like ions (z{N}) that get_system omits.
    sys = system if isinstance(system, System) else get_system(system)
    z = sys.Z
    mu = sys.mu_ratio.value

    def potential(r: np.ndarray) -> np.ndarray:
        return -z / r**p

    sol = solve_radial_with_error(potential, l=l, mu_ratio=mu, n_states=n_states)

    # Stamp the counterfactual potential into each level's provenance so the
    # NUMERICAL Quantity is self-describing in isolation: the solver only records
    # its own finite-difference method, not that V(r) was bent away from Coulomb.
    note = f"; counterfactual power-law potential V=-Z/r^{p:g}"

    def _tag(energy: Quantity) -> Quantity:
        return replace(
            energy,
            provenance=replace(energy.provenance, method=energy.provenance.method + note),
        )

    counterfactual = tuple(
        ForceLawLevel(radial_index=k, energy=_tag(sol.energies[k])) for k in range(n_states)
    )
    # radial index k <-> hydrogen level n = l + 1 + k (since n_r = n - l - 1 = k)
    reference = tuple(
        ReferenceLevel(n=l + 1 + k, energy=hydrogen_energy(l + 1 + k, Z=z, mu_ratio=mu))
        for k in range(n_states)
    )
    return ForceLawResult(
        p=p,
        l=l,
        z=z,
        system_key=sys.key,
        counterfactual=counterfactual,
        reference=reference,
    )
