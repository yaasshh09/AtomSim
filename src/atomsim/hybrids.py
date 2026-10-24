"""sp, sp2 and sp3 hybrids, built from hydrogen's own n = 2 states.

    h = c_s psi_2s + c_p (n_x psi_2px + n_y psi_2py + n_z psi_2pz),  c_s^2 + c_p^2 = 1

2s and 2p share one energy in the Coulomb field (the accidental degeneracy), so
every such h is itself an exact stationary state of hydrogen with E = E_2, and
its density is EXACT. That is a stronger statement than chemistry needs and a
narrower one than it uses: in carbon the 2s and 2p are not degenerate, and a
hybrid there is valence-bond bookkeeping for bond directions, not an eigenstate
of anything. The provenance says which of the two this is.

Orientation is a choice, and it is made for the y = 0 cross-section: the sp
lobes point along +-z, the three sp2 lobes lie in the xz plane at 120 degrees,
and the sp3 set has one lobe on +z and one in the xz plane, the other two
leaving the plane symmetrically.

Sampling is exact rejection sampling. By Cauchy-Schwarz,
|sum c_k psi_k|^2 <= (sum |c_k|) sum |c_k| |psi_k|^2, so drawing from the mixture
g = sum w_k |psi_k|^2 with w_k = |c_k| / sum|c_j| (each component through the
validated single-state sampler) and accepting with probability
|h|^2 / ((sum|c_k|) sum |c_k| |psi_k|^2) yields draws from |h|^2 exactly. The
acceptance rate is exactly 1/(sum|c_k|)^2: a half for sp, at least a quarter
for any sp3 direction.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from atomsim.analytic.hydrogen import energy
from atomsim.analytic.wavefunction import evaluate_state
from atomsim.plane import PlaneGrid, _plane_values, default_half_extent
from atomsim.provenance import Fidelity, Provenance
from atomsim.sampling import SampleCloud, sample_density

_TET = math.acos(-1.0 / 3.0)  # 109.47 degrees, the tetrahedral angle
_A = 2 * math.pi / 3


def _dir(theta: float, phi: float) -> tuple[float, float, float]:
    return (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))


#: key -> (s coefficient, unit lobe direction)
HYBRIDS: dict[str, tuple[float, tuple[float, float, float]]] = {
    "sp-1": (math.sqrt(1 / 2), (0.0, 0.0, 1.0)),
    "sp-2": (math.sqrt(1 / 2), (0.0, 0.0, -1.0)),
    "sp2-1": (math.sqrt(1 / 3), (0.0, 0.0, 1.0)),
    "sp2-2": (math.sqrt(1 / 3), (math.sin(_A), 0.0, math.cos(_A))),
    "sp2-3": (math.sqrt(1 / 3), (-math.sin(_A), 0.0, math.cos(_A))),
    "sp3-1": (0.5, (0.0, 0.0, 1.0)),
    "sp3-2": (0.5, _dir(_TET, 0.0)),
    "sp3-3": (0.5, _dir(_TET, _A)),
    "sp3-4": (0.5, _dir(_TET, -_A)),
}

#: (n, l, m) of the real-basis 2s, 2p_x, 2p_y, 2p_z.
_BASIS = ((2, 0, 0), (2, 1, 1), (2, 1, -1), (2, 1, 0))


@dataclass(frozen=True)
class Hybrid:
    key: str
    coefficients: tuple[float, float, float, float]  # on 2s, 2px, 2py, 2pz

    @property
    def family(self) -> str:
        return self.key.split("-")[0]


def hybrid(key: str) -> Hybrid:
    if key not in HYBRIDS:
        raise ValueError(f"unknown hybrid {key!r}; known: {sorted(HYBRIDS)}")
    cs, (nx, ny, nz) = HYBRIDS[key]
    cp = math.sqrt(1.0 - cs * cs)
    # The 2s enters with a minus sign. R_20 is positive inside its node at
    # r = 2/(Z mu') and negative outside, where most of its density is, and with
    # that sign <2s|z|2p0> = -3 a0: a plus would point the big lobe along -n.
    return Hybrid(key=key, coefficients=(-cs, cp * nx, cp * ny, cp * nz))


def evaluate_hybrid(
    key: str, positions: np.ndarray, Z: int = 1, mu_ratio: float = 1.0
) -> np.ndarray:
    """h at Cartesian points (bohr), real, bohr^-3/2."""
    h = hybrid(key)
    pos = np.asarray(positions, dtype=float)
    out = np.zeros(len(pos))
    for c, (n, l, m) in zip(h.coefficients, _BASIS):
        if c != 0.0:
            out += c * np.real(
                evaluate_state(n, l, m, pos, Z=Z, mu_ratio=mu_ratio, basis="real").values
            )
    return out


def _provenance(
    key: str, Z: int, mu_ratio: float, method: str, fidelity: Fidelity
) -> Provenance:
    h = hybrid(key)
    e = energy(2, Z=Z, mu_ratio=mu_ratio).value
    return Provenance(
        fidelity=fidelity,
        method=method,
        assumptions=(
            f"{h.family} hybrid of hydrogen's own 2s and 2p, which are degenerate in a "
            f"Coulomb field, so this is an exact stationary state with E = E_2 = "
            f"{e:.6g} hartree",
            "in a real carbon atom 2s and 2p are not degenerate, and a hybrid there is "
            "valence-bond bookkeeping, not an eigenstate",
            "orientation chosen so the lobes show on the y=0 plane (sp along z, sp2 in "
            "the xz plane, sp3 with one lobe on +z and one in the xz plane)",
        ),
    )


@dataclass(frozen=True)
class _Values:
    """The two attributes _plane_values reads off an evaluator's result."""

    values: np.ndarray
    provenance: Provenance


def hybrid_plane_grid(
    key: str,
    quantity: str = "density",
    Z: int = 1,
    mu_ratio: float = 1.0,
    resolution: int = 512,
    progress: Callable[[float], None] | None = None,
) -> PlaneGrid:
    if quantity not in ("density", "psi"):
        raise ValueError(f"quantity must be 'density' or 'psi', got {quantity!r}")
    he = default_half_extent(2, Z, mu_ratio)
    empty = Provenance(fidelity=Fidelity.EXACT, method="")
    values, axis, _ = _plane_values(
        lambda pos: _Values(evaluate_hybrid(key, pos, Z, mu_ratio), empty),
        quantity, resolution, he, progress,
    )
    what = "|h|^2" if quantity == "density" else "signed h"
    return PlaneGrid(
        values=values, axis=axis, quantity=quantity,
        unit="bohr^-3" if quantity == "density" else "bohr^-3/2",
        label=f"{what} for the {key} hybrid on the y=0 plane",
        n=2, l=1, m=0, Z=Z, mu_ratio=mu_ratio, basis="real",
        provenance=_provenance(
            key, Z, mu_ratio,
            f"evaluated {what} from the closed-form 2s and 2p on a "
            f"{resolution}x{resolution} y=0 grid, half-extent {he:g} bohr",
            Fidelity.EXACT,
        ),
    )


def sample_hybrid(
    key: str,
    count: int,
    Z: int = 1,
    mu_ratio: float = 1.0,
    seed: int = 0,
    progress: Callable[[float], None] | None = None,
) -> SampleCloud:
    """Draw `count` positions from |h|^2 by exact rejection from the component mixture."""
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    h = hybrid(key)
    live = [(abs(c), nlm) for c, nlm in zip(h.coefficients, _BASIS) if c != 0.0]
    total = sum(a for a, _ in live)
    weights = np.array([a / total for a, _ in live])
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    have = 0
    while have < count:
        want = max(int((count - have) * total * total * 1.2), 1024)
        picks = rng.multinomial(want, weights)
        proposals = [
            sample_density(
                n, l, m, int(k), Z=Z, mu_ratio=mu_ratio,
                seed=int(rng.integers(2**31)), basis="real",
            ).positions
            for k, (_, (n, l, m)) in zip(picks, live) if k > 0
        ]
        pos = np.concatenate(proposals).astype(np.float64)
        rng.shuffle(pos)
        target = evaluate_hybrid(key, pos, Z, mu_ratio) ** 2
        mixture = sum(
            a * np.abs(
                evaluate_state(n, l, m, pos, Z=Z, mu_ratio=mu_ratio, basis="real").values
            ) ** 2
            for a, (n, l, m) in live
        )
        accept = rng.random(len(pos)) * total * mixture <= target
        kept.append(pos[accept].astype(np.float32))
        have += int(accept.sum())
        if progress is not None:
            progress(min(have / count, 1.0))
    positions = np.concatenate(kept)[:count]
    return SampleCloud(
        positions=positions, n=2, l=1, m=0, Z=Z, mu_ratio=mu_ratio, basis="real",
        provenance=_provenance(
            key, Z, mu_ratio,
            f"sampled |h|^2 by exact rejection from the mixture sum_k |c_k||psi_k|^2 "
            f"(Cauchy-Schwarz envelope, acceptance 1/{total * total:.3g}); each "
            f"component through the validated factorized inverse-CDF sampler",
            Fidelity.NUMERICAL,
        ),
    )
