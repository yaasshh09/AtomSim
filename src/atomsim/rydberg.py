"""Rydberg series and quantum defects of the alkalis.

Far from the core, an alkali's valence electron sees one net charge and a
plain -1/r, so its levels follow hydrogen's formula with n shifted:

    E_nl = -1 / (2 (n - delta_l)^2)   [hartree, infinite nuclear mass]

The quantum defect delta_l is what the core does to the electron on the
fraction of each orbit it spends inside. It barely depends on n, because
every high-n orbital of one l has the same shape inside the core and only
its normalization changes, and it falls fast with l, because the centrifugal
barrier keeps high-l electrons out of the core. Sodium's s series sits 1.35
below hydrogen; its f series, 0.0016.

The computed defects come from the GSZ screened potential (APPROXIMATION)
solved on the exponential mesh to n = 20. Rydberg energies need far more
precision than the tridiagonal eigensolver delivers there: its eigenvalues are
good to machine epsilon times the matrix norm, and on a mesh reaching
1e-3 / Z bohr that norm is about 1e12 hartree, an absolute error near 1e-4,
while d delta = n^3 dE makes 1e-4 hartree cost 0.8 in delta at n = 20. So
each state is polished by shifted inverse iteration and a Rayleigh quotient,
which is accurate where the eigensolver's bound is not because the large
matrix entries sit where a Rydberg state is vanishingly small. The principal
number comes from counting the polished vector's nodes, never from its index,
because inverse iteration started from a poor eigenvalue can land on a
neighbour.

What remains is the mesh's own discretization error, which for these states
lives mostly in the long Coulomb tail and grows like n^2. It is the same
error for hydrogen on the same mesh, so the hydrogen defect computed alongside
(zero exactly, but not zero numerically) is subtracted. The two-step spread
of the corrected defect is its error bar.

The measured defects come from NIST term energies vendored in
data/nist_alkali_terms.json, never queried live.
"""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
from scipy import constants as _sc
from scipy.linalg import eigh_tridiagonal, solve_banded

from atomsim.numerics.mesh import mesh_for_atom_at_step
from atomsim.numerics.screening import screened_potential, screening_provenance
from atomsim.provenance import Fidelity, Provenance, Quantity

NIST_ALKALI_TERMS = json.loads(
    resources.files("atomsim.data")
    .joinpath("nist_alkali_terms.json")
    .read_text(encoding="utf-8")
)

#: The alkalis with both GSZ parameters and vendored terms.
RYDBERG_ATOMS = {"li": "Li", "na": "Na"}

_R_INF_CM = _sc.physical_constants["Rydberg constant"][0] / 100.0
_M_E_U = _sc.physical_constants["electron mass in u"][0]

N_MAX = 20
_BOX = 2500.0  # bohr; the n = 20 orbital is below 1e-12 of its peak by 2000
_STEP = 0.01
_POLISH_STEPS = 3

_METHOD = (
    "GSZ screened central field on an exponential mesh; each state polished by "
    "shifted inverse iteration and a Rayleigh quotient, n from its node count, "
    "and the same-mesh hydrogen defect subtracted"
)
_ASSUMPTIONS = (
    "delta = n - 1 / sqrt(-2 E): one net charge outside the core, infinite nuclear mass",
    "no spin-orbit: one central field for each l, compared against NIST's "
    "(2J+1)-averaged terms",
    "no core polarization beyond what the GSZ fit absorbs",
)


@dataclass(frozen=True)
class DefectPoint:
    n: int
    l: int
    energy: Quantity  # hartree
    defect: Quantity  # dimensionless


@dataclass(frozen=True)
class MeasuredDefect:
    n: int
    l: int
    defect: float
    uncertainty: float


#: States below this n start from the eigensolver's guess, whose absolute error
#: is harmless next to their level spacing. Higher ones start from the previous
#: state's defect, which barely moves with n.
_GUESS_FROM_SOLVER = 3


def _polished_levels(v_local: np.ndarray, mesh, l: int, count: int) -> dict[int, float]:
    """{n: E} for the lowest `count` states of channel l, polished.

    Raises if a polished state has the wrong node count: inverse iteration that
    lands on a neighbour is a bug to report, not a level to file under the
    wrong n.
    """
    d, o = mesh.hamiltonian_bands(v_local, l)
    guesses, vectors = eigh_tridiagonal(
        d, o, select="i", select_range=(0, min(count, _GUESS_FROM_SOLVER) - 1)
    )
    ab = np.zeros((3, d.size))
    ab[0, 1:] = o
    ab[2, :-1] = o
    out: dict[int, float] = {}
    for k in range(count):
        n = k + l + 1
        if k < len(guesses):
            s, e = vectors[:, k], float(guesses[k])
        else:
            prev = n - 1
            delta = prev - 1.0 / math.sqrt(-2.0 * out[prev])
            s, e = np.ones(d.size), -0.5 / (n - delta) ** 2
        for _ in range(_POLISH_STEPS):
            ab[1] = d - e
            s = solve_banded((1, 1), ab, s)
            s /= np.linalg.norm(s)
            hs = d * s
            hs[:-1] += o * s[1:]
            hs[1:] += o * s[:-1]
            e = float(s @ hs)
        big = s[np.abs(s) > 1e-8 * np.abs(s).max()]
        nodes = int(np.count_nonzero(np.diff(np.sign(big))))
        if nodes != k or e >= 0.0:
            raise RuntimeError(
                f"polishing state {k} of l={l} found {nodes} nodes at E={e:.3e}; "
                "inverse iteration converged on the wrong level"
            )
        out[n] = e
    return out


def _corrected(z: int, l: int, step: float) -> dict[int, tuple[float, float]]:
    """{n: (E, delta)} with the same-mesh hydrogen defect taken off delta."""
    mesh = mesh_for_atom_at_step(z, _BOX, step)
    count = N_MAX - l
    atom = _polished_levels(screened_potential(z, z)(mesh.r), mesh, l, count)
    hydrogen = _polished_levels(-1.0 / mesh.r, mesh, l, count)
    out = {}
    for n, e in atom.items():
        if n > N_MAX or n not in hydrogen:
            continue
        delta_h = n - 1.0 / math.sqrt(-2.0 * hydrogen[n])
        out[n] = (e, n - 1.0 / math.sqrt(-2.0 * e) - delta_h)
    return out


@lru_cache(maxsize=16)
def computed_defects(z: int, l: int) -> tuple[DefectPoint, ...]:
    """The GSZ quantum defects of channel l, every state from the lowest to n = 20."""
    if l not in range(4):
        raise ValueError(f"l must be 0..3, got {l}")
    fine = _corrected(z, l, _STEP / 2)
    coarse = _corrected(z, l, _STEP)
    base = screening_provenance(z, z)
    method = f"{base.method}; {_METHOD}"
    points = []
    for n in sorted(fine):
        if n not in coarse:
            continue
        e, delta = fine[n]
        e_c, delta_c = coarse[n]
        energy_q = Quantity(e, "hartree", f"E_{n}{'spdf'[l]}", Provenance(
            fidelity=Fidelity.APPROXIMATION, method=method,
            assumptions=base.assumptions, error_estimate=abs(e - e_c),
            refinement=base.refinement,
        ))
        points.append(DefectPoint(
            n=n, l=l,
            energy=energy_q,
            defect=Quantity(delta, "dimensionless", f"delta_{n}{'spdf'[l]}", Provenance(
                inputs=(energy_q,),
                fidelity=Fidelity.APPROXIMATION, method=method,
                assumptions=base.assumptions + _ASSUMPTIONS,
                error_estimate=abs(delta - delta_c),
                refinement=base.refinement,
            )),
        ))
    return tuple(points)


def measured_defects(symbol: str, l: int) -> tuple[MeasuredDefect, ...]:
    """NIST's defects for one l: delta = n - sqrt(R_M / (limit - E)), in cm-1."""
    atom = NIST_ALKALI_TERMS["atoms"][symbol]
    r_m = _R_INF_CM / (1.0 + _M_E_U / atom["isotope_mass_u"])
    out = []
    for t in atom["terms"]:
        if t["l"] != l:
            continue
        binding = atom["limit_cm1"] - t["energy_cm1"]
        n_star = math.sqrt(r_m / binding)
        # d n* / d binding = -n* / (2 binding); the level and the limit both carry error.
        spread = t["uncertainty_cm1"] + atom["limit_uncertainty_cm1"]
        out.append(MeasuredDefect(
            n=t["n"], l=l, defect=t["n"] - n_star,
            uncertainty=n_star / (2.0 * binding) * spread,
        ))
    return tuple(out)
