"""Green-Sellin-Zachor screened central potential for multi-electron atoms.

V_eff(r) = -(1/r) [ (Z - N + 1) + (N - 1) Omega(r) ],
Omega(r) = 1 / [ H (exp(r/d) - 1) + 1 ],  Omega(0)=1, Omega(inf)=0.

This is an APPROXIMATION: an analytic independent-particle model, not a
self-consistent solve. One electron (N=1) reduces to bare -Z/r with no
parameters -- the calibration anchor.

The (d, H) parameters are NOT a universal fit here: they are transcribed
per-atom from Szydlik & Green, Phys. Rev. A 9, 1885 (1974), "Independent-
particle-model potentials for ions and neutral atoms with Z <= 18", Table I.
The paper tabulates the pair (d, K) with K = H/d, so H = K * d; we vendor the
published (d, K) columns and derive H, keeping the source's own numbers visible.
Only neutral atoms (N=Z) He..P and Ar are sourced; neutral S and Cl are absent
from the paper and therefore have no preset (see atoms.NO_GSZ_PARAMETERS). Values
are pinned by tests (N=1 exact Coulomb limit + NIST valence ionization energy).
"""

from collections.abc import Callable

import numpy as np

from atomsim.provenance import Fidelity, Provenance

# Neutral-atom GSZ screening parameters (N = Z), transcribed from Szydlik & Green
# (1974), Table I. Columns are (d [bohr], K [dimensionless]); H = K * d. The
# per-atom Hartree-Fock energy deviation delta (ppm) is kept as a fidelity hint.
# Neutral S (Z=16) and Cl (Z=17) are not in the paper -- deliberately omitted.
_GSZ_NEUTRAL_DK: dict[int, tuple[float, float]] = {
    2: (0.381, 1.77),    # He   delta=0.5 ppm
    3: (0.462, 1.75),    # Li   240
    4: (0.769, 1.88),    # Be   113
    5: (0.970, 2.00),    # B    153
    6: (0.939, 2.13),    # C    133
    7: (0.848, 2.27),    # N    94
    8: (0.735, 2.41),    # O    65
    9: (0.663, 2.59),    # F    50
    10: (0.558, 2.71),   # Ne   42
    11: (0.584, 2.85),   # Na   68
    12: (0.670, 3.01),   # Mg   55
    13: (0.860, 3.17),   # Al   79
    14: (0.988, 3.26),   # Si   71
    15: (1.055, 3.33),   # P    57
    18: (1.045, 3.50),   # Ar   33
}


def gsz_parameters(z: int, n_electrons: int) -> tuple[float, float]:
    """(d, H) for the GSZ screening function; d in bohr, H dimensionless.

    Transcribed from Szydlik & Green (1974), Table I, as (d, K) with H = K * d.
    Only neutral atoms (N = Z) are sourced; anything else raises rather than
    inventing a parameter. Not needed for N=1 (the screening term vanishes).
    """
    if z < 1:
        raise ValueError(f"Z must be >= 1, got {z}")
    if not 1 <= n_electrons <= z + 1:
        raise ValueError(f"N must be in [1, Z+1], got {n_electrons} (Z={z})")
    if n_electrons != z or z not in _GSZ_NEUTRAL_DK:
        raise ValueError(
            f"no sourced GSZ parameters for Z={z}, N={n_electrons}; only neutral "
            f"He..P and Ar (Szydlik & Green 1974) are vendored"
        )
    d, k = _GSZ_NEUTRAL_DK[z]
    return d, k * d


def _omega(r: np.ndarray, d: float, h: float) -> np.ndarray:
    # At large r, expm1(r/d) overflows to +inf and Omega -> 0, the correct
    # asymptotic limit; silence the expected overflow rather than the physics.
    with np.errstate(over="ignore"):
        return 1.0 / (h * np.expm1(r / d) + 1.0)


def z_eff(z: int, n_electrons: int, r: np.ndarray) -> np.ndarray:
    core = float(z - n_electrons + 1)
    r = np.asarray(r, dtype=float)
    if n_electrons == 1:
        return np.full_like(r, core)
    d, h = gsz_parameters(z, n_electrons)
    return core + (n_electrons - 1) * _omega(r, d, h)


def screened_potential(z: int, n_electrons: int) -> Callable[[np.ndarray], np.ndarray]:
    def v(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return -z_eff(z, n_electrons, r) / r
    return v


def screening_provenance(z: int, n_electrons: int) -> Provenance:
    return Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=(
            "Green-Sellin-Zachor screened central potential, "
            "Szydlik-Green (1974) neutral-atom (d, K) parameters"
        ),
        assumptions=(
            f"independent-particle central field for Z={z}, N={n_electrons}",
            "no self-consistency; potential depends only on (Z, N)",
            "infinite nuclear mass (mu_ratio = 1)",
        ),
        error_estimate=None,  # quantified against NIST at the observable level
        refinement=(
            "self-consistent Hartree-Fock in hf_atom.py removes the model error"
        ),
    )
