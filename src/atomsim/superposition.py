"""Superpositions of hydrogen-like stationary states, evolving in time.

    Psi(r, t) = sum_k c_k psi_k(r) exp(-i E_k t)        (atomic units, hbar = 1)

is the exact solution of the time-dependent Schrodinger equation for an
isolated atom once the psi_k and E_k are exact, which they are here. The
evolution is drawn on the y=0 plane, where every psi_nlm is real in both
angular bases (see plane.py), so the whole movie is K real grids plus K
energies. The server evaluates the grids once and the browser runs the phase
sum each frame; `evolve_density` is the same sum in numpy, the reference the
tests hold the formula to.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from atomsim.analytic.angular import validate_angular
from atomsim.analytic.hydrogen import energy, validate_quantum_numbers
from atomsim.plane import default_half_extent, plane_grid
from atomsim.provenance import Fidelity, Provenance, Quantity

MAX_COMPONENTS = 4

#: Energies closer than this (hartree) count as degenerate. The Bohr levels are
#: rationals times mu Z^2, so a real gap is never anywhere near it.
DEGENERATE_TOL = 1e-12


@dataclass(frozen=True)
class Component:
    """One stationary state in the mix: a relative population and a phase (rad)."""

    n: int
    l: int
    m: int
    weight: float
    phase: float = 0.0


@dataclass(frozen=True)
class SuperpositionGrid:
    """The real psi_k on a shared y=0 grid, with normalized c_k and exact E_k."""

    components: tuple[Component, ...]
    coefficients: np.ndarray        # (K,) complex128, sum |c_k|^2 = 1
    energies: tuple[Quantity, ...]  # E_k, hartree
    psi: np.ndarray                 # (K, res, res) float64; [k, i, j] at (x=axis[j], 0, z=axis[i])
    axis: np.ndarray                # (res,) bohr
    basis: str
    Z: int
    mu_ratio: float
    provenance: Provenance

    def beat_frequencies(self) -> list[float]:
        """The distinct nonzero |E_j - E_k| (hartree) between populated states, ascending.

        Empty means the superposition is stationary: every populated component
        shares one energy, so no relative phase ever changes.
        """
        es = [q.value for q, c in zip(self.energies, self.coefficients) if abs(c) > 0]
        gaps: list[float] = []
        for i, a in enumerate(es):
            for b in es[i + 1:]:
                d = abs(a - b)
                if d > DEGENERATE_TOL and all(abs(d - g) > DEGENERATE_TOL for g in gaps):
                    gaps.append(d)
        return sorted(gaps)


def normalized_coefficients(components: Sequence[Component]) -> np.ndarray:
    """c_k = sqrt(w_k / sum w) e^{i phase_k}, so the populations sum to 1."""
    w = np.array([c.weight for c in components], dtype=float)
    if (w < 0).any() or not np.isfinite(w).all():
        raise ValueError("weights must be finite and >= 0")
    total = w.sum()
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    phases = np.array([c.phase for c in components], dtype=float)
    if not np.isfinite(phases).all():
        raise ValueError("phases must be finite")
    return np.sqrt(w / total) * np.exp(1j * phases)


def validate_components(components: Sequence[Component]) -> None:
    if not 1 <= len(components) <= MAX_COMPONENTS:
        raise ValueError(f"need 1 to {MAX_COMPONENTS} components, got {len(components)}")
    seen = set()
    for c in components:
        validate_quantum_numbers(c.n, c.l)
        validate_angular(c.l, c.m)
        key = (c.n, c.l, c.m)
        if key in seen:
            raise ValueError(f"state ({c.n}, {c.l}, {c.m}) appears twice; merge the weights")
        seen.add(key)


def superposition_grid(
    components: Sequence[Component],
    basis: str = "complex",
    Z: int = 1,
    mu_ratio: float = 1.0,
    resolution: int = 256,
    half_extent: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> SuperpositionGrid:
    """Evaluate each psi_k on one y=0 grid, framed for the largest n present."""
    comps = tuple(components)
    validate_components(comps)
    coeffs = normalized_coefficients(comps)
    he = (
        default_half_extent(max(c.n for c in comps), Z, mu_ratio)
        if half_extent is None else float(half_extent)
    )
    grids = []
    for k, c in enumerate(comps):
        def step(p, k=k):
            if progress is not None:
                progress((k + p) / len(comps))

        pg = plane_grid(
            c.n, c.l, c.m, quantity="psi", basis=basis, Z=Z, mu_ratio=mu_ratio,
            resolution=resolution, half_extent=he, progress=step,
        )
        grids.append(pg.values)
    energies = tuple(energy(c.n, Z=Z, mu_ratio=mu_ratio) for c in comps)

    provenance = Provenance(
        fidelity=Fidelity.EXACT,
        method=(
            "Psi(x,0,z,t) = sum_k c_k psi_k exp(-i E_k t / hbar), from closed-form "
            f"psi_nlm on a {resolution}x{resolution} y=0 grid (half-extent {he:g} bohr) "
            "and Bohr energies E_n = -mu' Z^2 / (2 n^2); the browser evaluates the "
            "phase sum each frame"
        ),
        assumptions=(
            "an isolated atom: no spontaneous emission, so no component decays "
            "(2p lives 1.6 ns, millions of beat periods)",
            "non-relativistic: states sharing n are degenerate and never beat; "
            "fine structure would split them by about alpha^2 of the gross gaps",
            "populations normalized so that sum_k |c_k|^2 = 1",
            "psi_k is real on y=0 (e^{i m phi} = +/-1 there), so the sum needs no "
            "complex grids",
        ),
        refinement="raise the resolution or change the extent",
    )
    return SuperpositionGrid(
        components=comps, coefficients=coeffs, energies=energies,
        psi=np.stack(grids), axis=pg.axis, basis=basis, Z=Z, mu_ratio=mu_ratio,
        provenance=provenance,
    )


def evolve_density(sg: SuperpositionGrid, t: float) -> np.ndarray:
    """|Psi(x,0,z,t)|^2 on the grid at time t (atomic units of time)."""
    e = np.array([q.value for q in sg.energies])
    amp = sg.coefficients * np.exp(-1j * e * t)
    psi_t = np.tensordot(amp, sg.psi, axes=1)
    return np.abs(psi_t) ** 2


def plane_gradients(sg: SuperpositionGrid) -> tuple[np.ndarray, np.ndarray]:
    """(d psi_k/dx, d psi_k/dz) on the grid by central differences (one-sided at the edges).

    Second order in the grid step, except across the nuclear cusp of an s state,
    where psi has a kink and no difference scheme converges at the cell holding
    r = 0. The browser takes the same differences on the same grids.
    """
    h = float(sg.axis[1] - sg.axis[0])
    dz, dx = np.gradient(sg.psi, h, axis=(1, 2))
    return dx, dz


def evolve_current(sg: SuperpositionGrid, t: float) -> tuple[np.ndarray, np.ndarray]:
    """(j_x, j_z) on y=0 at time t: Im(Psi* grad Psi) / mu, atomic units.

    With a_k = c_k e^{-i E_k t} and every psi_k real here,
        j = (1/mu) sum_{j,k} Im(a_j* a_k) psi_j grad psi_k,
    so a single stationary state (one k) carries no in-plane current at all;
    whatever flows in this picture is the beating. The out-of-plane j_y of the
    m != 0 components is not part of it.
    """
    e = np.array([q.value for q in sg.energies])
    amp = sg.coefficients * np.exp(-1j * e * t)
    # w[j, k] = Im(a_j* a_k), written out so the diagonal is exactly zero: a
    # complex product leaves ~1e-26 there, a current that is not in the physics.
    re, im = amp.real, amp.imag
    w = re[:, None] * im[None, :] - im[:, None] * re[None, :]
    dx, dz = plane_gradients(sg)
    jx = np.einsum("jk,jab,kab->ab", w, sg.psi, dx) / sg.mu_ratio
    jz = np.einsum("jk,jab,kab->ab", w, sg.psi, dz) / sg.mu_ratio
    return jx, jz
