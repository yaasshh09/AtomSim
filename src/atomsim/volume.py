"""|psi|^2 on a cubic grid, for the 3-D view's volume rendering.

The point cloud draws the density by sampling it and the isosurface by cutting
one contour of it. A volume render draws the whole field at once: each pixel
is a ray through the box, and what it shows is how much density that ray
crosses. The grid is the physics and is served as it is; everything the
browser does to it (trilinear interpolation between nodes, the transfer
function from density to colour and opacity, the gamma compression) is
presentation, disclosed as such on the client.

The box is fitted exactly as the isosurface's is: the smallest cube whose
inscribed sphere holds 99.9% of the electron, measured by a radial profile,
so the escaped mass is an upper bound rather than a hope. The grid values are
the evaluator's own: closed-form psi for the hydrogen-like systems (EXACT at
every node), the numerical screened or Hartree-Fock orbital otherwise.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from atomsim.analytic.angular import validate_angular
from atomsim.analytic.hydrogen import validate_quantum_numbers
from atomsim.analytic.wavefunction import evaluate_state
from atomsim.atoms import Configuration
from atomsim.hf_atom import evaluate_hf_state
from atomsim.isosurface import (
    _captured,
    _density_on_grid,
    _fit_box,
    default_half_width,
)
from atomsim.provenance import Fidelity, Provenance, Quantity
from atomsim.screened_atom import evaluate_screened_state

#: Edge lengths offered. 64^3 is a quarter of a million float32s, 1 MB, and
#: resolves every state the views offer; 96^3 is there for the large ones.
VOLUME_SIZES = (48, 64, 96)


@dataclass(frozen=True)
class VolumeGrid:
    values: np.ndarray  # (N, N, N) float32; [i, j, k] at (x=axis[i], y=axis[j], z=axis[k])
    half_width: float  # bohr
    captured: Quantity  # the fraction of the electron inside the cube, by quadrature
    label: str
    n: int
    l: int
    m: int
    basis: str
    provenance: Provenance


def _volume(
    evaluator: Callable,
    *,
    n: int,
    l: int,
    m: int,
    basis: str,
    start_half_width: float,
    resolution: int,
    fidelity: Fidelity,
    method_prefix: str,
    progress: Callable[[float], None] | None,
) -> VolumeGrid:
    if resolution not in VOLUME_SIZES:
        raise ValueError(f"resolution must be one of {VOLUME_SIZES}, got {resolution}")
    half_width, _, _ = _fit_box(evaluator, l, m, start_half_width)
    grid, _, psi_assumptions = _density_on_grid(evaluator, half_width, resolution, progress)
    captured = _captured(grid, half_width, resolution)
    return VolumeGrid(
        values=grid.astype(np.float32),
        half_width=half_width,
        captured=Quantity(captured, "dimensionless", "captured fraction", Provenance(
            fidelity=Fidelity.NUMERICAL,
            method="the grid's |psi|^2 summed times the cell volume",
            assumptions=(
                "a Riemann sum over the nodes; the box itself was fitted to hold "
                "99.9% of the electron by an exact radial profile",
            ),
        )),
        label=f"|psi_{n},{l},{m}|^2 in a {resolution}^3 box of half-width {half_width:.3g} bohr",
        n=n, l=l, m=m, basis=basis,
        provenance=Provenance(
            fidelity=fidelity,
            method=(
                f"{method_prefix} evaluated at every node of a {resolution}^3 cubic grid, "
                f"half-width {half_width:.4g} bohr"
            ),
            assumptions=psi_assumptions + (
                "the node values are the physics; interpolation between them and the "
                "mapping to colour and opacity are drawn on the client and disclosed there",
            ),
            refinement="raise the grid resolution",
        ),
    )


def density_volume(
    n: int, l: int, m: int, basis: str = "complex", Z: int = 1, mu_ratio: float = 1.0,
    resolution: int = 64, progress: Callable[[float], None] | None = None,
) -> VolumeGrid:
    """A hydrogen-like state's density on the grid: exact values at every node."""
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    return _volume(
        lambda pos: evaluate_state(n, l, m, pos, Z=Z, mu_ratio=mu_ratio, basis=basis),
        n=n, l=l, m=m, basis=basis,
        start_half_width=default_half_width(n, Z, mu_ratio),
        resolution=resolution, fidelity=Fidelity.EXACT,
        method_prefix="closed-form psi_nlm", progress=progress,
    )


def screened_density_volume(
    z: int, n_electrons: int, n: int, l: int, m: int, basis: str = "complex",
    resolution: int = 64, progress: Callable[[float], None] | None = None,
) -> VolumeGrid:
    """The same, for a screened (GSZ) orbital."""
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    return _volume(
        lambda pos: evaluate_screened_state(z, n_electrons, n, l, m, pos, basis=basis),
        n=n, l=l, m=m, basis=basis,
        start_half_width=default_half_width(n, max(z - n_electrons + 1, 1), 1.0),
        resolution=resolution, fidelity=Fidelity.APPROXIMATION,
        method_prefix="screened numerical psi_nlm", progress=progress,
    )


def hf_density_volume(
    z: int, n_electrons: int, n: int, l: int, m: int, basis: str = "complex",
    resolution: int = 64, progress: Callable[[float], None] | None = None, *,
    config: Configuration | None = None, exchange: bool = True, pauli: bool = True,
) -> VolumeGrid:
    """The same, for a Hartree-Fock orbital; the tier is read off the solve."""
    validate_quantum_numbers(n, l)
    validate_angular(l, m)

    def evaluator(pos):
        return evaluate_hf_state(
            z, n_electrons, n, l, m, pos,
            basis=basis, config=config, exchange=exchange, pauli=pauli,
        )

    fidelity = evaluator(np.zeros((1, 3))).provenance.fidelity
    return _volume(
        evaluator, n=n, l=l, m=m, basis=basis,
        start_half_width=default_half_width(n, max(z - n_electrons + 1, 1), 1.0),
        resolution=resolution, fidelity=fidelity,
        method_prefix="Hartree-Fock psi_nlm", progress=progress,
    )
