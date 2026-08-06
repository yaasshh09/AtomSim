"""2-D cross-sections of psi_nlm on the y=0 plane (contains the z quantization axis).

This is where the classic "hydrogen poster" pictures live. On y=0 the azimuth
is phi = 0 (x >= 0) or pi (x < 0), so e^{i m phi} = +/-1 and psi is real-valued
in BOTH angular bases: a signed-psi plot is honest here, and the plot label
must state exactly which quantity is shown (spec 7.2 honesty fix over the
poster's contradictory -/+ "probability density" colorbar).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from atomsim.analytic.angular import validate_angular
from atomsim.analytic.hydrogen import validate_quantum_numbers
from atomsim.analytic.wavefunction import evaluate_state
from atomsim.atoms import Configuration
from atomsim.hf_atom import evaluate_hf_state
from atomsim.provenance import Fidelity, Provenance
from atomsim.screened_atom import evaluate_screened_state

_ROW_CHUNKS = 16


def _plane_values(evaluator, quantity, resolution, half_extent, progress):
    """Evaluate `quantity` on a (resolution x resolution) y=0 grid via `evaluator`.

    `evaluator(pos)` takes (N, 3) Cartesian positions (bohr) and returns a
    WavefunctionValues; returns (values 2-D, axis 1-D, psi assumptions).
    """
    axis = np.linspace(-half_extent, half_extent, resolution)
    values = np.zeros((resolution, resolution))
    psi_assumptions: tuple[str, ...] = ()
    starts = np.linspace(0, resolution, _ROW_CHUNKS + 1).astype(int)
    for k in range(_ROW_CHUNKS):
        i0, i1 = int(starts[k]), int(starts[k + 1])
        if i1 == i0:
            continue
        zz, xx = np.meshgrid(axis[i0:i1], axis, indexing="ij")
        pos = np.stack([xx.ravel(), np.zeros(xx.size), zz.ravel()], axis=1)
        psi = evaluator(pos)
        psi_assumptions = psi.provenance.assumptions
        block = psi.values.reshape(i1 - i0, resolution)
        if quantity == "density":
            values[i0:i1] = np.abs(block) ** 2
        else:
            values[i0:i1] = np.real(block)
        if progress is not None:
            progress(i1 / resolution)
    return values, axis, psi_assumptions


@dataclass(frozen=True)
class PlaneGrid:
    """psi-derived values on a square y=0 grid. Container carries provenance."""

    values: np.ndarray  # (resolution, resolution) float64; [i, j] = (x=axis[j], 0, z=axis[i])
    axis: np.ndarray    # (resolution,) shared x/z axis, bohr
    quantity: str       # "density" (|psi|^2) or "psi" (signed psi, real on y=0)
    unit: str           # bohr^-3 for density, bohr^-3/2 for psi
    label: str
    n: int
    l: int
    m: int
    Z: int
    mu_ratio: float
    basis: str
    provenance: Provenance


def default_half_extent(n: int, Z: int = 1, mu_ratio: float = 1.0) -> float:
    """Display framing: ~2.5 n^2 a0/(Z mu) keeps outer lobes visible at <1% peak density."""
    return 2.5 * n * n / (Z * mu_ratio)


def plane_grid(
    n: int,
    l: int,
    m: int,
    quantity: str = "density",
    basis: str = "complex",
    Z: int = 1,
    mu_ratio: float = 1.0,
    resolution: int = 512,
    half_extent: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> PlaneGrid:
    """Evaluate |psi|^2 or signed psi on a (resolution x resolution) y=0 grid."""
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    if quantity not in ("density", "psi"):
        raise ValueError(f"quantity must be 'density' or 'psi', got {quantity!r}")
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    he = default_half_extent(n, Z, mu_ratio) if half_extent is None else float(half_extent)
    if he <= 0.0:
        raise ValueError(f"half_extent must be positive, got {he}")

    def evaluator(pos):
        return evaluate_state(n, l, m, pos, Z=Z, mu_ratio=mu_ratio, basis=basis)

    values, axis, psi_assumptions = _plane_values(
        evaluator, quantity, resolution, he, progress
    )

    if quantity == "density":
        unit = "bohr^-3"
        label = f"|psi_{n},{l},{m}|^2 on the y=0 plane"
        qdesc = "|psi|^2 (probability density)"
        extra = ("plane y=0 contains the z quantization axis",)
    else:
        unit = "bohr^-3/2"
        label = f"psi_{n},{l},{m} on the y=0 plane"
        qdesc = "signed psi"
        extra = (
            "plane y=0 contains the z quantization axis",
            "psi is real on y=0 (e^{i m phi} = +/-1 there), so a signed plot is honest",
        )
    provenance = Provenance(
        fidelity=Fidelity.EXACT,
        method=(
            f"{qdesc} from closed-form psi_nlm on a {resolution}x{resolution} "
            f"y=0 grid, half-extent {he:g} bohr"
        ),
        assumptions=psi_assumptions + extra,
        refinement="increase resolution or adjust extent",
    )
    return PlaneGrid(
        values=values, axis=axis, quantity=quantity, unit=unit, label=label,
        n=n, l=l, m=m, Z=Z, mu_ratio=mu_ratio, basis=basis, provenance=provenance,
    )


def screened_plane_grid(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    quantity: str = "density",
    basis: str = "complex",
    resolution: int = 512,
    half_extent: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> PlaneGrid:
    """|psi|^2 or signed psi for a screened GSZ/GJG atom on the y=0 plane."""
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    if quantity not in ("density", "psi"):
        raise ValueError(f"quantity must be 'density' or 'psi', got {quantity!r}")
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    z_net = max(z - n_electrons + 1, 1)  # asymptotic core charge sets display extent
    he = default_half_extent(n, z_net, 1.0) if half_extent is None else float(half_extent)
    if he <= 0.0:
        raise ValueError(f"half_extent must be positive, got {he}")

    def evaluator(pos):
        return evaluate_screened_state(z, n_electrons, n, l, m, pos, basis=basis)

    values, axis, psi_assumptions = _plane_values(
        evaluator, quantity, resolution, he, progress
    )

    if quantity == "density":
        unit = "bohr^-3"
        label = f"|psi_{n},{l},{m}|^2 on the y=0 plane"
        qdesc = "|psi|^2 (probability density)"
        extra = ("plane y=0 contains the z quantization axis",)
    else:
        unit = "bohr^-3/2"
        label = f"psi_{n},{l},{m} on the y=0 plane"
        qdesc = "signed psi"
        extra = (
            "plane y=0 contains the z quantization axis",
            "psi is real on y=0 (e^{i m phi} = +/-1 there), so a signed plot is honest",
        )
    provenance = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=(
            f"{qdesc} from a numerical screened psi_nlm on a {resolution}x{resolution} "
            f"y=0 grid, half-extent {he:g} bohr"
        ),
        assumptions=psi_assumptions + extra,
        refinement="increase resolution, extent, or radial solver resolution",
    )
    return PlaneGrid(
        values=values, axis=axis, quantity=quantity, unit=unit, label=label,
        n=n, l=l, m=m, Z=z, mu_ratio=1.0, basis=basis, provenance=provenance,
    )


def hf_plane_grid(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    quantity: str = "density",
    basis: str = "complex",
    resolution: int = 512,
    half_extent: float | None = None,
    progress: Callable[[float], None] | None = None,
    *,
    config: Configuration | None = None,
    exchange: bool = True,
    pauli: bool = True,
) -> PlaneGrid:
    """|psi|^2 or signed psi for a Hartree-Fock orbital on the y=0 plane.

    Structurally the screened routine with one line changed, which is the
    point: the two many-electron models have to be comparable in the same
    frame, and a plane that fitted its own extent differently would put half
    the difference between the pictures into the axes.
    """
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    if quantity not in ("density", "psi"):
        raise ValueError(f"quantity must be 'density' or 'psi', got {quantity!r}")
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    z_net = max(z - n_electrons + 1, 1)  # asymptotic core charge sets display extent
    he = default_half_extent(n, z_net, 1.0) if half_extent is None else float(half_extent)
    if he <= 0.0:
        raise ValueError(f"half_extent must be positive, got {he}")

    def evaluator(pos):
        return evaluate_hf_state(
            z, n_electrons, n, l, m, pos,
            basis=basis, config=config, exchange=exchange, pauli=pauli,
        )

    values, axis, psi_assumptions = _plane_values(
        evaluator, quantity, resolution, he, progress
    )
    # The tier the solve came back with, read off the psi that was just
    # evaluated rather than assumed from the model's name. Costs one extra
    # interpolation against an already-cached solve, and buys a badge that
    # cannot drift out of step with the physics it sits over.
    fidelity = evaluator(np.zeros((1, 3))).provenance.fidelity

    if quantity == "density":
        unit = "bohr^-3"
        label = f"|psi_{n},{l},{m}|^2 on the y=0 plane"
        qdesc = "|psi|^2 (probability density)"
        extra = ("plane y=0 contains the z quantization axis",)
    else:
        unit = "bohr^-3/2"
        label = f"psi_{n},{l},{m} on the y=0 plane"
        qdesc = "signed psi"
        extra = (
            "plane y=0 contains the z quantization axis",
            "psi is real on y=0 (e^{i m phi} = +/-1 there), so a signed plot is honest",
        )
    provenance = Provenance(
        fidelity=fidelity,
        method=(
            f"{qdesc} from a Hartree-Fock psi_nlm on a {resolution}x{resolution} "
            f"y=0 grid, half-extent {he:g} bohr"
        ),
        assumptions=psi_assumptions + extra,
        refinement="increase resolution, extent, or the solver mesh refinement",
    )
    return PlaneGrid(
        values=values, axis=axis, quantity=quantity, unit=unit, label=label,
        n=n, l=l, m=m, Z=z, mu_ratio=1.0, basis=basis, provenance=provenance,
    )
