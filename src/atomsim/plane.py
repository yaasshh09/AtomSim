"""2-D cross-sections of psi_nlm on the y=0 plane, which contains the z
quantization axis.

This is where the classic "hydrogen poster" pictures live. On y=0 the azimuth
is phi = 0 (x >= 0) or pi (x < 0), so e^{i m phi} = +/-1 and psi is real-valued
in BOTH angular bases: a signed-psi plot is honest here, and the label states
exactly which quantity is shown (spec 7.2, the honesty fix over the poster's
contradictory -/+ "probability density" colorbar).
"""

import dataclasses
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


@dataclass(frozen=True)
class PlaneOrientation:
    """Which plane is drawn: its normal at polar angle theta and azimuth phi
    (degrees), shifted `offset` bohr from the nucleus along that normal.

    The default, theta = phi = 90 and no offset, is the y = 0 plane every
    view has always drawn, and its frame is the exact unit axes, so the
    default grid is the same floats it always was.
    """

    theta: float = 90.0
    phi: float = 90.0
    offset: float = 0.0

    @property
    def is_y0(self) -> bool:
        return (self.theta, self.phi, self.offset) == (90.0, 90.0, 0.0)

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(e1, e2, normal): grid column j runs along e1, row i along e2.

        e2 is minus the polar unit vector at the normal, which is +z projected
        into the plane whenever the plane is not horizontal, and never
        degenerates when it is. e1 = normal x e2, which for the default gives
        x across and z up with the normal, +y, pointing away from the viewer:
        the classic x-z picture, and the convention every tilt keeps.
        """
        if self.is_y0:
            return (
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 1.0, 0.0]),
            )
        t, f = np.radians(self.theta), np.radians(self.phi)
        normal = np.array([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)])
        e2 = -np.array([np.cos(t) * np.cos(f), np.cos(t) * np.sin(f), -np.sin(t)])
        return np.cross(normal, e2), e2, normal

    def phrase(self) -> str:
        if self.is_y0:
            return "the y=0 plane"
        where = (
            f"{self.offset:g} bohr from the nucleus" if self.offset else "through the nucleus"
        )
        return f"the plane with normal at theta={self.theta:g}, phi={self.phi:g} deg, {where}"


Y0 = PlaneOrientation()


def psi_is_real_on(orientation: PlaneOrientation, basis: str, m: int) -> bool:
    """Whether psi is real on this plane, so a signed plot of it is honest.

    A real orbital is real everywhere and an m = 0 state has no phase to
    carry. Otherwise e^{i m phi} is +/-1 only on y = 0; on any other plane it
    turns, and dropping its imaginary part would draw half a function.
    """
    return basis == "real" or m == 0 or orientation.is_y0


def _describe(orientation: PlaneOrientation, quantity: str, n: int, l: int, m: int):
    """(unit, label, qdesc, extra assumptions, grid word) for one grid.

    The y=0 strings are exactly the ones every plane always carried.
    """
    where = orientation.phrase()
    if orientation.is_y0:
        grid_word = "y=0 grid"
        plane_note = ("the plane y=0 contains the z quantization axis",)
        real_note = (
            "psi is real on y=0 (e^{i m phi} = +/-1 there), so the signed plot is honest"
        )
    else:
        e1, e2, _ = orientation.frame()
        grid_word = f"tilted grid ({where})"
        plane_note = (
            f"{where}; the horizontal axis runs along "
            f"({e1[0]:.3f}, {e1[1]:.3f}, {e1[2]:.3f}) and the vertical along "
            f"({e2[0]:.3f}, {e2[1]:.3f}, {e2[2]:.3f})",
        )
        real_note = (
            "psi is real everywhere in this basis (a real orbital, or m = 0), "
            "so the signed plot is honest"
        )
    if quantity == "density":
        return ("bohr^-3", f"|psi_{n},{l},{m}|^2 on {where}",
                "|psi|^2 (probability density)", plane_note, grid_word)
    return ("bohr^-3/2", f"psi_{n},{l},{m} on {where}", "signed psi",
            plane_note + (real_note,), grid_word)


def _plane_values(evaluator, quantity, resolution, half_extent, progress, orientation=Y0):
    """Evaluate `quantity` on a (resolution x resolution) grid in the plane,
    through `evaluator`.

    `evaluator(pos)` takes (N, 3) Cartesian positions (bohr) and hands back a
    WavefunctionValues. Returns (values 2-D, axis 1-D, psi assumptions).
    """
    axis = np.linspace(-half_extent, half_extent, resolution)
    e1, e2, normal = orientation.frame()
    shift = orientation.offset * normal
    values = np.zeros((resolution, resolution))
    psi_assumptions: tuple[str, ...] = ()
    starts = np.linspace(0, resolution, _ROW_CHUNKS + 1).astype(int)
    for k in range(_ROW_CHUNKS):
        i0, i1 = int(starts[k]), int(starts[k + 1])
        if i1 == i0:
            continue
        zz, xx = np.meshgrid(axis[i0:i1], axis, indexing="ij")
        if orientation.is_y0:
            pos = np.stack([xx.ravel(), np.zeros(xx.size), zz.ravel()], axis=1)
        else:
            pos = xx.ravel()[:, None] * e1 + zz.ravel()[:, None] * e2 + shift
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
    """The psi-derived values on a square y=0 grid, with their provenance."""

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
    orientation: PlaneOrientation = Y0


def default_half_extent(n: int, Z: int = 1, mu_ratio: float = 1.0) -> float:
    """How the display is framed: ~2.5 n^2 a0/(Z mu) keeps the outer lobes visible."""
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
    orientation: PlaneOrientation = Y0,
) -> PlaneGrid:
    """Evaluate |psi|^2 or signed psi on a (resolution x resolution) y=0 grid."""
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    if quantity not in ("density", "psi"):
        raise ValueError(f"quantity must be 'density' or 'psi', got {quantity!r}")
    if quantity == "psi" and not psi_is_real_on(orientation, basis, m):
        raise ValueError(
            "psi is complex on this plane (complex basis, m != 0), so a signed plot "
            "would drop half of it; draw the density or use the real basis"
        )
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    he = default_half_extent(n, Z, mu_ratio) if half_extent is None else float(half_extent)
    if he <= 0.0:
        raise ValueError(f"half_extent must be positive, got {he}")

    def evaluator(pos):
        return evaluate_state(n, l, m, pos, Z=Z, mu_ratio=mu_ratio, basis=basis)

    values, axis, psi_assumptions = _plane_values(
        evaluator, quantity, resolution, he, progress, orientation
    )

    unit, label, qdesc, extra, grid_word = _describe(orientation, quantity, n, l, m)
    provenance = Provenance(
        fidelity=Fidelity.EXACT,
        method=(
            f"evaluated {qdesc} from closed-form psi_nlm on a "
            f"{resolution}x{resolution} {grid_word}, half-extent {he:g} bohr"
        ),
        assumptions=psi_assumptions + extra,
        refinement="raise the resolution or change the extent",
    )
    return PlaneGrid(
        values=values, axis=axis, quantity=quantity, unit=unit, label=label,
        n=n, l=l, m=m, Z=Z, mu_ratio=mu_ratio, basis=basis, provenance=provenance,
        orientation=orientation,
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
    orientation: PlaneOrientation = Y0,
) -> PlaneGrid:
    """Evaluate |psi|^2 or signed psi for a screened GSZ/GJG atom on the y=0 plane."""
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    if quantity not in ("density", "psi"):
        raise ValueError(f"quantity must be 'density' or 'psi', got {quantity!r}")
    if quantity == "psi" and not psi_is_real_on(orientation, basis, m):
        raise ValueError(
            "psi is complex on this plane (complex basis, m != 0), so a signed plot "
            "would drop half of it; draw the density or use the real basis"
        )
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    z_net = max(z - n_electrons + 1, 1)  # the asymptotic core charge sets the extent
    he = default_half_extent(n, z_net, 1.0) if half_extent is None else float(half_extent)
    if he <= 0.0:
        raise ValueError(f"half_extent must be positive, got {he}")

    def evaluator(pos):
        return evaluate_screened_state(z, n_electrons, n, l, m, pos, basis=basis)

    values, axis, psi_assumptions = _plane_values(
        evaluator, quantity, resolution, he, progress, orientation
    )

    unit, label, qdesc, extra, grid_word = _describe(orientation, quantity, n, l, m)
    provenance = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=(
            f"evaluated {qdesc} from a numerical screened psi_nlm on a "
            f"{resolution}x{resolution} {grid_word}, half-extent {he:g} bohr"
        ),
        assumptions=psi_assumptions + extra,
        refinement=(
            "raise the resolution, the extent, or the radial solver resolution"
        ),
    )
    return PlaneGrid(
        values=values, axis=axis, quantity=quantity, unit=unit, label=label,
        n=n, l=l, m=m, Z=z, mu_ratio=1.0, basis=basis, provenance=provenance,
        orientation=orientation,
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
    orientation: PlaneOrientation = Y0,
) -> PlaneGrid:
    """Evaluate |psi|^2 or signed psi for a Hartree-Fock orbital on the y=0 plane.

    This is the screened routine with one line changed, which is the point: the
    two many-electron models have to be comparable in the same frame, and a
    plane that fitted its own extent differently would put half the difference
    between the pictures into the axes.
    """
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    if quantity not in ("density", "psi"):
        raise ValueError(f"quantity must be 'density' or 'psi', got {quantity!r}")
    if quantity == "psi" and not psi_is_real_on(orientation, basis, m):
        raise ValueError(
            "psi is complex on this plane (complex basis, m != 0), so a signed plot "
            "would drop half of it; draw the density or use the real basis"
        )
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
        evaluator, quantity, resolution, he, progress, orientation
    )
    # The tier is read off the psi just evaluated rather than assumed from the
    # model's name. It costs one extra interpolation against an already cached
    # solve, and it buys a badge that cannot drift out of step with the physics
    # it sits over.
    fidelity = evaluator(np.zeros((1, 3))).provenance.fidelity

    unit, label, qdesc, extra, grid_word = _describe(orientation, quantity, n, l, m)
    provenance = Provenance(
        fidelity=fidelity,
        method=(
            f"evaluated {qdesc} from a Hartree-Fock psi_nlm on a "
            f"{resolution}x{resolution} {grid_word}, half-extent {he:g} bohr"
        ),
        assumptions=psi_assumptions + extra,
        refinement="raise the resolution, the extent, or the solver mesh refinement",
    )
    return PlaneGrid(
        values=values, axis=axis, quantity=quantity, unit=unit, label=label,
        n=n, l=l, m=m, Z=z, mu_ratio=1.0, basis=basis, provenance=provenance,
        orientation=orientation,
    )


def current_through_plane(pg: PlaneGrid) -> PlaneGrid:
    """j_y, the probability current through the y=0 plane, from a density grid.

    A stationary psi = R(r) Y_lm(theta, phi) e^{-iEt} has one current component,
    j_phi = hbar m |psi|^2 / (mu r sin theta), circling the z axis. On y=0 the
    azimuthal direction is +y at x > 0 and -y at x < 0 while r sin theta = |x|,
    so both halves give j_y = m |psi|^2 / (mu x) in atomic units: the flow goes in
    through one side of the axis and out through the other. A real orbital is a
    real function, so its current is identically zero, and so is any m = 0 state.

    The tier is the density's own: this is a closed-form identity applied to it,
    so it adds no error of its own.
    """
    if pg.quantity != "density":
        raise ValueError(f"need a density grid, got {pg.quantity!r}")
    if not pg.orientation.is_y0:
        raise ValueError("the current through the plane is drawn on y=0 only")
    values = np.zeros_like(pg.values)
    if pg.basis == "complex" and pg.m != 0:
        x = pg.axis[np.newaxis, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(x != 0.0, pg.m * pg.values / (pg.mu_ratio * x), 0.0)
    why_zero = (
        ("a real orbital is a real function, so Im(psi* grad psi) = 0 everywhere",)
        if pg.basis == "real" else
        (("m = 0: nothing circulates about z",) if pg.m == 0 else ())
    )
    provenance = Provenance(
        fidelity=pg.provenance.fidelity,
        method=(
            pg.provenance.method
            + "; then j_y = hbar m |psi|^2 / (mu x), the azimuthal current "
            "j_phi = hbar m |psi|^2 / (mu r sin theta) seen through y=0"
        ),
        assumptions=pg.provenance.assumptions + (
            "a stationary state: its current is purely azimuthal",
            "the in-plane components j_x and j_z are zero; only flow through the plane is drawn",
        ) + why_zero,
        error_estimate=pg.provenance.error_estimate,
        refinement=pg.provenance.refinement,
    )
    return dataclasses.replace(
        pg, values=values, quantity="current", unit="hbar/(m_e bohr^4)",
        label=f"j_y of psi_{pg.n},{pg.l},{pg.m} through the y=0 plane",
        provenance=provenance,
    )
