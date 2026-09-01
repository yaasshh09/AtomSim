"""My isosurfaces of |psi|^2, at the level that encloses a fraction of the
electron you state.

The textbook lobe is the least honest picture in chemistry. Every book draws
one, none of them says what it is a picture of, and the reader comes away
believing an orbital has an edge. It does not. A surface through |psi|^2 is a
choice of contour, and the only thing that makes it a claim about an atom rather
than about a rendering setting is the fraction of the electron it contains.

So my control here is that fraction. Ask me for 0.9 and I solve for the level
rather than picking it, and my answer carries the part textbooks leave out: the
electron is outside this surface one time in ten.

Two things can turn that number into a lie, and I measure both rather than
assuming them. The first is my box: probability that falls outside my grid
cannot be enclosed by anything, so if 2% of the electron is out there then a
"90%" contour I computed inside the box is really 90% of 98%, and no amount of
refinement fixes it. So I grow the box until it holds essentially everything,
and I report what still escapes. The second is my grid: the level that encloses
90% of a Riemann sum is not exactly the level that encloses 90% of the integral,
so I repeat the solve at half the resolution and take the difference as my error
bar.

My geometry lives in `numerics/marching_tets.py` and knows nothing about atoms.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from atomsim.analytic.angular import validate_angular
from atomsim.analytic.hydrogen import validate_quantum_numbers
from atomsim.analytic.wavefunction import evaluate_state
from atomsim.atoms import Configuration
from atomsim.hf_atom import evaluate_hf_state
from atomsim.numerics.marching_tets import (
    Mesh,
    connected_components,
    enclosed_volume,
    marching_tets,
    surface_area,
)
from atomsim.provenance import Fidelity, Provenance, Quantity
from atomsim.screened_atom import evaluate_screened_state

#: The grid sizes I will build a surface on. Odd counts put a grid point exactly
#: on the nucleus, where 1s has its cusp, which is where my Riemann sum is least
#: forgiving; even counts straddle it. I offer both and hide neither, since my
#: resolution appears in the provenance.
GRID_SIZES = (48, 64, 96, 128)

#: How much of the electron my box must hold before I solve a level on it. I
#: disclose the remainder rather than quietly rolling it into my answer.
_BOX_CAPTURE = 0.999

#: The radial samples I use to measure the tail. I keep them uniform, so my
#: trapezoid error is h^2 on a smooth r^2 |R(r)|^2, and 1024 intervals over tens
#: of bohr leave it far below the 1e-3 I size my box to.
_TAIL_SAMPLES = 1024

_EVAL_CHUNK = 200_000


@dataclass(frozen=True)
class Isosurface:
    """A closed surface of |psi|^2 I drew, and the measurements that make it a
    claim.

    I use a container rather than a `Field`: a mesh is not samples on a 1-D
    grid, and my `PlaneGrid` already set the precedent for a container that
    carries its own `Provenance`.
    """

    vertices: np.ndarray        # (M, 3) bohr
    triangles: np.ndarray       # (K, 3) int32 indices into vertices
    vertex_phase: np.ndarray    # (M,) radians: arg(psi) at each vertex, 0 or pi when psi is real
    target_fraction: float
    enclosed_fraction: Quantity  # what my level actually encloses on this grid
    level: Quantity              # the |psi|^2 contour value, bohr^-3
    escaped_fraction: Quantity   # probability outside my box entirely
    mesh_volume: Quantity        # bohr^3, from my triangles
    voxel_volume: Quantity       # bohr^3, from counting my cells above the level
    area: Quantity               # bohr^2
    components: int
    half_width: float
    resolution: int
    n: int
    l: int
    m: int
    Z: int
    basis: str
    label: str
    provenance: Provenance

    @property
    def outside_fraction(self) -> float:
        """The complement, which is the sentence textbooks leave out."""
        return 1.0 - self.enclosed_fraction.value


def default_half_width(n: int, Z: int = 1, mu_ratio: float = 1.0) -> float:
    """My starting box for a hydrogen-like state, in bohr.

    I keep it deliberately generous rather than tuned: it is only a starting
    point for my growth loop below, and that loop stops on a measurement. For 1s
    it gives me 8 bohr where 99.9% of the electron is inside 5.61, and guessing
    high costs me a longer radial profile, not a wrong answer.
    """
    return (2.0 * n * n + 6.0 * n) / (Z * mu_ratio)


def _density_on_grid(evaluator, half_width: float, resolution: int, progress=None):
    """|psi|^2 on my cubic grid, plus the psi assumptions the evaluator declares."""
    axis = np.linspace(-half_width, half_width, resolution)
    grid = np.zeros((resolution, resolution, resolution))
    assumptions: tuple[str, ...] = ()
    rows_per_chunk = max(1, _EVAL_CHUNK // (resolution * resolution))
    for i0 in range(0, resolution, rows_per_chunk):
        i1 = min(i0 + rows_per_chunk, resolution)
        xx, yy, zz = np.meshgrid(axis[i0:i1], axis, axis, indexing="ij")
        pos = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        psi = evaluator(pos)
        assumptions = psi.provenance.assumptions
        grid[i0:i1] = (np.abs(psi.values) ** 2).reshape(i1 - i0, resolution, resolution)
        if progress is not None:
            progress(i1 / resolution)
    return grid, axis, assumptions


def _cell_volume(half_width: float, resolution: int) -> float:
    return (2.0 * half_width / (resolution - 1)) ** 3


def _captured(grid: np.ndarray, half_width: float, resolution: int) -> float:
    return float(grid.sum() * _cell_volume(half_width, resolution))


def radial_mass(evaluator, l: int, m: int, r_max: float, samples: int = _TAIL_SAMPLES):
    """The probability I enclose within each radius out to `r_max`, by spherical
    average.

    I cannot measure the tail on the same cubic grid I draw the surface on. A
    Riemann sum over a fixed number of points gets coarser as my box grows, so
    what it reports is the physical tail plus a quadrature error that grows
    faster than the tail shrinks: for a hydrogen 1s my sum fell from 0.997 at 8
    bohr to 0.0005 at 134, and any loop that grows the box until the sum is
    large enough grows it forever.

    So I measure the tail in the coordinate it actually lives in. For a single
    state psi = R(r) Y_lm, the mass in a shell is 4 pi r^2 <|psi|^2> over the
    sphere, and that average is a polynomial in cos(theta) of degree 2l times a
    trigonometric polynomial in phi of degree 2|m|. Gauss-Legendre with l + 2
    nodes and a uniform phi rule with 2|m| + 2 points are each exact on those,
    so my only approximation left is the 1-D radial quadrature.

    I return `(r, cumulative)`, where `cumulative[k]` is the probability inside
    radius `r[k]`. A normalized state has `cumulative[-1]` approaching 1, which
    is a check on the evaluator rather than something I assume about it.
    """
    n_theta = max(2, l + 2)
    n_phi = max(4, 2 * abs(m) + 2)
    u, w_u = np.polynomial.legendre.leggauss(n_theta)
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    sin_theta = np.sqrt(1.0 - u**2)

    # My weights average over the sphere: sum(w_u) = 2 for the u integral and
    # 1 / n_phi for the uniform phi rule, so the two together are 1/(4 pi) times
    # the solid-angle integral.
    directions = np.stack(
        [
            np.outer(sin_theta, np.cos(phi)).ravel(),
            np.outer(sin_theta, np.sin(phi)).ravel(),
            np.repeat(u, n_phi),
        ],
        axis=1,
    )
    weights = np.repeat(w_u, n_phi) / (2.0 * n_phi)

    r = np.linspace(0.0, float(r_max), samples + 1)
    pos = (r[:, None, None] * directions[None, :, :]).reshape(-1, 3)
    psi = evaluator(pos)
    sphere_mean = (np.abs(psi.values) ** 2).reshape(r.size, -1) @ weights
    shell = 4.0 * np.pi * r**2 * sphere_mean

    cumulative = np.zeros_like(r)
    cumulative[1:] = np.cumsum(0.5 * (shell[1:] + shell[:-1]) * np.diff(r))
    return r, cumulative


def _fit_box(evaluator, l: int, m: int, start: float, growths: int = 8):
    """My smallest radius holding `_BOX_CAPTURE` of the electron, and the profile.

    I test the inscribed sphere rather than the cube, so the box I hand back
    holds at least the stated fraction and the escaped mass I report from it is
    an upper bound rather than an estimate.
    """
    r_max = float(start)
    r, cumulative = radial_mass(evaluator, l, m, r_max)
    for _ in range(growths):
        if cumulative[-1] >= _BOX_CAPTURE:
            break
        r_max *= 1.6
        r, cumulative = radial_mass(evaluator, l, m, r_max)
    half_width = float(np.interp(_BOX_CAPTURE, cumulative, r))
    return half_width, r, cumulative


def solve_level(grid: np.ndarray, cell_volume: float, target: float) -> float:
    """The |psi|^2 level whose super-level set holds `target` of the electron.

    I sort my cell densities descending, accumulate `rho dV`, and interpolate
    where the running total crosses the target. That is exact on my grid, one
    sort, with no tolerance to tune and no root-finder to fail.

    My target is a fraction of the whole electron, not of the part that landed
    inside my box. That choice is what makes my escaped mass a disclosure
    instead of a silent renormalization.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"target fraction must be in (0, 1), got {target}")
    flat = np.sort(grid.reshape(-1))[::-1]
    cumulative = np.cumsum(flat) * cell_volume
    if cumulative[-1] < target:
        raise ValueError(
            f"my box holds {cumulative[-1]:.4f} of the probability, which cannot "
            f"enclose {target:.4f}: widen the box"
        )
    idx = int(np.searchsorted(cumulative, target))
    if idx == 0:
        return float(flat[0])
    span = cumulative[idx] - cumulative[idx - 1]
    t = 0.0 if span == 0.0 else (target - cumulative[idx - 1]) / span
    return float(flat[idx - 1] + t * (flat[idx] - flat[idx - 1]))


def fraction_above(grid: np.ndarray, cell_volume: float, level: float) -> float:
    """The probability my region at or above `level` holds, on this grid."""
    return float(grid[grid >= level].sum() * cell_volume)


def _phase_at(evaluator, vertices: np.ndarray) -> np.ndarray:
    """arg(psi) at each of my vertices.

    I draw the surface through |psi|^2, which is blind to sign, so this is what
    gives me the two-coloured lobes of a p orbital. In my real basis psi is real
    and the answer is 0 or pi; in my complex basis it is the true phase, and it
    feeds the same hue map my point cloud uses.
    """
    if vertices.shape[0] == 0:
        return np.zeros(0)
    return np.angle(evaluator(vertices).values)


def _build(
    evaluator,
    *,
    n: int,
    l: int,
    m: int,
    z: int,
    basis: str,
    target_fraction: float,
    resolution: int,
    half_width: float | None,
    start_half_width: float,
    fidelity: Fidelity,
    method_prefix: str,
    extra_assumptions: tuple[str, ...],
    refinement: str,
    progress: Callable[[float], None] | None,
) -> Isosurface:
    if resolution not in GRID_SIZES:
        raise ValueError(f"resolution must be one of {GRID_SIZES}, got {resolution}")
    if not 0.0 < target_fraction < 1.0:
        raise ValueError(f"target fraction must be in (0, 1), got {target_fraction}")

    requested = None if half_width is None else float(half_width)
    if requested is not None and requested <= 0.0:
        raise ValueError(f"half_width must be positive, got {requested}")

    # I measure the tail out to at least the box I actually use, so a box you
    # set by hand gets the same measured escape one I fitted does.
    probe = start_half_width if requested is None else max(start_half_width, requested)
    fitted, tail_r, tail_cumulative = _fit_box(evaluator, l, m, probe)
    half_width = fitted if requested is None else requested
    escaped = float(max(0.0, 1.0 - np.interp(half_width, tail_r, tail_cumulative)))
    normalization = float(tail_cumulative[-1])

    grid, axis, psi_assumptions = _density_on_grid(evaluator, half_width, resolution, progress)
    cell = _cell_volume(half_width, resolution)
    captured = _captured(grid, half_width, resolution)

    level = solve_level(grid, cell, target_fraction)
    achieved = fraction_above(grid, cell, level)

    spacing = float(axis[1] - axis[0])
    mesh: Mesh = marching_tets(grid, level, origin=(-half_width,) * 3, spacing=spacing)
    phase = _phase_at(evaluator, mesh.vertices)
    mesh_vol = enclosed_volume(mesh)
    voxel_vol = float((grid >= level).sum() * cell)
    components = connected_components(mesh)

    # I halve the grid twice over, because the two things this surface of mine
    # claims do not converge at the same rate and one of them is nearly blind.
    #
    # My level itself barely moves under halving. For a hydrogen 2p at 50% my
    # coarse and fine levels agree to fifteen digits, so the enclosed fraction
    # comes back accurate to 1e-4 and an error bar built only from it reads as
    # near zero. That is a true statement about my fraction and it would be a
    # false impression of my surface: WHERE the contour sits carries far more
    # error than WHAT it encloses, and volume is the quantity that sees it.
    coarse = grid[::2, ::2, ::2]
    try:
        # Every second point, so my cells are twice as wide and eight times as big.
        coarse_level = solve_level(coarse, 8.0 * cell, target_fraction)
        fraction_error = abs(fraction_above(grid, cell, coarse_level) - achieved)
        coarse_mesh = marching_tets(
            coarse, coarse_level, origin=(-half_width,) * 3, spacing=2.0 * spacing
        )
        volume_error = abs(enclosed_volume(coarse_mesh) - mesh_vol)
    except ValueError:
        # My halved grid resolves the peak too poorly to hold the target at
        # all. That is a statement about the coarse grid and not about this one,
        # and inventing an error bar from it would be worse than saying so.
        fraction_error = None
        volume_error = None

    outside = 1.0 - achieved
    assumptions = psi_assumptions + extra_assumptions + (
        f"my surface encloses {achieved:.4f} of the electron, so the electron is "
        f"found outside the surface {outside:.1%} of the time: an orbital has no "
        f"boundary, and this contour is a choice of enclosed fraction, not an edge",
        f"{escaped:.2e} of the probability lies outside the sphere of radius "
        f"{half_width:g} bohr inscribed in the box, an upper bound on what no contour "
        f"I draw in the box can enclose",
        f"my {resolution}^3 sum of |psi|^2 dV over the box closes to {captured:.6f} "
        f"and my radial quadrature puts {normalization:.6f} of the state inside "
        f"{tail_r[-1]:g} bohr: each checks the other, and I assumed neither",
        f"my mesh volume {mesh_vol:.4f} bohr^3 against {voxel_vol:.4f} bohr^3 counted "
        f"by cells above the level: the difference is my triangulation's discretization",
        f"my surface comes out in {components} piece{'' if components == 1 else 's'} on "
        f"a {spacing:.3f} bohr cell; I do not resolve a gap narrower than that, so lobes "
        f"separated by less appear joined. A p orbital's lobes touch at the node, and "
        f"the looser the contour the nearer the node they part",
        (
            f"I solved the level on a {resolution}^3 grid; halving it moves my "
            f"enclosed fraction by {fraction_error:.2e} and my enclosed volume by "
            f"{volume_error:.2e} bohr^3 "
            f"({volume_error / mesh_vol if mesh_vol > 0 else float('nan'):.2%}). The second "
            f"number is my honest one for the shape: the level is nearly grid-"
            f"independent, so my fraction converges long before the surface does"
            if fraction_error is not None
            else f"I solved the level on a {resolution}^3 grid; my halved grid could "
            f"not hold {target_fraction:.4g} of the probability, so I quote no "
            f"grid-halving estimate"
        ),
    )
    provenance = Provenance(
        fidelity=fidelity,
        method=(
            f"I drew a {method_prefix} isosurface at the |psi|^2 level enclosing "
            f"{target_fraction:.4g} of the electron, by marching tetrahedra on a "
            f"{resolution}^3 grid of half-width {half_width:g} bohr"
        ),
        assumptions=assumptions,
        error_estimate=fraction_error,
        refinement=refinement,
    )

    def scalar(value, unit, label, error=None):
        return Quantity(
            value=float(value),
            unit=unit,
            label=label,
            provenance=Provenance(
                fidelity=fidelity,
                method=provenance.method,
                assumptions=assumptions,
                error_estimate=error,
                refinement=refinement,
            ),
        )

    return Isosurface(
        vertices=mesh.vertices,
        triangles=mesh.triangles,
        vertex_phase=phase,
        target_fraction=float(target_fraction),
        enclosed_fraction=scalar(achieved, "1", "enclosed probability", fraction_error),
        level=scalar(level, "bohr^-3", "|psi|^2 contour level"),
        escaped_fraction=scalar(escaped, "1", "probability outside the box"),
        mesh_volume=scalar(
            mesh_vol, "bohr^3", "volume enclosed by the surface", volume_error
        ),
        voxel_volume=scalar(voxel_vol, "bohr^3", "volume of cells above the level"),
        area=scalar(surface_area(mesh), "bohr^2", "surface area"),
        components=components,
        half_width=half_width,
        resolution=resolution,
        n=n,
        l=l,
        m=m,
        Z=z,
        basis=basis,
        label=f"|psi_{n},{l},{m}|^2 contour enclosing {target_fraction:.0%}",
        provenance=provenance,
    )


def isosurface(
    n: int,
    l: int,
    m: int,
    target_fraction: float = 0.9,
    basis: str = "complex",
    Z: int = 1,
    mu_ratio: float = 1.0,
    resolution: int = 96,
    half_width: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> Isosurface:
    """My enclosed-probability isosurface of a hydrogen-like |psi_nlm|^2.

    I call it `NUMERICAL` even though psi is closed form: my grid and my linear
    interpolation along cell edges are the approximation here, and no exactness
    upstream makes a triangulated contour exact.
    """
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    if basis not in ("complex", "real"):
        raise ValueError(f"basis must be 'complex' or 'real', got {basis!r}")

    def evaluator(pos):
        return evaluate_state(n, l, m, pos, Z=Z, mu_ratio=mu_ratio, basis=basis)

    return _build(
        evaluator,
        n=n, l=l, m=m, z=Z, basis=basis,
        target_fraction=target_fraction,
        resolution=resolution,
        half_width=half_width,
        start_half_width=default_half_width(n, Z, mu_ratio),
        fidelity=Fidelity.NUMERICAL,
        method_prefix="closed-form psi_nlm",
        extra_assumptions=(
            "psi is exact; my grid, my level solve and my triangulation are not",
        ),
        refinement="I could raise my grid resolution or widen my box",
        progress=progress,
    )


def screened_isosurface(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    target_fraction: float = 0.9,
    basis: str = "complex",
    resolution: int = 96,
    half_width: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> Isosurface:
    """The same surface, for a screened (GSZ/GJG) atom.

    I take `APPROXIMATION`, the weaker of my two tiers: my screening model is a
    bigger departure from the real atom than my grid is from the model.
    """
    validate_quantum_numbers(n, l)
    validate_angular(l, m)

    def evaluator(pos):
        return evaluate_screened_state(z, n_electrons, n, l, m, pos, basis=basis)

    z_net = max(z - n_electrons + 1, 1)
    return _build(
        evaluator,
        n=n, l=l, m=m, z=z, basis=basis,
        target_fraction=target_fraction,
        resolution=resolution,
        half_width=half_width,
        start_half_width=default_half_width(n, z_net, 1.0),
        fidelity=Fidelity.APPROXIMATION,
        method_prefix="screened numerical psi_nlm",
        extra_assumptions=(
            "my enclosed fraction is exact for the screened model's psi, which is "
            "itself an approximation to the atom",
        ),
        refinement="I could raise my grid resolution, widen my box, or use Hartree-Fock",
        progress=progress,
    )


def hf_isosurface(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    target_fraction: float = 0.9,
    basis: str = "complex",
    resolution: int = 96,
    half_width: float | None = None,
    progress: Callable[[float], None] | None = None,
    *,
    config: Configuration | None = None,
    exchange: bool = True,
    pauli: bool = True,
) -> Isosurface:
    """The same surface, for a self-consistent Hartree-Fock orbital.

    I take `APPROXIMATION`, the weaker of my two tiers, for the reason
    screened_isosurface gives: my model's distance from the real atom is larger
    than my grid's distance from the model. With either counterfactual switch
    thrown it is COUNTERFACTUAL instead, and I read that off the solve rather
    than deciding it here.
    """
    validate_quantum_numbers(n, l)
    validate_angular(l, m)

    def evaluator(pos):
        return evaluate_hf_state(
            z, n_electrons, n, l, m, pos,
            basis=basis, config=config, exchange=exchange, pauli=pauli,
        )

    fidelity = evaluator(np.zeros((1, 3))).provenance.fidelity
    counterfactual = fidelity is Fidelity.COUNTERFACTUAL
    z_net = max(z - n_electrons + 1, 1)
    return _build(
        evaluator,
        n=n, l=l, m=m, z=z, basis=basis,
        target_fraction=target_fraction,
        resolution=resolution,
        half_width=half_width,
        start_half_width=default_half_width(n, z_net, 1.0),
        fidelity=fidelity,
        method_prefix=(
            "Hartree psi_nlm" if counterfactual else "Hartree-Fock psi_nlm"
        ),
        extra_assumptions=(
            (
                "my enclosed fraction is exact for the orbital this solve of mine "
                "produced, and that solve is a counterfactual: the fraction is "
                "not a statement about any real atom"
            )
            if counterfactual
            else (
                "my enclosed fraction is exact for the Hartree-Fock orbital, "
                "which neglects correlation"
            ),
        ),
        refinement=(
            "I could turn the altered rule back on"
            if counterfactual
            else "I could raise my grid resolution or widen my box"
        ),
        progress=progress,
    )
