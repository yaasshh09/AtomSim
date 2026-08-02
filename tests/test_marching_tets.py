"""The geometry kernel, checked against shapes whose area and volume are known.

No physics here on purpose. A broken triangulation hides extremely well behind a
plausible-looking orbital, so it is tested against a sphere and a plane, where
the right answer is a formula and a wrong facet shows up as a number.

See docs/superpowers/specs/2026-08-03-phase25-isosurfaces-design.md.
"""

import numpy as np
import pytest

from atomsim.numerics.marching_tets import (
    connected_components,
    edge_use_counts,
    enclosed_volume,
    marching_tets,
    surface_area,
)


def _sphere_field(n: int, half_width: float = 2.0):
    """-(r^2), so the region at or above a level is a ball. Returns grid + step."""
    axis = np.linspace(-half_width, half_width, n)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return -(x**2 + y**2 + z**2), axis[1] - axis[0]


def _sphere_mesh(n: int, radius: float, half_width: float = 2.0):
    field, step = _sphere_field(n, half_width)
    return marching_tets(field, -(radius**2), origin=(-half_width,) * 3, spacing=step)


# --------------------------------------------------------------------------
# The derived case table
# --------------------------------------------------------------------------


def test_every_tetrahedron_case_emits_a_closed_ring_of_cut_edges():
    """A cut edge has one end inside and one outside. Nothing else is cut.

    This is the property the table was derived from, checked back against the
    table, so a typo in the derivation cannot pass as a convention.
    """
    from atomsim.numerics.marching_tets import _TET_CASES, _TET_EDGES

    for code in range(16):
        inside = {c for c in range(4) if code & (1 << c)}
        expected = {
            e for e, (a, b) in enumerate(map(tuple, _TET_EDGES)) if (a in inside) != (b in inside)
        }
        used = {int(e) for tri in _TET_CASES[code] for e in tri if e >= 0}
        assert used == expected, f"case {code} cuts the wrong edges"


def test_the_two_and_two_cases_emit_two_triangles_and_the_rest_emit_one():
    from atomsim.numerics.marching_tets import _TET_CASES

    for code in range(16):
        n_tris = int((_TET_CASES[code][:, 0] >= 0).sum())
        n_inside = bin(code).count("1")
        expected = 0 if n_inside in (0, 4) else (2 if n_inside == 2 else 1)
        assert n_tris == expected


def test_the_six_tetrahedra_tile_the_cell_exactly():
    """Kuhn's decomposition, checked by volume rather than by inspection."""
    from atomsim.numerics.marching_tets import _CORNER_OFFSETS, _TETS

    total = 0.0
    for tet in _TETS:
        p = _CORNER_OFFSETS[tet].astype(float)
        total += abs(np.dot(p[1] - p[0], np.cross(p[2] - p[0], p[3] - p[0]))) / 6
    assert total == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# Closed forms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [24, 48, 72])
def test_sphere_volume_and_area_converge_on_the_formulas(n):
    radius = 1.0
    mesh = _sphere_mesh(n, radius)
    # Measured, not guessed: a piecewise-linear surface cuts the corners off a
    # sphere, so it comes in low, and the deficit falls with the cell size.
    tolerance = {24: 0.03, 48: 8e-3, 72: 4e-3}[n]
    assert enclosed_volume(mesh) == pytest.approx(4 / 3 * np.pi * radius**3, rel=tolerance)
    assert surface_area(mesh) == pytest.approx(4 * np.pi * radius**2, rel=tolerance)


def test_refining_the_grid_actually_reduces_the_error():
    """Convergence, not just closeness at one resolution.

    A surface that is wrong by a fixed bias would pass a loose tolerance at
    every n. This fails unless refinement buys something.
    """
    exact = 4 / 3 * np.pi
    errors = [abs(enclosed_volume(_sphere_mesh(n, 1.0)) - exact) for n in (24, 48, 96)]
    assert errors[1] < errors[0] / 2
    assert errors[2] < errors[1] / 2


def test_vertices_land_on_the_sphere_to_first_order():
    """Linear interpolation puts vertices on the level set, not on cell corners.

    The radial spread of the vertices is the interpolation error, and it has to
    be far smaller than the cell size or the surface is a voxel staircase
    wearing triangles.
    """
    n, half_width = 48, 2.0
    step = 2 * half_width / (n - 1)
    radii = np.linalg.norm(_sphere_mesh(n, 1.0, half_width).vertices, axis=1)
    assert np.abs(radii - 1.0).max() < 0.1 * step


def test_a_plane_is_reproduced_exactly():
    """No curvature, so linear interpolation is not an approximation at all.

    Any offset here is a bug in the interpolation or the indexing, with no
    discretization error to hide behind.
    """
    axis = np.linspace(-1.0, 1.0, 21)
    x, _, _ = np.meshgrid(axis, axis, axis, indexing="ij")
    mesh = marching_tets(-x, -0.31, origin=(-1.0,) * 3, spacing=axis[1] - axis[0])
    assert np.abs(mesh.vertices[:, 0] - 0.31).max() < 1e-12


# --------------------------------------------------------------------------
# Mesh integrity
# --------------------------------------------------------------------------


def test_the_surface_is_watertight():
    """Every edge in exactly two triangles.

    This is what the shared Kuhn diagonal buys, and it is the property that
    makes the enclosed volume mean anything: an open surface still has a
    divergence-theorem number, it is just not a volume.
    """
    counts = edge_use_counts(_sphere_mesh(48, 1.0))
    assert set(np.unique(counts).tolist()) == {2}


def test_orientation_is_outward_everywhere():
    """One flipped facet leaves the volume nearly right, so check the facets.

    Each triangle's own signed contribution about the sphere's centre must be
    positive; summing first would let a flipped pair cancel invisibly.
    """
    mesh = _sphere_mesh(36, 1.0)
    v = mesh.vertices[mesh.triangles]
    signed = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2]))
    assert (signed > 0).all()


def test_vertices_are_welded_rather_than_duplicated_per_triangle():
    """Unwelded, a closed sphere would report thousands of components.

    Welding is also what keeps the buffer the client downloads roughly half the
    size of the naive one.
    """
    mesh = _sphere_mesh(36, 1.0)
    assert connected_components(mesh) == 1
    assert mesh.vertices.shape[0] < mesh.triangles.shape[0]


def test_two_separated_balls_come_out_as_two_components():
    axis = np.linspace(-3.0, 3.0, 61)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    left = (x + 1.5) ** 2 + y**2 + z**2
    right = (x - 1.5) ** 2 + y**2 + z**2
    mesh = marching_tets(
        -np.minimum(left, right), -0.36, origin=(-3.0,) * 3, spacing=axis[1] - axis[0]
    )
    assert connected_components(mesh) == 2


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------


def test_a_level_no_cell_crosses_gives_an_empty_mesh_not_a_crash():
    mesh = _sphere_mesh(16, 0.0001)
    assert mesh.is_empty
    assert enclosed_volume(mesh) == 0.0


def test_a_level_the_whole_box_is_above_gives_an_empty_mesh():
    """The surface has left the box. Empty is the honest answer; the caller
    above knows the box was too small, and this module does not guess."""
    field = np.ones((8, 8, 8))
    assert marching_tets(field, 0.5).is_empty


def test_a_field_that_is_not_three_dimensional_is_refused():
    with pytest.raises(ValueError, match="3-D"):
        marching_tets(np.zeros((4, 4)), 0.5)


def test_chunking_does_not_change_the_answer():
    """The slab loop is a memory strategy, so it must be invisible in the mesh.

    Slabs meet at cell boundaries where vertices are shared, and a welding key
    that was local to a slab would silently tear the surface there.
    """
    field, step = _sphere_field(40)
    whole = marching_tets(field, -1.0, origin=(-2.0,) * 3, spacing=step, chunk=1000)
    slabbed = marching_tets(field, -1.0, origin=(-2.0,) * 3, spacing=step, chunk=3)
    assert whole.vertices.shape == slabbed.vertices.shape
    assert enclosed_volume(whole) == pytest.approx(enclosed_volume(slabbed), rel=1e-12)
    assert set(np.unique(edge_use_counts(slabbed)).tolist()) == {2}
