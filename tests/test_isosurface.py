"""Phase 25: the enclosed-probability isosurface.

The failure this file exists to catch is that a wrong surface still looks like an
orbital. A bug in the level solve, the box, or the interpolation produces a
smooth closed shape that reads as a p orbital to any eye and to most smoke tests.
So every assertion here is something a plausible-looking wrong answer fails:
a closed form nobody in this repo chose, a cross-check against the KS-validated
sampler from Phase 1, a mesh volume against an independently counted one, and
convergence under refinement.

See docs/specs/2026-08-03-phase25-isosurfaces-design.md.
"""

import numpy as np
import pytest

from atomsim.isosurface import (
    GRID_SIZES,
    default_half_width,
    fraction_above,
    hf_isosurface,
    isosurface,
    screened_isosurface,
    solve_level,
)
from atomsim.numerics.marching_tets import edge_use_counts
from atomsim.provenance import Fidelity
from atomsim.sampling import sample_density


def hydrogen_1s_enclosed(radius: float) -> float:
    """P(r < a) for the hydrogen ground state, in bohr.

    The closed form: 1 - e^(-2a)(1 + 2a + 2a^2). Written here rather than
    imported so the test does not check the engine against itself.
    """
    return 1.0 - np.exp(-2 * radius) * (1 + 2 * radius + 2 * radius**2)


def radius_enclosing(fraction: float) -> float:
    """Invert the above by bisection. Independent of everything under test."""
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if hydrogen_1s_enclosed(mid) < fraction:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# The level solve, on arrays whose answer is arithmetic
# --------------------------------------------------------------------------


def test_solve_level_on_a_uniform_box_splits_it_by_count():
    """Every cell equal, so any level below the common value takes everything.

    The degenerate case is worth pinning because the interpolation divides by
    the gap between neighbouring densities, which is zero here.
    """
    grid = np.full((4, 4, 4), 0.5)
    cell = 1.0 / (0.5 * 64)  # normalize so the box holds exactly 1
    assert fraction_above(grid, cell, solve_level(grid, cell, 0.5)) == pytest.approx(1.0)


def test_solve_level_refuses_a_target_the_box_cannot_hold():
    """The disclosure this protects is the whole point of measuring the box.

    Renormalizing to what is inside would answer "90%" for a box holding half
    the electron, which is a lie with a number attached.
    """
    grid = np.full((4, 4, 4), 0.1)
    with pytest.raises(ValueError, match="cannot enclose"):
        solve_level(grid, 0.001, 0.9)


@pytest.mark.parametrize("target", [-0.1, 0.0, 1.0, 2.0])
def test_solve_level_rejects_targets_outside_the_open_unit_interval(target):
    with pytest.raises(ValueError, match="must be in"):
        solve_level(np.ones((3, 3, 3)), 1.0, target)


# --------------------------------------------------------------------------
# The closed form
# --------------------------------------------------------------------------


def test_hydrogen_1s_at_ninety_percent_is_the_textbook_sphere():
    """2.6612 bohr, from a formula, checked three ways on one surface.

    1s has no angular dependence, so its contour is a sphere of known radius.
    Vertex radii, enclosed volume, and area all have to agree with it, and they
    are three different functions of the same mesh: a level that is off moves
    all three, and a triangulation bug moves the last two only.
    """
    expected_r = radius_enclosing(0.9)
    assert expected_r == pytest.approx(2.6612, abs=1e-3)

    surface = isosurface(1, 0, 0, target_fraction=0.9, resolution=96)
    radii = np.linalg.norm(surface.vertices, axis=1)

    assert radii.mean() == pytest.approx(expected_r, rel=2e-3)
    assert radii.std() < 0.02
    assert surface.mesh_volume.value == pytest.approx(4 / 3 * np.pi * expected_r**3, rel=0.01)
    assert surface.area.value == pytest.approx(4 * np.pi * expected_r**2, rel=0.01)


@pytest.mark.parametrize("fraction", [0.5, 0.95])
def test_the_radius_tracks_the_requested_fraction(fraction):
    """Not just one lucky contour: the map from fraction to radius is the claim.

    Looser than the 90% case above, and the reason is the cusp. A tight contour
    sits close in, where |psi|^2 has its kink at the nucleus and the cell sum
    converges first order rather than second: the 50% radius is off by 2.7% at
    48^3 and 0.07% at 128^3. Refinement is asserted separately below; here the
    tolerance is set to what 64^3 honestly delivers.
    """
    surface = isosurface(1, 0, 0, target_fraction=fraction, resolution=64)
    radii = np.linalg.norm(surface.vertices, axis=1)
    assert radii.mean() == pytest.approx(radius_enclosing(fraction), rel=1e-2)


def test_a_tighter_contour_sits_inside_a_looser_one():
    """Monotonicity. Cheap, and it catches a sign or an inverted sort."""
    inner = isosurface(1, 0, 0, target_fraction=0.5, resolution=48)
    outer = isosurface(1, 0, 0, target_fraction=0.95, resolution=48)
    assert inner.level.value > outer.level.value
    assert inner.mesh_volume.value < outer.mesh_volume.value


# --------------------------------------------------------------------------
# The independent cross-check: the Phase 1 sampler
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n,l,m", [(1, 0, 0), (2, 1, 0), (3, 2, 1)])
def test_the_sampler_lands_inside_the_surface_at_the_stated_rate(n, l, m):
    """Points drawn from |psi|^2 fall inside the 90% contour 90% of the time.

    This is the check that ties the surface to machinery validated by a
    different method entirely (KS tests against the analytic CDFs), and it goes
    through the level rather than the triangles: a point is inside exactly when
    its density is at or above the level, no ray casting needed.

    The tolerance is binomial: 40000 draws give a standard error of 0.0015 on a
    proportion of 0.9, so 0.01 is more than six sigma and still fails on any
    real bias in the level.
    """
    from atomsim.analytic.wavefunction import evaluate_state

    surface = isosurface(n, l, m, target_fraction=0.9, resolution=96)
    cloud = sample_density(n, l, m, 40_000, seed=7)
    density = np.abs(evaluate_state(n, l, m, cloud.positions).values) ** 2
    inside = float((density >= surface.level.value).mean())
    assert inside == pytest.approx(0.9, abs=0.01)


# --------------------------------------------------------------------------
# The box
# --------------------------------------------------------------------------


def test_the_box_grows_until_it_holds_essentially_everything():
    """A 4f is far more diffuse than a 1s, and neither may lose its tail."""
    for n, l in ((1, 0), (4, 3)):
        surface = isosurface(n, l, 0, target_fraction=0.9, resolution=48)
        assert surface.escaped_fraction.value < 2e-3
        assert surface.half_width > default_half_width(n) * 0.5


def test_a_box_too_small_for_the_target_is_refused_rather_than_renormalized():
    """Hand-set half_width, so the caller asked for it and gets told.

    1 bohr holds about 32% of a hydrogen 1s, so 90% is not available in there.
    """
    with pytest.raises(ValueError, match="widen the box"):
        isosurface(1, 0, 0, target_fraction=0.9, resolution=48, half_width=1.0)


def test_the_escaped_mass_is_reported_even_when_it_is_tiny():
    surface = isosurface(1, 0, 0, target_fraction=0.9, resolution=64)
    assert surface.escaped_fraction.value >= 0.0
    assert any("outside the" in a and "box" in a for a in surface.provenance.assumptions)


# --------------------------------------------------------------------------
# Mesh against grid, and convergence
# --------------------------------------------------------------------------


def test_the_mesh_volume_agrees_with_the_cells_it_was_cut_from():
    """Two different objects: an interpolated surface and a staircase of voxels.

    They must agree to within the discretization, and the gap is reported. If
    they agreed exactly the interpolation would not be doing anything.
    """
    surface = isosurface(2, 1, 0, target_fraction=0.9, resolution=96)
    assert surface.mesh_volume.value == pytest.approx(surface.voxel_volume.value, rel=0.05)
    assert surface.mesh_volume.value != surface.voxel_volume.value


def test_refinement_moves_the_answer_toward_the_closed_form():
    exact = 4 / 3 * np.pi * radius_enclosing(0.9) ** 3
    errors = [
        abs(isosurface(1, 0, 0, target_fraction=0.9, resolution=n).mesh_volume.value - exact)
        for n in (48, 96)
    ]
    assert errors[1] < errors[0]


def test_the_surface_is_watertight_for_a_lobed_orbital():
    """The topology tests below mean nothing on a mesh full of holes."""
    surface = isosurface(3, 2, 0, target_fraction=0.9, resolution=64)
    counts = edge_use_counts(
        type("M", (), {"vertices": surface.vertices, "triangles": surface.triangles})()
    )
    assert set(np.unique(counts).tolist()) == {2}


# --------------------------------------------------------------------------
# Topology: the textbook lobes are lobes
# --------------------------------------------------------------------------


def test_the_s_orbital_is_one_piece_and_the_p_orbital_is_two():
    """"Two lobes" is a claim, so it gets asserted rather than admired.

    2p_z in the real basis vanishes on the whole z = 0 plane, so in the
    continuum every contour of it is two pieces. On a grid it is two pieces only
    while the gap between the lobes is wider than a cell, and that gap closes as
    the contour loosens: the surface meets the z axis at 0.60 bohr for a 30%
    contour and at 0.11 bohr for a 90% one, because a looser contour is a lower
    level and a lower level is crossed nearer the node.

    So the assertion is the one that is actually true of this mesh, with the
    reason measured rather than assumed: two lobes when the gap clears a cell,
    fused when it does not. Separating a 90% p orbital needs about 0.1 bohr
    cells across a 30 bohr box, which is 300^3 and not on offer.
    """
    assert isosurface(1, 0, 0, target_fraction=0.9, resolution=64).components == 1

    split = isosurface(2, 1, 0, target_fraction=0.5, basis="real", resolution=64)
    gap = 2 * np.abs(split.vertices[:, 2]).min()
    assert gap > 2 * split.half_width / 63
    assert split.components == 2

    fused = isosurface(2, 1, 0, target_fraction=0.9, basis="real", resolution=64)
    assert 2 * np.abs(fused.vertices[:, 2]).min() < 2 * fused.half_width / 63
    assert fused.components == 1
    assert any("narrower than" in a for a in fused.provenance.assumptions)


def test_the_two_lobes_carry_opposite_signs_of_psi():
    """The surface is drawn through |psi|^2, which cannot tell them apart.

    The colouring comes from psi evaluated at the vertices, and if that step
    were dropped the picture would be a single-coloured dumbbell that teaches
    the wrong thing about bonding.
    """
    p_z = isosurface(2, 1, 0, target_fraction=0.9, basis="real", resolution=64)
    upper = p_z.vertex_phase[p_z.vertices[:, 2] > 0]
    lower = p_z.vertex_phase[p_z.vertices[:, 2] < 0]
    assert np.allclose(upper, 0.0)
    assert np.allclose(np.abs(lower), np.pi)


@pytest.mark.parametrize("m", [1, -1])
def test_a_complex_orbital_carries_a_real_phase_rather_than_a_sign(m):
    """m = +-1 in the complex basis winds through 2 pi around the z axis.

    What is asserted is `arg(psi) - m phi` being the same at every vertex, not
    it being zero: Y_1,1 carries the Condon-Shortley minus sign, so its phase is
    phi + pi while Y_1,-1's is -phi. The winding is the physics; the constant is
    a convention, and pinning the convention here would make the test fail if
    the engine ever adopted the other one without anything being wrong.
    """
    surface = isosurface(2, 1, m, target_fraction=0.9, basis="complex", resolution=64)
    phase = surface.vertex_phase
    assert phase.max() - phase.min() > 5.0
    azimuth = np.arctan2(surface.vertices[:, 1], surface.vertices[:, 0])
    residual = np.exp(1j * (phase - m * azimuth))
    assert np.abs(residual - residual[0]).max() < 1e-9


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_an_exact_wavefunction_still_gives_a_numerical_surface():
    """The grid is the approximation. EXACT here would be a lie about the mesh."""
    surface = isosurface(1, 0, 0, target_fraction=0.9, resolution=48)
    assert surface.provenance.fidelity is Fidelity.NUMERICAL
    assert surface.level.provenance.fidelity is Fidelity.NUMERICAL


def test_the_disclosure_states_the_complement_not_just_the_fraction():
    """"90% enclosed" is only half the sentence, and the other half is the lesson."""
    surface = isosurface(1, 0, 0, target_fraction=0.9, resolution=48)
    text = " ".join(surface.provenance.assumptions)
    assert "outside the surface" in text
    assert "no boundary" in text
    assert surface.outside_fraction == pytest.approx(1 - surface.enclosed_fraction.value)


def test_the_error_bar_is_a_grid_halving_on_the_fraction():
    surface = isosurface(1, 0, 0, target_fraction=0.9, resolution=64)
    assert surface.provenance.error_estimate is not None
    assert 0.0 <= surface.provenance.error_estimate < 0.02
    assert surface.enclosed_fraction.provenance.error_estimate is not None


def test_the_volume_gets_its_own_error_bar_because_the_fraction_one_is_blind():
    """Two claims, two convergence rates, and quoting only the fast one lies.

    The level barely moves under halving, so the fraction it encloses comes back
    right to about 1e-4 or better - for a 1s at 90% the two grids agree exactly
    and the fraction error is 0.0. Where the contour sits is a different matter:
    the same halving moves the enclosed volume by half a percent. An error bar
    of zero beside a surface that is half a percent off in size is precisely the
    quiet lie this project exists to prevent, so the volume carries its own.
    """
    surface = isosurface(1, 0, 0, target_fraction=0.9, resolution=96)
    fraction_error = surface.provenance.error_estimate
    volume_error = surface.mesh_volume.provenance.error_estimate

    assert fraction_error is not None and volume_error is not None
    assert fraction_error < 1e-3
    # The point of the test: the geometric error is orders larger, and real.
    assert volume_error / surface.mesh_volume.value > 1e-3
    text = " ".join(surface.provenance.assumptions)
    assert "enclosed volume by" in text
    assert "converges long before the surface does" in text


def test_the_volume_error_bar_shrinks_as_the_grid_refines():
    """It is an error estimate, so it has to behave like one."""
    errors = [
        isosurface(2, 1, 0, target_fraction=0.9, resolution=n, basis="real")
        for n in (48, 96)
    ]
    relative = [s.mesh_volume.provenance.error_estimate / s.mesh_volume.value for s in errors]
    assert relative[1] < relative[0]


def test_the_achieved_fraction_is_what_is_reported_not_the_request():
    """They are close, and the reported one is the measured one anyway."""
    surface = isosurface(2, 1, 0, target_fraction=0.9, resolution=64)
    assert surface.enclosed_fraction.value == pytest.approx(0.9, abs=5e-3)
    assert surface.target_fraction == 0.9


@pytest.mark.parametrize("resolution", [7, 1000])
def test_an_unsupported_grid_size_is_refused(resolution):
    with pytest.raises(ValueError, match="resolution must be one of"):
        isosurface(1, 0, 0, resolution=resolution)


def test_the_offered_grid_sizes_are_the_ones_that_work():
    for size in GRID_SIZES:
        assert isosurface(1, 0, 0, target_fraction=0.5, resolution=size).components == 1


# --------------------------------------------------------------------------
# Many-electron states
# --------------------------------------------------------------------------


def test_a_screened_orbital_inherits_the_weaker_tier():
    """The screening model is a bigger departure than the grid is."""
    surface = screened_isosurface(11, 11, 3, 0, 0, target_fraction=0.9, resolution=48)
    assert surface.provenance.fidelity is Fidelity.APPROXIMATION
    assert surface.enclosed_fraction.value == pytest.approx(0.9, abs=0.01)


def test_a_hartree_fock_orbital_gets_a_surface_too():
    surface = hf_isosurface(4, 4, 2, 0, 0, target_fraction=0.9, resolution=48)
    assert surface.provenance.fidelity is Fidelity.APPROXIMATION
    assert surface.enclosed_fraction.value == pytest.approx(0.9, abs=0.01)
    assert surface.components >= 1


def test_the_hartree_fock_valence_orbital_is_bigger_than_the_core_one():
    """A shape check with physics in it, on the model's own terms.

    Beryllium's 2s contour has to enclose the 1s one. If the radial function
    were being read off the wrong channel this would invert.
    """
    core = hf_isosurface(4, 4, 1, 0, 0, target_fraction=0.9, resolution=48)
    valence = hf_isosurface(4, 4, 2, 0, 0, target_fraction=0.9, resolution=48)
    assert valence.mesh_volume.value > core.mesh_volume.value
