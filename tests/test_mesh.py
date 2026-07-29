import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal

from atomsim.analytic.hydrogen import energy
from atomsim.numerics.hartree_fock import local_hamiltonian_bands
from atomsim.numerics.mesh import (
    RadialMesh,
    exponential_mesh,
    mesh_for_atom,
    mesh_for_atom_at_step,
    uniform_mesh,
)


def solve(mesh: RadialMesh, z: float, l: int, n_states: int):
    """Lowest eigenvalues of -1/2 d2/dr2 + l(l+1)/2r^2 - Z/r on this mesh."""
    diag, offdiag = mesh.hamiltonian_bands(-z / mesh.r, l)
    return eigh_tridiagonal(
        diag, offdiag, select="i", select_range=(0, n_states - 1), eigvals_only=True
    )


def relative_error(mesh: RadialMesh, z: int, n: int, l: int) -> float:
    want = energy(n, Z=z).value
    got = solve(mesh, float(z), l, n - l)[n - l - 1]
    return abs(got - want) / abs(want)


def without_the_wall_correction(mesh: RadialMesh) -> RadialMesh:
    """The same mesh with a plain hard wall at r_min."""
    return RadialMesh(
        r=mesh.r, jacobian=mesh.jacobian, step=mesh.step,
        kinetic_diag=mesh.kinetic_diag, kinetic_offdiag=mesh.kinetic_offdiag,
        inner_wall_coupling=0.0, inner_ghost_ratio=0.0,
    )


class TestTheUniformMeshIsTheOldCodePath:
    """The assembly must reproduce the hand-written uniform stencil, not merely
    resemble it. That is what lets one mesh type replace two code paths without
    re-litigating every result the uniform grid produced."""

    def test_bands_reproduce_local_hamiltonian_bands(self):
        mesh = uniform_mesh(30.0, 2000)
        v = -3.0 / mesh.r
        want_diag, want_off = local_hamiltonian_bands(v, 2, mesh.r)
        got_diag, got_off = mesh.hamiltonian_bands(v, 2)
        assert got_diag == pytest.approx(want_diag, rel=1e-15)
        assert got_off == pytest.approx(want_off, rel=1e-15)

    def test_its_wall_correction_is_identically_zero(self):
        """A uniform mesh's wall sits on the origin, where P really is zero.
        The r^(l+1) correction must not perturb it at all."""
        mesh = uniform_mesh(30.0, 500)
        v = -1.0 / mesh.r
        assert mesh.hamiltonian_bands(v, 0)[0][0] == local_hamiltonian_bands(
            v, 0, mesh.r
        )[0][0]

    def test_integrate_is_the_trapezoid_rule_in_r(self):
        mesh = uniform_mesh(20.0, 500)
        f = np.exp(-mesh.r) * mesh.r**2
        assert mesh.integrate(f) == pytest.approx(np.trapezoid(f, mesh.r), rel=1e-15)


class TestTheExponentialMeshSolvesHydrogen:
    """Analytic hydrogen is EXACT, so it is ground truth for both meshes."""

    @pytest.mark.parametrize("z", [1, 18])
    @pytest.mark.parametrize("n,l", [(1, 0), (2, 0), (2, 1), (3, 2)])
    def test_eigenvalues_match_the_closed_form(self, z, n, l):
        """5e-5 rather than something rounder because that is what the mesh
        actually delivers on excited states. The conditioning floor is an
        ABSOLUTE error, near 1e-6 hartree at Z = 1 and 1e-3 at Z = 18, so a
        shallow level wears the same error as a deep one and shows a larger
        relative one. Measured worst case here: 4.5e-5, on argon's 3s."""
        assert relative_error(mesh_for_atom(z, 60.0, 1200), z, n, l) < 5e-5

    def test_it_beats_a_uniform_grid_of_the_same_size_by_orders(self):
        """The whole reason for the mesh. A uniform grid spends its points on
        vacuum; at Z = 18 that is the difference between a usable answer and an
        unusable one, on identical point counts."""
        curved = relative_error(mesh_for_atom(18, 40.0, 1200), 18, 1, 0)
        flat = relative_error(uniform_mesh(40.0, 1200), 18, 1, 0)
        assert curved < flat / 1000


class TestTheInnerWallIsWhereTheAccuracyGoes:
    def test_the_ghost_node_correction_buys_two_orders(self):
        """Without it a hard wall at r_min costs 2 r_min on hydrogen, linearly.
        Deleting the correction must make the answer measurably worse - that is
        what stops it being cargo cult."""
        mesh = exponential_mesh(1e-2, 60.0, 800)
        blunt = relative_error(without_the_wall_correction(mesh), 1, 1, 0)
        assert relative_error(mesh, 1, 1, 0) < blunt / 50

    @pytest.mark.parametrize("a", [1e-2, 1e-3])
    def test_a_hard_wall_error_is_linear_in_the_inner_radius(self, a):
        """2 pi a |psi(0)|^2 = 2a for hydrogen. Pinning the mechanism means the
        floor below is understood rather than merely observed."""
        blunt = without_the_wall_correction(exponential_mesh(a, 60.0, 800))
        assert relative_error(blunt, 1, 1, 0) == pytest.approx(2 * a / 0.5, rel=0.2)

    @pytest.mark.parametrize("z", [1, 18])
    def test_the_floor_is_reached_and_is_the_same_at_every_z(self, z):
        """Wall truncation falls as r_min^2 while conditioning noise rises as
        1/r_min^2, so the optimum is Z-independent and near 2.4e-6. This is the
        mesh's accuracy claim, made falsifiable."""
        assert relative_error(mesh_for_atom(z, 60.0, 1200), z, 1, 0) < 5e-6

    def test_refining_past_the_floor_does_not_help(self):
        """Stated so nobody later 'fixes' the mesh by adding points, and so the
        no-Richardson decision in hf_atom has something to point at."""
        coarse = relative_error(mesh_for_atom(1, 60.0, 1200), 1, 1, 0)
        fine = relative_error(mesh_for_atom(1, 60.0, 4800), 1, 1, 0)
        assert fine > coarse / 10


class TestSizingByStepInsteadOfPointCount:
    """Why this exists: inverting `mesh_for_atom` by hand needs r_min, and a
    caller that keeps its own copy of that constant gets a mesh at the wrong
    step, silently, the moment the two drift apart."""

    @pytest.mark.parametrize("z", [1, 2, 10, 18])
    @pytest.mark.parametrize("step", [0.02, 0.01, 0.005])
    def test_it_lands_within_one_point_of_the_requested_step(self, z, step):
        """The point count floors, so the delivered step is never finer than
        asked and overshoots by at most one point's worth. Pinned in that
        direction on purpose: a caller sizing a mesh by step should know which
        way the rounding goes rather than assume it is symmetric."""
        mesh = mesh_for_atom_at_step(z, 60.0, step)
        assert mesh.step >= step
        assert mesh.step == pytest.approx(step, rel=1.0 / (mesh.points - 1))

    @pytest.mark.parametrize("z", [1, 2, 10, 18])
    def test_it_agrees_with_mesh_for_atom_on_the_same_point_count(self, z):
        """The two constructors must place identical nodes, or the mesh a
        solve runs on would depend on which door it came through."""
        by_step = mesh_for_atom_at_step(z, 60.0, 0.01)
        by_count = mesh_for_atom(z, 60.0, by_step.points)
        assert np.array_equal(by_step.r, by_count.r)
        assert by_step.step == by_count.step

    def test_halving_the_step_keeps_both_endpoints_exactly(self):
        """The refinement pair hf_atom quotes an error from is only a
        statement about the step if nothing else moved between the two."""
        coarse = mesh_for_atom_at_step(18, 60.0, 0.01)
        fine = mesh_for_atom_at_step(18, 60.0, 0.005)
        assert fine.r[0] == coarse.r[0]
        assert fine.r[-1] == coarse.r[-1]
        assert fine.points > coarse.points

    def test_a_non_positive_step_is_rejected(self):
        with pytest.raises(ValueError, match="step must be positive"):
            mesh_for_atom_at_step(2, 60.0, 0.0)

    def test_a_box_inside_the_inner_radius_is_rejected(self):
        with pytest.raises(ValueError, match="must exceed inner radius"):
            mesh_for_atom_at_step(18, 1e-9, 0.01)


class TestTheMeshCarriesItsOwnQuadrature:
    def test_a_normalized_hydrogen_orbital_integrates_to_one(self):
        mesh = mesh_for_atom(1, 60.0, 2000)
        p = 2.0 * mesh.r * np.exp(-mesh.r)
        assert mesh.integrate(p**2) == pytest.approx(1.0, rel=1e-8)

    def test_cumulative_ends_at_the_total(self):
        mesh = mesh_for_atom(1, 40.0, 1500)
        f = mesh.r * np.exp(-mesh.r)
        assert mesh.cumulative(f)[-1] == pytest.approx(mesh.integrate(f), rel=1e-12)

    def test_the_transform_round_trips_an_orbital(self):
        mesh = mesh_for_atom(1, 60.0, 2000)
        p = 2.0 * mesh.r * np.exp(-mesh.r)
        assert mesh.to_p(mesh.to_s(p)) == pytest.approx(p, rel=1e-9)

    def test_to_s_makes_the_euclidean_norm_the_physical_one(self):
        """The point of the sqrt(J) substitution: LOBPCG works in the plain dot
        product, and this is what makes that the L2(dr) product."""
        mesh = mesh_for_atom(1, 60.0, 2000)
        s = mesh.to_s(2.0 * mesh.r * np.exp(-mesh.r))
        assert float(s @ s) == pytest.approx(1.0, rel=1e-6)


class TestItRefusesMeshesItCannotDiscretize:
    def test_a_decreasing_mesh_is_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            RadialMesh(
                r=np.array([3.0, 2.0, 1.0]), jacobian=np.ones(3), step=1.0,
                kinetic_diag=np.ones(3), kinetic_offdiag=np.ones(2),
                inner_wall_coupling=0.0, inner_ghost_ratio=0.0,
            )

    def test_a_mesh_reaching_the_origin_is_rejected(self):
        with pytest.raises(ValueError, match="strictly above zero"):
            exponential_mesh(0.0, 40.0, 100)

    def test_the_kinetic_bands_must_match_the_node_count(self):
        with pytest.raises(ValueError, match="one entry per interior interval"):
            RadialMesh(
                r=np.array([1.0, 2.0, 3.0]), jacobian=np.ones(3), step=1.0,
                kinetic_diag=np.ones(3), kinetic_offdiag=np.ones(3),
                inner_wall_coupling=0.0, inner_ghost_ratio=0.0,
            )

    def test_a_ghost_node_above_the_first_point_is_rejected(self):
        with pytest.raises(ValueError, match=r"below r\[0\]"):
            RadialMesh(
                r=np.array([1.0, 2.0, 3.0]), jacobian=np.ones(3), step=1.0,
                kinetic_diag=np.ones(3), kinetic_offdiag=np.ones(2),
                inner_wall_coupling=0.0, inner_ghost_ratio=1.5,
            )

    def test_a_potential_off_the_mesh_is_rejected(self):
        mesh = uniform_mesh(10.0, 100)
        with pytest.raises(ValueError, match="sampled on this mesh"):
            mesh.hamiltonian_bands(np.ones(50), 0)
