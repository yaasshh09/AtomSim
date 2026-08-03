"""Hartree-Fock reaching the picture views.

The load-bearing test here is the first one. Every other check in this file
would pass on a build that silently drew the aufbau configuration under any
label the user picked, which is exactly the failure this phase exists to
prevent, so the configuration is asserted to reach the orbital before anything
else is asserted about the orbital.
"""

import numpy as np
import pytest

from atomsim.atoms import aufbau_configuration, parse_config
from atomsim.hf_atom import evaluate_hf_state, hf_radial
from atomsim.provenance import Fidelity


def test_explicit_configuration_reaches_the_orbital():
    """A non-aufbau configuration must change the orbital it produces.

    Sodium's 3s sits outside a closed neon core. Promote it to 3p and the 3s is
    gone; ask instead for the 2p, which BOTH configurations occupy, and the
    orbital still has to differ, because the Fock operator for the 2p is built
    from the other occupied orbitals and one of them moved.
    """
    ground = aufbau_configuration(11)
    excited = parse_config("1s2 2s2 2p6 3p1")

    r_ground, _ = hf_radial(11, 11, 2, 1, points=200, config=ground)
    r_excited, _ = hf_radial(11, 11, 2, 1, points=200, config=excited)

    assert not np.allclose(r_ground.values, r_excited.values, atol=1e-9)


def test_exchange_off_reaches_the_orbital_and_the_badge():
    """The Hartree 2p is a different curve, and says so in its own tier."""
    config = aufbau_configuration(10)
    r_hf, _ = hf_radial(10, 10, 2, 1, points=200, config=config)
    r_hartree, _ = hf_radial(
        10, 10, 2, 1, points=200, config=config, exchange=False
    )

    assert not np.allclose(r_hf.values, r_hartree.values, atol=1e-9)
    assert r_hf.provenance.fidelity is Fidelity.APPROXIMATION
    assert r_hartree.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_evaluate_hf_state_inherits_the_counterfactual_tier():
    """The 3-D evaluator must not staple APPROXIMATION onto a Hartree orbital.

    It used to: the tier was a literal in the Provenance constructor rather
    than something read off the solve, so every picture came back
    APPROXIMATION whatever the flags said.
    """
    pos = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 0.5]])
    real = evaluate_hf_state(10, 10, 2, 1, 0, pos)
    hartree = evaluate_hf_state(10, 10, 2, 1, 0, pos, exchange=False)

    assert real.provenance.fidelity is Fidelity.APPROXIMATION
    assert hartree.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_pauli_off_refuses_every_subshell_but_the_one_that_exists():
    """With the cap lifted the configuration is 1s^N and nothing else is there.

    The refusal has to name the reason. A bare "not occupied" would read as a
    contingent fact about this atom rather than as the consequence of the
    switch the caller just flipped.
    """
    collapsed = aufbau_configuration(10, pauli=False)
    with pytest.raises(ValueError, match="occupancy cap"):
        hf_radial(
            10, 10, 2, 1, points=200,
            config=collapsed, exchange=False, pauli=False,
        )
    # And the one that does exist still comes back.
    r, _ = hf_radial(
        10, 10, 1, 0, points=200, config=collapsed, exchange=False, pauli=False
    )
    assert r.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_orbital_carries_the_not_an_observable_claim():
    """Every Hartree-Fock picture routes through hf_radial, so the claim does."""
    r, p = hf_radial(10, 10, 2, 1, points=200)
    joined = " ".join(r.provenance.assumptions)
    assert "not an observable" in joined
    assert "spherical" in joined
    assert joined == " ".join(p.provenance.assumptions)


def test_hf_sampling_reduces_to_hydrogen():
    """At Z=1, N=1 the Fock operator IS the bare Coulomb Hamiltonian.

    There is no other electron, so no direct term, no exchange term, and
    nothing for self-consistency to do. The sampler therefore has to reproduce
    the closed-form 1s radial CDF, 1 - e^(-2r)(1 + 2r + 2r^2), and a KS test is
    the check the analytic sampler already gets held to.

    A ground truth this tier rarely has, which is why it is spent here.
    """
    from scipy import stats

    from atomsim.sampling import sample_hf_density

    cloud = sample_hf_density(1, 1, 1, 0, 0, 20_000, seed=7)
    r = np.linalg.norm(cloud.positions.astype(np.float64), axis=1)

    def cdf(x):
        return 1.0 - np.exp(-2.0 * x) * (1.0 + 2.0 * x + 2.0 * x * x)

    assert stats.kstest(r, cdf).pvalue > 0.01


def test_hf_cloud_carries_the_solve_and_the_claim():
    from atomsim.sampling import sample_hf_density

    cloud = sample_hf_density(10, 10, 2, 1, 0, 2_000, seed=1)
    joined = " ".join(cloud.provenance.assumptions)
    assert cloud.provenance.fidelity is Fidelity.APPROXIMATION
    assert "not an observable" in joined
    assert "correlation" in joined  # the solve's own disclosure survived


def test_hf_cloud_goes_counterfactual_with_exchange_off():
    from atomsim.sampling import sample_hf_density

    cloud = sample_hf_density(10, 10, 2, 1, 0, 2_000, seed=1, exchange=False)
    assert cloud.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_hf_plane_agrees_with_the_evaluator_it_is_built_on():
    """The grid is not allowed to be its own authority.

    A plane routine can be wrong in two ways that look identical on screen: a
    transposed axis pair, and an off-by-one in the half-extent. Both survive
    every self-consistent check the grid can run on itself, and both die
    against psi evaluated directly at the same Cartesian points.
    """
    from atomsim.plane import hf_plane_grid

    pg = hf_plane_grid(10, 10, 2, 1, 0, quantity="psi", resolution=33)
    axis = pg.axis
    # Row i is z = axis[i], column j is x = axis[j]; see the layout string the
    # server publishes for this array.
    for i in (3, 16, 29):
        for j in (5, 16, 27):
            direct = evaluate_hf_state(
                10, 10, 2, 1, 0,
                np.array([[axis[j], 0.0, axis[i]]]),
            )
            assert pg.values[i, j] == pytest.approx(
                float(np.real(direct.values[0])), rel=1e-9, abs=1e-12
            )


def test_hf_psi_is_real_on_the_y_zero_plane():
    """e^(i m phi) = +/-1 there, so a signed plot is honest and is labeled so."""
    from atomsim.plane import hf_plane_grid

    pg = hf_plane_grid(10, 10, 2, 1, 1, quantity="psi", resolution=33)
    pos = np.array([[0.7, 0.0, 0.9], [-1.3, 0.0, 0.4]])
    psi = evaluate_hf_state(10, 10, 2, 1, 1, pos).values
    assert np.max(np.abs(np.imag(psi))) < 1e-12
    assert "psi is real on y=0" in " ".join(pg.provenance.assumptions)


def test_hf_plane_inherits_the_counterfactual_tier():
    from atomsim.plane import hf_plane_grid

    real = hf_plane_grid(10, 10, 2, 1, 0, resolution=17)
    hartree = hf_plane_grid(10, 10, 2, 1, 0, resolution=17, exchange=False)
    assert real.provenance.fidelity is Fidelity.APPROXIMATION
    assert hartree.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert not np.allclose(real.values, hartree.values)


def test_hf_isosurface_reduces_to_the_closed_form_hydrogen_radius():
    """2.6612 bohr for the 1s at 90%, which Phase 25 validated closed-form.

    Z=1, N=1 makes the Fock operator the bare Coulomb Hamiltonian, so the
    numerical orbital has an exact answer to be held to and the surface built
    on it inherits that.
    """
    from atomsim.isosurface import hf_isosurface

    surf = hf_isosurface(1, 1, 1, 0, 0, target_fraction=0.9, resolution=96)
    radii = np.linalg.norm(surf.vertices, axis=1)
    assert radii.mean() == pytest.approx(2.6612, rel=5e-3)


def test_helium_hartree_and_hartree_fock_surfaces_are_bit_identical():
    """Helium's exchange energy is exactly zero, so the orbital is too.

    Exchange couples same-spin pairs only, and 1s2 holds one spin up and one
    spin down, so exchange_operator builds no terms at all. Phase 22
    established this on the energy; the surface has to inherit it to the bit,
    not merely to a tolerance, because a tolerance would hide a small real
    difference that would mean the toggle was reaching something it should not.
    """
    from atomsim.isosurface import hf_isosurface

    with_x = hf_isosurface(2, 2, 1, 0, 0, resolution=64)
    without = hf_isosurface(2, 2, 1, 0, 0, resolution=64, exchange=False)
    assert np.array_equal(with_x.vertices, without.vertices)
    # And the badge still flips, because the model the caller asked for is a
    # different model even where its answer coincides.
    assert with_x.provenance.fidelity is Fidelity.APPROXIMATION
    assert without.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_multi_shell_atom_surfaces_differ_with_exchange_off():
    """Neon has same-spin pairs, so removing exchange has to move the 2p."""
    from atomsim.isosurface import hf_isosurface

    with_x = hf_isosurface(10, 10, 2, 1, 0, resolution=64)
    without = hf_isosurface(10, 10, 2, 1, 0, resolution=64, exchange=False)
    assert not np.array_equal(with_x.vertices, without.vertices)


def _orbital_mean_radius(z, n_electrons, n, l, config, exchange, pauli):
    """<r> for ONE subshell, by quadrature on hf_radial's display grid.

    Deliberately not hf_mean_radius, which is a whole-atom average over every
    occupied subshell. See the test below for why the distinction matters.
    """
    _, p = hf_radial(
        z, n_electrons, n, l, points=4000,
        config=config, exchange=exchange, pauli=pauli,
    )
    return float(
        np.trapezoid(p.grid * p.values, p.grid) / np.trapezoid(p.values, p.grid)
    )


def test_pauli_collapse_orders_two_independent_measures_the_same_way():
    """Cross-checked rather than asserted.

    No direction is claimed for the collapsed 1s radius here, because a
    direction asserted without a derivation is a guess that a passing test then
    protects. Instead two separately computed sizes must order the collapsed
    and the real atom the same way: the 90% enclosure radius off the
    triangulated surface, and a 1-D quadrature on the display grid. Marching
    tetrahedra through an interpolated 3-D grid and a trapezoid rule over
    r^2 R^2 share nothing but the solve, so agreeing on the sign is a real
    check on whichever direction it turns out to be.

    Both measures must be about the SAME orbital, which is the trap this test
    fell into when it was first written. hf_mean_radius is a whole-atom
    average over every occupied subshell, and for beryllium the two questions
    have opposite answers: the atom shrinks under the collapse (1.53 -> 0.53
    bohr, because the 2s is gone) while its 1s swells (0.42 -> 0.53 bohr,
    because four electrons in one orbital repel each other harder than two do,
    q - 1 being 3 rather than 1). Neither number is wrong and the surface draws
    the 1s, so the 1s is what it gets compared against.
    """
    from atomsim.isosurface import hf_isosurface

    real_cfg = aufbau_configuration(4)
    collapsed_cfg = aufbau_configuration(4, pauli=False)

    real_surf = hf_isosurface(4, 4, 1, 0, 0, resolution=64, config=real_cfg)
    collapsed_surf = hf_isosurface(
        4, 4, 1, 0, 0, resolution=64,
        config=collapsed_cfg, exchange=False, pauli=False,
    )
    surface_sign = np.sign(
        np.linalg.norm(collapsed_surf.vertices, axis=1).mean()
        - np.linalg.norm(real_surf.vertices, axis=1).mean()
    )

    quadrature_sign = np.sign(
        _orbital_mean_radius(4, 4, 1, 0, collapsed_cfg, False, False)
        - _orbital_mean_radius(4, 4, 1, 0, real_cfg, True, True)
    )

    assert surface_sign != 0
    assert surface_sign == quadrature_sign


def test_the_collapsed_atom_shrinks_while_its_1s_swells():
    """Both are true, and a reader who conflates them will call one a bug.

    Pinned because the cross-check above was written wrong on the first pass
    for exactly this reason. The whole-atom radius falls because the 2s stops
    existing; the 1s radius rises because it now holds every electron and each
    one sees three others instead of one.
    """
    from atomsim.hf_atom import hf_mean_radius, solve_hartree_fock

    real_cfg = aufbau_configuration(4)
    collapsed_cfg = aufbau_configuration(4, pauli=False)

    atom_real = hf_mean_radius(solve_hartree_fock(4, 4, real_cfg)).value
    atom_collapsed = hf_mean_radius(
        solve_hartree_fock(4, 4, collapsed_cfg, False, False)
    ).value
    assert atom_collapsed < atom_real

    orbital_real = _orbital_mean_radius(4, 4, 1, 0, real_cfg, True, True)
    orbital_collapsed = _orbital_mean_radius(
        4, 4, 1, 0, collapsed_cfg, False, False
    )
    assert orbital_collapsed > orbital_real

    # The collapsed atom holds nothing but its 1s, so its two radii are the
    # same number reached two ways: a quadrature on the solver mesh and one on
    # the uniform display grid. They must agree to well inside either grid.
    assert atom_collapsed == pytest.approx(orbital_collapsed, rel=1e-3)
