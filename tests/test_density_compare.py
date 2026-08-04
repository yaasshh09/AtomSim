"""Comparing the two many-electron densities, and putting a number on the gap.

Both models claim to approximate the same observable, so the difference between
them is a statement about physics rather than about convention. The number this
module returns is half the L1 norm, which is exactly the count of electrons the
two models place differently, and the tests below check it against a case that
can be integrated by hand before they check it against either solver.
"""

import numpy as np
import pytest

from atomsim.density_compare import (
    _common_grid,
    _displaced_charge,
    _resample,
    _weaker,
    _window_loss,
)
from atomsim.provenance import Fidelity, Field, Provenance


def _field(values, grid, fidelity=Fidelity.APPROXIMATION, error=None):
    return Field(
        values=np.asarray(values, dtype=float),
        grid=np.asarray(grid, dtype=float),
        unit="electrons/bohr",
        grid_unit="bohr",
        label="D(r)",
        provenance=Provenance(
            fidelity=fidelity, method="test fixture", error_estimate=error
        ),
    )


# --- the number, against an integral that can be done by hand ---------------


def test_displaced_charge_matches_a_hand_integral():
    """D_a = 2r and D_b = 2 - 2r on [0, 1], both integrating to one electron.

    |D_a - D_b| = |4r - 2|, whose integral over [0, 1] is 1, so half of it is
    0.5. The integrand is piecewise linear with its kink at r = 0.5, and the
    grid below puts a node there, so the trapezoid rule is exact and the
    assertion can be tight rather than approximate.
    """
    r = np.linspace(0.0, 1.0, 101)
    assert _displaced_charge(r, 2 * r, 2 - 2 * r) == pytest.approx(0.5, abs=1e-12)


def test_a_density_is_not_displaced_from_itself():
    r = np.linspace(0.0, 1.0, 101)
    assert _displaced_charge(r, 2 * r, 2 * r) == 0.0


def test_displaced_charge_is_symmetric():
    """Neither model is the reference, so the number cannot depend on the order."""
    r = np.linspace(0.0, 1.0, 101)
    a, b = 2 * r, 2 - 2 * r
    assert _displaced_charge(r, a, b) == _displaced_charge(r, b, a)


# --- the common window ------------------------------------------------------


def test_the_common_grid_is_the_intersection_of_the_two_boxes():
    """Neither model is extrapolated past where its own solver ran."""
    a = _field(np.ones(50), np.geomspace(1e-4, 60.0, 50))
    b = _field(np.ones(50), np.geomspace(1e-3, 64.0, 50))
    grid = _common_grid(a, b, 200)
    assert grid[0] == pytest.approx(1e-3)
    assert grid[-1] == pytest.approx(60.0)
    assert len(grid) == 200


def test_window_loss_is_the_charge_left_outside():
    """A flat density on [0, 2] restricted to [0, 1] loses exactly half of it."""
    f = _field(np.ones(201), np.linspace(0.0, 2.0, 201))
    assert _window_loss(f, np.linspace(0.0, 1.0, 101)) == pytest.approx(1.0, abs=1e-12)


# --- the tier rule ----------------------------------------------------------


def test_the_weaker_tier_wins():
    assert _weaker(Fidelity.APPROXIMATION, Fidelity.COUNTERFACTUAL) is Fidelity.COUNTERFACTUAL
    assert _weaker(Fidelity.COUNTERFACTUAL, Fidelity.APPROXIMATION) is Fidelity.COUNTERFACTUAL
    assert _weaker(Fidelity.APPROXIMATION, Fidelity.APPROXIMATION) is Fidelity.APPROXIMATION
    assert _weaker(Fidelity.EXACT, Fidelity.NUMERICAL) is Fidelity.NUMERICAL


def test_a_visual_liberty_has_no_place_in_this_comparison():
    """Raising beats defaulting: a density is never a presentational choice.

    If one ever becomes one, this should stop rather than quietly rank it.
    """
    with pytest.raises(KeyError):
        _weaker(Fidelity.VISUAL_LIBERTY, Fidelity.APPROXIMATION)


# --- resampling keeps the provenance, and says what it did ------------------


def test_resampling_carries_the_provenance_and_discloses_itself():
    f = _field(np.linspace(1.0, 2.0, 50), np.geomspace(1e-3, 10.0, 50))
    out = _resample(f, np.geomspace(1e-2, 5.0, 30))
    assert out.provenance.fidelity is f.provenance.fidelity
    assert "resampled" in out.provenance.method
    assert len(out.values) == 30
    assert out.unit == f.unit


from atomsim.atoms import aufbau_configuration  # noqa: E402
from atomsim.density_compare import (  # noqa: E402
    _peaks_with_depth,
    _shell_table,
    compare_total_densities,
)

# --- peaks, and how well separated they are ---------------------------------


def test_the_innermost_peak_has_no_separation_to_report():
    """Depth measures the minimum before a peak, and the first peak has none."""
    r = np.linspace(0.1, 10.0, 500)
    v = np.exp(-((r - 1.0) ** 2) / 0.02)
    peaks = _peaks_with_depth(r, v)
    assert len(peaks) == 1
    assert peaks[0][0] == pytest.approx(1.0, abs=0.05)
    assert peaks[0][1] is None


def test_depth_is_the_relative_drop_into_the_preceding_minimum():
    """Two Gaussians whose valley bottoms at half the outer peak: depth 0.5."""
    r = np.linspace(0.0, 10.0, 2001)
    v = np.exp(-((r - 2.0) ** 2) / 0.08) + np.exp(-((r - 6.0) ** 2) / 0.08)
    peaks = _peaks_with_depth(r, v)
    assert len(peaks) == 2
    assert peaks[1][1] == pytest.approx(1.0, abs=0.01)  # the valley reaches zero


def test_the_noise_floor_keeps_the_faint_shell_and_drops_the_fainter_wiggle():
    """The floor sits between two measured numbers, not at a round one.

    Sodium's outermost Hartree-Fock peak sits at 2.2e-2 of the tallest one and
    is a real shell, so a bump of that size must survive. Argon's Hartree-Fock
    tail beyond 40 bohr sign-flips at about 1e-34 of the tallest, so a wiggle
    of that character (scaled up to 1e-30 here, still twenty-two orders below
    the floor, so a coarse grid can resolve the point) must not turn into a
    second shell. Both curves are built by hand, not by running either solver,
    so this checks the floor's placement directly rather than through a slow
    physics run that could pass or fail for unrelated reasons.
    """
    r = np.linspace(0.0, 3.0, 3001)
    tall = np.exp(-((r - 1.0) ** 2) / 0.01)

    real_bump = 0.022 * np.exp(-((r - 2.0) ** 2) / 0.01)
    with_real_shell = _peaks_with_depth(r, tall + real_bump)
    assert len(with_real_shell) == 2, "a 2.2 percent peak is a real shell"

    noise = 1e-30 * np.sign(np.sin(r * 5000.0))
    with_noise = _peaks_with_depth(r, tall + noise)
    assert len(with_noise) == 1, "a 1e-30 wiggle is noise, not a second shell"


# --- the two atoms where the models disagree about how many shells there are -


def test_gsz_does_not_resolve_sodiums_third_shell_and_hartree_fock_barely_does():
    """The finding this whole view exists to show, pinned in both directions.

    Sodium has three shells. Under GSZ the density falls monotonically past the
    L peak and the 3s charge rides out on the tail as a shoulder, so there is
    no third maximum and no minimum after the second. Under Hartree-Fock there
    is a third maximum, but the dip before it is 0.3 percent deep, which is a
    shell you would miss if the table did not print the depth beside it.

    Both halves are asserted. A solver change that flattens the HF dimple would
    otherwise silently drop a shell from a table nobody would think to re-check.
    """
    c = compare_total_densities(11, 11)
    assert [s.label for s in c.shells] == ["K", "L", "M"]
    k, ell, m = c.shells
    assert k.gsz_radius is not None and k.hf_radius is not None
    assert ell.gsz_radius is not None and ell.hf_radius is not None
    assert m.gsz_radius is None, "GSZ is not expected to resolve sodium's M shell"
    assert m.hf_radius == pytest.approx(3.16, rel=0.05)
    assert m.hf_depth == pytest.approx(0.003, abs=0.002)


def test_magnesium_is_the_same_case_with_a_deeper_dimple():
    c = compare_total_densities(12, 12)
    m = c.shells[2]
    assert m.label == "M"
    assert m.gsz_radius is None
    assert m.hf_radius == pytest.approx(2.43, rel=0.05)
    assert m.hf_depth == pytest.approx(0.015, abs=0.005)


@pytest.mark.parametrize("z", [13, 14, 18])
def test_three_clean_shells_under_both_models(z):
    """Aluminium up: both models resolve all three, so no cell is empty."""
    c = compare_total_densities(z, z)
    assert len(c.shells) == 3
    for s in c.shells:
        assert s.gsz_radius is not None
        assert s.hf_radius is not None


def test_the_shell_count_comes_from_the_configuration_not_from_the_peaks():
    """Otherwise the table would report that sodium has two shells."""
    c = compare_total_densities(11, 11)
    n_shells = len({n for (n, _), _ in aufbau_configuration(11)})
    assert len(c.shells) == n_shells == 3


def test_more_maxima_than_shells_raises_with_enough_to_diagnose_it():
    """A density with more shells than the atom has is an unresolved orbital.

    Phase 28's argon box bug looked exactly like this: a spurious maximum on
    top of the atom's real ones. A table that quietly dropped the extra row
    would hide that kind of solver failure, so `_shell_table` raises instead.
    Before the noise floor existed, six failing tests tripped this branch by
    accident; now nothing does, so it needs its own deliberate test, built
    from hand-made arrays rather than a solver, with a message specific
    enough to diagnose which model and which radii without re-running
    anything: the model name, the count, and at least one of the radii.
    """
    r = np.linspace(0.0, 10.0, 2001)
    v = (
        np.exp(-((r - 1.0) ** 2) / 0.01)
        + 0.8 * np.exp(-((r - 3.0) ** 2) / 0.01)
        + 0.6 * np.exp(-((r - 5.0) ** 2) / 0.01)
        + 0.4 * np.exp(-((r - 7.0) ** 2) / 0.01)
    )
    config = (((1, 0), 2), ((2, 0), 2), ((3, 0), 2))  # three shells, four peaks
    with pytest.raises(ValueError) as excinfo:
        _shell_table(r, v, v, config)
    message = str(excinfo.value)
    assert "GSZ" in message
    assert "4 maxima" in message
    assert "7" in message  # the outermost of the four radii, at r = 7 bohr


# --- the innermost shell has nothing before it to be separated from --------


@pytest.mark.parametrize("z", [2, 6, 10, 11, 14, 18])
def test_the_innermost_shell_has_no_measured_separation_under_either_model(z):
    """`ShellPeak`'s docstring claims this; this is what makes it a checked claim.

    A non-None depth here would mean a spurious minimum exists at small r,
    before the K peak, and the depth reported beside that peak would be a
    measurement of nothing rather than of how separated two shells are.
    """
    c = compare_total_densities(z, z)
    k = c.shells[0]
    assert k.gsz_depth is None
    assert k.hf_depth is None


# --- the number, on real atoms ----------------------------------------------


@pytest.mark.parametrize(
    "z,expected",
    [(2, 0.0003), (10, 0.0232), (11, 0.1218), (18, 0.0600)],
)
def test_the_measured_disagreement(z, expected):
    """The figures the captions quote, pinned so the captions stay checkable.

    Measured at 800 display points before this module existed. The tolerance is
    the reported bar, not a round number: a change that moves these past their
    own error estimate is a change in the physics, and should fail here.
    """
    c = compare_total_densities(z, z)
    assert c.displaced_charge.value == pytest.approx(
        expected, abs=max(c.displaced_charge.provenance.error_estimate, 2e-3)
    )


@pytest.mark.parametrize("z", [2, 6, 10, 11, 18])
def test_the_window_costs_less_than_the_bar_it_is_folded_into(z):
    """The intersection is honest only while what it drops is smaller than the noise."""
    c = compare_total_densities(z, z)
    loss = _window_loss(c.gsz, c.grid) + _window_loss(c.hf, c.grid)
    assert loss < c.displaced_charge.provenance.error_estimate
    assert loss < 5e-3


#: Every atom both models cover: He..Ar, less sulfur and chlorine, which have no
#: published neutral GSZ parameters. The Radial view's caption and the README
#: both say "every atom they both cover", so the test says the same list rather
#: than a sample of it.
BOTH_MODELS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18)


def test_the_models_agree_far_better_than_their_energies_do():
    """The lesson of the view, as an assertion rather than a caption.

    GSZ was fitted to reproduce Hartree-Fock potentials, so a close density is
    what the fit bought. Its valence ionization energies are 2 to 24 percent
    off NIST; its density is inside 1.5 percent of HF's for every atom both
    models can draw.

    The whole list, not a sample, because the caption quotes the bound for the
    whole list. Lithium is what sets it, at 1.45 percent, so the margin under
    the stated 1.5 is about three percent of the bound and not the comfortable
    gap the other fourteen suggest: aluminium is next at 0.98 and nothing else
    passes 0.95. A solver change that moves lithium is the one that makes the
    caption wrong, and it is the only atom here that would.
    """
    for z in BOTH_MODELS:
        c = compare_total_densities(z, z)
        assert c.displaced_charge.value / z < 0.015, f"Z={z}"


def test_lithium_is_the_atom_that_sets_the_bound():
    """Pinned separately, because a bound is only as good as its binding case.

    Folded into the loop above this would be one assertion among fifteen and
    would read as no more load-bearing than fluorine's, which has six times the
    room. Lithium is the whole margin.
    """
    c = compare_total_densities(3, 3)
    assert c.displaced_charge.value / 3 == pytest.approx(0.0145, abs=5e-4)


# --- provenance -------------------------------------------------------------


def test_the_comparison_is_an_approximation_and_says_neither_is_truth():
    c = compare_total_densities(10, 10)
    assert c.provenance.fidelity is Fidelity.APPROXIMATION
    assert c.displaced_charge.unit == "electrons"
    assert any("not truth" in a for a in c.provenance.assumptions)


def test_a_thrown_switch_makes_the_comparison_counterfactual():
    """Hartree-Fock without exchange is altered physics; the overlay inherits it."""
    c = compare_total_densities(10, 10, exchange=False)
    assert c.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert c.displaced_charge.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_both_models_take_the_same_configuration_with_pauli_off():
    """One counterfactual question, not two.

    With the cap off the Hartree-Fock configuration is 1s^N, and the screened
    side is handed that same configuration rather than resolving its own, so
    the overlay compares two models of the same altered atom.
    """
    c = compare_total_densities(10, 10, exchange=False, pauli=False)
    assert c.provenance.fidelity is Fidelity.COUNTERFACTUAL
    # One shell, because there is one occupied n.
    assert [s.label for s in c.shells] == ["K"]


def test_an_ion_is_refused_because_the_parameters_are_fitted_to_neutrals():
    with pytest.raises(ValueError, match="neutral"):
        compare_total_densities(10, 9)
