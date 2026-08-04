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
