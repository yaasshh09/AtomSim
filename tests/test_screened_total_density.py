"""The total radial density under the screened model.

Phase 27 built D(r) = sum_a q_a P_a(r)^2 for Hartree-Fock and stopped there,
noting that the screened orbitals are normalized the same way so the rest was
a few lines. The few lines turned out to be the easy part: the box the screened
solver has always used for a single orbital is far too generous for a whole
atom, and on it argon's density loses 0.13 of an electron and grows a second K
peak that is not a shell. So the checks below are mostly about resolution, and
the two that matter are the electron count and the number of shells.

Neither is a smoke test. An unresolved core orbital is smooth, plausible and
wrong, and the only two things that catch it are integrating the curve and
counting its peaks.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from atomsim.atoms import aufbau_configuration
from atomsim.provenance import Fidelity
from atomsim.screened_atom import screened_total_radial_density
from atomsim.server.app import create_app


def _peaks(field, floor=0.01):
    """Interior maxima above `floor` of the tallest, i.e. the shells."""
    v = field.values
    big = v > floor * v.max()
    return [
        field.grid[i]
        for i in range(1, len(v) - 1)
        if big[i] and v[i] > v[i - 1] and v[i] > v[i + 1]
    ]


# --- the count of electrons, which is what makes it a density ---------------


@pytest.mark.parametrize("z", [2, 3, 6, 10, 18])
def test_the_curve_integrates_to_the_electron_count(z):
    d = screened_total_radial_density(z, z)
    assert np.trapezoid(d.values, d.grid) == pytest.approx(z, abs=5e-3)


def test_the_residual_is_reported_as_the_error_bar():
    """In electrons, which is the unit of the thing it is an error in.

    Every u_a is normalized so integral u^2 dr = 1 on the solver's own mesh,
    which makes integral D dr = N exact there and the shortfall after
    resampling a real error with the right dimension. This is the one shape in
    the screened model entitled to an error estimate, for the same reason the
    Hartree-Fock one is: see hf_atom.hf_total_radial_density.
    """
    d = screened_total_radial_density(18, 18)
    assert d.unit == "electrons/bohr"
    assert d.provenance.error_estimate is not None
    assert d.provenance.error_estimate < 5e-3
    assert d.provenance.error_estimate == pytest.approx(
        abs(np.trapezoid(d.values, d.grid) - 18.0), rel=1e-9
    )


def test_refining_the_display_grid_does_not_make_it_worse():
    """It converges to the solver mesh's own bias rather than to zero.

    Linear interpolation of u followed by squaring sits slightly below the true
    u^2 between solver nodes, so the residual settles at a floor set by the
    solve, not by the display grid. That floor is small and it is reported;
    what would not be acceptable is a residual that grows, which is what a
    display grid too coarse for the core looks like.
    """
    errs = [
        abs(np.trapezoid(d.values, d.grid) - 18.0)
        for d in (screened_total_radial_density(18, 18, points=p)
                  for p in (400, 800, 1600))
    ]
    assert max(errs) < 5e-3
    assert errs[2] <= errs[0] * 1.5


# --- the shells, which is what the curve is for -----------------------------


@pytest.mark.parametrize("z,shells", [(2, 1), (10, 2), (18, 3)])
def test_it_has_one_peak_per_shell(z, shells):
    """Helium K; neon K and L; argon K, L and M. The periodic table, drawn.

    This is the check that catches an unresolved core. On the box the screened
    solver uses for one orbital, argon's 1s is covered by about four points and
    the resampled density splits it into two maxima at 0.054 and 0.066 bohr, a
    fourth shell that no atom has.
    """
    assert len(_peaks(screened_total_radial_density(z, z))) == shells


def test_the_k_shell_integrates_to_more_than_two_electrons():
    """The caption in RadialView says "near 2.2, not 2, under either model".

    A number written into the interface has to be checkable, and this one is
    also the honest part of the claim above it: cutting a density at its minima
    does not partition the electrons, because the shells overlap and the dip
    between them is not a wall. Both models are checked, because the caption
    says both.
    """
    from atomsim.hf_atom import hf_total_radial_density

    for d in (screened_total_radial_density(18, 18, points=4000),
              hf_total_radial_density(18, 18, points=4000)):
        v, g = d.values, d.grid
        first_min = next(
            i for i in range(1, len(v) - 1)
            if v[i] < v[i - 1] and v[i] < v[i + 1] and v[i] > 1e-3 * v.max()
        )
        k_shell = np.trapezoid(v[: first_min + 1], g[: first_min + 1])
        assert k_shell == pytest.approx(2.2, abs=0.05)


def test_the_shells_land_where_the_shells_are():
    """Argon's K, L and M, in order and an order of magnitude apart."""
    k, l_, m = _peaks(screened_total_radial_density(18, 18))
    assert 0.03 < k < 0.09
    assert 0.2 < l_ < 0.45
    assert 0.9 < m < 1.8


# --- and it says what it is -------------------------------------------------


def test_it_is_labelled_as_the_screened_model_and_not_as_hartree_fock():
    d = screened_total_radial_density(18, 18)
    assert d.provenance.fidelity is Fidelity.APPROXIMATION
    assert "N = 18" in d.label
    joined = " ".join(d.provenance.assumptions).lower()
    # The model was fitted to a potential, not to a density. Saying so is the
    # difference between this curve and the Hartree-Fock one it sits beside.
    assert "fitted" in joined or "not fitted" in joined


def test_a_non_ground_configuration_is_honoured():
    """The density follows the configuration it is given, not the Aufbau one."""
    assert aufbau_configuration(10) == (((1, 0), 2), ((2, 0), 2), ((2, 1), 6))
    excited = (((1, 0), 2), ((2, 0), 2), ((2, 1), 5), ((3, 0), 1))
    d = screened_total_radial_density(10, 10, config=excited)
    assert np.trapezoid(d.values, d.grid) == pytest.approx(10.0, abs=5e-3)


def test_sulfur_is_refused_here_too():
    """No parameters, no potential, no density. See tests/test_sulfur_chlorine."""
    with pytest.raises(ValueError, match="no sourced GSZ parameters"):
        screened_total_radial_density(16, 16)


# --- and it reaches the browser --------------------------------------------


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def test_the_radial_endpoint_ships_it_under_the_screened_model(client):
    d = client.get("/api/radial/3/1?system=ar").json()["total_density"]
    assert np.trapezoid(d["values"], d["grid"]) == pytest.approx(18.0, abs=5e-3)
    assert d["unit"] == "electrons/bohr"


def test_it_follows_the_configuration_the_request_names(client):
    """The orbital on this branch cannot depend on the configuration; this can.

    In a central field the shape of a 3p is fixed by (Z, N) alone, so the two
    plots above the density are the same whatever configuration is asked for.
    The density is the first thing on the screened radial branch that has to
    read it, which is exactly the kind of parameter that gets dropped.
    """
    excited = "1s2 2s2 2p6 3s2 3p5 4s1"
    a = client.get("/api/radial/3/1?system=ar").json()["total_density"]
    b = client.get(f"/api/radial/3/1?system=ar&config={excited}").json()["total_density"]
    assert np.trapezoid(b["values"], b["grid"]) == pytest.approx(18.0, abs=5e-3)
    # Moving one electron from 3p to 4s moves it outward, so the tail grows.
    assert sum(b["values"][-40:]) > sum(a["values"][-40:])


def test_a_one_electron_system_still_gets_none(client):
    """Unchanged by this: hydrogen's density IS the orbital already plotted."""
    assert client.get("/api/radial/3/1?system=h").json()["total_density"] is None
