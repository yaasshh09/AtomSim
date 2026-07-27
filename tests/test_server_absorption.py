"""/api/absorption: a whole line list eating a continuum, served honestly.

The engine's physics is validated in `test_absorption.py`. What is left for
the endpoint is that it hands the physics over intact: same-length arrays, a
saturation number travelling with the curve that says how much of the census
is being lost, provenance that survives the trip, and refusals where a
plausible-looking curve would be a lie.
"""

import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _absorb(client, **params):
    r = client.get("/api/absorption", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_the_arrays_describe_one_spectrum(client):
    body = _absorb(client, column_density_m2=1e20)
    n = len(body["wavelength_nm"])
    assert n > 100
    assert len(body["transmission"]) == n
    assert len(body["optical_depth"]) == n
    assert all(0.0 <= t <= 1.0 for t in body["transmission"])
    assert all(t >= 0.0 for t in body["optical_depth"])
    assert body["wavelength_nm"] == sorted(body["wavelength_nm"])


def test_every_line_arrives_with_its_own_strength_and_column(client):
    """Degenerate lines must not be collapsed on the way out either.

    Hydrogen to n=4 puts fourteen lines on six wavelengths, so a response
    keyed or de-duplicated by wavelength would drop eight of them.
    """
    body = _absorb(client, n_max=4, column_density_m2=1e20)
    assert len(body["lines"]) == 14
    balmer = [
        d for d in body["lines"] if abs(d["wavelength_nm"] - 656.4696) < 1e-3
    ]
    assert len(balmer) == 3
    assert len({d["label"] for d in balmer}) == 3
    assert len({round(d["oscillator_strength"], 6) for d in balmer}) == 3


def test_lyman_absorbs_where_balmer_does_not(client):
    """The one fact the emission endpoint could not represent.

    At 10,000 K essentially every neutral atom sits in n = 1, so Lyman-alpha
    absorbs out of a full level and Balmer-alpha out of a nearly empty one.
    One gas, one column, optical depths orders of magnitude apart.
    """
    body = _absorb(client, n_max=4, column_density_m2=1e20)
    lyman = min(
        body["lines"], key=lambda d: abs(d["wavelength_nm"] - 121.567)
    )
    balmer = max(
        (d for d in body["lines"] if abs(d["wavelength_nm"] - 656.4696) < 1e-3),
        key=lambda d: d["tau_centre"],
    )
    assert lyman["tau_centre"] > 1.0
    assert balmer["tau_centre"] < 1.0
    assert lyman["lower_column_m2"] > 1e3 * balmer["lower_column_m2"]


def test_saturation_travels_with_the_curve(client):
    """The number that stops a deeper line being read as more gas."""
    thin = _absorb(client, column_density_m2=1e15)
    thick = _absorb(client, column_density_m2=1e22)
    assert thin["saturation"] == pytest.approx(1.0, abs=0.02)
    assert thick["saturation"] < 0.5
    assert thick["equivalent_width_nm"] > thin["equivalent_width_nm"]
    assert thick["equivalent_width_nm"] < thick["thin_limit_width_nm"]


def test_a_black_core_is_disclosed_not_merely_present(client):
    body = _absorb(client, column_density_m2=1e22)
    assert any("black core" in a for a in body["provenance"]["assumptions"])
    assert any(d["regime"] != "linear" for d in body["lines"])


def test_the_column_says_it_is_a_knob(client):
    """A user-chosen column is counterfactual and must not pose as measured."""
    body = _absorb(client, column_density_m2=1e20)
    assert body["column_provenance"]["fidelity"] == "counterfactual"
    assert body["provenance"]["fidelity"] == "approximation"
    text = " ".join(body["provenance"]["assumptions"])
    assert "stimulated emission" in text
    assert "continuous opacity" in text


def test_the_grid_reports_its_own_quadrature_error(client):
    body = _absorb(client, column_density_m2=1e18)
    assert body["flux_closure"] == pytest.approx(1.0, abs=0.02)


def test_zero_column_is_a_transparent_slab_not_an_error(client):
    """Nothing in the way is a legitimate answer, and it is a flat line."""
    body = _absorb(client, column_density_m2=0.0)
    assert all(t == pytest.approx(1.0) for t in body["transmission"])
    assert body["equivalent_width_nm"] == pytest.approx(0.0, abs=1e-15)


def test_a_window_can_be_asked_for(client):
    body = _absorb(
        client, column_density_m2=1e20, lambda_min=100.0, lambda_max=130.0
    )
    assert body["wavelength_nm"][0] >= 50.0
    assert all(d["wavelength_nm"] <= 130.0 for d in body["lines"])


# --------------------------------------------------------------------------
# Refusals: the cases where a curve would be a lie.
# --------------------------------------------------------------------------

def test_the_gas_that_produced_the_spectrum_comes_back_with_it(client):
    """The conditions are not optional here the way they are for emission.

    Which level the atoms are in *is* what each line absorbs with, so the
    endpoint assumes a gas rather than refusing, and then says which one. A
    transmission curve with no conditions attached would leave the whole
    Lyman/Balmer contrast unexplained.
    """
    body = _absorb(client, temperature_k=12_000.0, electron_density_cm3=1e13)
    assert body["thermal"]["temperature_k"] == pytest.approx(12_000.0)
    assert body["thermal"]["electron_density_cm3"] == pytest.approx(1e13)
    assert 0.0 <= body["thermal"]["ionized_fraction"]["value"] <= 1.0


@pytest.mark.parametrize(
    "params",
    [
        {"temperature_k": 1.0},
        {"electron_density_cm3": 1.0},
    ],
)
def test_conditions_outside_the_displayable_range_are_refused(client, params):
    r = client.get("/api/absorption", params=params)
    assert r.status_code == 422


@pytest.mark.parametrize("column", [-1.0, 1e40])
def test_an_impossible_column_is_refused(client, column):
    r = client.get("/api/absorption", params={"column_density_m2": column})
    assert r.status_code == 422
    assert "column_density_m2" in r.text


def test_a_half_open_window_is_refused(client):
    r = client.get("/api/absorption", params={"lambda_min": 100.0})
    assert r.status_code == 422


def test_an_unphysical_resolving_power_is_refused(client):
    r = client.get("/api/absorption", params={"resolving_power": 5})
    assert r.status_code == 422
