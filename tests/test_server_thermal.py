"""API contract for thermal conditions on /api/spectrum.

The physics is validated in test_populations.py and test_spectra_thermal.py.
These check the boundary: that the two knobs are honoured together or not at
all, that the conditions come back attached to the answer rather than having to
be remembered by the caller, and that the LTE assumptions survive the trip.
"""

import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app

WARM = "temperature_k=10000&electron_density_cm3=1e13"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _line(body, n_up, n_low):
    for ln in body["lines"]:
        if (ln["n_upper"], ln["n_lower"]) == (n_up, n_low):
            return ln
    raise AssertionError(f"no line {n_up} -> {n_low}")


def test_no_conditions_means_no_emissivity(client):
    body = client.get("/api/spectrum?system=h&n_max=4&intensities=true").json()
    assert body["thermal"] is None
    assert all(ln["emissivity"] is None for ln in body["lines"])


def test_conditions_produce_emissivity_on_every_line(client):
    body = client.get(f"/api/spectrum?system=h&n_max=6&{WARM}").json()
    assert body["lines"]
    assert all(ln["emissivity"] is not None for ln in body["lines"])
    assert all(ln["einstein_a_s"] is not None for ln in body["lines"]), (
        "thermal implies intensities"
    )


def test_the_conditions_come_back_with_the_answer(client):
    """The caller should not have to remember what it asked for to read what it
    got, and the ionized fraction is the number that says whether a dim
    spectrum is dim because it is cold or because it is gone.
    """
    body = client.get(f"/api/spectrum?system=h&n_max=6&{WARM}").json()
    t = body["thermal"]
    assert t["temperature_k"] == 10000.0
    assert t["electron_density_cm3"] == 1e13
    assert 0.0 <= t["ionized_fraction"]["value"] <= 1.0
    assert t["partition_function"]["value"] >= 2.0


def test_the_lte_assumptions_survive_the_boundary(client):
    body = client.get(f"/api/spectrum?system=h&n_max=4&{WARM}").json()
    prov = _line(body, 2, 1)["emissivity"]["provenance"]
    assert prov["fidelity"] == "approximation"
    assert any("LTE" in a for a in prov["assumptions"])
    assert any("optically thin" in a.lower() for a in prov["assumptions"])
    ion = body["thermal"]["ionized_fraction"]["provenance"]
    assert any("self-consist" in a.lower() for a in ion["assumptions"])


def test_the_partition_function_reports_its_cutoff(client):
    body = client.get(f"/api/spectrum?system=h&n_max=6&{WARM}").json()
    prov = body["thermal"]["partition_function"]["provenance"]
    assert any("truncat" in a.lower() for a in prov["assumptions"])
    assert any("n_max=6" in a for a in prov["assumptions"])


def test_one_knob_without_the_other_is_rejected(client):
    """Half of Saha is not a state anyone can read."""
    for query in ("temperature_k=10000", "electron_density_cm3=1e13"):
        r = client.get(f"/api/spectrum?system=h&{query}")
        assert r.status_code == 422
        assert "together" in r.json()["detail"]


@pytest.mark.parametrize(
    "query",
    [
        "temperature_k=1&electron_density_cm3=1e13",
        "temperature_k=1e9&electron_density_cm3=1e13",
        "temperature_k=10000&electron_density_cm3=1",
        "temperature_k=10000&electron_density_cm3=1e30",
    ],
)
def test_conditions_outside_the_display_range_are_rejected(client, query):
    assert client.get(f"/api/spectrum?system=h&{query}").status_code == 422


def test_conditions_do_not_disturb_wavelengths(client):
    plain = client.get("/api/spectrum?system=h&n_max=6").json()
    warm = client.get(f"/api/spectrum?system=h&n_max=6&{WARM}").json()
    assert [ln["wavelength_nm"]["value"] for ln in plain["lines"]] == [
        ln["wavelength_nm"]["value"] for ln in warm["lines"]
    ]


def test_heating_the_gas_changes_which_lines_are_bright(client):
    """The whole point of the control. If the ordering never moved, the
    temperature would be decoration.
    """
    def order(t):
        body = client.get(
            f"/api/spectrum?system=h&n_max=6&temperature_k={t}&electron_density_cm3=1e13"
        ).json()
        ranked = sorted(body["lines"], key=lambda ln: -ln["emissivity"]["value"])
        return [(ln["n_upper"], ln["n_lower"]) for ln in ranked]

    assert order(3000) != order(20000)


def test_a_screened_atom_gets_conditions_too(client):
    body = client.get(f"/api/spectrum?system=li&{WARM}").json()
    assert body["thermal"] is not None
    assert all(ln["emissivity"] is not None for ln in body["lines"])
    ion = body["thermal"]["ionized_fraction"]["provenance"]
    assert any("Koopmans" in a for a in ion["assumptions"])


def test_lithium_is_more_ionized_than_hydrogen_at_the_same_conditions(client):
    """Li gives up its valence electron at about 5.4 eV against hydrogen's
    13.6, so the boundary must be passing each atom's own chi through.
    """
    warm = "temperature_k=6000&electron_density_cm3=1e13"
    li = client.get(f"/api/spectrum?system=li&{warm}").json()
    h = client.get(f"/api/spectrum?system=h&{warm}").json()
    assert (
        li["thermal"]["ionized_fraction"]["value"]
        > h["thermal"]["ionized_fraction"]["value"]
    )
