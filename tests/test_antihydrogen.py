"""Antihydrogen: hydrogen's numbers, with CPT stated as the reason, everywhere."""

import pytest
from fastapi.testclient import TestClient

from atomsim.analytic.hyperfine import hyperfine_report
from atomsim.provenance import Fidelity
from atomsim.server.app import create_app
from atomsim.spectra import transition_lines
from atomsim.systems import get_system


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_same_masses_with_cpt_disclosed():
    a, h = get_system("anti-h"), get_system("h")
    assert a.mu_ratio.value == h.mu_ratio.value and a.Z == h.Z
    assert any("CPT" in s for s in a.mu_ratio.provenance.assumptions)
    assert any("positron" in s for s in a.mu_ratio.provenance.assumptions)
    # Borrowed by symmetry, never measured: not a reference measurement.
    assert a.nuclear_radius.provenance.fidelity is Fidelity.APPROXIMATION
    assert a.nuclear_radius.value == h.nuclear_radius.value


def test_spectrum_is_hydrogens():
    a = transition_lines(get_system("anti-h"), n_max=4).lines
    h = transition_lines(get_system("h"), n_max=4).lines
    assert [ln.wavelength.value for ln in a] == [ln.wavelength.value for ln in h]


def test_hyperfine_is_hydrogens():
    a = hyperfine_report(1, get_system("anti-h"))
    h = hyperfine_report(1, get_system("h"))
    assert a.available and a.nucleus_name == "antiproton"
    assert a.A.value == pytest.approx(h.A.value, rel=1e-15)
    assert "CPT" in a.note


def test_served_like_any_preset(client):
    keys = [s["key"] for s in client.get("/api/systems").json()["systems"]]
    assert "anti-h" in keys
    lv = client.get("/api/levels", params={"system": "anti-h"})
    assert lv.status_code == 200, lv.text
    fs = client.get("/api/finite-size", params={"system": "anti-h"}).json()
    assert fs["radius_fm"]["provenance"]["fidelity"] == "approximation"
