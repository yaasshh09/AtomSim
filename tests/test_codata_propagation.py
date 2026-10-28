"""CODATA uncertainty propagation: an exact closed form carries the measured
constants' uncertainty, and nothing more, into the values a reader sees."""

import math

import pytest
from fastapi.testclient import TestClient
from scipy import constants as sc

from atomsim.server.app import create_app
from atomsim.systems import get_system

REL_EV = (
    sc.physical_constants["Hartree energy in eV"][2]
    / sc.physical_constants["Hartree energy in eV"][0]
)


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_hydrogen_energy_in_ev_carries_the_conversion_uncertainty(client):
    body = client.get("/api/state/1/0/0").json()
    e_ev = body["energy_ev"]
    mu = get_system("h").mu_ratio
    rel_mu = mu.provenance.error_estimate / mu.value
    expected = abs(e_ev["value"]) * math.hypot(REL_EV, rel_mu)
    assert e_ev["provenance"]["fidelity"] == "exact"
    assert e_ev["provenance"]["error_estimate"] == pytest.approx(expected, rel=1e-6)
    # Small, as it should be: a few parts in 10^11, set by the proton-electron mass ratio.
    assert e_ev["provenance"]["error_estimate"] / abs(e_ev["value"]) < 1e-10


def test_muonic_energy_is_limited_by_the_muon_mass(client):
    """For muonic hydrogen the reduced mass is known far less well than the
    hartree, so the mass term dominates the error bar."""
    body = client.get("/api/state/1/0/0", params={"system": "mu-h"}).json()
    e = body["energy"]
    mu = get_system("mu-h").mu_ratio
    rel_mu = mu.provenance.error_estimate / mu.value
    assert rel_mu > 10 * REL_EV
    assert e["provenance"]["error_estimate"] == pytest.approx(abs(e["value"]) * rel_mu, rel=1e-9)
    assert any(q["label"] == mu.label for q in e["provenance"]["inputs"])


def test_mean_radius_goes_as_one_over_mu(client):
    body = client.get("/api/state/2/1/0", params={"system": "mu-h"}).json()
    r = body["mean_radius"]
    mu = get_system("mu-h").mu_ratio
    assert r["provenance"]["error_estimate"] == pytest.approx(
        r["value"] * mu.provenance.error_estimate / mu.value, rel=1e-9
    )
