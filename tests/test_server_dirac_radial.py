"""/api/dirac-radial: every j of a state, against Schrodinger, with the heavy-nucleus note."""

import math

import pytest
from fastapi.testclient import TestClient

from atomsim.constants import ALPHA
from atomsim.server.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_2p_has_both_j_and_contracts_at_high_z(client):
    r = client.get("/api/dirac-radial/2/1", params={"z": 80})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [c["j"] for c in body["components"]] == [0.5, 1.5]
    assert [c["kappa"] for c in body["components"]] == [1, -2]
    schr = body["schrodinger_mean_r"]["value"]
    # Relativity pulls the j = 1/2 orbital in, and by more than j = 3/2.
    half, three_half = (c["mean_r"]["value"] for c in body["components"])
    assert half < three_half < schr
    assert body["components"][0]["probability"]["provenance"]["fidelity"] == "exact"
    assert any("infinitely heavy" in a
               for a in body["components"][0]["large"]["provenance"]["assumptions"])
    assert body["schrodinger"]["grid"][0] > 0


def test_1s_moments_are_the_closed_forms(client):
    body = client.get("/api/dirac-radial/1/0", params={"z": 92}).json()
    gam = math.sqrt(1 - (92 * ALPHA) ** 2)
    (c,) = body["components"]
    assert c["mean_r"]["value"] == pytest.approx((2 * gam + 1) / (2 * 92), rel=1e-8)
    assert c["small_weight"]["value"] == pytest.approx((1 - gam) / 2, rel=1e-7)


@pytest.mark.parametrize("path,params", [
    ("/api/dirac-radial/2/2", {}),
    ("/api/dirac-radial/7/0", {}),
    ("/api/dirac-radial/1/0", {"z": 0}),
    ("/api/dirac-radial/1/0", {"z": 119}),
])
def test_refusals(client, path, params):
    assert client.get(path, params=params).status_code == 422


def test_dirac_binds_tighter_than_bohr(client):
    body = client.get("/api/dirac-radial/1/0", params={"z": 92}).json()
    bohr = body["schrodinger_energy"]["value"]
    assert bohr == pytest.approx(-(92**2) / 2, rel=1e-12)
    assert body["components"][0]["energy"]["value"] < bohr
