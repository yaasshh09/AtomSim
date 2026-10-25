"""/api/rabi: a driven transition served with its tier, and the refusals."""

import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_default_is_lyman_alpha_closed_pair(client):
    r = client.get("/api/rabi", params={"intensity": 1e5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lower"] == [1, 0, 0] and body["upper"] == [2, 1, 0]
    assert body["steady_excited"] is not None
    assert body["excited"]["provenance"]["fidelity"] == "approximation"
    assert len(body["excited"]["values"]) == len(body["excited_undamped"]["grid"])


def test_leaky_pair_has_no_steady_formula(client):
    body = client.get("/api/rabi", params={"n2": 3}).json()
    assert body["steady_excited"] is None
    assert body["lost"]["values"][-1] > 0


@pytest.mark.parametrize("params,code", [
    ({"system": "na"}, 400),
    ({"intensity": 1e20}, 422),
    ({"intensity": -1}, 422),
    ({"l2": 2, "n2": 3}, 422),
    ({"n2": 11, "l2": 1}, 422),
    ({"detuning": 1e9}, 422),
])
def test_refusals(client, params, code):
    assert client.get("/api/rabi", params=params).status_code == code
