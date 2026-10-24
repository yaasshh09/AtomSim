"""/api/cascade: the decay cascade served, provenance and all."""

import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_3p_cascade(client):
    r = client.get("/api/cascade/3/1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["start"] == [3, 1]
    terminal = {(lv["n"], lv["l"]): lv for lv in body["levels"] if lv["terminal"]}
    assert set(terminal) == {(2, 0), (1, 0)}
    assert all(lv["lifetime"] is None for lv in terminal.values())
    assert terminal[(2, 0)]["visit_probability"]["value"] == pytest.approx(0.1183, abs=1e-4)
    assert body["mean_settle_time"]["provenance"]["fidelity"] == "numerical"
    assert len(body["populations"]) == len(body["levels"])
    assert body["populations"][0]["grid_unit"] == "s"


def test_reduced_mass_slows_positronium(client):
    # A scales as mu, so halving mu doubles every lifetime.
    h = client.get("/api/cascade/2/1").json()["mean_settle_time"]["value"]
    ps = client.get("/api/cascade/2/1", params={"system": "ps"}).json()
    assert ps["mean_settle_time"]["value"] == pytest.approx(2 * h, rel=1e-3)


def test_refusals(client):
    assert client.get("/api/cascade/3/1", params={"system": "na"}).status_code == 400
    assert client.get("/api/cascade/11/1").status_code == 422
    assert client.get("/api/cascade/2/2").status_code == 422
