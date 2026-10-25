"""/api/finite-size: the shift for a real nucleus, and a refusal where there is none."""

import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_hydrogen(client):
    r = client.get("/api/finite-size", params={"system": "h"})
    assert r.status_code == 200, r.text
    body = r.json()
    s1 = body["levels"][0]
    assert (s1["n"], s1["l"]) == (1, 0)
    assert s1["shift_mhz"]["value"] == pytest.approx(1.1, rel=0.03)
    # The error bar travels with the unit.
    mhz, mev = s1["shift_mhz"], s1["shift_mev"]
    assert mhz["provenance"]["error_estimate"] / mhz["value"] == pytest.approx(
        mev["provenance"]["error_estimate"] / mev["value"], rel=1e-12)
    p2 = body["levels"][3]
    assert (p2["l"], p2["shift_mev"]["value"]) == (1, 0.0)
    assert body["coefficient_ours"]["value"] == pytest.approx(
        body["coefficient_published"], rel=0.01)
    assert {p["key"] for p in body["predictions"]} == {"muonic", "codata_2010"}
    assert "arXiv:1301.0905" in body["citation"]


def test_muonic_shift_is_millions_of_times_bigger(client):
    h = client.get("/api/finite-size", params={"system": "h"}).json()["levels"][1]
    mu = client.get("/api/finite-size", params={"system": "mu-h"}).json()["levels"][1]
    ratio = mu["fraction"] / h["fraction"]
    assert 1e4 < ratio < 1e5   # (m_mu/m_e)^3 on the shift, over (m_mu/m_e) on E_n


def test_refusals(client):
    assert client.get("/api/finite-size", params={"system": "ps"}).status_code == 400
    assert client.get("/api/finite-size", params={"system": "na"}).status_code == 400
