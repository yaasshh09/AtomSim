"""The comparison over HTTP, and the two ways it is refused."""

import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_compare_is_off_by_default(client):
    r = client.get("/api/radial/2/1?system=ne")
    assert r.status_code == 200
    assert r.json()["density_comparison"] is None


def test_compare_returns_both_curves_on_one_grid(client):
    r = client.get("/api/radial/2/1?system=ne&compare=true")
    assert r.status_code == 200
    c = r.json()["density_comparison"]
    assert c["gsz"]["grid"] == c["hf"]["grid"]
    assert len(c["gsz"]["values"]) == len(c["hf"]["values"])
    assert c["displaced_charge"]["unit"] == "electrons"
    assert c["displaced_charge"]["provenance"]["error_estimate"] > 0


def test_the_shell_table_names_the_model_that_resolves_no_peak(client):
    r = client.get("/api/radial/3/0?system=na&compare=true")
    assert r.status_code == 200
    shells = r.json()["density_comparison"]["shells"]
    assert [s["label"] for s in shells] == ["K", "L", "M"]
    assert shells[2]["gsz_radius"] is None
    assert shells[2]["hf_radius"] is not None


def test_compare_works_from_either_model(client):
    """The overlay is symmetric, so the radio the user left it on cannot matter."""
    gsz = client.get("/api/radial/2/1?system=ne&compare=true").json()
    hf = client.get("/api/radial/2/1?system=ne&model=hf&compare=true").json()
    assert gsz["density_comparison"]["displaced_charge"]["value"] == pytest.approx(
        hf["density_comparison"]["displaced_charge"]["value"]
    )


def test_compare_does_not_need_the_drawn_orbital_to_be_occupied(client):
    """3d is empty in neon, and a density does not care.

    The orbital plots are refused on their own terms under model=hf; this asks
    for the screened orbital, which exists, plus the comparison, which is about
    the atom rather than about (n, l).
    """
    r = client.get("/api/radial/3/2?system=ne&compare=true")
    assert r.status_code == 200
    assert r.json()["density_comparison"] is not None


# --- the two refusals -------------------------------------------------------


def test_sulfur_is_refused_by_name(client):
    r = client.get("/api/radial/3/1?system=s&model=hf&compare=true")
    assert r.status_code == 400
    detail = r.json()["detail"].lower()
    assert "green" in detail


def test_a_one_electron_system_is_refused(client):
    r = client.get("/api/radial/2/1?system=h&compare=true")
    assert r.status_code == 422
    assert "one-electron" in r.json()["detail"]


def test_a_thrown_switch_reaches_the_comparison_badge(client):
    r = client.get("/api/radial/1/0?system=ne&model=hf&exchange=false&compare=true")
    assert r.status_code == 200
    c = r.json()["density_comparison"]
    assert c["provenance"]["fidelity"] == "counterfactual"
    assert "exchange" in c["provenance"]["method"]
