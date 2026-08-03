"""Sulfur and chlorine: the two atoms the engine solves and the app could not show.

Szydlik and Green never published neutral GSZ parameters for Z = 16 or 17, so
`ATOM_KEYS` left them out and the whole application - every view, including the
Levels view that runs on Hartree-Fock - could not select them. Hartree-Fock
needs no fitted table, which is the entire argument for having it, and until
now the one place that argument was checkable was a unit test.

So `ATOM_KEYS` splits into two questions it had been answering with one list:
which keys name an atom, and which of those atoms the GSZ model can speak for.
The tests below are mostly about the second one staying honest, because making
the key valid opens every screened endpoint to an atom they cannot serve, and a
raw ValueError from deep in the screening table is not a refusal.
"""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from atomsim.atoms import (
    ATOM_KEYS,
    GSZ_ATOM_KEYS,
    atom_for_key,
    has_gsz_parameters,
    is_atom_key,
)
from atomsim.server.app import create_app


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def _wait_done(client, job_id, deadline_s=120.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline_s:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish")


# --- the split itself ------------------------------------------------------


def test_sulfur_and_chlorine_are_atoms_the_app_knows():
    for key in ("s", "cl"):
        assert is_atom_key(key)
        assert key in ATOM_KEYS
    assert atom_for_key("s").z == 16
    assert atom_for_key("cl").z == 17


def test_they_are_not_atoms_the_screened_model_can_speak_for():
    """The capability list is separate now, and still excludes exactly two."""
    assert set(ATOM_KEYS) - set(GSZ_ATOM_KEYS) == {"s", "cl"}
    assert not has_gsz_parameters(16)
    assert not has_gsz_parameters(17)
    assert has_gsz_parameters(18)  # argon is tabulated, sulfur's neighbour is not


def test_the_gsz_list_did_not_otherwise_change():
    """A guard on the split: nothing else gained or lost parameters."""
    assert GSZ_ATOM_KEYS[0] == "he" and GSZ_ATOM_KEYS[-1] == "ar"
    assert len(GSZ_ATOM_KEYS) == 15
    assert len(ATOM_KEYS) == 17


# --- the server refuses the screened model, with the reason ----------------


@pytest.mark.parametrize(
    "path",
    [
        "/api/radial/3/1?system=s",
        "/api/levels?system=s",
        "/api/spectrum?system=s",
    ],
)
def test_screened_endpoints_refuse_sulfur_with_the_citation(client, path):
    """400 and a reason, not a 500 out of the screening table.

    The refusal names the paper that omits it and the model that does not need
    it, because "no parameters" without either is a dead end rather than a
    redirection.
    """
    r = client.get(path)
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "Szydlik" in detail
    assert "Hartree-Fock" in detail


def test_screened_jobs_refuse_sulfur_too(client):
    r = client.post(
        "/api/jobs/sample", json=dict(n=3, l=1, m=0, count=2_000, system="s")
    )
    assert r.status_code == 400
    assert "Szydlik" in r.json()["detail"]


# --- and Hartree-Fock serves them, which was the whole point ---------------


def test_sulfur_radial_under_hartree_fock(client):
    body = client.get("/api/radial/3/1?system=s&model=hf").json()
    assert body["r_wavefunction"]["provenance"]["fidelity"] == "approximation"
    # Sixteen electrons, all accounted for by the density this now ships.
    d = body["total_density"]
    assert np.trapezoid(d["values"], d["grid"]) == pytest.approx(16.0, rel=1e-3)


def test_chlorine_cloud_under_hartree_fock(client):
    r = client.post(
        "/api/jobs/sample",
        json=dict(n=3, l=1, m=0, count=2_000, system="cl", model="hf"),
    )
    assert r.status_code == 200, r.text
    done = _wait_done(client, r.json()["id"])
    assert done["status"] == "done", done.get("error")
    meta = client.get(f"/api/jobs/{r.json()['id']}/meta").json()
    assert meta["model"] == "hf"
    assert meta["provenance"]["fidelity"] == "approximation"


def test_the_systems_list_offers_them_and_says_what_can_draw_them(client):
    systems = {s["key"]: s for s in client.get("/api/systems").json()["systems"]}
    assert systems["s"]["has_gsz"] is False
    assert systems["ar"]["has_gsz"] is True
    # The description has to carry it too: a picker that greys a control needs
    # a sentence next to it, and this is where the sentence comes from.
    assert "Hartree-Fock" in systems["s"]["description"]
    assert "GSZ" in systems["ar"]["description"]
