"""Hartree-Fock over the job boundary.

Mirrors test_server_iso.py in shape. What is checked here that is not checked
in the engine tests: that the four new fields survive the trip and change the
answer, that the defaults cannot hand a client a model or a counterfactual it
did not ask for, and that the refusals arrive with their reasons rather than as
a bare 422.
"""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

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


def _run(client, path, **body):
    r = client.post(path, json=body)
    assert r.status_code == 200, r.text
    job = r.json()["id"]
    done = _wait_done(client, job)
    assert done["status"] == "done", done.get("error")
    return job


def test_sample_job_defaults_to_the_screened_model(client):
    """A client that has never heard of `model` cannot get Hartree-Fock."""
    job = _run(client, "/api/jobs/sample", n=2, l=1, m=0, count=2_000, system="ne")
    meta = client.get(f"/api/jobs/{job}/meta").json()
    assert meta["model"] == "gsz"
    assert "screened" in meta["provenance"]["method"]


def test_sample_job_under_hartree_fock(client):
    job = _run(
        client, "/api/jobs/sample",
        n=2, l=1, m=0, count=2_000, system="ne", model="hf",
    )
    meta = client.get(f"/api/jobs/{job}/meta").json()
    assert meta["model"] == "hf"
    assert "Hartree-Fock" in meta["provenance"]["method"]
    assert meta["provenance"]["fidelity"] == "approximation"


def test_plane_job_under_hartree_fock_differs_from_the_screened_one(client):
    """Two models, two pictures. If they matched, one branch was not taken."""
    values = {}
    for model in ("gsz", "hf"):
        job = _run(
            client, "/api/jobs/plane",
            n=2, l=1, m=0, system="ne", resolution=64, model=model,
        )
        r = client.get(f"/api/jobs/{job}/data")
        values[model] = np.frombuffer(r.content, dtype=np.float32)
    assert not np.allclose(values["gsz"], values["hf"])


def test_iso_job_carries_the_counterfactual_badge(client):
    job = _run(
        client, "/api/jobs/isosurface",
        n=2, l=1, m=0, system="ne", resolution=48, model="hf", exchange=False,
    )
    meta = client.get(f"/api/jobs/{job}/meta").json()
    assert meta["model"] == "hf"
    assert meta["provenance"]["fidelity"] == "counterfactual"
    joined = " ".join(meta["provenance"]["assumptions"])
    assert "distinguishable" in joined


def test_explicit_config_reaches_the_picture(client):
    """The configuration trap, over the wire this time."""
    values = {}
    for config in (None, "1s2 2s2 2p5 3s1"):
        body = dict(n=2, l=1, m=0, system="ne", resolution=64, model="hf")
        if config is not None:
            body["config"] = config
        job = _run(client, "/api/jobs/plane", **body)
        r = client.get(f"/api/jobs/{job}/data")
        values[str(config)] = np.frombuffer(r.content, dtype=np.float32)
    assert not np.allclose(values["None"], values["1s2 2s2 2p5 3s1"])


def test_refuses_an_unoccupied_subshell_with_the_reason(client):
    r = client.post(
        "/api/jobs/sample",
        json=dict(n=3, l=2, m=0, count=2_000, system="ne", model="hf"),
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "not occupied" in detail
    assert "Fock operator" in detail


def test_refuses_a_non_1s_orbital_with_the_cap_lifted(client):
    r = client.post(
        "/api/jobs/sample",
        json=dict(
            n=2, l=1, m=0, count=2_000, system="ne",
            model="hf", exchange=False, pauli=False,
        ),
    )
    assert r.status_code == 422
    assert "occupancy cap" in r.json()["detail"]


def test_refuses_pauli_off_with_exchange_on(client):
    r = client.post(
        "/api/jobs/isosurface",
        json=dict(n=1, l=0, m=0, system="ne", model="hf", pauli=False),
    )
    assert r.status_code == 422
    assert "antisymmetry" in str(r.json()["detail"])


def test_refuses_a_one_electron_system(client):
    """Hartree-Fock of hydrogen is a question the other views already answer."""
    r = client.post(
        "/api/jobs/plane", json=dict(n=1, l=0, m=0, system="h", model="hf")
    )
    assert r.status_code == 422
    assert "electron count" in r.json()["detail"]


def test_counterfactual_flags_are_ignored_under_the_screened_model(client):
    """They name Hartree-Fock's rules, and GSZ has no exchange term to remove.

    Accepted and unused rather than refused: the client's model selector and
    its counterfactual switches are separate controls, and a user who leaves a
    switch set while switching models has not asked for anything incoherent.
    What must not happen is a screened picture wearing a COUNTERFACTUAL badge.
    """
    job = _run(
        client, "/api/jobs/sample",
        n=2, l=1, m=0, count=2_000, system="ne", exchange=False,
    )
    meta = client.get(f"/api/jobs/{job}/meta").json()
    assert meta["model"] == "gsz"
    assert meta["provenance"]["fidelity"] == "approximation"


def test_radial_under_hartree_fock(client):
    r = client.get("/api/radial/2/1?system=ne&model=hf")
    assert r.status_code == 200
    body = r.json()
    joined = " ".join(body["r_wavefunction"]["provenance"]["assumptions"])
    assert "not an observable" in joined
    assert body["r_wavefunction"]["provenance"]["fidelity"] == "approximation"


def test_radial_hartree_fock_differs_from_screened(client):
    hf = client.get("/api/radial/2/1?system=ne&model=hf").json()
    gsz = client.get("/api/radial/2/1?system=ne").json()
    assert not np.allclose(
        hf["r_wavefunction"]["values"], gsz["r_wavefunction"]["values"]
    )


def test_radial_refuses_an_unoccupied_subshell(client):
    r = client.get("/api/radial/3/2?system=ne&model=hf")
    assert r.status_code == 422
    assert "not occupied" in r.json()["detail"]


def test_radial_carries_the_total_density_under_hartree_fock(client):
    body = client.get("/api/radial/2/1?system=ne&model=hf").json()
    d = body["total_density"]
    assert d is not None
    assert d["unit"] == "electrons/bohr"
    assert "observable" in " ".join(d["provenance"]["assumptions"])
    # It accounts for all ten electrons, which is the point of sending it.
    assert np.trapezoid(d["values"], d["grid"]) == pytest.approx(10.0, rel=1e-3)


def test_screened_radial_has_no_total_density(client):
    """Null rather than absent: the field exists and this model does not fill it."""
    body = client.get("/api/radial/2/1?system=ne").json()
    assert body["total_density"] is None


def test_radial_counterfactual_flips_the_tier(client):
    r = client.get("/api/radial/2/1?system=ne&model=hf&exchange=false")
    assert r.status_code == 200
    prov = r.json()["r_wavefunction"]["provenance"]
    assert prov["fidelity"] == "counterfactual"
