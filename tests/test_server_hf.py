"""The Hartree-Fock server surface.

The solve takes seconds, so it runs as a background job like sampling and
plane grids, and reuses their endpoints rather than growing a parallel set:
POST /api/jobs/hf starts it, /api/jobs/{id} reports status, /api/jobs/{id}/meta
returns the energies with their provenance, and /api/jobs/{id}/data serves the
orbital amplitudes as raw float32.

Most of what is checked here is that provenance survives the trip. An energy
that arrives in the browser without its fidelity tier is exactly the quiet lie
this project exists to prevent, and the boundary is where it would be lost.
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


def _run(client, **body):
    """POST a solve, wait for it, return (job body, meta body)."""
    r = client.post("/api/jobs/hf", json=body)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    else:
        raise TimeoutError(f"job {job_id} did not finish")
    assert status["status"] == "done", status["error"]
    meta = client.get(f"/api/jobs/{job_id}/meta")
    assert meta.status_code == 200, meta.text
    return job_id, meta.json()


def test_hartree_fock_runs_as_a_job_not_a_blocking_request(client):
    r = client.post("/api/jobs/hf", json={"z": 2})
    assert r.status_code == 200
    body = r.json()
    assert "id" in body
    assert body["status"] in ("pending", "running", "done")


def test_the_energy_carries_its_fidelity_to_the_browser(client):
    _, meta = _run(client, z=2)
    assert meta["kind"] == "hf"
    assert meta["total_energy"]["provenance"]["fidelity"] == "approximation"
    assert meta["total_energy"]["unit"] == "hartree"
    assert meta["total_energy"]["value"] == pytest.approx(-2.8617, abs=1e-3)


def test_the_diagnostics_are_labelled_numerical_not_approximation(client):
    """The virial ratio is a property of the converged solve, not of helium.

    If it arrived tagged APPROXIMATION a view would be entitled to draw it as
    physics, so the tier is the thing keeping it honest and it is worth a test
    of its own.
    """
    _, meta = _run(client, z=2)
    for key in ("virial_ratio", "kinetic", "potential"):
        assert meta[key]["provenance"]["fidelity"] == "numerical", key
    assert meta["virial_ratio"]["value"] == pytest.approx(2.0, rel=1e-3)


def test_the_error_estimate_survives_the_boundary(client):
    _, meta = _run(client, z=2)
    err = meta["total_energy"]["provenance"]["error_estimate"]
    assert err is not None and err > 0.0


def test_convergence_fields_reach_the_client(client):
    _, meta = _run(client, z=2)
    assert meta["converged"] is True
    assert meta["iterations"] > 0
    assert meta["coarse_iterations"] > 0


def test_energies_arrive_in_ev_as_well_as_hartree(client):
    _, meta = _run(client, z=2)
    ev = meta["total_energy_ev"]
    assert ev["unit"] == "eV"
    assert ev["value"] == pytest.approx(meta["total_energy"]["value"] * 27.2114, rel=1e-4)


def test_the_orbital_amplitudes_come_back_as_binary(client):
    job_id, meta = _run(client, z=4)
    labels = [o["label"] for o in meta["orbitals"]]
    assert labels == ["1s", "2s"]

    grid = client.get(f"/api/jobs/{job_id}/data?channel=grid")
    assert grid.status_code == 200
    r = np.frombuffer(grid.content, dtype=np.float32)
    assert r.size == meta["grid_points"]
    assert np.all(np.diff(r) > 0)

    for orbital in meta["orbitals"]:
        raw = client.get(f"/api/jobs/{job_id}/data?channel={orbital['channel']}")
        assert raw.status_code == 200
        p = np.frombuffer(raw.content, dtype=np.float32)
        assert p.size == r.size
        # P = r*R, normalized to 1 under int P^2 dr. It vanishes at the outer
        # wall, but NOT at the inner one: the exponential mesh starts at a
        # small nonzero r_min, and P(r_min) ~ r_min * R(0) is a few parts in a
        # thousand for beryllium's 1s rather than zero. Asserting zero there
        # would be asserting that the mesh reaches the origin, which it does
        # not and deliberately does not.
        assert abs(p[-1]) < 1e-4
        assert abs(p[0]) < 0.05 * np.abs(p).max()
        assert np.trapezoid(p.astype(np.float64) ** 2, r.astype(np.float64)) == (
            pytest.approx(1.0, rel=1e-3)
        )


def test_an_unknown_channel_names_the_ones_that_exist(client):
    job_id, _ = _run(client, z=2)
    r = client.get(f"/api/jobs/{job_id}/data?channel=P_7f")
    assert r.status_code == 422
    assert "P_1s" in r.json()["detail"]


def test_a_neutral_alkali_beyond_argon_is_refused_with_the_reason(client):
    """Potassium is a well-posed request this solver cannot answer honestly.

    The refusal has to say which part is the problem, because the obvious
    reading - that Z is too large - is wrong, and a user who believes it would
    not try the ions that do work.
    """
    r = client.post("/api/jobs/hf", json={"z": 19})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "n = 4" in detail
    assert "n <= 3" in detail


def test_an_argon_like_ion_above_argon_is_accepted(client):
    """The other half of the claim the refusal makes.

    If this failed, the refusal message would be describing a limit the server
    does not actually have, which is its own kind of dishonesty.
    """
    _, meta = _run(client, z=20, n_electrons=18)
    assert meta["symbol"] is None  # past the preset table, and says so
    assert meta["z"] == 20 and meta["n_electrons"] == 18
    assert meta["virial_ratio"]["value"] == pytest.approx(2.0, rel=1e-3)


def test_z_beyond_the_tested_range_is_refused(client):
    r = client.post("/api/jobs/hf", json={"z": 60, "n_electrons": 18})
    assert r.status_code == 400
    assert "36" in r.json()["detail"]


def test_an_electron_count_that_contradicts_the_config_is_refused(client):
    r = client.post("/api/jobs/hf", json={"z": 10, "n_electrons": 8, "config": "1s2 2s2 2p6"})
    assert r.status_code == 400
    assert "10 electrons" in r.json()["detail"]


def test_a_malformed_config_is_a_different_status_than_an_unsupported_one(client):
    """422 means "not understood", 400 means "understood and declined"."""
    unreadable = client.post("/api/jobs/hf", json={"z": 10, "config": "1s2 2s9"})
    assert unreadable.status_code == 422
    declined = client.post("/api/jobs/hf", json={"z": 19})
    assert declined.status_code == 400


def test_an_excited_configuration_is_allowed_and_flagged_as_not_ground(client):
    _, meta = _run(client, z=4, config="1s2 2p2")
    assert meta["is_ground"] is False
    assert meta["config"] == "1s2 2p2"


def test_an_open_shell_discloses_the_configuration_average_and_neon_does_not(client):
    _, carbon = _run(client, z=6)
    _, neon = _run(client, z=10)
    assert "not per term" in " ".join(carbon["total_energy"]["provenance"]["assumptions"])
    assert "not per term" not in " ".join(neon["total_energy"]["provenance"]["assumptions"])


def test_the_neglected_relativity_is_quantified_for_a_heavy_atom(client):
    """Helium's is 0.005% and argon's 0.44%, so a phrase that reads the same in
    both would not be telling the reader anything."""
    _, argon = _run(client, z=18)
    joined = " ".join(argon["total_energy"]["provenance"]["assumptions"])
    assert "neglects relativity" in joined and "%" in joined

    _, helium = _run(client, z=2)
    assert "neglects relativity" not in " ".join(
        helium["total_energy"]["provenance"]["assumptions"]
    )
