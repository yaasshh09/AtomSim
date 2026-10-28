"""The volume grid: exact values at the nodes, a box that holds the electron,
the state's own symmetry, and the job that serves it."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from atomsim.analytic.wavefunction import evaluate_state
from atomsim.provenance import Fidelity
from atomsim.server.app import create_app
from atomsim.volume import density_volume


def test_node_values_are_the_closed_form_density():
    vol = density_volume(3, 2, 1, resolution=48)
    assert vol.provenance.fidelity is Fidelity.EXACT
    axis = np.linspace(-vol.half_width, vol.half_width, 48)
    idx = [(5, 30, 41), (24, 24, 10), (47, 0, 23)]
    pos = np.array([[axis[i], axis[j], axis[k]] for i, j, k in idx])
    exact = np.abs(evaluate_state(3, 2, 1, pos).values) ** 2
    got = np.array([vol.values[i, j, k] for i, j, k in idx])
    assert np.allclose(got, exact, rtol=1e-6)  # float32 storage


def test_the_box_holds_the_electron():
    """Fitted to hold 99.9% in its inscribed sphere, so the cube holds at least that,
    less what a 64-point Riemann sum misses."""
    vol = density_volume(2, 1, 0, resolution=64)
    assert vol.captured.value == pytest.approx(1.0, abs=5e-3)


def test_2p0_is_odd_in_z_so_its_density_is_even():
    v = density_volume(2, 1, 0, resolution=48).values
    assert np.allclose(v, v[:, :, ::-1], rtol=1e-5, atol=1e-12)


def test_volume_job():
    body = {"n": 1, "l": 0, "m": 0}
    with TestClient(create_app()) as client:
        refused = client.post("/api/jobs/volume", json={**body, "resolution": 50})
        assert refused.status_code == 422
        job = client.post("/api/jobs/volume", json={**body, "resolution": 48}).json()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = client.get(f"/api/jobs/{job['id']}").json()
            if status["status"] in ("done", "error"):
                break
            time.sleep(0.05)
        assert status["status"] == "done", status["error"]
        meta = client.get(f"/api/jobs/{job['id']}/meta").json()
        assert meta["kind"] == "volume"
        assert meta["provenance"]["fidelity"] == "exact"
        data = client.get(f"/api/jobs/{job['id']}/data").content
        assert len(data) == 48**3 * 4
