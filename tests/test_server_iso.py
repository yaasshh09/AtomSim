"""The isosurface server surface.

The mesh is the first result this project sends over the wire that is three
arrays of three different shapes and dtypes, so most of what is checked here is
that it arrives assembled: triangles that index vertices that exist, a phase per
vertex, and the byte counts the meta promised.

The rest is the disclosure. A surface whose enclosed fraction and escaped mass
did not survive the trip is exactly the textbook lobe this phase exists to
replace, and the boundary is where they would be lost.
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


def _wait_done(client, job_id, deadline_s=60.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline_s:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish")


def _run(client, **body):
    r = client.post("/api/jobs/isosurface", json=body)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    status = _wait_done(client, job_id)
    assert status["status"] == "done", status["error"]
    return job_id, client.get(f"/api/jobs/{job_id}/meta").json()


def _channel(client, job_id, name, dtype):
    raw = client.get(f"/api/jobs/{job_id}/data?channel={name}").content
    return np.frombuffer(raw, dtype=dtype)


def test_isosurface_job_end_to_end(client):
    """One 1s surface, checked as geometry rather than as a byte count.

    The closed-form radius is 2.6612 bohr, and it is asserted on the vertices
    that came back through the wire rather than on the engine's, so a transposed
    reshape or a truncated read fails here.
    """
    job_id, meta = _run(client, n=1, l=0, m=0, fraction=0.9, resolution=64)

    assert meta["kind"] == "isosurface"
    assert meta["resolution"] == 64
    assert meta["target_fraction"] == 0.9
    assert meta["enclosed_fraction"]["value"] == pytest.approx(0.9, abs=5e-3)
    assert meta["outside_fraction"] == pytest.approx(
        1 - meta["enclosed_fraction"]["value"]
    )
    assert meta["escaped_fraction"]["value"] < 2e-3
    assert meta["components"] == 1
    assert meta["axis_unit"] == "bohr"

    vertices = _channel(client, job_id, "vertices", np.float32).reshape(-1, 3)
    triangles = _channel(client, job_id, "triangles", np.uint32).reshape(-1, 3)
    phase = _channel(client, job_id, "phase", np.float32)

    assert vertices.shape[0] == meta["vertex_count"]
    assert triangles.shape[0] == meta["triangle_count"]
    assert phase.shape[0] == meta["vertex_count"]
    assert triangles.max() < vertices.shape[0]

    radii = np.linalg.norm(vertices, axis=1)
    assert radii.mean() == pytest.approx(2.6612, rel=5e-3)
    # 1s is real and positive everywhere, so every vertex is on the same side.
    assert np.allclose(phase, 0.0)


def test_the_default_channel_is_the_vertices(client):
    """A client that asks for no channel gets the one it cannot render without."""
    job_id, meta = _run(client, n=1, l=0, m=0, fraction=0.5, resolution=48)
    default = client.get(f"/api/jobs/{job_id}/data").content
    named = client.get(f"/api/jobs/{job_id}/data?channel=vertices").content
    assert default == named
    assert len(default) == meta["vertex_count"] * 3 * 4


def test_an_unknown_channel_names_the_ones_that_exist(client):
    job_id, _ = _run(client, n=1, l=0, m=0, resolution=48)
    r = client.get(f"/api/jobs/{job_id}/data?channel=normals")
    assert r.status_code == 422
    assert "vertices, triangles, phase" in r.json()["detail"]


def test_the_provenance_and_its_disclosures_survive_the_trip(client):
    """The tier, the complement, and the escaped mass all reach the browser."""
    _, meta = _run(client, n=2, l=1, m=0, fraction=0.9, resolution=48, basis="real")
    prov = meta["provenance"]
    assert prov["fidelity"] == "numerical"
    text = " ".join(prov["assumptions"])
    assert "no boundary" in text
    assert "outside the surface" in text
    assert "inscribed in the box" in text
    assert prov["error_estimate"] is not None
    # Every scalar carries its own, not just the container.
    assert meta["level"]["provenance"]["fidelity"] == "numerical"
    assert meta["level"]["unit"] == "bohr^-3"
    assert meta["mesh_volume"]["unit"] == "bohr^3"
    assert meta["area"]["unit"] == "bohr^2"


def test_a_screened_atom_arrives_as_an_approximation(client):
    """Sodium's valence orbital, on the same endpoint and a weaker tier."""
    _, meta = _run(client, n=3, l=0, m=0, fraction=0.9, resolution=48, system="na")
    assert meta["provenance"]["fidelity"] == "approximation"
    assert meta["system"] == "na"
    assert meta["enclosed_fraction"]["value"] == pytest.approx(0.9, abs=0.01)


def test_the_reduced_mass_of_the_system_reaches_the_surface(client):
    """Muonic hydrogen's 1s is 186 times smaller, and the endpoint has to say so.

    A system key that was accepted and then ignored would return a hydrogen
    surface under a muonic label, which is the failure mode this catches.
    """
    job_id, _ = _run(client, n=1, l=0, m=0, fraction=0.9, resolution=48, system="mu-h")
    muonic = np.linalg.norm(
        _channel(client, job_id, "vertices", np.float32).reshape(-1, 3), axis=1
    ).mean()
    assert muonic == pytest.approx(2.6612 / 186.0, rel=0.05)


@pytest.mark.parametrize(
    "body",
    [
        {"n": 1, "l": 0, "m": 0, "fraction": 0.0},
        {"n": 1, "l": 0, "m": 0, "fraction": 1.0},
        {"n": 1, "l": 0, "m": 0, "fraction": 1.5},
        {"n": 1, "l": 0, "m": 0, "resolution": 300},
        {"n": 1, "l": 1, "m": 0},
        {"n": 2, "l": 1, "m": 2},
    ],
)
def test_requests_that_do_not_name_a_surface_are_refused(client, body):
    """422 up front rather than a job that dies in a thread.

    The bounds on the fraction are the interesting ones: 0 and 1 are not
    contours. The level enclosing everything is zero, which is not a surface,
    and the level enclosing nothing is the peak, which is a point.
    """
    assert client.post("/api/jobs/isosurface", json=body).status_code == 422


def test_a_tighter_fraction_returns_a_smaller_surface(client):
    """The control is the fraction, so it has to control something."""
    _, tight = _run(client, n=1, l=0, m=0, fraction=0.5, resolution=48)
    _, loose = _run(client, n=1, l=0, m=0, fraction=0.95, resolution=48)
    assert tight["level"]["value"] > loose["level"]["value"]
    assert tight["mesh_volume"]["value"] < loose["mesh_volume"]["value"]
    assert tight["outside_fraction"] > loose["outside_fraction"]
