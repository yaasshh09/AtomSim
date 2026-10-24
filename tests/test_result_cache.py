import numpy as np
import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import PlaneRequest, SampleRequest, create_app
from atomsim.server.result_cache import ResultCache, nbytes


def test_nbytes_walks_nested_results_once():
    a = np.zeros(1000)
    assert nbytes((a, {"b": [a, np.zeros(10, np.float32)]})) == 8000 + 40  # `a` counted once


def test_eviction_is_by_bytes_oldest_first_and_reads_refresh():
    c = ResultCache(max_bytes=3000)
    c.put("a", np.zeros(125))  # 1000 bytes each
    c.put("b", np.zeros(125))
    c.put("c", np.zeros(125))
    assert c.get("a") is not None  # now the freshest
    c.put("d", np.zeros(125))
    assert c.get("b") is None and c.get("a") is not None and len(c) == 3
    c.put("huge", np.zeros(1000))  # bigger than the whole budget: refused, nothing evicted
    assert c.get("huge") is None and len(c) == 3


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def _wait(client, job_id):
    for _ in range(600):
        if client.get(f"/api/jobs/{job_id}").json()["status"] in ("done", "error"):
            return
    raise TimeoutError(job_id)


def _sample(client, **body):
    body = {"n": 2, "l": 1, "m": 0, "count": 2000, **body}
    job = client.post("/api/jobs/sample", json=body).json()
    _wait(client, job["id"])
    return client.get(f"/api/jobs/{job['id']}/data?channel=positions").content


def test_an_identical_request_is_answered_from_the_cache(client):
    results = client.app.state.results
    first = _sample(client)
    hits = results.hits
    assert _sample(client) == first  # same request, same points, bit for bit
    assert results.hits == hits + 1
    # A different seed is a different request, so a different cloud.
    assert _sample(client, seed=7) != first


def test_the_cache_key_is_the_validated_request(client):
    """Omitting a default and spelling it out must land on the same entry."""
    results = client.app.state.results
    _sample(client)
    hits = results.hits
    _sample(client, basis="complex", seed=0, system="h")
    assert results.hits == hits + 1


def test_the_warmer_fills_the_cache_and_leaves_the_job_store_alone(client):
    app = client.app
    app.state.popular = [
        ("sample", SampleRequest(n=1, l=0, m=0, count=1000)),
        ("plane", PlaneRequest(n=2, l=1, m=0, resolution=32)),
    ]
    before = len(app.state.results)
    client.portal.call(app.state.warm)
    assert len(app.state.results) == before + 2
    assert len(app.state.jobs) == 0  # warm jobs are discarded once cached


def test_the_default_warm_list_names_real_states():
    app = create_app()
    kinds = {k for k, _ in app.state.popular}
    assert kinds == {"sample", "plane", "hf"}
    hf = [r for k, r in app.state.popular if k == "hf"]
    assert all(r.n_electrons == r.z for r in hf)  # as the client sends it
