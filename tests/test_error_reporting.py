import logging

import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app
from atomsim.server.jobs import JobStore


def test_a_failed_job_leaves_its_traceback_in_the_log(caplog):
    store = JobStore()
    job = store.create()

    def boom(progress):
        raise RuntimeError("the solver fell over")

    with caplog.at_level(logging.ERROR, logger="atomsim.server.jobs"):
        store.run(job.id, boom)
    assert job.error == "RuntimeError: the solver fell over"
    record = next(r for r in caplog.records if "failed" in r.getMessage())
    assert record.exc_info is not None  # the traceback, not just the message


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def test_a_client_error_is_one_log_line(client, caplog):
    with caplog.at_level(logging.WARNING, logger="atomsim.server.app"):
        r = client.post("/api/client-error", json={
            "message": "TypeError: x is undefined",
            "stack": "at a (main.js:1)\nINFO:     forged line\nat b (main.js:2)",
            "page": "/", "version": "0.1.0",
        })
    assert r.status_code == 204
    line = next(r.getMessage() for r in caplog.records if "client error" in r.getMessage())
    assert "\n" not in line and "forged line" in line  # kept, but cannot pose as our own line


def test_reports_are_capped_and_rate_limited(client):
    assert client.post("/api/client-error", json={"message": "x" * 501}).status_code == 422
    codes = [client.post("/api/client-error", json={"message": "m"}).status_code for _ in range(7)]
    assert codes[:5] == [204] * 5 and codes[-1] == 429
