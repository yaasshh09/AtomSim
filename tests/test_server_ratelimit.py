"""The limiter's wiring: which requests are charged, and what a refusal says.

`test_ratelimit.py` pins the bucket arithmetic. This file pins the questions
only the app can answer: that reads stay free, that the refusal is a 429 that
tells the client when to come back, and that the proxy header is trusted only
when it has been named.
"""

import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app

SAMPLE = {"n": 1, "l": 0, "m": 0, "count": 1000}


@pytest.fixture
def limited(monkeypatch):
    """An app whose bucket holds exactly two jobs and refills slowly."""
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT", "on")
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT_BURST", "2")
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT_PERIOD", "600")
    with TestClient(create_app()) as client:
        yield client


def test_the_burst_is_allowed_and_the_next_job_is_refused(limited):
    assert limited.post("/api/jobs/sample", json=SAMPLE).status_code == 200
    assert limited.post("/api/jobs/sample", json=SAMPLE).status_code == 200

    refused = limited.post("/api/jobs/sample", json=SAMPLE)
    assert refused.status_code == 429
    assert "retry" in refused.json()["detail"].lower()


def test_a_refusal_says_when_to_come_back(limited):
    for _ in range(3):
        response = limited.post("/api/jobs/sample", json=SAMPLE)
    assert response.status_code == 429
    # 2 tokens per 600 s is one per 300 s, and the header must be a whole
    # number of seconds a client can actually wait on.
    retry_after = int(response.headers["Retry-After"])
    assert retry_after == 300


def test_reads_are_never_charged(limited):
    """Exhaust the bucket, then confirm the app itself still works."""
    for _ in range(4):
        limited.post("/api/jobs/sample", json=SAMPLE)

    assert limited.get("/api/systems").status_code == 200
    assert limited.get("/api/state/1/0/0").status_code == 200
    assert limited.get("/api/levels?system=h&n_max=3").status_code == 200


def test_the_limiter_is_off_when_the_environment_says_so(monkeypatch):
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT", "off")
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT_BURST", "1")
    with TestClient(create_app()) as client:
        codes = [client.post("/api/jobs/sample", json=SAMPLE).status_code for _ in range(5)]
    assert codes == [200] * 5


def test_a_named_proxy_header_separates_clients_behind_one_address(monkeypatch):
    """Without this, everyone behind the deploy proxy shares a single bucket."""
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT", "on")
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT_BURST", "1")
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT_PERIOD", "600")
    monkeypatch.setenv("ATOMSIM_CLIENT_IP_HEADER", "Fly-Client-IP")
    with TestClient(create_app()) as client:
        first = client.post("/api/jobs/sample", json=SAMPLE, headers={"Fly-Client-IP": "1.1.1.1"})
        same = client.post("/api/jobs/sample", json=SAMPLE, headers={"Fly-Client-IP": "1.1.1.1"})
        other = client.post("/api/jobs/sample", json=SAMPLE, headers={"Fly-Client-IP": "2.2.2.2"})
    assert first.status_code == 200
    assert same.status_code == 429  # same visitor, bucket spent
    assert other.status_code == 200  # different visitor, own bucket


def test_a_spoofed_prefix_does_not_buy_a_fresh_bucket(monkeypatch):
    """The caller writes the left of the list; the trusted proxy appends the right.

    A forwarding proxy appends rather than replaces, so a client sending its own
    address arrives as `<whatever they typed>, <their real address>`. Charging
    the leftmost entry charges a string the caller chose, and rotating it per
    request would mint an unlimited supply of full buckets: a limiter that
    refuses nobody.
    """
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT", "on")
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT_BURST", "1")
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT_PERIOD", "600")
    monkeypatch.setenv("ATOMSIM_CLIENT_IP_HEADER", "X-Forwarded-For")
    with TestClient(create_app()) as client:
        first = client.post(
            "/api/jobs/sample", json=SAMPLE, headers={"X-Forwarded-For": "spoof-a, 3.3.3.3"}
        )
        rotated = client.post(
            "/api/jobs/sample", json=SAMPLE, headers={"X-Forwarded-For": "spoof-b, 3.3.3.3"}
        )
    assert first.status_code == 200
    assert rotated.status_code == 429  # same real address, bucket already spent


def test_an_unnamed_proxy_header_is_ignored(monkeypatch):
    """A spoofable header must not become a way around the limiter."""
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT", "on")
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT_BURST", "1")
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT_PERIOD", "600")
    monkeypatch.delenv("ATOMSIM_CLIENT_IP_HEADER", raising=False)
    with TestClient(create_app()) as client:
        first = client.post("/api/jobs/sample", json=SAMPLE, headers={"Fly-Client-IP": "1.1.1.1"})
        spoofed = client.post("/api/jobs/sample", json=SAMPLE, headers={"Fly-Client-IP": "9.9.9.9"})
    assert first.status_code == 200
    assert spoofed.status_code == 429
