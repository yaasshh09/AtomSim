"""Where the built frontend came from, and what happens when it did not.

The mount is conditional, so a wrong path does not fail: it serves an API with
no application in front of it and says nothing. These tests pin both halves,
the override that a container needs and the disclosure that a silent 404 was
missing.
"""

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import _web_dist, create_app


@pytest.fixture()
def built(tmp_path):
    """A directory shaped like a real `vite build` output."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text('<div id="root"></div>', encoding="utf-8")
    return dist


def test_the_default_is_the_source_checkout(monkeypatch):
    # An override in the ambient environment would make this pass for the
    # wrong reason, so clear it: this test is about the fallback.
    monkeypatch.delenv("ATOMSIM_WEB_DIST", raising=False)
    # parents[3] of src/atomsim/server/app.py is the repo root, and parents[1]
    # of this file is the same directory.
    expected = Path(__file__).resolve().parents[1] / "web" / "dist"
    assert _web_dist() == expected


def test_the_override_names_where_the_build_is(monkeypatch, built):
    monkeypatch.setenv("ATOMSIM_WEB_DIST", str(built))
    assert _web_dist() == built

    with TestClient(create_app()) as client:
        served = client.get("/")
    assert served.status_code == 200
    assert 'id="root"' in served.text


def test_a_missing_build_is_said_out_loud(monkeypatch, tmp_path, caplog):
    absent = tmp_path / "never-built"
    monkeypatch.setenv("ATOMSIM_WEB_DIST", str(absent))

    with caplog.at_level(logging.WARNING, logger="atomsim.server.app"):
        with TestClient(create_app()) as client:
            served = client.get("/")

    assert served.status_code == 404
    assert str(absent) in caplog.text


def test_a_mounted_build_is_said_out_loud(monkeypatch, built, caplog):
    monkeypatch.setenv("ATOMSIM_WEB_DIST", str(built))

    with caplog.at_level(logging.INFO, logger="atomsim.server.app"):
        create_app()

    assert str(built) in caplog.text


def test_the_api_still_answers_without_a_build(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOMSIM_WEB_DIST", str(tmp_path / "never-built"))
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["status"] == "ok"
