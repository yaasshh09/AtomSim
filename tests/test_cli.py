import subprocess
import sys

import pytest

import atomsim.cli as cli


def test_parser_defaults():
    args = cli.build_parser().parse_args(["serve"])
    assert args.command == "serve"
    assert args.port == 8000
    assert args.no_browser is False


def test_serve_invokes_uvicorn_on_loopback(monkeypatch):
    captured = {}

    def fake_run(app, host, port):
        captured["host"] = host
        captured["port"] = port

    opened = []
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    monkeypatch.setattr(cli, "_open_browser_soon", lambda url: opened.append(url))

    cli.main(["serve", "--port", "8123"])
    assert captured == {"host": "127.0.0.1", "port": 8123}
    assert opened == ["http://127.0.0.1:8123"]


def test_no_browser_flag(monkeypatch):
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, host, port: None)
    opened = []
    monkeypatch.setattr(cli, "_open_browser_soon", lambda url: opened.append(url))
    cli.main(["serve", "--no-browser"])
    assert opened == []


@pytest.mark.parametrize("module", ["atomsim", "atomsim.cli"])
def test_module_entry_point_runs_the_parser(module):
    """`python -m atomsim[.cli] --help` must print help, not exit silently."""
    done = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "usage: atomsim" in done.stdout
    assert "serve" in done.stdout


@pytest.mark.parametrize("module", ["atomsim", "atomsim.cli"])
def test_module_entry_point_requires_a_subcommand(module):
    done = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2
    assert "required" in done.stderr
