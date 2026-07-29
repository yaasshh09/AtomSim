import subprocess
import sys

import pytest
import uvicorn

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
    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(cli, "_open_browser_soon", lambda url: opened.append(url))

    cli.main(["serve", "--port", "8123"])
    assert captured == {"host": "127.0.0.1", "port": 8123}
    assert opened == ["http://127.0.0.1:8123"]


def test_no_browser_flag(monkeypatch):
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: None)
    opened = []
    monkeypatch.setattr(cli, "_open_browser_soon", lambda url: opened.append(url))
    cli.main(["serve", "--no-browser"])
    assert opened == []


def test_importing_the_cli_does_not_drag_in_the_server_stack():
    """Parsing arguments must not cost a web framework.

    Importing atomsim.server.app at module scope was 5.4 of the 6.3 seconds it
    took to load atomsim.cli, and `atomsim --help` paid all of it. The import
    is inside main() now, and this pins it there: the failure mode is someone
    adding a convenient top-level import and nobody noticing that startup went
    back to six seconds, since nothing about the behaviour would change.
    """
    probe = (
        "import sys, atomsim.cli; "
        "print(','.join(m for m in ('fastapi', 'uvicorn', 'matplotlib') "
        "if m in sys.modules))"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "", (
        f"atomsim.cli eagerly imported: {done.stdout.strip()}"
    )


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
