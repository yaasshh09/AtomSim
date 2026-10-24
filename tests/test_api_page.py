"""The API guide (web/api.html) may only describe routes that exist."""

import re
from pathlib import Path

from atomsim.server.app import create_app

PAGE = (Path(__file__).resolve().parents[1] / "web" / "api.html").read_text("utf-8")


def _pattern(route_path: str) -> re.Pattern:
    return re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", route_path) + "$")


def test_every_path_the_page_names_is_a_real_route():
    routes = [_pattern(r.path) for r in create_app().routes if hasattr(r, "path")]
    named = set(re.findall(r"(/(?:api|ws)/[A-Za-z0-9_/{}.-]+)", PAGE))
    assert named, "the page should name some routes"
    for path in named:
        concrete = re.sub(r"\{[^}]+\}", "1", path)
        assert any(p.match(concrete) for p in routes), f"{path} is not a route"


def test_the_quoted_limits_are_the_server_defaults():
    from atomsim.server.ratelimit import DEFAULT_CAPACITY, DEFAULT_PERIOD

    assert f"burst of {DEFAULT_CAPACITY}" in PAGE
    assert "one\n        job every three seconds" in PAGE or "job every three seconds" in PAGE
    assert DEFAULT_PERIOD / DEFAULT_CAPACITY == 3.0
