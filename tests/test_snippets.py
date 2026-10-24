"""Every snippet the app offers has to run, and has to print a fidelity tier."""

import contextlib
import io

import pytest

from atomsim.snippets import snippet

CASES = [
    ("cloud", {"n": 3, "l": 1, "m": -1, "system": "mu-h"}),
    ("cloud", {"n": 2, "l": 1, "m": 1, "basis": "real", "system": "z3"}),
    ("cloud", {"system": "h", "hybrid": "sp3-2", "n": 2, "l": 1, "basis": "real"}),
    ("plane", {"n": 3, "l": 2, "m": 1, "plane": "current"}),
    ("plane", {"n": 2, "l": 1, "m": 0, "plane": "psi", "system": "he+"}),
    ("plane", {"system": "d", "hybrid": "sp2-1", "n": 2, "l": 1, "basis": "real"}),
    ("radial", {"n": 4, "l": 2, "system": "ps"}),
    ("spectrum", {"system": "h"}),
    ("evolve", {"sup": ["1_0_0_1_0", "2_1_1_0.5_0.25"]}),
    ("levels", {"system": "t"}),
    ("whatif", {}),
    ("cloud", {"system": "he"}),            # screened, GSZ
    ("levels", {"system": "he", "model": "hf"}),
    ("radial", {"system": "s"}),            # no GSZ fit, so Hartree-Fock regardless
]


@pytest.mark.parametrize("view,kw", CASES)
def test_every_snippet_runs_and_reports_its_tier(view, kw):
    src = snippet(view, count=1000, **kw)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(src, f"<snippet {view}>", "exec"), {})
    printed = out.getvalue()
    if view != "spectrum":
        assert any(t in printed for t in ("EXACT", "NUMERICAL", "APPROXIMATION")), printed
    assert printed.strip()


def test_the_radial_snippet_is_normalized():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(snippet("radial", n=3, l=1), {})
    assert float(out.getvalue().split("=")[1].split()[0]) == pytest.approx(1.0, abs=1e-3)


def test_the_endpoint_serves_runnable_code_and_refuses_unknown_systems():
    from fastapi.testclient import TestClient

    from atomsim.server.app import create_app

    with TestClient(create_app()) as c:
        r = c.get("/api/snippet?view=plane&n=2&l=1&m=1&system=mu-h&plane=current")
        assert r.status_code == 200
        assert "current_through_plane" in r.json()["code"]
        assert c.get("/api/snippet?view=cloud&system=unobtainium").status_code == 422
        assert c.get("/api/snippet?view=cloud&n=1&l=1").status_code == 422
        evolve = c.get("/api/snippet?view=evolve&sup=1_0_0_1_0&sup=3_2_1_1_0.5").json()["code"]
        assert "Component(3, 2, 1, 1.0, 0.5 * math.pi)" in evolve
