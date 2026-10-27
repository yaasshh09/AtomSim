"""Quantum defects: the polished solver on hydrogen, the defect's flatness in n,
an error bar that brackets a finer mesh, and the measured side against NIST."""

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from atomsim.numerics.mesh import mesh_for_atom_at_step
from atomsim.provenance import Fidelity
from atomsim.rydberg import (
    _BOX,
    N_MAX,
    _corrected,
    _polished_levels,
    computed_defects,
    measured_defects,
)
from atomsim.server.app import create_app


def test_polished_hydrogen_converges_at_second_order():
    """Raw hydrogen defects on the mesh: not zero, but falling 4x per halving.

    This is the error the same-mesh subtraction removes, so it had better be a
    smooth discretization error and not eigensolver noise.
    """
    raw = {}
    for step in (0.01, 0.005):
        mesh = mesh_for_atom_at_step(11, _BOX, step)
        levels = _polished_levels(-1.0 / mesh.r, mesh, 0, N_MAX)
        raw[step] = np.array([n - 1 / math.sqrt(-2 * levels[n]) for n in range(5, 21)])
    ratio = raw[0.01] / raw[0.005]
    assert np.all(np.abs(ratio - 4.0) < 0.1)


@pytest.mark.parametrize("z,l", [(3, 0), (11, 0), (11, 1), (11, 2)])
def test_defect_is_flat_in_n(z, l):
    """The Rydberg formula's whole claim: delta barely moves along a series."""
    points = {p.n: p.defect.value for p in computed_defects(z, l)}
    assert abs(points[20] - points[10]) < 2e-3


def test_error_bar_brackets_a_finer_mesh():
    fine = _corrected(11, 0, 0.0025)
    for p in computed_defects(11, 0):
        if p.n >= 4:
            assert abs(p.defect.value - fine[p.n][1]) <= p.defect.provenance.error_estimate


def test_every_level_is_found_once():
    for l in range(4):
        assert [p.n for p in computed_defects(11, l)] == list(range(l + 1, N_MAX + 1))


def test_measured_sodium_matches_textbook_defects():
    by_l = {l: {m.n: m for m in measured_defects("Na", l)} for l in range(4)}
    assert by_l[0][10].defect == pytest.approx(1.3488, abs=1e-3)
    assert by_l[1][10].defect == pytest.approx(0.8562, abs=1e-3)
    assert by_l[2][10].defect == pytest.approx(0.0145, abs=1e-3)
    assert by_l[3][9].defect == pytest.approx(0.0017, abs=3e-4)


def test_gsz_sits_close_to_sodium_with_the_gap_disclosed():
    """GSZ overshoots s and p by about 0.035: an APPROXIMATION, labelled so."""
    for l in (0, 1):
        c = {p.n: p.defect for p in computed_defects(11, l)}
        m = {x.n: x.defect for x in measured_defects("Na", l)}
        assert c[10].provenance.fidelity is Fidelity.APPROXIMATION
        assert 0 < c[10].value - m[10] < 0.04


def test_endpoint():
    client = TestClient(create_app())
    body = client.get("/api/rydberg/na").json()
    assert body["symbol"] == "Na"
    assert [s["l"] for s in body["series"]] == [0, 1, 2, 3]
    s = body["series"][0]
    assert s["computed"][0]["n"] == 1
    assert s["measured"][0]["n"] == 3
    assert client.get("/api/rydberg/k").status_code == 422
