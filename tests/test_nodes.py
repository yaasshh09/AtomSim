import math

import numpy as np
import pytest

from atomsim.analytic.nodes import nodes
from atomsim.analytic.wavefunction import evaluate_state

CASES = [(n, l, m) for n in range(1, 6) for l in range(n) for m in range(-l, l + 1)]


def _psi(n, l, m, pts, basis, Z=1, mu=1.0):
    pts = np.asarray(pts, dtype=float)
    return evaluate_state(n, l, m, pts, Z=Z, mu_ratio=mu, basis=basis).values


def _at(r, th, ph):
    return [r * math.sin(th) * math.cos(ph), r * math.sin(th) * math.sin(ph), r * math.cos(th)]


@pytest.mark.parametrize("n,l,m", CASES)
@pytest.mark.parametrize("basis", ["complex", "real"])
def test_psi_vanishes_on_every_reported_node_and_the_counts_are_right(n, l, m, basis):
    ns = nodes(n, l, m, Z=2, mu_ratio=0.9, basis=basis)
    assert len(ns.radial) == n - l - 1
    assert len(ns.cones) == l - abs(m)
    assert len(ns.planes) == (abs(m) if basis == "real" else 0)
    assert ns.z_axis == (m != 0)
    # A generic direction, off every angular node, and a generic radius.
    th, ph, r0 = 1.1, 0.37, 1.7
    ray = [_at(r, th, ph) for r in np.linspace(0.1, 4 * n * n, 400)]
    scale = np.max(np.abs(_psi(n, l, m, ray, basis, 2, 0.9)))
    for rq in ns.radial:
        r = rq.value
        p = [r * math.sin(th) * math.cos(ph), r * math.sin(th) * math.sin(ph), r * math.cos(th)]
        assert abs(_psi(n, l, m, [p], basis, 2, 0.9)[0]) < 1e-9 * scale
    for c in ns.cones:
        t = c.value
        p = [r0 * math.sin(t) * math.cos(ph), r0 * math.sin(t) * math.sin(ph), r0 * math.cos(t)]
        assert abs(_psi(n, l, m, [p], basis, 2, 0.9)[0]) < 1e-9 * scale
    for pl in ns.planes:
        f = pl.value
        p = [r0 * math.sin(th) * math.cos(f), r0 * math.sin(th) * math.sin(f), r0 * math.cos(th)]
        assert abs(_psi(n, l, m, [p], basis, 2, 0.9)[0]) < 1e-9 * scale
    if ns.z_axis:
        assert abs(_psi(n, l, m, [[0, 0, r0]], basis, 2, 0.9)[0]) < 1e-12 * scale


def test_p_orbitals_put_their_planes_where_their_names_say():
    px, py = nodes(2, 1, 1, basis="real"), nodes(2, 1, -1, basis="real")
    assert px.planes[0].value == pytest.approx(math.pi / 2)  # p_x vanishes on x = 0
    assert py.planes[0].value == pytest.approx(0.0)  # p_y vanishes on y = 0
    assert nodes(3, 2, 0).cones[0].value == pytest.approx(math.acos(1 / math.sqrt(3)))  # d_z2


def test_the_nodes_endpoint_is_hydrogenic_only():
    from fastapi.testclient import TestClient

    from atomsim.server.app import create_app

    with TestClient(create_app()) as c:
        body = c.get("/api/nodes/3/1/1?basis=real&system=he%2B").json()
        assert [len(body[k]) for k in ("radial", "cones", "planes")] == [1, 0, 1]
        assert body["provenance"]["fidelity"] == "exact"
        assert c.get("/api/nodes/3/1/1?system=na").status_code == 400
        assert c.get("/api/nodes/3/1/2").status_code == 422
