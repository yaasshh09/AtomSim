import math

import numpy as np
import pytest

from atomsim.hybrids import HYBRIDS, evaluate_hybrid, hybrid, hybrid_plane_grid, sample_hybrid
from atomsim.provenance import Fidelity

FAMILIES = {"sp": 2, "sp2": 3, "sp3": 4}


def test_each_family_is_orthonormal():
    for fam, k in FAMILIES.items():
        hs = [np.array(hybrid(f"{fam}-{i}").coefficients) for i in range(1, k + 1)]
        gram = np.array([[a @ b for b in hs] for a in hs])
        assert np.allclose(gram, np.eye(k), atol=1e-12), fam


def test_sp3_lobes_sit_at_the_tetrahedral_angle():
    dirs = [np.array(HYBRIDS[f"sp3-{i}"][1]) for i in range(1, 5)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert math.degrees(math.acos(dirs[i] @ dirs[j])) == pytest.approx(109.4712, abs=1e-3)


@pytest.mark.parametrize(
    "key,dipole",
    [("sp-1", 3.0), ("sp3-3", 1.5 * math.sqrt(3)), ("sp2-2", 2 * math.sqrt(2))],
)
def test_samples_are_drawn_from_h_squared_and_the_lobe_points_along_n(key, dipole):
    """<r> = 2 c_s c_p <2s|r.n|2p_n> n, with <2s|z|2p0> = -3 a0 and c_s < 0.

    That is a closed-form first moment of |h|^2, and the sampler has to hit it:
    sp 3 bohr, sp2 2 sqrt(2), sp3 3 sqrt(3)/2, all along the lobe direction.
    """
    cloud = sample_hybrid(key, 60_000, seed=3)
    assert cloud.positions.shape == (60_000, 3)
    mean = cloud.positions.astype(np.float64).mean(axis=0)
    spread = cloud.positions.astype(np.float64).std(axis=0) / math.sqrt(60_000)
    expected = dipole * np.array(HYBRIDS[key][1])
    assert np.all(np.abs(mean - expected) < 5 * spread + 1e-9), (mean, expected)
    assert cloud.provenance.fidelity is Fidelity.NUMERICAL


def test_the_plane_is_the_closed_form_density_and_exact():
    pg = hybrid_plane_grid("sp2-2", resolution=33)
    i, j = 7, 25
    pos = np.array([[pg.axis[j], 0.0, pg.axis[i]]])
    assert pg.values[i, j] == pytest.approx(evaluate_hybrid("sp2-2", pos)[0] ** 2, rel=1e-12)
    assert pg.provenance.fidelity is Fidelity.EXACT
    assert any("valence-bond bookkeeping" in a for a in pg.provenance.assumptions)


def test_unknown_hybrids_are_refused():
    with pytest.raises(ValueError):
        hybrid("sp4-1")


def test_hybrids_travel_through_the_sample_and_plane_jobs():
    from fastapi.testclient import TestClient

    from atomsim.server.app import create_app

    with TestClient(create_app()) as c:
        def run(path, body):
            job = c.post(path, json=body)
            assert job.status_code == 200, job.text
            jid = job.json()["id"]
            for _ in range(2000):
                if c.get(f"/api/jobs/{jid}").json()["status"] in ("done", "error"):
                    break
            return c.get(f"/api/jobs/{jid}/meta").json()

        base = {"n": 2, "l": 1, "m": 0, "basis": "real", "hybrid": "sp3-1"}
        meta = run("/api/jobs/sample", {**base, "count": 2000})
        assert meta["hybrid"] == "sp3-1"
        assert any("degenerate" in a for a in meta["provenance"]["assumptions"])
        plane = run("/api/jobs/plane", {**base, "resolution": 32})
        assert plane["hybrid"] == "sp3-1" and plane["provenance"]["fidelity"] == "exact"
        # Refused before any job exists: an unknown key, and a non-degenerate atom.
        assert c.post("/api/jobs/sample", json={**base, "hybrid": "sp4-1"}).status_code == 422
        assert c.post("/api/jobs/plane", json={**base, "system": "na"}).status_code == 400
