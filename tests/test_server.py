import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import atomsim
from atomsim.constants import ALPHA
from atomsim.server.app import create_app
from atomsim.server.jobs import JobStatus


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def _wait_done(client, job_id, deadline_s=30.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline_s:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish")


def test_health_reports_version(client):
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "version": atomsim.__version__}


def test_state_carries_exact_provenance(client):
    r = client.get("/api/state/2/1/0")
    assert r.status_code == 200
    body = r.json()
    # mu-aware since M2: the M1 endpoint silently assumed infinite nuclear mass
    assert body["energy"]["value"] == pytest.approx(-0.125 * 0.9994557, rel=1e-6)
    assert body["energy"]["unit"] == "hartree"
    assert body["energy"]["provenance"]["fidelity"] == "exact"
    assert body["energy_ev"]["unit"] == "eV"
    assert body["energy_ev"]["value"] == pytest.approx(-3.40, abs=0.01)
    assert body["mean_radius"]["value"] == pytest.approx(5.0 / 0.9994557, rel=1e-6)


def test_state_rejects_invalid_quantum_numbers(client):
    assert client.get("/api/state/1/1/0").status_code == 422   # l == n
    assert client.get("/api/state/2/1/2").status_code == 422   # |m| > l
    assert client.get("/api/state/0/0/0").status_code == 422   # n < 1


def test_sample_job_end_to_end(client):
    r = client.post(
        "/api/jobs/sample", json={"n": 1, "l": 0, "m": 0, "count": 5000, "seed": 7}
    )
    assert r.status_code == 200
    job_id = r.json()["id"]

    final = _wait_done(client, job_id)
    assert final["status"] == "done"
    assert final["progress"] == pytest.approx(1.0)

    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["count"] == 5000
    assert meta["dtype"] == "float32"
    assert meta["layout"] == "xyz-interleaved"
    assert meta["unit"] == "bohr"
    assert meta["provenance"]["fidelity"] == "numerical"

    raw = client.get(f"/api/jobs/{job_id}/data")
    assert raw.headers["content-type"].startswith("application/octet-stream")
    positions = np.frombuffer(raw.content, dtype=np.float32).reshape(-1, 3)
    assert positions.shape == (5000, 3)
    assert np.isfinite(positions).all()


def test_sample_rejects_invalid_body(client):
    assert (
        client.post("/api/jobs/sample", json={"n": 1, "l": 1, "m": 0}).status_code == 422
    )
    assert (
        client.post(
            "/api/jobs/sample", json={"n": 1, "l": 0, "m": 0, "count": 10}
        ).status_code
        == 422
    )  # below pydantic ge=1000


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404
    assert client.get("/api/jobs/deadbeef/meta").status_code == 404
    assert client.get("/api/jobs/deadbeef/data").status_code == 404


def test_meta_before_done_is_409(client):
    job = client.app.state.jobs.create()  # created but never run
    assert client.get(f"/api/jobs/{job.id}/meta").status_code == 409
    assert client.get(f"/api/jobs/{job.id}/data").status_code == 409


def test_websocket_streams_progress_to_done(client):
    r = client.post(
        "/api/jobs/sample", json={"n": 2, "l": 1, "m": 0, "count": 20000, "seed": 1}
    )
    job_id = r.json()["id"]
    messages = []
    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        for _ in range(600):
            msg = ws.receive_json()
            messages.append(msg)
            if msg["status"] in ("done", "error"):
                break
    assert messages[-1]["status"] == "done"
    assert messages[-1]["progress"] == pytest.approx(1.0)


def test_job_error_surfaces_via_status(client):
    jobs = client.app.state.jobs
    job = jobs.create()

    def bad(progress):
        raise RuntimeError("sampler exploded")

    jobs.run(job.id, bad)
    body = client.get(f"/api/jobs/{job.id}").json()
    assert body["status"] == "error"
    assert "sampler exploded" in body["error"]
    assert body["status"] == JobStatus.ERROR.value


def test_systems_endpoint_lists_presets(client):
    body = client.get("/api/systems").json()
    keys = [s["key"] for s in body["systems"]]
    # Hydrogenic presets stay first and unchanged; screened atoms are appended.
    assert keys[:6] == ["h", "d", "t", "mu-h", "ps", "he+"]
    mu = body["systems"][0]["mu_ratio"]
    assert mu["provenance"]["fidelity"] == "exact"
    assert "CODATA" in mu["provenance"]["method"]


def test_state_accepts_system_and_fine_structure(client):
    body = client.get("/api/state/2/1/0?system=he%2B&fine_structure=true").json()
    assert body["system"]["key"] == "he+"
    assert body["energy"]["value"] == pytest.approx(-0.5 * 0.999863, rel=1e-4)
    js = sorted(lvl["j"] for lvl in body["levels"])
    assert js == [0.5, 1.5]
    assert body["levels"][0]["shift"]["provenance"]["fidelity"] == "approximation"


def test_state_defaults_stay_hydrogen_gross(client):
    body = client.get("/api/state/2/1/0").json()
    assert body["system"]["key"] == "h"
    assert body["levels"] == []
    # mu-scaled now: -0.125 * mu'
    assert body["energy"]["value"] == pytest.approx(-0.125 * 0.9994557, rel=1e-6)


def test_state_unknown_system_is_422(client):
    assert client.get("/api/state/1/0/0?system=xenon").status_code == 422


def test_radial_endpoint_fields(client):
    body = client.get("/api/radial/2/1?points=200").json()
    rw, rp = body["r_wavefunction"], body["radial_probability"]
    assert len(rw["grid"]) == 200 and len(rw["values"]) == 200
    assert rw["provenance"]["fidelity"] == "exact"
    p = np.array(rp["values"])
    g = np.array(rp["grid"])
    assert np.trapezoid(p, g) == pytest.approx(1.0, abs=5e-3)  # P(r) normalized


def test_spectrum_endpoint_with_comparison(client):
    body = client.get("/api/spectrum?system=h&n_max=6").json()
    assert body["reference_citation"] and "NIST" in body["reference_citation"]
    assert len(body["lines"]) > 10
    assert body["comparison"] is not None
    assert all(c["within_tolerance"] for c in body["comparison"])


def test_spectrum_without_reference_data(client):
    body = client.get("/api/spectrum?system=ps&n_max=3").json()
    assert body["comparison"] is None
    assert body["reference_citation"] is None
    assert len(body["lines"]) > 0


def test_sample_job_real_basis_and_system(client):
    r = client.post(
        "/api/jobs/sample",
        json={"n": 2, "l": 1, "m": 1, "count": 5000, "seed": 3,
              "basis": "real", "system": "d"},
    )
    job_id = r.json()["id"]
    final = _wait_done(client, job_id)
    assert final["status"] == "done"
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["basis"] == "real"
    assert meta["system"] == "d"


def test_sample_job_rejects_bad_basis(client):
    r = client.post(
        "/api/jobs/sample",
        json={"n": 1, "l": 0, "m": 0, "basis": "cartoon"},
    )
    assert r.status_code == 422


def test_sample_channels_complex_basis(client):
    r = client.post(
        "/api/jobs/sample", json={"n": 2, "l": 1, "m": 1, "count": 4000, "seed": 5}
    )
    job_id = r.json()["id"]
    assert _wait_done(client, job_id)["status"] == "done"

    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["kind"] == "sample"
    names = [c["name"] for c in meta["channels"]]
    assert names == ["positions", "density", "phase"]
    assert all(c["dtype"] == "float32" for c in meta["channels"])
    assert meta["channels"][1]["unit"] == "bohr^-3"
    assert meta["channels"][2]["unit"] == "rad"
    assert meta["channels"][2]["provenance"]["fidelity"] == "exact"

    density = np.frombuffer(
        client.get(f"/api/jobs/{job_id}/data?channel=density").content, dtype=np.float32
    )
    phase = np.frombuffer(
        client.get(f"/api/jobs/{job_id}/data?channel=phase").content, dtype=np.float32
    )
    assert density.shape == (4000,)
    assert (density >= 0.0).all() and density.max() > 0.0
    assert phase.shape == (4000,)
    assert (np.abs(phase) <= np.pi + 1e-6).all()

    # wiring check: channels must equal a direct re-evaluation at the sampled points
    # (with the H system's reduced mass, exactly as the server evaluates)
    from atomsim.analytic.wavefunction import evaluate_state
    from atomsim.systems import get_system

    xyz = np.frombuffer(
        client.get(f"/api/jobs/{job_id}/data").content, dtype=np.float32
    ).reshape(-1, 3)
    mu = get_system("h").mu_ratio.value
    psi = evaluate_state(2, 1, 1, xyz.astype(np.float64), mu_ratio=mu).values
    assert np.allclose(density, (np.abs(psi) ** 2).astype(np.float32), rtol=1e-5, atol=0)
    assert np.allclose(phase, np.angle(psi).astype(np.float32), atol=1e-5)


def test_sample_channels_real_basis_has_no_phase(client):
    r = client.post(
        "/api/jobs/sample",
        json={"n": 2, "l": 1, "m": 1, "count": 2000, "seed": 5, "basis": "real"},
    )
    job_id = r.json()["id"]
    assert _wait_done(client, job_id)["status"] == "done"
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert [c["name"] for c in meta["channels"]] == ["positions", "density"]
    assert client.get(f"/api/jobs/{job_id}/data?channel=phase").status_code == 422
    assert client.get(f"/api/jobs/{job_id}/data?channel=vibes").status_code == 422


def test_levels_endpoint_gross(client):
    body = client.get("/api/levels?n_max=3").json()
    assert body["n_max"] == 3 and body["fine"] is None
    assert [g["n"] for g in body["gross"]] == [1, 2, 3]
    assert [g["degeneracy"] for g in body["gross"]] == [2, 8, 18]
    e1 = body["gross"][0]["energy"]
    assert e1["unit"] == "hartree"
    assert e1["value"] == pytest.approx(-0.4997278, rel=1e-5)  # reduced-mass H
    assert body["gross"][0]["energy_ev"]["unit"] == "eV"


def test_levels_endpoint_fine_structure(client):
    body = client.get("/api/levels?n_max=2&fine_structure=true").json()
    fine = body["fine"]
    assert [(f["n"], f["l"], f["j"]) for f in fine] == [
        (1, 0, 0.5), (2, 0, 0.5), (2, 1, 0.5), (2, 1, 1.5),
    ]
    for f in fine:
        assert f["shift"]["provenance"]["fidelity"] == "approximation"
        assert f["shift_ev"]["unit"] == "eV"
    # 2p_1/2 lies below 2p_3/2
    assert fine[2]["energy"]["value"] < fine[3]["energy"]["value"]


def test_levels_endpoint_validation(client):
    assert client.get("/api/levels?n_max=0").status_code == 422
    assert client.get("/api/levels?n_max=11").status_code == 422
    assert client.get("/api/levels?system=unobtainium").status_code == 422


def test_state_readouts(client):
    body = client.get("/api/state/3/2/1").json()
    assert body["angular_momentum"]["value"] == pytest.approx(6.0**0.5)
    assert body["angular_momentum"]["unit"] == "hbar"
    assert body["radial_nodes"] == 0
    assert body["angular_nodes"] == 2
    r_bohr = body["mean_radius"]["value"]
    r_pm = body["mean_radius_pm"]["value"]
    assert body["mean_radius_pm"]["unit"] == "pm"
    assert r_pm == pytest.approx(r_bohr * 52.9177, rel=1e-4)


def test_state_fine_structure_shift_ev(client):
    body = client.get("/api/state/2/1/0?fine_structure=true").json()
    assert len(body["levels"]) == 2
    for lev in body["levels"]:
        assert lev["shift_ev"]["unit"] == "eV"


def test_plane_job_end_to_end(client):
    r = client.post(
        "/api/jobs/plane",
        json={"n": 2, "l": 1, "m": 0, "resolution": 64},
    )
    assert r.status_code == 200
    job_id = r.json()["id"]
    assert _wait_done(client, job_id)["status"] == "done"

    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["kind"] == "plane"
    assert meta["resolution"] == 64
    assert meta["quantity"] == "density"
    assert meta["unit"] == "bohr^-3"
    assert meta["half_extent"] == pytest.approx(10.0 / 0.9994557, rel=1e-4)
    assert meta["provenance"]["fidelity"] == "exact"

    raw = client.get(f"/api/jobs/{job_id}/data").content
    assert len(raw) == 64 * 64 * 4
    values = np.frombuffer(raw, dtype=np.float32).reshape(64, 64)

    # deterministic: must equal a direct engine computation at the system's mu
    from atomsim.plane import plane_grid
    from atomsim.systems import get_system

    mu = get_system("h").mu_ratio.value
    expected = plane_grid(2, 1, 0, mu_ratio=mu, resolution=64).values.astype(np.float32)
    assert np.array_equal(values, expected)

    # plane jobs have exactly one channel
    assert client.get(f"/api/jobs/{job_id}/data?channel=density").status_code == 422


def test_plane_job_psi_quantity_and_validation(client):
    r = client.post(
        "/api/jobs/plane",
        json={"n": 2, "l": 1, "m": 0, "quantity": "psi", "resolution": 32},
    )
    job_id = r.json()["id"]
    assert _wait_done(client, job_id)["status"] == "done"
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["quantity"] == "psi"
    assert meta["unit"] == "bohr^-3/2"

    assert client.post(
        "/api/jobs/plane", json={"n": 1, "l": 0, "m": 0, "resolution": 4096}
    ).status_code == 422
    assert client.post(
        "/api/jobs/plane", json={"n": 1, "l": 0, "m": 0, "quantity": "vibes"}
    ).status_code == 422
    assert client.post(
        "/api/jobs/plane", json={"n": 1, "l": 1, "m": 0}
    ).status_code == 422


def test_systems_carry_nuclear_radius(client):
    systems = {s["key"]: s for s in client.get("/api/systems").json()["systems"]}
    r_bohr = systems["h"]["nuclear_radius"]
    r_fm = systems["h"]["nuclear_radius_fm"]
    assert r_bohr["unit"] == "bohr"
    assert r_fm["unit"] == "fm"
    assert r_fm["value"] == pytest.approx(0.84075, rel=1e-6)
    assert r_fm["provenance"]["fidelity"] == "exact"
    assert "CODATA" in r_fm["provenance"]["method"]
    assert r_fm["provenance"]["error_estimate"] == pytest.approx(6.4e-4, rel=1e-2)
    # point lepton: honestly absent, not zero
    assert systems["ps"]["nuclear_radius"] is None
    assert systems["ps"]["nuclear_radius_fm"] is None
    # the embedded system on /api/state carries it too
    state_sys = client.get("/api/state/1/0/0?system=mu-h").json()["system"]
    assert state_sys["nuclear_radius_fm"]["value"] == pytest.approx(0.84075, rel=1e-6)


def test_levels_default_alpha_is_real_and_approximation(client):
    body = client.get("/api/levels?fine_structure=true").json()
    assert body["alpha"] == pytest.approx(ALPHA)
    assert body["fine"][0]["shift"]["provenance"]["fidelity"] == "approximation"


def test_levels_altered_alpha_is_counterfactual(client):
    real = client.get("/api/levels?fine_structure=true").json()
    alt = client.get("/api/levels?fine_structure=true&alpha=0.05").json()
    assert alt["alpha"] == pytest.approx(0.05)
    assert alt["fine"][0]["shift"]["provenance"]["fidelity"] == "counterfactual"
    # bigger alpha -> deeper (more negative) fine shift
    assert abs(alt["fine"][0]["shift"]["value"]) > abs(real["fine"][0]["shift"]["value"])


def test_levels_generic_z_system_resolves(client):
    body = client.get("/api/levels?system=z3&fine_structure=true").json()
    assert body["system"]["z"] == 3


def test_levels_rejects_bad_alpha_and_z(client):
    assert client.get("/api/levels?alpha=0").status_code == 422
    assert client.get("/api/levels?alpha=0.6").status_code == 422
    assert client.get("/api/levels?system=z0").status_code == 422
    assert client.get("/api/levels?system=z99").status_code == 422


def test_constants_default_is_real_and_exact(client):
    body = client.get("/api/constants").json()
    assert body["altered"] is False
    assert body["alpha"]["quantity"]["value"] == pytest.approx(ALPHA, rel=1e-6)
    assert body["alpha"]["quantity"]["provenance"]["fidelity"] == "exact"
    assert body["alpha"]["changed"] is False


def test_constants_degeneracy_pair_changes_nothing_observable(client):
    body = client.get("/api/constants?e=2&eps0=4").json()
    assert body["altered"] is True
    assert body["alpha"]["changed"] is False
    assert body["bohr_radius_pm"]["changed"] is False
    assert body["hartree_ev"]["changed"] is False
    assert body["alpha"]["quantity"]["provenance"]["fidelity"] == "counterfactual"


def test_constants_electron_mass_scales_size_and_binding(client):
    body = client.get("/api/constants?m_e=0.5").json()
    assert body["bohr_radius_pm"]["ratio"] == pytest.approx(2.0)
    assert body["hartree_ev"]["ratio"] == pytest.approx(0.5)


def test_constants_rejects_out_of_range(client):
    assert client.get("/api/constants?e=0.1").status_code == 422
    assert client.get("/api/constants?hbar=5").status_code == 422


def test_classical_hydrogen_ground(client):
    r = client.get("/api/classical?system=h&n=1")
    assert r.status_code == 200
    body = r.json()
    assert body["z"] == 1
    assert body["collapse_time_s"]["value"] == pytest.approx(1.556e-11, rel=0.02)
    assert body["collapse_time_s"]["provenance"]["fidelity"] == "counterfactual"
    assert body["orbits"][0]["radius_bohr"]["provenance"]["fidelity"] == "approximation"


def test_classical_orbits_cover_1_to_n(client):
    body = client.get("/api/classical?system=h&n=3").json()
    assert [o["n"] for o in body["orbits"]] == [1, 2, 3]


def test_classical_rejects_n_below_one(client):
    assert client.get("/api/classical?system=h&n=0").status_code == 422


def test_forcelaw_rejects_out_of_range_p(client):
    assert client.get("/api/forcelaw?p=2.5&l=0").status_code == 422
    assert client.get("/api/forcelaw?p=0.1&l=0").status_code == 422


def test_forcelaw_backcompat_defaults_to_powerlaw(client):
    r = client.get("/api/forcelaw?p=1.0&l=0&system=h&n_states=3")
    assert r.status_code == 200
    body = r.json()
    assert body["preset"] == "powerlaw"
    assert body["reference"]["kind"] == "levels"
    assert len(body["counterfactual"]) == 3
    assert body["potential_curve"]["v_ev"]


def test_forcelaw_yukawa(client):
    r = client.get("/api/forcelaw?preset=yukawa&lambda=3.0&l=0&system=h&n_states=4")
    assert r.status_code == 200
    body = r.json()
    assert body["preset"] == "yukawa"
    assert body["params"]["lambda"] == 3.0
    assert body["bound_count"] <= body["requested_count"]


def test_forcelaw_harmonic_reference_is_levels(client):
    r = client.get("/api/forcelaw?preset=harmonic&omega=0.5&l=0&n_states=2")
    assert r.status_code == 200
    body = r.json()
    assert body["reference"]["kind"] == "levels"
    assert body["reference"]["items"][0]["energy"]["provenance"]["fidelity"] == "exact"


def test_forcelaw_finitewell_markers(client):
    r = client.get("/api/forcelaw?preset=finitewell&v0=2.0&a=3.0&l=0&n_states=3")
    assert r.status_code == 200
    assert r.json()["reference"]["kind"] == "markers"


def test_forcelaw_rejects_unknown_preset(client):
    assert client.get("/api/forcelaw?preset=bogus&l=0").status_code == 422


def test_forcelaw_rejects_out_of_range_param(client):
    assert client.get("/api/forcelaw?preset=yukawa&lambda=999&l=0").status_code == 422


def test_forcelaw_rejects_negative_l(client):
    assert client.get("/api/forcelaw?preset=harmonic&omega=0.5&l=-1").status_code == 422


# --- Phase 6: screened atoms ---

def test_systems_list_includes_screened_atoms(client):
    body = client.get("/api/systems").json()
    keys = {s["key"]: s for s in body["systems"]}
    assert "na" in keys
    assert keys["na"]["kind"] == "screened"
    assert keys["na"]["n_electrons"] == 11
    assert keys["h"]["kind"] == "hydrogenic"


def test_systems_list_excludes_unsourced_sulfur_chlorine(client):
    keys = {s["key"] for s in client.get("/api/systems").json()["systems"]}
    assert "s" not in keys and "cl" not in keys  # no published GSZ parameters


def test_cloud_job_screened_atom_end_to_end(client):
    r = client.post(
        "/api/jobs/sample",
        json={"n": 3, "l": 0, "m": 0, "system": "na", "count": 2000, "seed": 4},
    )
    assert r.status_code == 200
    job_id = r.json()["id"]
    final = _wait_done(client, job_id)
    assert final["status"] == "done"
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["count"] == 2000
    assert meta["provenance"]["fidelity"] == "approximation"
    raw = client.get(f"/api/jobs/{job_id}/data")
    positions = np.frombuffer(raw.content, dtype=np.float32).reshape(-1, 3)
    assert positions.shape == (2000, 3)
    assert np.isfinite(positions).all()


def test_plane_job_screened_atom_end_to_end(client):
    r = client.post(
        "/api/jobs/plane",
        json={"n": 3, "l": 0, "m": 0, "system": "na", "quantity": "psi", "resolution": 64},
    )
    assert r.status_code == 200
    job_id = r.json()["id"]
    final = _wait_done(client, job_id)
    assert final["status"] == "done"
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["quantity"] == "psi"
    assert meta["provenance"]["fidelity"] == "approximation"
    raw = client.get(f"/api/jobs/{job_id}/data")
    grid = np.frombuffer(raw.content, dtype=np.float32)
    assert grid.size == 64 * 64
    assert np.isfinite(grid).all()


def test_levels_screened_lithium_lifts_degeneracy(client):
    body = client.get("/api/levels?system=li").json()
    assert body["system"]["kind"] == "screened"
    assert body["is_ground"] is True
    e = {(o["n"], o["l"]): o["energy"]["value"] for o in body["orbitals"]}
    assert e[(2, 0)] < e[(2, 1)]  # 2s below 2p
    assert any(o["occupancy"] > 0 for o in body["orbitals"])
    # Orbital energies are APPROXIMATION and carry an eV conversion.
    val = next(o for o in body["orbitals"] if o["occupancy"] > 0)
    assert val["energy"]["provenance"]["fidelity"] == "approximation"
    assert val["energy_ev"]["unit"] == "eV"


def test_levels_screened_bad_config_422(client):
    assert client.get("/api/levels?system=li&config=1s5").status_code == 422


def test_levels_screened_wrong_electron_count_422(client):
    # A valid config but wrong electron count for the element.
    assert client.get("/api/levels?system=li&config=1s2").status_code == 422


def test_levels_screened_excited_config_not_ground(client):
    body = client.get("/api/levels?system=na&config=1s2 2s2 2p6 3p1").json()
    assert body["is_ground"] is False
    assert body["config"] == "1s2 2s2 2p6 3p1"


def test_levels_hydrogenic_unchanged(client):
    body = client.get("/api/levels?system=h&n_max=3").json()
    assert body["system"]["kind"] == "hydrogenic"
    assert body["gross"][0]["degeneracy"] == 2


def test_radial_screened_sodium_3s(client):
    body = client.get("/api/radial/3/0?system=na&points=300").json()
    assert body["system"]["kind"] == "screened"
    assert len(body["radial_probability"]["values"]) == 300
    prov = body["r_wavefunction"]["provenance"]
    assert prov["fidelity"] == "approximation"
    assert "Green" in prov["method"] or "GSZ" in prov["method"]


def test_radial_hydrogenic_unchanged(client):
    body = client.get("/api/radial/2/1?system=h&points=200").json()
    assert body["system"]["kind"] == "hydrogenic"
    assert body["r_wavefunction"]["provenance"]["fidelity"] == "exact"


def test_spectrum_screened_sodium_returns_dipole_lines(client):
    body = client.get("/api/spectrum?system=na").json()
    assert body["system"]["kind"] == "screened"
    assert body["fine_structure"] is False
    assert body["lines"]  # non-empty dipole-allowed set
    for ln in body["lines"]:
        assert abs(ln["l_upper"] - ln["l_lower"] ) == 1
        assert ln["energy_ev"]["provenance"]["fidelity"] == "approximation"
    # Na I NIST reference is vendored: the comparison + citation reach the client.
    assert body["reference_citation"] is not None and "NIST" in body["reference_citation"]
    assert body["comparison"]  # the D line and other transitions are associated
    assert body["tolerance_relative"] == 0.05


def test_spectrum_hydrogenic_unchanged(client):
    body = client.get("/api/spectrum?system=h&n_max=3").json()
    assert body["system"]["kind"] == "hydrogenic"
    assert body["reference_citation"] is not None  # H I NIST data still compared


def test_forcelaw_custom_recovers_hydrogen(client):
    r = client.get("/api/forcelaw", params={"preset": "custom", "expr": "-1/r", "l": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["expression"] == "-1/r"
    assert body["preset"] == "custom"
    assert body["counterfactual"][0]["trusted"] is True
    assert body["counterfactual"][0]["energy"]["provenance"]["fidelity"] == "numerical"


def test_forcelaw_custom_missing_expr_is_422(client):
    r = client.get("/api/forcelaw", params={"preset": "custom", "l": 0})
    assert r.status_code == 422


def test_forcelaw_custom_rejects_unsafe_expr(client):
    r = client.get("/api/forcelaw", params={"preset": "custom", "expr": "r.__class__"})
    assert r.status_code == 422
    assert "not allowed" in r.json()["detail"]


def test_levels_dirac_is_exact_and_degenerate(client):
    r = client.get("/api/levels", params={"system": "h", "n_max": 3, "dirac": "true"})
    assert r.status_code == 200
    body = r.json()
    assert body["dirac"] is True
    fine = body["fine"]
    assert fine[0]["energy"]["provenance"]["fidelity"] == "exact"
    # 2s1/2 (l=0,j=0.5) and 2p1/2 (l=1,j=0.5) must share one Dirac energy
    n2 = [f for f in fine if f["n"] == 2 and f["j"] == 0.5]
    assert len(n2) == 2
    assert n2[0]["energy"]["value"] == pytest.approx(n2[1]["energy"]["value"], abs=1e-14)


def test_levels_perturbative_still_default(client):
    r = client.get("/api/levels", params={"system": "h", "n_max": 2, "fine_structure": "true"})
    body = r.json()
    assert body["dirac"] is False
    assert body["fine"][0]["energy"]["provenance"]["fidelity"] == "approximation"


def test_levels_dirac_supercritical_is_422(client):
    r = client.get("/api/levels", params={"system": "z200", "n_max": 1, "dirac": "true"})
    assert r.status_code == 422
