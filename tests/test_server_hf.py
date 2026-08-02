"""The Hartree-Fock server surface.

The solve takes seconds, so it runs as a background job like sampling and
plane grids, and reuses their endpoints rather than growing a parallel set:
POST /api/jobs/hf starts it, /api/jobs/{id} reports status, /api/jobs/{id}/meta
returns the energies with their provenance, and /api/jobs/{id}/data serves the
orbital amplitudes as raw float32.

Most of what is checked here is that provenance survives the trip. An energy
that arrives in the browser without its fidelity tier is exactly the quiet lie
this project exists to prevent, and the boundary is where it would be lost.
"""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def _run(client, **body):
    """POST a solve, wait for it, return (job body, meta body)."""
    r = client.post("/api/jobs/hf", json=body)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    else:
        raise TimeoutError(f"job {job_id} did not finish")
    assert status["status"] == "done", status["error"]
    meta = client.get(f"/api/jobs/{job_id}/meta")
    assert meta.status_code == 200, meta.text
    return job_id, meta.json()


def test_hartree_fock_runs_as_a_job_not_a_blocking_request(client):
    r = client.post("/api/jobs/hf", json={"z": 2})
    assert r.status_code == 200
    body = r.json()
    assert "id" in body
    assert body["status"] in ("pending", "running", "done")


def test_the_energy_carries_its_fidelity_to_the_browser(client):
    _, meta = _run(client, z=2)
    assert meta["kind"] == "hf"
    assert meta["total_energy"]["provenance"]["fidelity"] == "approximation"
    assert meta["total_energy"]["unit"] == "hartree"
    assert meta["total_energy"]["value"] == pytest.approx(-2.8617, abs=1e-3)


def test_the_diagnostics_are_labelled_numerical_not_approximation(client):
    """The virial ratio is a property of the converged solve, not of helium.

    If it arrived tagged APPROXIMATION a view would be entitled to draw it as
    physics, so the tier is the thing keeping it honest and it is worth a test
    of its own.
    """
    _, meta = _run(client, z=2)
    for key in ("virial_ratio", "kinetic", "potential"):
        assert meta[key]["provenance"]["fidelity"] == "numerical", key
    assert meta["virial_ratio"]["value"] == pytest.approx(2.0, rel=1e-3)


def test_the_error_estimate_survives_the_boundary(client):
    _, meta = _run(client, z=2)
    err = meta["total_energy"]["provenance"]["error_estimate"]
    assert err is not None and err > 0.0


def test_convergence_fields_reach_the_client(client):
    _, meta = _run(client, z=2)
    assert meta["converged"] is True
    assert meta["iterations"] > 0
    assert meta["coarse_iterations"] > 0


def test_energies_arrive_in_ev_as_well_as_hartree(client):
    _, meta = _run(client, z=2)
    ev = meta["total_energy_ev"]
    assert ev["unit"] == "eV"
    assert ev["value"] == pytest.approx(meta["total_energy"]["value"] * 27.2114, rel=1e-4)


def test_the_orbital_amplitudes_come_back_as_binary(client):
    job_id, meta = _run(client, z=4)
    labels = [o["label"] for o in meta["orbitals"]]
    assert labels == ["1s", "2s"]

    grid = client.get(f"/api/jobs/{job_id}/data?channel=grid")
    assert grid.status_code == 200
    r = np.frombuffer(grid.content, dtype=np.float32)
    assert r.size == meta["grid_points"]
    assert np.all(np.diff(r) > 0)

    for orbital in meta["orbitals"]:
        raw = client.get(f"/api/jobs/{job_id}/data?channel={orbital['channel']}")
        assert raw.status_code == 200
        p = np.frombuffer(raw.content, dtype=np.float32)
        assert p.size == r.size
        # P = r*R, normalized to 1 under int P^2 dr. It vanishes at the outer
        # wall, but NOT at the inner one: the exponential mesh starts at a
        # small nonzero r_min, and P(r_min) ~ r_min * R(0) is a few parts in a
        # thousand for beryllium's 1s rather than zero. Asserting zero there
        # would be asserting that the mesh reaches the origin, which it does
        # not and deliberately does not.
        assert abs(p[-1]) < 1e-4
        assert abs(p[0]) < 0.05 * np.abs(p).max()
        assert np.trapezoid(p.astype(np.float64) ** 2, r.astype(np.float64)) == (
            pytest.approx(1.0, rel=1e-3)
        )


def test_an_unknown_channel_names_the_ones_that_exist(client):
    job_id, _ = _run(client, z=2)
    r = client.get(f"/api/jobs/{job_id}/data?channel=P_7f")
    assert r.status_code == 422
    assert "P_1s" in r.json()["detail"]


def test_a_neutral_alkali_beyond_argon_is_refused_with_the_reason(client):
    """Potassium is a well-posed request this solver cannot answer honestly.

    The refusal has to say which part is the problem, because the obvious
    reading - that Z is too large - is wrong, and a user who believes it would
    not try the ions that do work.
    """
    r = client.post("/api/jobs/hf", json={"z": 19})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "n = 4" in detail
    assert "n <= 3" in detail


def test_an_argon_like_ion_above_argon_is_accepted(client):
    """The other half of the claim the refusal makes.

    If this failed, the refusal message would be describing a limit the server
    does not actually have, which is its own kind of dishonesty.
    """
    _, meta = _run(client, z=20, n_electrons=18)
    assert meta["symbol"] is None  # past the preset table, and says so
    assert meta["z"] == 20 and meta["n_electrons"] == 18
    assert meta["virial_ratio"]["value"] == pytest.approx(2.0, rel=1e-3)


def test_z_beyond_the_tested_range_is_refused(client):
    r = client.post("/api/jobs/hf", json={"z": 60, "n_electrons": 18})
    assert r.status_code == 400
    assert "36" in r.json()["detail"]


def test_an_electron_count_that_contradicts_the_config_is_refused(client):
    r = client.post("/api/jobs/hf", json={"z": 10, "n_electrons": 8, "config": "1s2 2s2 2p6"})
    assert r.status_code == 400
    assert "10 electrons" in r.json()["detail"]


def test_a_malformed_config_is_a_different_status_than_an_unsupported_one(client):
    """422 means "not understood", 400 means "understood and declined"."""
    unreadable = client.post("/api/jobs/hf", json={"z": 10, "config": "1s2 2s9"})
    assert unreadable.status_code == 422
    declined = client.post("/api/jobs/hf", json={"z": 19})
    assert declined.status_code == 400


def test_an_excited_configuration_is_allowed_and_flagged_as_not_ground(client):
    _, meta = _run(client, z=4, config="1s2 2p2")
    assert meta["is_ground"] is False
    assert meta["config"] == "1s2 2p2"


def test_an_open_shell_discloses_the_configuration_average_and_neon_does_not(client):
    _, carbon = _run(client, z=6)
    _, neon = _run(client, z=10)
    assert "not per term" in " ".join(carbon["total_energy"]["provenance"]["assumptions"])
    assert "not per term" not in " ".join(neon["total_energy"]["provenance"]["assumptions"])


def test_the_neglected_relativity_is_quantified_for_a_heavy_atom(client):
    """Helium's is 0.005% and argon's 0.44%, so a phrase that reads the same in
    both would not be telling the reader anything."""
    _, argon = _run(client, z=18)
    joined = " ".join(argon["total_energy"]["provenance"]["assumptions"])
    assert "neglects relativity" in joined and "%" in joined

    _, helium = _run(client, z=2)
    assert "neglects relativity" not in " ".join(
        helium["total_energy"]["provenance"]["assumptions"]
    )


def test_a_solve_defaults_to_real_physics(client):
    """A client that has never heard of the toggle cannot get the
    counterfactual by accident."""
    _, meta = _run(client, z=4)
    assert meta["exchange"] is True
    assert meta["exchange_energy"] is None
    assert meta["total_energy"]["provenance"]["fidelity"] == "approximation"


def test_turning_exchange_off_arrives_labelled_counterfactual(client):
    _, meta = _run(client, z=4, exchange=False)
    assert meta["exchange"] is False
    assert meta["total_energy"]["provenance"]["fidelity"] == "counterfactual"
    # The orbital shapes are counterfactual too, and their channels say so.
    assert all(
        c["provenance"]["fidelity"] == "counterfactual"
        for c in meta["channels"]
    )


def test_the_counterfactual_solve_brings_the_comparison_with_it(client):
    """The exchange energy is a difference between two solves, so the server
    computes it from two solves it ran itself.

    A client that fetched the two energies from two jobs and subtracted would
    be free to difference results from different meshes and report the gap
    between two calculations as the gap between two models.
    """
    _, meta = _run(client, z=4, exchange=False)
    assert meta["exchange_energy"] is not None
    assert meta["exchange_energy"]["unit"] == "hartree"
    assert meta["exchange_energy"]["value"] < 0  # exchange stabilizes
    assert meta["exchange_energy"]["provenance"]["fidelity"] == "counterfactual"
    # And in eV, because that is what the view shows.
    assert meta["exchange_energy_ev"]["unit"] == "eV"
    assert meta["exchange_energy_ev"]["value"] < meta["exchange_energy"]["value"]


def test_helium_reports_an_exchange_energy_of_zero_rather_than_omitting_it(client):
    """Zero is the answer, not the absence of one.

    A closed 1s shell has no same-spin pair, so there is nothing for exchange
    to do. Dropping the field for helium would let a view show nothing where
    it should show a teaching moment.
    """
    _, meta = _run(client, z=2, exchange=False)
    assert meta["exchange_energy"]["value"] == 0.0


def test_the_disclosure_reaches_the_browser_intact(client):
    _, meta = _run(client, z=10, exchange=False)
    joined = " ".join(meta["total_energy"]["provenance"]["assumptions"]).lower()
    assert "distinguishable" in joined
    assert "pauli principle is not switched off" in joined


def test_the_orbital_amplitudes_still_come_back_as_float32(client):
    """The wrapper the job now returns must not have broken /data."""
    job_id, meta = _run(client, z=4, exchange=False)
    r = client.get(f"/api/jobs/{job_id}/data", params={"channel": "grid"})
    assert r.status_code == 200
    grid = np.frombuffer(r.content, dtype=np.float32)
    assert grid.size == meta["grid_points"]
    assert np.all(np.diff(grid) > 0)


# --------------------------------------------------------------------------
# Phase 24: the occupancy cap lifted as well
# --------------------------------------------------------------------------


def test_pauli_defaults_on_so_the_collapse_cannot_arrive_by_accident(client):
    _, meta = _run(client, z=4)
    assert meta["pauli"] is True
    assert meta["collapse"] is None


def test_pauli_off_with_exchange_on_is_refused_by_the_schema(client):
    """422, and the body says why.

    Not 400: 400 is the server declining a well-posed request, and this one is
    not well posed. There is no such model to decline.
    """
    r = client.post("/api/jobs/hf", json={"z": 4, "pauli": False, "exchange": True})
    assert r.status_code == 422
    assert "antisymmetry" in r.text


def test_pauli_off_collapses_the_configuration_to_one_orbital(client):
    _, meta = _run(client, z=10, pauli=False, exchange=False)
    assert meta["pauli"] is False
    assert meta["exchange"] is False
    assert meta["config"] == "1s10"
    assert len(meta["orbitals"]) == 1
    assert meta["orbitals"][0]["occupancy"] == 10
    # Ground for its own rule, which is the only rule in force here.
    assert meta["is_ground"] is True


def test_the_collapsed_energy_arrives_counterfactual_everywhere(client):
    _, meta = _run(client, z=10, pauli=False, exchange=False)
    assert meta["total_energy"]["provenance"]["fidelity"] == "counterfactual"
    assert all(
        c["provenance"]["fidelity"] == "counterfactual" for c in meta["channels"]
    )
    assert all(
        o["energy"]["provenance"]["fidelity"] == "counterfactual"
        for o in meta["orbitals"]
    )


def test_the_collapsed_solve_brings_the_real_atom_with_it(client):
    """Same reason the exchange energy travels with its solve.

    A view that fetched "the real neon" from one job and "collapsed neon" from
    another could difference a fine mesh against a coarse one and call the
    remainder the cost of the exclusion principle.
    """
    _, meta = _run(client, z=10, pauli=False, exchange=False)
    collapse = meta["collapse"]
    assert collapse is not None
    assert collapse["real_config"] == "1s2 2s2 2p6"
    # More bound, and by a lot: nothing holds the electrons out of the well.
    assert collapse["binding_change"]["value"] < 0
    assert collapse["binding_change_ev"]["unit"] == "eV"
    assert collapse["real_total_energy"]["value"] == pytest.approx(-128.55, abs=0.05)
    # Smaller, which is the half of it a picture can show.
    assert collapse["radius_ratio"]["value"] < 1.0
    assert (
        collapse["collapsed_radius"]["value"] < collapse["real_radius"]["value"]
    )
    assert collapse["real_radius"]["unit"] == "bohr"


def test_the_external_check_travels_to_the_browser(client):
    """The closed-form bound, so the page can show what the number was tested
    against rather than asking to be believed."""
    _, meta = _run(client, z=10, pauli=False, exchange=False)
    collapse = meta["collapse"]
    # zeta* = 10 - 45/16.
    assert collapse["variational_zeta"]["value"] == pytest.approx(7.1875, abs=1e-9)
    assert collapse["variational_energy"]["value"] == pytest.approx(-258.30, abs=0.01)
    # And the SCF came in at or below it, which is the whole point of sending it.
    assert meta["total_energy"]["value"] <= collapse["variational_energy"]["value"]


def test_the_stronger_disclosure_replaces_the_weaker_one_over_the_wire(client):
    """Phase 22 tells the browser the cap is still on. This must not."""
    _, meta = _run(client, z=10, pauli=False, exchange=False)
    joined = " ".join(meta["total_energy"]["provenance"]["assumptions"]).lower()
    assert "occupancy cap is gone" in joined
    assert "pauli principle is not switched off" not in joined
    assert "term structure is undefined" in joined


def test_a_hand_written_collapsed_configuration_is_accepted_without_a_comparison(
    client,
):
    """1s3 2s1 is a legal configuration with the cap off, and has no twin.

    There is no "same atom with Pauli on" for it to be measured against, so the
    server sends no comparison rather than one against the Aufbau ground state,
    which would report the distance between two different configurations as the
    cost of the exclusion principle.
    """
    _, meta = _run(client, z=4, pauli=False, exchange=False, config="1s3 2s1")
    assert meta["config"] == "1s3 2s1"
    assert meta["collapse"] is None
    assert meta["is_ground"] is False


def test_the_cap_still_binds_when_pauli_is_on(client):
    """The same configuration, refused, because the rule is back."""
    r = client.post("/api/jobs/hf", json={"z": 4, "config": "1s3 2s1"})
    assert r.status_code == 422
    assert "exceeds capacity" in r.text


def test_the_collapsed_orbital_still_serves_its_amplitude(client):
    job_id, meta = _run(client, z=10, pauli=False, exchange=False)
    channel = meta["orbitals"][0]["channel"]
    assert channel == "P_1s"
    r = client.get(f"/api/jobs/{job_id}/data", params={"channel": channel})
    assert r.status_code == 200
    amplitude = np.frombuffer(r.content, dtype=np.float32)
    assert amplitude.size == meta["grid_points"]
