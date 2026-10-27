"""K-alpha lines: the relativity estimate against exact Dirac energies, Moseley's
law against measurement, and Delta-SCF Hartree-Fock against NIST."""

import warnings

import pytest

from atomsim.analytic.dirac import dirac_energy
from atomsim.atoms import format_config
from atomsim.constants import HARTREE_EV
from atomsim.provenance import Fidelity
from atomsim.xray import (
    XRAY_Z,
    hole_configurations,
    kalpha,
    measured_kalpha,
    moseley_energy,
    relativity_estimate,
)


def _measured_mean_ev(z: int) -> float:
    """(2 K-alpha1 + K-alpha2) / 3: the line a calculation without spin-orbit predicts."""
    m = measured_kalpha(z)
    return (2 * m["ka1"]["experiment_ev"] + m["ka2"]["experiment_ev"]) / 3


def test_hole_configurations():
    one_s, two_p = hole_configurations(10)
    assert format_config(one_s) == "1s1 2s2 2p6"
    assert format_config(two_p) == "1s2 2s2 2p5"


@pytest.mark.parametrize("z", [11, 19, 27, 36])
def test_relativity_estimate_matches_exact_dirac_at_leading_order(z):
    """The estimate is the (Z alpha)^2 term of the exact hydrogenic Dirac line."""
    zeff = z - 1

    def dirac_shift(n, j):
        return dirac_energy(n, j, Z=zeff).value - (-(zeff**2) / (2 * n * n))

    exact = -dirac_shift(1, 0.5) + (2 * dirac_shift(2, 1.5) + dirac_shift(2, 0.5)) / 3
    est = relativity_estimate(z).value
    # The next order is (Z alpha)^2 smaller again.
    assert est == pytest.approx(exact, rel=2 * (zeff / 137.036) ** 2)


@pytest.mark.parametrize("z", [12, 26, 36])
def test_moseley_law_is_good_to_a_few_percent(z):
    e = moseley_energy(z).value * HARTREE_EV
    assert e == pytest.approx(_measured_mean_ev(z), rel=0.03)


@pytest.mark.parametrize("z", [14, 18])
def test_delta_scf_with_relativity_lands_near_nist(z):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = kalpha(z)
    assert k.delta_scf.provenance.fidelity is Fidelity.APPROXIMATION
    hf = k.delta_scf.value * HARTREE_EV
    rel = k.relativity.value * HARTREE_EV
    measured = _measured_mean_ev(z)
    # Delta-SCF alone is low by about the relativity; adding the estimate closes
    # most of the gap. Correlation and QED are what is left.
    assert hf < measured
    assert abs(hf + rel - measured) < abs(hf - measured)
    assert abs(hf + rel - measured) < 2.0e-3 * measured


def test_every_vendored_element_has_both_lines():
    assert XRAY_Z == tuple(range(10, 37))
    for z in XRAY_Z:
        m = measured_kalpha(z)
        assert m["ka1"]["experiment_ev"] >= m["ka2"]["experiment_ev"]


def test_xray_job_serves_the_line_with_its_provenance():
    import time

    from fastapi.testclient import TestClient

    from atomsim.server.app import create_app

    with TestClient(create_app()) as client:
        assert client.post("/api/jobs/xray", json={"z": 9}).status_code == 400
        job_id = client.post("/api/jobs/xray", json={"z": 10}).json()["id"]
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            status = client.get(f"/api/jobs/{job_id}").json()
            if status["status"] in ("done", "error"):
                break
            time.sleep(0.05)
        assert status["status"] == "done", status["error"]
        meta = client.get(f"/api/jobs/{job_id}/meta").json()
        assert meta["kind"] == "kalpha"
        assert meta["hole_1s_config"] == "1s1 2s2 2p6"
        assert meta["delta_scf_ev"]["unit"] == "eV"
        assert meta["delta_scf_ev"]["provenance"]["fidelity"] == "approximation"
        assert meta["ka1"]["experiment_kind"] == "direct"
        assert client.get(f"/api/jobs/{job_id}/data").status_code == 422
