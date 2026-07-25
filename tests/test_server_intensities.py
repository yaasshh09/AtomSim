"""API contract for line intensities on /api/spectrum.

The physics is validated in test_transitions.py and test_spectra_intensities.py.
These check the boundary: that the flag is honoured, that provenance survives
the trip, and that the two cases with no honest answer come back with their
reason attached rather than as silent nulls.
"""

import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _line(body, n_up, l_up, n_low, l_low):
    for ln in body["lines"]:
        if (ln["n_upper"], ln["l_upper"], ln["n_lower"], ln["l_lower"]) == (
            n_up, l_up, n_low, l_low,
        ):
            return ln
    raise AssertionError(f"no line {n_up}{l_up} -> {n_low}{l_low} in response")


def test_intensities_absent_unless_requested(client):
    body = client.get("/api/spectrum?system=h&n_max=4").json()
    assert all(ln["einstein_a_s"] is None for ln in body["lines"])
    assert body["intensity_note"] is None


def test_intensities_served_when_requested(client):
    body = client.get("/api/spectrum?system=h&n_max=4&intensities=true").json()
    lya = _line(body, 2, 1, 1, 0)
    assert lya["einstein_a_s"]["value"] == pytest.approx(6.27e8, rel=3e-3)
    assert lya["einstein_a_s"]["unit"] == "s^-1"
    assert lya["oscillator_strength"]["value"] == pytest.approx(0.4162, rel=1e-2)
    assert body["intensity_note"] is None


def test_intensity_provenance_survives_the_boundary(client):
    body = client.get("/api/spectrum?system=h&n_max=3&intensities=true").json()
    prov = body["lines"][0]["einstein_a_s"]["provenance"]
    assert prov["fidelity"] == "numerical"
    assert "Gauss-Laguerre" in prov["method"] or "dE^3" in prov["method"]
    assert prov["error_estimate"] is not None


def test_every_served_line_has_a_positive_rate(client):
    body = client.get("/api/spectrum?system=h&n_max=5&intensities=true").json()
    assert all(ln["einstein_a_s"]["value"] > 0.0 for ln in body["lines"])


def test_fine_structure_serves_j_resolved_intensities(client):
    body = client.get(
        "/api/spectrum?system=h&n_max=4&fine_structure=true&intensities=true"
    ).json()
    assert body["lines"] and all(ln["einstein_a_s"] is not None for ln in body["lines"])
    assert all(ln["einstein_a_s"]["value"] > 0.0 for ln in body["lines"])
    assert body["intensity_note"] is None
    # Every line is j-resolved on both ends, and the 6j is named in provenance.
    assert all(ln["j_upper"] is not None for ln in body["lines"])
    assert "6j" in body["lines"][0]["einstein_a_s"]["provenance"]["method"]


def test_screened_atom_returns_null_intensities_with_a_reason(client):
    body = client.get("/api/spectrum?system=he&intensities=true").json()
    assert all(ln["einstein_a_s"] is None for ln in body["lines"])
    assert body["intensity_note"] and "screen" in body["intensity_note"].lower()


def test_intensities_do_not_change_wavelengths_or_comparison(client):
    plain = client.get("/api/spectrum?system=h&n_max=6").json()
    loud = client.get("/api/spectrum?system=h&n_max=6&intensities=true").json()
    assert [ln["wavelength_nm"]["value"] for ln in plain["lines"]] == [
        ln["wavelength_nm"]["value"] for ln in loud["lines"]
    ]
    assert plain["comparison"] == loud["comparison"]


def test_reduced_mass_reaches_the_served_rate(client):
    """Positronium's mu = 1/2 halves A relative to hydrogen (A ~ mu)."""
    h = _line(client.get("/api/spectrum?system=h&n_max=3&intensities=true").json(),
              2, 1, 1, 0)
    ps = _line(client.get("/api/spectrum?system=ps&n_max=3&intensities=true").json(),
               2, 1, 1, 0)
    assert ps["einstein_a_s"]["value"] / h["einstein_a_s"]["value"] == pytest.approx(
        0.5, rel=1e-3
    )
