"""/api/spectrum profile block: the curve, its widths, and its disclosures."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _profile(client, **params):
    r = client.get("/api/spectrum", params={"profile": True, **params})
    assert r.status_code == 200, r.text
    return r.json()


def test_profile_is_absent_unless_asked_for(client):
    body = client.get("/api/spectrum", params={"intensities": True}).json()
    assert body["profile"] is None
    assert body["profile_note"] is None


def test_profile_returns_a_curve_with_matching_axes(client):
    body = _profile(client, intensities=True)
    prof = body["profile"]
    assert len(prof["wavelength_nm"]) == len(prof["intensity"])
    assert len(prof["wavelength_nm"]) > 100
    assert all(np.isfinite(prof["intensity"]))
    assert all(v >= 0.0 for v in prof["intensity"])
    assert prof["wavelength_nm"] == sorted(prof["wavelength_nm"])


def test_profile_weight_follows_the_bars(client):
    """Curve and bars must be driven by the same quantity, or the two
    renderings of one spectrum would disagree about which line is brightest."""
    rate = _profile(client, intensities=True)["profile"]
    assert rate["weight_kind"] == "rate"
    assert rate["unit"] == "s^-1 per nm"
    lte = _profile(
        client, intensities=True, temperature_k=1e4, electron_density_cm3=1e12
    )["profile"]
    assert lte["weight_kind"] == "emissivity"
    assert lte["unit"] == "eV/s per atom per nm"


def test_profile_reports_its_own_flux_closure(client):
    body = _profile(
        client, intensities=True, temperature_k=1e4, electron_density_cm3=1e12
    )
    assert body["profile"]["flux_closure"] == pytest.approx(1.0, rel=5e-3)


def test_profile_carries_provenance_and_what_it_omits(client):
    prov = _profile(client, intensities=True)["profile"]["provenance"]
    assert prov["fidelity"] == "approximation"
    text = " ".join(prov["assumptions"])
    assert "collisional" in text
    assert "self-absorption" in text


def test_widths_break_down_by_mechanism(client):
    """Pointing at a line has to answer 'what set this width', so the terms
    that contributed are named per line rather than summed away."""
    body = _profile(
        client, intensities=True, temperature_k=1e4, electron_density_cm3=1e12,
        resolving_power=5000,
    )
    widths = body["profile"]["widths"]
    assert widths
    for w in widths:
        assert set(w["terms"]) >= {"natural", "Doppler", "instrumental"}
        assert w["fwhm_nm"] > 0.0
        assert w["sigma_nm"] > 0.0


def test_no_width_source_returns_a_note_not_a_curve(client):
    """Without intensities, thermal conditions or an instrument there is no
    width to draw, and the server must say which knob is missing."""
    body = _profile(client, intensities=False)
    assert body["profile"] is None
    assert body["profile_note"] is not None
    assert "resolving power" in body["profile_note"]


def test_instrument_alone_is_enough_to_draw(client):
    body = _profile(client, intensities=False, resolving_power=2000)
    assert body["profile"] is not None
    assert body["profile"]["weight_kind"] == "uniform"


def test_resolving_power_is_bounded(client):
    r = client.get(
        "/api/spectrum", params={"profile": True, "resolving_power": 1e9}
    )
    assert r.status_code == 422


def test_lower_resolving_power_widens_every_line(client):
    sharp = _profile(client, intensities=True, resolving_power=100000)["profile"]
    blunt = _profile(client, intensities=True, resolving_power=1000)["profile"]
    for a, b in zip(sharp["widths"], blunt["widths"], strict=True):
        assert b["fwhm_nm"] > a["fwhm_nm"]


def test_hotter_gas_widens_every_line(client):
    cool = _profile(
        client, intensities=True, temperature_k=3e3, electron_density_cm3=1e12
    )["profile"]
    hot = _profile(
        client, intensities=True, temperature_k=3e4, electron_density_cm3=1e12
    )["profile"]
    for a, b in zip(cool["widths"], hot["widths"], strict=True):
        assert b["fwhm_nm"] > a["fwhm_nm"]


def test_dense_plasma_reports_the_broadening_it_does_not_model(client):
    body = _profile(
        client, intensities=True, temperature_k=1e4, electron_density_cm3=1e17
    )
    prof = body["profile"]
    assert prof["stark_span_nm"] is not None
    assert prof["stark_span_nm"]["value"] > 0.0
    assert prof["stark_note"] is not None
    assert "Stark" in prof["stark_note"]


def test_thin_gas_does_not_cry_wolf(client):
    prof = _profile(
        client, intensities=True, temperature_k=1e4, electron_density_cm3=1e6
    )["profile"]
    assert prof["stark_note"] is None


def test_screened_atom_gets_a_profile_but_no_stark_estimate(client):
    """Sodium has no degenerate l manifold, so the linear Stark estimate does
    not apply to it and must not be reported as though it did."""
    prof = _profile(
        client, system="na", intensities=True,
        temperature_k=1e4, electron_density_cm3=1e17,
    )["profile"]
    assert len(prof["wavelength_nm"]) > 100
    assert prof["stark_span_nm"] is None
    assert prof["stark_note"] is None


def test_fine_structure_profile_stays_in_the_optical_window(client):
    """Within-n components sit out at millimetres. Letting them set the range
    would spend the whole grid on empty space."""
    prof = _profile(
        client, intensities=True, fine_structure=True, n_max=5,
        temperature_k=1e4, electron_density_cm3=1e12,
    )["profile"]
    assert max(prof["wavelength_nm"]) < 1e5


def test_full_range_includes_the_within_n_components(client):
    narrow = _profile(
        client, intensities=True, fine_structure=True, n_max=5,
        temperature_k=1e4, electron_density_cm3=1e12,
    )["profile"]
    wide = _profile(
        client, intensities=True, fine_structure=True, n_max=5,
        temperature_k=1e4, electron_density_cm3=1e12, full_range=True,
    )["profile"]
    assert max(wide["wavelength_nm"]) > max(narrow["wavelength_nm"])
    assert len(wide["widths"]) > len(narrow["widths"])


def test_positronium_lines_are_dramatically_wider_than_hydrogen(client):
    """The exotic presets get real Doppler widths from their own masses, and
    positronium is 30 times lighter per emitter than hydrogen."""
    h = _profile(
        client, system="h", intensities=True,
        temperature_k=1e4, electron_density_cm3=1e12,
    )["profile"]
    ps = _profile(
        client, system="ps", intensities=True,
        temperature_k=1e4, electron_density_cm3=1e12,
    )["profile"]
    # Compare the fractional width, since the two systems' lines sit at
    # different wavelengths (positronium's Rydberg is half of hydrogen's).
    h_rel = h["widths"][0]["sigma_nm"] / h["widths"][0]["wavelength_nm"]
    ps_rel = ps["widths"][0]["sigma_nm"] / ps["widths"][0]["wavelength_nm"]
    assert ps_rel / h_rel == pytest.approx(30.3, rel=0.05)


def test_generic_z_has_no_thermal_width_and_says_why(client):
    """An infinitely massive nucleus cannot recoil. The zero must be explained
    rather than served as a sharp line."""
    prof = _profile(
        client, system="z3", intensities=True,
        temperature_k=1e4, electron_density_cm3=1e12,
    )["profile"]
    assert all("Doppler" not in w["terms"] for w in prof["widths"])
    assert any(
        "finite emitter mass" in a for a in prof["provenance"]["assumptions"]
    )
