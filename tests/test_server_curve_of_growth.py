"""/api/curve-of-growth: the three branches, served with their labels."""

import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _cog(client, **params):
    r = client.get("/api/curve-of-growth", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_returns_all_three_branches(client):
    body = _cog(client, lambda_nm=656.28)
    assert set(body["regime"]) == {"linear", "saturated", "damping"}
    n = len(body["column_density_m2"])
    assert len(body["equivalent_width_nm"]) == n
    assert len(body["slope"]) == n
    assert len(body["tau_centre"]) == n


def test_picks_the_line_nearest_the_requested_wavelength(client):
    assert _cog(client, lambda_nm=656.0)["label"] == "3->2"
    assert _cog(client, lambda_nm=486.0)["label"] == "4->2"
    assert _cog(client, lambda_nm=121.5)["label"] == "2->1"


def test_the_branches_have_their_textbook_slopes(client):
    body = _cog(client, lambda_nm=656.28)
    a = body["damping_parameter"]
    pairs = list(zip(body["tau_centre"], body["slope"], strict=True))
    linear = [s for t, s in pairs if t < 0.01]
    damping = [s for t, s in pairs if a * t > 100]
    assert linear and damping
    assert all(abs(s - 1.0) < 0.05 for s in linear)
    assert all(abs(s - 0.5) < 0.05 for s in damping)


def test_equivalent_width_rises_with_column(client):
    w = _cog(client, lambda_nm=656.28)["equivalent_width_nm"]
    assert all(b > a for a, b in zip(w, w[1:], strict=False))


def test_saturation_is_visible_in_the_numbers(client):
    """Across the saturated branch the width must grow far slower than the
    column. This is the claim the whole phase rests on."""
    body = _cog(client, lambda_nm=656.28)
    sat = [
        (n, w) for n, w, r in zip(
            body["column_density_m2"], body["equivalent_width_nm"], body["regime"],
            strict=True,
        )
        if r == "saturated"
    ]
    assert len(sat) > 5
    column_ratio = sat[-1][0] / sat[0][0]
    width_ratio = sat[-1][1] / sat[0][1]
    assert column_ratio > 100
    assert width_ratio < 5


def test_a_hotter_gas_moves_the_knee(client):
    """The Doppler width sets where saturation starts, which is what makes a
    curve-of-growth fit able to measure the temperature."""
    def knee(t):
        b = _cog(client, lambda_nm=656.28, temperature_k=t)
        for n, r in zip(b["column_density_m2"], b["regime"], strict=True):
            if r != "linear":
                return n
        raise AssertionError("never left the linear branch")

    assert knee(40000) > 1.5 * knee(2500)


def test_carries_the_widths_it_was_computed_from(client):
    body = _cog(client, lambda_nm=656.28, temperature_k=10000)
    assert body["sigma_nm"] > 0
    assert body["gamma_nm"] > 0
    assert body["damping_parameter"] == pytest.approx(
        body["gamma_nm"] / (body["sigma_nm"] * 2**0.5), rel=1e-9
    )


def test_an_instrument_does_not_change_the_curve(client):
    """Equivalent width is instrument-independent, so a spectrograph must
    widen the profile and leave the curve of growth alone."""
    plain = _cog(client, lambda_nm=656.28)
    blurred = _cog(client, lambda_nm=656.28, resolving_power=3000)
    assert blurred["sigma_nm"] > plain["sigma_nm"]
    # The two runs do not share columns (the range is built from the widths),
    # so compare W/N at the thin end, which the closed form fixes by f alone.
    thin_a = plain["equivalent_width_nm"][0] / plain["column_density_m2"][0]
    thin_b = blurred["equivalent_width_nm"][0] / blurred["column_density_m2"][0]
    assert thin_b == pytest.approx(thin_a, rel=1e-3)


def test_states_what_it_cannot_do(client):
    text = " ".join(_cog(client, lambda_nm=656.28)["provenance"]["assumptions"])
    assert "never reverses" in text
    assert "stimulated emission" in text


def test_screened_atom_gets_a_curve(client):
    body = _cog(client, system="na", lambda_nm=589.0)
    assert set(body["regime"]) >= {"linear", "saturated"}
    assert body["oscillator_strength"] > 0


def test_rejects_a_nonsense_wavelength(client):
    r = client.get("/api/curve-of-growth", params={"lambda_nm": 0})
    assert r.status_code == 422


def test_rejects_an_out_of_range_resolving_power(client):
    r = client.get(
        "/api/curve-of-growth", params={"lambda_nm": 656.28, "resolving_power": 5}
    )
    assert r.status_code == 422
