"""The continuous nuclear-mass family: an electron around a +1 point charge of
mass M m_e, for any M between positronium (1) and well past tritium.

The ground truth is the presets themselves. At M = 1 the family has to be
positronium exactly, and at the CODATA proton mass it has to be hydrogen to
machine precision, because both are the same two-body formula.
"""

import pytest
from fastapi.testclient import TestClient
from scipy import constants as sc

from atomsim.analytic.fine_structure import fine_structure_shift
from atomsim.analytic.hydrogen import energy
from atomsim.provenance import Fidelity
from atomsim.server.app import create_app
from atomsim.systems import (
    MASS_RATIO_MAX,
    MASS_RATIO_MIN,
    get_system,
    is_mass_key,
    nuclear_mass_system,
)

R_P = sc.physical_constants["proton-electron mass ratio"][0]


def test_reduced_mass_is_the_two_body_formula():
    for M in (1.0, 2.5, 206.768, 1000.0, 9999.0):
        s = nuclear_mass_system(M)
        assert s.Z == 1
        assert s.mu_ratio.value == pytest.approx(M / (1.0 + M), rel=1e-15)
        assert s.m_over_M == pytest.approx(1.0 / M, rel=1e-15)
        assert s.nuclear_radius is None  # a point charge, said by absence


def test_lower_end_is_positronium_exactly():
    ps = get_system("ps")
    s = nuclear_mass_system(1.0)
    assert s.mu_ratio.value == ps.mu_ratio.value == 0.5
    assert s.m_over_M == ps.m_over_M
    assert energy(1, mu_ratio=s.mu_ratio.value).value == energy(1, mu_ratio=0.5).value


def test_proton_mass_reproduces_hydrogen():
    h = get_system("h")
    s = nuclear_mass_system(R_P)
    assert s.mu_ratio.value == pytest.approx(h.mu_ratio.value, rel=1e-15)
    assert s.m_over_M == pytest.approx(h.m_over_M, rel=1e-15)
    for n in (1, 2, 5):
        assert energy(n, mu_ratio=s.mu_ratio.value).value == pytest.approx(
            energy(n, mu_ratio=h.mu_ratio.value).value, rel=1e-15
        )


def test_binding_rises_monotonically_with_the_nuclear_mass():
    masses = [1.0, 3.0, 10.0, 100.0, 1000.0, 10000.0]
    e1 = [energy(1, mu_ratio=nuclear_mass_system(M).mu_ratio.value).value for M in masses]
    assert all(a > b for a, b in zip(e1, e1[1:]))  # deeper as the nucleus gets heavier
    assert e1[0] == -0.25 and e1[-1] > -0.5  # never past the infinite-mass limit


def test_recoil_error_bar_grows_toward_positronium():
    # The fine-structure recoil scale is m/M: it must widen as M shrinks, so the
    # slider can never walk into positronium carrying hydrogen's confidence.
    errs = []
    for M in (1836.0, 100.0, 1.0):
        s = nuclear_mass_system(M)
        sh = fine_structure_shift(2, 1, 1.5, Z=1, mu_ratio=s.mu_ratio.value, m_over_M=s.m_over_M)
        errs.append(sh.provenance.error_estimate)
    assert errs[0] < errs[1] < errs[2]


def test_hypothetical_nucleus_is_counterfactual_and_says_why():
    s = nuclear_mass_system(500.0)
    p = s.mu_ratio.provenance
    assert p.fidelity is Fidelity.COUNTERFACTUAL
    assert "M = 500 m_e" in p.method
    joined = " ".join(p.assumptions)
    assert "point" in joined and "+1" in joined
    assert "positronium" in s.description and "hydrogen" in s.description


def test_keys_round_trip_through_get_system():
    assert is_mass_key("m206.768") and is_mass_key("m1") and is_mass_key("m1836.15")
    assert not is_mass_key("mu-h") and not is_mass_key("m") and not is_mass_key("m-3")
    assert not is_mass_key("m1e3")  # one spelling per mass, so caches and URLs agree
    s = get_system("m206.768")
    assert s.key == "m206.768"
    assert s.mu_ratio.value == pytest.approx(206.768 / 207.768, rel=1e-15)
    assert "M = 206.768" in s.name


@pytest.mark.parametrize("key", ["m0.5", "m0", "m10001", "m99999"])
def test_out_of_range_masses_are_refused(key):
    with pytest.raises(ValueError, match="nuclear mass"):
        get_system(key)


def test_range_spans_positronium_to_past_tritium():
    assert MASS_RATIO_MIN == 1.0
    assert MASS_RATIO_MAX >= sc.physical_constants["triton-electron mass ratio"][0]


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def test_state_endpoint_serves_a_mass_key(client):
    r = client.get("/api/state/1/0/0", params={"system": "m1"})
    assert r.status_code == 200
    body = r.json()
    assert body["energy"]["value"] == -0.25
    assert body["system"]["key"] == "m1"
    assert body["system"]["mu_ratio"]["provenance"]["fidelity"] == "counterfactual"


def test_system_endpoint_describes_one_key(client):
    r = client.get("/api/system/m1000")
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "hydrogenic"
    assert body["mu_ratio"]["value"] == pytest.approx(1000 / 1001, rel=1e-12)
    assert client.get("/api/system/h").json()["name"] == "Hydrogen"


@pytest.mark.parametrize("key", ["m0.5", "m20000", "m"])
def test_bad_mass_keys_are_422(client, key):
    assert client.get("/api/state/1/0/0", params={"system": key}).status_code == 422
    assert client.get(f"/api/system/{key}").status_code == 422


def test_mass_key_has_no_hyperfine_and_says_so(client):
    # A chosen mass carries no magnetic moment or charge radius, and neither is invented.
    r = client.get("/api/levels", params={"system": "m1000", "hyperfine": True})
    assert r.status_code == 200
    shell = r.json()["hyperfine_shells"][0]
    assert shell["available"] is False and "no identified nucleus" in shell["reason"]
    r = client.get("/api/finite-size", params={"system": "m1000"})
    assert r.status_code == 400 and "no nuclear charge radius" in r.json()["detail"]
