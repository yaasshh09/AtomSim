"""The exchange hole: its sum rule, the on-top half, helium's whole-spin hole,
and a conditional density that never goes negative."""

import warnings

import numpy as np
import pytest
from fastapi.testclient import TestClient

from atomsim.exchange_hole import CLOSED_SHELL_Z, exchange_hole
from atomsim.provenance import Fidelity
from atomsim.server.app import create_app


def _hole(z, a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return exchange_hole(z, a)


def test_closed_shells_are_the_noble_gases_and_the_full_s_shells():
    assert CLOSED_SHELL_Z == (2, 4, 10, 12, 18, 20, 30, 36)


@pytest.mark.parametrize("z,a", [(4, 2.0), (10, 0.3), (10, 1.0), (18, 1.5)])
def test_hole_holds_exactly_one_electron(z, a):
    """The sum rule depends on the orbitals being orthonormal in the solver's
    own quadrature, so it is a real check, not an identity."""
    h = _hole(z, a)
    assert h.sum_rule.provenance.fidelity is Fidelity.NUMERICAL
    assert h.sum_rule.value == pytest.approx(-1.0, abs=1e-6)


@pytest.mark.parametrize("z,a", [(10, 0.3), (18, 1.5)])
def test_half_the_density_is_left_on_top_and_none_goes_negative(z, a):
    h = _hole(z, a)
    j = int(np.flatnonzero(h.s == h.a)[0])
    # Only the other spin remains where the electron sits.
    assert h.conditional[j] / h.density[j] == pytest.approx(0.5, abs=1e-12)
    assert np.all(h.conditional >= -1e-12 * h.density.max())
    assert np.all(h.conditional <= h.density * (1 + 1e-12))


def test_helium_hole_is_the_whole_same_spin_density():
    """One orbital per spin: the hole is that orbital, everywhere, not just on top.
    The other electron has the opposite spin, and Hartree-Fock lets it ignore the first."""
    h = _hole(2, 0.5)
    assert np.allclose(h.conditional, 0.5 * h.density, rtol=1e-10, atol=0)


def test_endpoint():
    client = TestClient(create_app())
    body = client.get("/api/exchange-hole", params={"z": 10, "a": 1.0}).json()
    assert body["a_bohr"] == pytest.approx(1.0, rel=0.01)
    assert body["a_bohr"] in body["conditional"]["grid"]
    assert body["density"]["provenance"]["fidelity"] == "approximation"
    assert body["sum_rule"]["value"] == pytest.approx(-1.0, abs=1e-6)
    assert len(body["density"]["values"]) <= 1700
    assert client.get("/api/exchange-hole", params={"z": 11}).status_code == 400
    assert client.get("/api/exchange-hole", params={"z": 10, "a": 9}).status_code == 422
