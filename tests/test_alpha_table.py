"""The table under a changed alpha: the edge where Z alpha reaches 1, and
relativity's share growing as (Z alpha)^2."""

import pytest
from fastapi.testclient import TestClient

from atomsim.alpha_table import alpha_table
from atomsim.constants import ALPHA
from atomsim.provenance import Fidelity
from atomsim.server.app import create_app


def test_every_known_element_has_a_1s_in_this_world():
    t = alpha_table(1.0)
    assert t.z_edge == 118
    assert all(s is not None for s in t.relativistic_share)
    assert t.relativistic_share[0].provenance.fidelity is Fidelity.EXACT


def test_doubling_alpha_ends_the_table_near_68():
    t = alpha_table(2.0)
    assert t.z_edge == 68
    assert t.relativistic_share[67] is not None and t.relativistic_share[68] is None
    assert t.relativistic_share[0].provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_share_grows_as_z_alpha_squared_for_light_atoms():
    """Leading order: (Z alpha)^2 / 4 of the 1s binding."""
    t = alpha_table(1.0)
    for z in (1, 6, 10):
        share = t.relativistic_share[z - 1].value
        assert share == pytest.approx((z * ALPHA) ** 2 / 4, rel=0.01)


def test_endpoint():
    body = TestClient(create_app()).get("/api/alpha-table", params={"alpha_multiplier": 2}).json()
    assert body["z_edge"] == 68
    assert len(body["shares"]) == 118
    assert body["shares"][100] is None
    assert body["shares"][0]["provenance"]["fidelity"] == "counterfactual"
