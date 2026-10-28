"""Hydrogen in 2, 3 and 4 dimensions: the closed form against an independent
numerical solve, the degeneracy counts, and the endpoint."""

import pytest
from fastapi.testclient import TestClient

from atomsim.dimensions import DIMENSIONS, degeneracy, energy_d, numerical_energy
from atomsim.provenance import Fidelity
from atomsim.server.app import create_app


@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("n,l", [(1, 0), (2, 0), (2, 1), (3, 2), (4, 0)])
def test_closed_form_matches_the_numerical_radial_solve(dim, n, l):
    """Includes the two-dimensional s states, whose attractive -1/(8 r^2) and
    logarithmic irregular solution make them the hard case for any grid."""
    assert numerical_energy(n, l, dim) == pytest.approx(energy_d(n, dim).value, abs=5e-6)


def test_two_dimensional_hydrogen_is_four_times_deeper():
    assert energy_d(1, 2).value == pytest.approx(4 * energy_d(1, 3).value, rel=1e-15)


def test_degeneracies():
    assert [degeneracy(n, 2) for n in range(1, 6)] == [1, 3, 5, 7, 9]
    assert [degeneracy(n, 3) for n in range(1, 6)] == [1, 4, 9, 16, 25]
    expected_4d = [n * (n + 1) * (2 * n + 1) // 6 for n in range(1, 6)]
    assert [degeneracy(n, 4) for n in range(1, 6)] == expected_4d


def test_only_three_dimensions_is_real():
    assert energy_d(2, 3).provenance.fidelity is Fidelity.EXACT
    for dim in (2, 4):
        prov = energy_d(2, dim).provenance
        assert prov.fidelity is Fidelity.COUNTERFACTUAL
        assert any("Gauss's law" in a for a in prov.assumptions)


def test_endpoint():
    client = TestClient(create_app())
    body = client.get("/api/dimensions", params={"n_max": 3}).json()
    assert [d["dim"] for d in body["dimensions"]] == [2, 3, 4]
    two = body["dimensions"][0]["levels"]
    assert [lv["degeneracy"] for lv in two] == [1, 3, 5]
    assert two[0]["energy_ev"]["unit"] == "eV"
    assert two[0]["energy_ev"]["provenance"]["fidelity"] == "counterfactual"
    assert client.get("/api/dimensions", params={"system": "na"}).status_code == 400
