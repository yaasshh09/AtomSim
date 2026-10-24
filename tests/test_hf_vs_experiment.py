"""Hartree-Fock's Koopmans ionization energies held against NIST's measured ones.

The pattern is the physics, so the tests pin the pattern and not just a
tolerance: Koopmans freezes the orbitals, which overestimates the IE, and
Hartree-Fock drops correlation, which underestimates it. Where removing an
electron reshapes what is left (helium, neon) the first wins and Koopmans comes
out high; for an alkali the lone valence electron leaves a closed core almost
undisturbed, relaxation is small, and the missing correlation shows instead.
"""

import pytest

from atomsim.atoms import aufbau_configuration
from atomsim.hf_atom import hf_valence_ionization_energy, solve_hartree_fock
from atomsim.hf_reference import NIST_IONIZATION, nist_ionization_energy

HARTREE_EV = 27.211386245988


def test_the_vendored_table_covers_he_to_ca_with_its_source():
    atoms = NIST_IONIZATION["atoms"]
    assert [a["z"] for a in atoms] == list(range(2, 21))
    assert nist_ionization_energy(2)["ie_ev"] == pytest.approx(24.587389011)
    assert all(a["uncertainty_ev"] > 0 for a in atoms)
    assert "NIST" in NIST_IONIZATION["citation"] and NIST_IONIZATION["retrieved"]


def _koopmans_minus_nist(z: int) -> float:
    res = solve_hartree_fock(z, z, aufbau_configuration(z))
    ie = hf_valence_ionization_energy(res).value * HARTREE_EV
    return ie - nist_ionization_energy(z)["ie_ev"]


@pytest.mark.parametrize(
    "z,low,high",
    [
        (2, 0.30, 0.45),   # He: relaxation wins, Koopmans high by 0.39 eV
        (10, 1.40, 1.75),  # Ne: the same, and larger
        (3, -0.12, 0.02),  # Li: an alkali, close
        (19, -0.45, -0.20),  # K: correlation wins, Koopmans low
    ],
)
def test_koopmans_errs_the_way_the_physics_says(z, low, high):
    assert low < _koopmans_minus_nist(z) < high


def test_the_endpoint_serves_the_table_with_its_citation():
    from fastapi.testclient import TestClient

    from atomsim.server.app import create_app

    with TestClient(create_app()) as c:
        body = c.get("/api/reference/ionization").json()
    assert body["retrieved"] == NIST_IONIZATION["retrieved"]
    assert len(body["atoms"]) == 19
