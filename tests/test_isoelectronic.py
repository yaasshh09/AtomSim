"""Isoelectronic sequences: the 1/Z coefficients against their closed forms,
and Hartree-Fock settling onto the known second-order coefficient."""

import warnings

import pytest
from fastapi.testclient import TestClient

from atomsim.atoms import aufbau_configuration
from atomsim.hf_atom import solve_hartree_fock
from atomsim.isoelectronic import (
    SEQUENCE_ELECTRONS,
    first_order_coefficient,
    hydrogenic_coefficient,
    sequence_coefficients,
    sequence_reference,
)
from atomsim.provenance import Fidelity
from atomsim.server.app import create_app

# First-order repulsion over hydrogenic orbitals, from the Slater integrals
# F0(1s,1s) = 5/8, F0(1s,2s) = 17/81, G0(1s,2s) = 16/729, F0(2s,2s) = 77/512.
_E1_CLOSED = {
    2: 5 / 8,
    3: 5 / 8 + 2 * 17 / 81 - 16 / 729,
    4: 5 / 8 + 4 * 17 / 81 - 2 * 16 / 729 + 77 / 512,
}

# The Hartree-Fock second-order coefficient for the helium sequence.
_E2_HF_HELIUM = -0.111003


@pytest.mark.parametrize("n", sorted(_E1_CLOSED))
def test_first_order_matches_closed_form(n):
    e1 = first_order_coefficient(aufbau_configuration(n))
    assert e1.provenance.fidelity is Fidelity.NUMERICAL
    err = abs(e1.value - _E1_CLOSED[n])
    assert err < 1e-5 * _E1_CLOSED[n]
    # The error bar brackets the real error rather than decorating it.
    assert err <= e1.provenance.error_estimate


def test_hydrogenic_limit_is_exact():
    e0 = hydrogenic_coefficient(aufbau_configuration(10))
    assert e0.provenance.fidelity is Fidelity.EXACT
    assert e0.value == -(2 / 2 + 8 / 8)


def test_every_sequence_has_coefficients_and_ions():
    for n in SEQUENCE_ELECTRONS:
        c = sequence_coefficients(n)
        ions = sequence_reference(n)["ions"]
        assert [ion["z"] for ion in ions] == list(range(n, n + 11))
        assert c.e1.value > 0


def test_helium_sequence_settles_on_second_order_coefficient():
    """E_HF - E0 Z^2 - E1 Z is E2 + E3/Z + ..., so it closes on E2 like 1/Z."""
    c = sequence_coefficients(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for z in (2, 4, 6, 8):
            hf = solve_hartree_fock(z, 2, c.config).total_energy
            residual = hf.value - c.e0.value * z * z - c.e1.value * z
            bar = hf.provenance.error_estimate + z * c.e1.provenance.error_estimate
            assert abs(residual - _E2_HF_HELIUM) < 2e-3 / z + bar


def test_endpoint():
    client = TestClient(create_app())
    body = client.get("/api/isoelectronic/3").json()
    assert body["isoelectronic_with"] == "Li"
    assert body["config"] == "1s2 2s1"
    assert body["e1"]["value"] == pytest.approx(_E1_CLOSED[3], abs=1e-5)
    assert body["e0"]["provenance"]["fidelity"] == "exact"
    first = body["ions"][0]
    assert (first["z"], first["symbol"], first["kind"]) == (3, "Li", "measured")
    assert client.get("/api/isoelectronic/5").status_code == 422
