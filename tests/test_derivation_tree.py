"""Derivation trees: a number's provenance names what it was computed from,
down to the solves and constants at its leaves."""

import pytest
from fastapi.testclient import TestClient
from scipy import constants as sc

from atomsim.constants import HARTREE_EV, codata_factor
from atomsim.provenance import Fidelity
from atomsim.server.app import create_app


def test_codata_factor_carries_the_constants_own_uncertainty():
    q = codata_factor("Hartree energy in eV", HARTREE_EV * 1e3, "meV per hartree")
    value, _, unc = sc.physical_constants["Hartree energy in eV"]
    assert q.provenance.fidelity is Fidelity.EXACT
    assert q.provenance.error_estimate == pytest.approx(1e3 * unc, rel=1e-12)
    assert q.provenance.inputs == ()


def test_correlation_energy_tree_reaches_both_solves_and_the_constant():
    body = TestClient(create_app()).get("/api/helium-correlation").json()
    corr = body["correlation_ev"]
    # eV <- (E_corr in hartree, the CODATA factor)
    top = corr["provenance"]["inputs"]
    assert [q["unit"] for q in top] == ["hartree", "eV per hartree"]
    assert top[1]["label"] == "CODATA Hartree energy in eV"
    # E_corr <- (Hylleraas energy, Hartree-Fock energy): the two solves, as leaves.
    leaves = top[0]["provenance"]["inputs"]
    assert {q["label"] for q in leaves} == {"E_Hylleraas", "E_total"}
    assert all(q["provenance"]["inputs"] == [] for q in leaves)
    # The arithmetic holds through the tree.
    e_hyl = next(q["value"] for q in leaves if q["label"] == "E_Hylleraas")
    e_hf = next(q["value"] for q in leaves if q["label"] == "E_total")
    assert corr["value"] == pytest.approx((e_hyl - e_hf) * top[1]["value"], rel=1e-12)


def test_a_closed_form_names_only_the_measured_mass_it_was_given():
    body = TestClient(create_app()).get("/api/state/2/1/0").json()
    energy = body["energy"]
    assert energy["provenance"]["fidelity"] == "exact"
    # The formula is exact; its one measured input is the reduced-mass ratio.
    inputs = energy["provenance"]["inputs"]
    assert [q["label"] for q in inputs] == ["mu/m_e (Hydrogen)"]
    assert inputs[0]["provenance"]["inputs"] == []
