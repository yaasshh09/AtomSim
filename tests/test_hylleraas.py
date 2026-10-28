"""Hylleraas helium: two closed-form anchors, the variational bound, the virial
theorem, and the correlation energy Hartree-Fock leaves out."""

import warnings

import pytest

from atomsim.constants import HARTREE_EV
from atomsim.hf_reference import nist_ionization_energy
from atomsim.hylleraas import (
    HELIUM_EXACT,
    REPORT_OMEGA,
    helium_report,
    hylleraas_series,
    solve_basis,
)
from atomsim.provenance import Fidelity


def test_one_term_is_the_screened_hydrogenic_estimate():
    """e^(-zeta s) alone: zeta = Z - 5/16 and E = -(27/16)^2, the textbook result."""
    level = solve_basis(((0, 0, 0),))
    assert level.zeta == pytest.approx(27 / 16, abs=1e-6)
    assert level.energy == pytest.approx(-((27 / 16) ** 2), abs=1e-12)


def test_hylleraas_1929_three_terms():
    """1, u, t^2: Hylleraas's own three-parameter function, -2.90243 hartree."""
    assert solve_basis(((0, 0, 0), (0, 0, 1), (0, 1, 0))).energy == pytest.approx(
        -2.90243, abs=2e-5
    )


def test_bounds_fall_toward_the_exact_energy_from_above():
    series = hylleraas_series(REPORT_OMEGA)
    energies = [lv.energy for lv in series]
    assert all(e > HELIUM_EXACT for e in energies)
    assert all(b < a for a, b in zip(energies, energies[1:], strict=False))
    assert energies[-1] - HELIUM_EXACT < 1e-6
    # The quoted spread brackets the true distance to the exact value.
    assert energies[-1] - HELIUM_EXACT < energies[-2] - energies[-1]


def test_virial_theorem_holds_at_the_optimal_zeta():
    """Optimizing the scale makes -V/T exactly 2: a check on T and V separately."""
    for level in hylleraas_series(REPORT_OMEGA):
        assert level.virial == pytest.approx(2.0, abs=1e-6)


def test_correlation_energy_and_ionization_energy():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = helium_report()
    assert r.correlated.provenance.fidelity is Fidelity.NUMERICAL
    # The literature correlation energy, E_exact - E_HF-limit, is -0.04204 hartree.
    bar = r.correlation.provenance.error_estimate + 1e-5
    assert r.correlation.value == pytest.approx(-0.04204, abs=bar)
    measured = nist_ionization_energy(2)["ie_ev"]
    assert abs(r.ie_hartree_fock.value * HARTREE_EV - measured) > 1.0  # HF is 1.1 eV low
    # With correlation and the reduced mass, what is left is mass polarization,
    # relativity and QED: about a milli-electronvolt.
    assert abs(r.ie_reduced_mass.value * HARTREE_EV - measured) < 2e-3


def test_endpoint():
    from fastapi.testclient import TestClient

    from atomsim.server.app import create_app

    body = TestClient(create_app()).get("/api/helium-correlation").json()
    assert len(body["series"]) == REPORT_OMEGA + 1
    assert body["series"][-1]["terms"] == 50
    assert body["correlation_ev"]["unit"] == "eV"
    assert body["correlation_ev"]["value"] == pytest.approx(-1.144, abs=0.002)
    assert body["measured_ie_ev"] == pytest.approx(24.587389, abs=1e-6)
