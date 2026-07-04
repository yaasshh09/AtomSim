import numpy as np
import pytest
from scipy.constants import electron_mass, physical_constants, proton_mass

from atomsim.analytic.hydrogen import (
    energy,
    mean_radius,
    radial_wavefunction,
    validate_quantum_numbers,
)
from atomsim.constants import HARTREE_EV
from atomsim.provenance import Fidelity


def test_hydrogen_ground_state_is_minus_half_hartree():
    q = energy(1)
    assert q.value == pytest.approx(-0.5, abs=1e-15)
    assert q.unit == "hartree"
    assert q.provenance.fidelity is Fidelity.EXACT


def test_energy_scales_as_z_squared_over_n_squared():
    assert energy(2, Z=3).value == pytest.approx(-0.5 * 9 / 4, rel=1e-14)


def test_helium_plus_ground_state():
    assert energy(1, Z=2).value == pytest.approx(-2.0, rel=1e-14)


def test_positronium_via_reduced_mass():
    # mu = m_e/2 exactly -> binding 6.803 eV
    q = energy(1, mu_ratio=0.5)
    assert q.value == pytest.approx(-0.25, rel=1e-14)
    assert q.value * HARTREE_EV == pytest.approx(-6.803, abs=0.001)


def test_muonic_hydrogen_ground_state_in_ev():
    m_mu = physical_constants["muon mass"][0]
    mu_ratio = (m_mu * proton_mass / (m_mu + proton_mass)) / electron_mass
    e_ev = energy(1, mu_ratio=mu_ratio).value * HARTREE_EV
    assert e_ev == pytest.approx(-2528.5, abs=2.0)  # known ~ -2.53 keV


def test_invalid_quantum_numbers_raise():
    with pytest.raises(ValueError):
        energy(0)
    with pytest.raises(ValueError):
        energy(-3)


def test_provenance_names_its_assumptions():
    p = energy(1).provenance
    joined = " ".join(p.assumptions).lower()
    assert "non-relativistic" in joined
    assert "point nucleus" in joined


def test_invalid_l_raises():
    with pytest.raises(ValueError):
        validate_quantum_numbers(1, 1)   # l == n
    with pytest.raises(ValueError):
        validate_quantum_numbers(2, 2)   # l == n
    with pytest.raises(ValueError):
        validate_quantum_numbers(2, -1)  # l < 0
    validate_quantum_numbers(3, 2)       # valid: must NOT raise


def _grid():
    return np.linspace(1e-8, 150.0, 200_001)


def test_radial_wavefunctions_are_normalized():
    r = _grid()
    for n, l in [(1, 0), (2, 0), (2, 1), (3, 1), (5, 3)]:
        norm = np.trapezoid(radial_wavefunction(n, l, r) ** 2 * r**2, r)
        assert norm == pytest.approx(1.0, abs=1e-6), (n, l)


def test_node_counts_are_n_minus_l_minus_1():
    r = _grid()
    for n, l in [(1, 0), (2, 0), (3, 0), (3, 2), (4, 1)]:
        R = radial_wavefunction(n, l, r)
        mask = np.abs(R) > 1e-6 * np.abs(R).max()
        signs = np.sign(R[mask])
        nodes = int(np.sum(signs[1:] != signs[:-1]))
        assert nodes == n - l - 1, (n, l)


def test_mean_radius_matches_exact_formula_and_integral():
    r = _grid()
    for n, l in [(1, 0), (2, 1), (3, 0)]:
        R = radial_wavefunction(n, l, r)
        integral = np.trapezoid(R**2 * r**3, r)
        exact = mean_radius(n, l)
        assert integral == pytest.approx(exact, rel=1e-5), (n, l)
    assert mean_radius(1, 0) == pytest.approx(1.5)  # 1s: <r> = 1.5 bohr


def test_orthogonality_same_l():
    r = _grid()
    overlap = np.trapezoid(
        radial_wavefunction(1, 0, r) * radial_wavefunction(2, 0, r) * r**2, r
    )
    assert abs(overlap) < 1e-6


def test_scaling_with_z_and_reduced_mass():
    # length scale ~ 1/(Z mu'): heavier reduced mass shrinks the atom
    assert mean_radius(1, 0, Z=2) == pytest.approx(0.75)
    assert mean_radius(1, 0, mu_ratio=2.0) == pytest.approx(0.75)
