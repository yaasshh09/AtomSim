import numpy as np
import pytest

from atomsim.analytic.hydrogen import energy as exact_energy
from atomsim.analytic.hydrogen import radial_wavefunction
from atomsim.numerics.radial_solver import solve_radial, solve_radial_with_error
from atomsim.provenance import Fidelity


def test_3d_harmonic_oscillator_l0_energies():
    # V = r^2/2, mu'=1: exact E = 2k + l + 3/2 -> 1.5, 3.5, 5.5
    sol = solve_radial(lambda r: 0.5 * r**2, l=0, r_max=12.0, n_points=2400, n_states=3)
    got = [q.value for q in sol.energies]
    assert got == pytest.approx([1.5, 3.5, 5.5], abs=1e-4)


def test_3d_harmonic_oscillator_l1_energies():
    sol = solve_radial(lambda r: 0.5 * r**2, l=1, r_max=12.0, n_points=2400, n_states=2)
    got = [q.value for q in sol.energies]
    assert got == pytest.approx([2.5, 4.5], abs=1e-4)


def test_solutions_are_normalized_and_sign_fixed():
    sol = solve_radial(lambda r: 0.5 * r**2, l=0, r_max=12.0, n_points=2400, n_states=3)
    for k in range(3):
        norm = np.trapezoid(sol.u[k] ** 2, sol.r)
        assert norm == pytest.approx(1.0, abs=1e-8)
        first = np.argmax(np.abs(sol.u[k]) > 0.01 * np.abs(sol.u[k]).max())
        assert sol.u[k][first] > 0


def test_state_k_has_k_nodes():
    from atomsim.numerics.analysis import count_sign_changes

    sol = solve_radial(lambda r: 0.5 * r**2, l=0, r_max=12.0, n_points=2400, n_states=4)
    for k in range(4):
        assert count_sign_changes(sol.u[k]) == k


def test_energies_carry_numerical_provenance():
    sol = solve_radial(lambda r: 0.5 * r**2, l=0, r_max=12.0, n_points=1200, n_states=1)
    p = sol.energies[0].provenance
    assert p.fidelity is Fidelity.NUMERICAL
    assert "finite-difference" in p.method
    assert any("grid" in a for a in p.assumptions)


def _coulomb(z):
    return lambda r: -z / r


def test_hydrogen_energies_match_analytic():
    sol = solve_radial(_coulomb(1), l=0, n_states=3)
    for k, q in enumerate(sol.energies):
        exact = exact_energy(k + 1).value
        assert abs(q.value - exact) / abs(exact) < 1e-3, k


def test_helium_plus_and_positronium_match_analytic():
    sol = solve_radial(_coulomb(2), l=0, r_max=60.0, n_states=2)
    assert sol.energies[0].value == pytest.approx(exact_energy(1, Z=2).value, rel=1e-3)

    sol = solve_radial(_coulomb(1), mu_ratio=0.5, l=0, r_max=200.0, n_states=1)
    assert sol.energies[0].value == pytest.approx(
        exact_energy(1, mu_ratio=0.5).value, rel=1e-3
    )


def test_l1_states_match_analytic():
    sol = solve_radial(_coulomb(1), l=1, n_states=2)
    # lowest l=1 state is n=2, then n=3
    assert sol.energies[0].value == pytest.approx(exact_energy(2).value, rel=1e-3)
    assert sol.energies[1].value == pytest.approx(exact_energy(3).value, rel=1e-3)


def test_numerical_1s_wavefunction_overlaps_analytic():
    sol = solve_radial(_coulomb(1), l=0, n_states=1)
    u_exact = sol.r * radial_wavefunction(1, 0, sol.r)
    overlap = np.trapezoid(sol.u[0] * u_exact, sol.r)
    assert overlap > 0.99999


def test_error_estimate_bounds_true_error():
    sol = solve_radial_with_error(_coulomb(1), l=0, n_points=6000, n_states=2)
    for k, q in enumerate(sol.energies):
        est = q.provenance.error_estimate
        assert est is not None and est > 0
        true_err = abs(q.value - exact_energy(k + 1).value)
        assert true_err <= 2.0 * est + 1e-12, (k, true_err, est)
        assert est < 1e-3 * abs(q.value)  # and the estimate itself is small
