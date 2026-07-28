"""Pins the matrix-free Fock channel solve.

The grid convention here is not cosmetic. The 3-point stencil imposes the
Dirichlet condition u = 0 one step below r[0], so the grid has to satisfy
r[0] == h for that step to land on the origin. A grid starting at 1e-5 with
h = 2e-3 misplaces the origin and costs 4.6% on hydrogen's ground state while
still looking perfectly smooth. Hence radial_grid below, and hence the explicit
check in local_hamiltonian_bands.
"""

import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal

from atomsim.numerics.hartree_fock import (
    HFConvergenceError,
    fock_operator,
    local_hamiltonian_bands,
    solve_channel,
)
from atomsim.numerics.hf_terms import Subshell


def radial_grid(r_max=40.0, n_points=20000):
    """Uniform grid with r[0] == h, matching radial_solver.solve_radial."""
    h = r_max / (n_points + 1)
    return h * np.arange(1, n_points + 1)


@pytest.fixture
def grid():
    return radial_grid()


def coulomb(z):
    return lambda r: -z / r


def hydrogenic_1s(r, z):
    return 2.0 * z**1.5 * r * np.exp(-z * r)


def test_one_electron_channel_reproduces_hydrogen(grid):
    """With q = 1 there is no interaction at all, so this must return the
    analytic hydrogen levels: -1/2, -1/8, -1/18."""
    shells = (Subshell(n=1, l=0, q=1, p=np.zeros_like(grid)),)
    sol = solve_channel(shells, 0, coulomb(1.0), l=0, r=grid, n_states=3,
                        guess=None)
    assert sol.energies[0] == pytest.approx(-0.5, rel=2e-4)
    assert sol.energies[1] == pytest.approx(-0.125, rel=2e-4)
    assert sol.energies[2] == pytest.approx(-1.0 / 18.0, rel=2e-3)


def test_one_electron_p_channel_reproduces_hydrogen(grid):
    """l = 1 exercises the centrifugal term, which the s channel cannot."""
    shells = (Subshell(n=2, l=1, q=1, p=np.zeros_like(grid)),)
    sol = solve_channel(shells, 0, coulomb(1.0), l=1, r=grid, n_states=2,
                        guess=None)
    assert sol.energies[0] == pytest.approx(-0.125, rel=2e-4)
    assert sol.energies[1] == pytest.approx(-1.0 / 18.0, rel=2e-3)


def test_returned_orbitals_are_normalized(grid):
    shells = (Subshell(n=1, l=0, q=1, p=np.zeros_like(grid)),)
    sol = solve_channel(shells, 0, coulomb(1.0), l=0, r=grid, n_states=3,
                        guess=None)
    for u in sol.orbitals:
        assert np.trapezoid(u**2, grid) == pytest.approx(1.0, rel=1e-8)


def test_returned_orbitals_are_orthogonal(grid):
    shells = (Subshell(n=1, l=0, q=1, p=np.zeros_like(grid)),)
    sol = solve_channel(shells, 0, coulomb(1.0), l=0, r=grid, n_states=3,
                        guess=None)
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = np.trapezoid(sol.orbitals[i] * sol.orbitals[j], grid)
            assert abs(overlap) < 1e-8


def test_fock_operator_is_symmetric_on_random_vectors(grid):
    """<x, F y> = <F x, y> to quadrature accuracy. The pair-potential
    quadrature is not exactly symmetric, so this is a tolerance, not an
    identity, and the tolerance is what justifies the explicit
    re-orthogonalization in solve_channel."""
    rng = np.random.default_rng(0)
    p = grid * np.exp(-grid)
    p /= np.sqrt(np.trapezoid(p**2, grid))
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    op = fock_operator(shells, 0, coulomb(2.0), l=0, r=grid)
    x = rng.standard_normal(grid.size)
    y = rng.standard_normal(grid.size)
    left = float(x @ op.matvec(y))
    right = float(op.matvec(x) @ y)
    assert left == pytest.approx(right, rel=1e-6)


def test_helium_direct_term_changes_the_answer(grid):
    """Helium's 1s eigenvalue must sit well above the bare Z=2 hydrogenic -2.0.

    Note this guards the DIRECT term, not exchange: a single closed s subshell
    has no exchange term at all (test_helium_has_no_exchange_term pins that),
    so nothing here could detect a broken exchange operator. That guard is
    test_exchange_shifts_the_eigenvalue below, which needs two subshells.
    """
    p = hydrogenic_1s(grid, 2.0)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    sol = solve_channel(shells, 0, coulomb(2.0), l=0, r=grid, n_states=1,
                        guess=None)
    assert -2.0 < sol.energies[0] < -0.5


def test_exchange_shifts_the_eigenvalue(grid):
    """The real guard against a silently-zero exchange operator.

    Solve the 1s channel of a Be-like 1s2 2s2 configuration, then solve the
    same channel with the local part only. Exchange is attractive here, so
    dropping it must move the eigenvalue by a visible amount.
    """
    p1 = hydrogenic_1s(grid, 4.0)
    p2 = np.sqrt(2.0) * grid * (1.0 - grid) * np.exp(-grid)
    p2 /= np.sqrt(np.trapezoid(p2**2, grid))
    shells = (Subshell(n=1, l=0, q=2, p=p1), Subshell(n=2, l=0, q=2, p=p2))

    full = solve_channel(shells, 0, coulomb(4.0), l=0, r=grid, n_states=1,
                         guess=None)

    from atomsim.numerics.hf_terms import direct_potential

    v_local = coulomb(4.0)(grid) + direct_potential(shells, 0, grid)
    diag, offdiag = local_hamiltonian_bands(v_local, 0, grid)
    local_only = eigh_tridiagonal(diag, offdiag, select="i",
                                  select_range=(0, 0), eigvals_only=True)[0]

    assert abs(full.energies[0] - local_only) > 1e-3
    assert full.energies[0] < local_only  # exchange lowers it


def test_warm_start_does_not_cost_more_iterations(grid):
    """The preconditioner claim, made falsifiable: restarting from a converged
    orbital must not take more LOBPCG iterations than starting cold."""
    p = hydrogenic_1s(grid, 2.0)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    cold = solve_channel(shells, 0, coulomb(2.0), l=0, r=grid, n_states=1,
                         guess=None)
    warm = solve_channel(shells, 0, coulomb(2.0), l=0, r=grid, n_states=1,
                         guess=cold.orbitals)
    assert warm.iterations <= cold.iterations
    assert warm.energies[0] == pytest.approx(cold.energies[0], rel=1e-8)


def test_iteration_count_shows_the_preconditioner_is_working(grid):
    """Unpreconditioned LOBPCG on this operator needs hundreds of iterations.
    If this ever climbs past ~50 the preconditioner has silently stopped
    being applied."""
    p = hydrogenic_1s(grid, 2.0)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    sol = solve_channel(shells, 0, coulomb(2.0), l=0, r=grid, n_states=1,
                        guess=None)
    assert sol.iterations < 50


def test_solution_reports_the_residual_it_achieved(grid):
    """The convergence claim has to be inspectable, not just asserted."""
    p = hydrogenic_1s(grid, 2.0)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    sol = solve_channel(shells, 0, coulomb(2.0), l=0, r=grid, n_states=1,
                        guess=None)
    assert 0.0 < sol.residual < 1e-4


def test_unreachable_accuracy_raises_instead_of_returning_a_number(grid):
    """The gate that was missing. An impossible ceiling must produce an error,
    not a plausible-looking eigenvalue: LOBPCG stagnates above its quadrature
    floor without saying so in any way that len(history) reveals.
    """
    p = hydrogenic_1s(grid, 2.0)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    with pytest.raises(HFConvergenceError, match="residual"):
        solve_channel(shells, 0, coulomb(2.0), l=0, r=grid, n_states=1,
                      guess=None, tol=1e-30, residual_ceiling=1e-30,
                      maxiter=10)


def test_exchange_channel_converges_within_the_ceiling(grid):
    """The case that exposed the missing gate: a channel with exchange active
    stagnates near 1e-6, which must count as converged rather than as failure.
    """
    p1 = hydrogenic_1s(grid, 4.0)
    p2 = np.sqrt(2.0) * grid * (1.0 - grid) * np.exp(-grid)
    p2 /= np.sqrt(np.trapezoid(p2**2, grid))
    shells = (Subshell(n=1, l=0, q=2, p=p1), Subshell(n=2, l=0, q=2, p=p2))
    sol = solve_channel(shells, 0, coulomb(4.0), l=0, r=grid, n_states=1,
                        guess=None)
    assert sol.residual < 1e-4


def test_local_bands_reject_a_grid_whose_origin_is_misplaced():
    """r[0] != h means the Dirichlet step lands somewhere other than r = 0.
    That is a 4.6% error on hydrogen that no smoothness check would catch, so
    it has to be refused rather than absorbed."""
    bad = np.linspace(1e-5, 40.0, 20000)
    with pytest.raises(ValueError, match="r\\[0\\]"):
        local_hamiltonian_bands(-1.0 / bad, 0, bad)


def test_local_bands_reject_a_non_uniform_grid():
    bad = np.geomspace(1e-3, 40.0, 20000)
    with pytest.raises(ValueError, match="uniform"):
        local_hamiltonian_bands(-1.0 / bad, 0, bad)


def test_local_bands_match_the_radial_solver_discretization(grid):
    """The two engines must agree where they overlap, or hydrogen would come
    out differently depending on which solver was asked."""
    from atomsim.numerics.radial_solver import solve_radial

    diag, offdiag = local_hamiltonian_bands(coulomb(1.0)(grid), 0, grid)
    bands = eigh_tridiagonal(diag, offdiag, select="i", select_range=(0, 2),
                             eigvals_only=True)
    reference = solve_radial(coulomb(1.0), l=0, r_max=40.0,
                             n_points=grid.size, n_states=3)
    assert np.allclose(bands, [e.value for e in reference.energies], rtol=1e-12)
