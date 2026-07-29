"""Pins the SCF loop and the total energy computed three independent ways.

Route 1 assembles the energy functional term by term. Route 2 uses the orbital
identity E = 1/2 sum_a q_a (I(a) + eps_a). The two are algebraically identical
but share no code beyond the one-electron integral, so a wrong angular
coefficient shows up as a disagreement instead of as a wrong number in both.
Route 3 is the virial ratio, which is a property of the converged solution
rather than of the code that assembled it.

The SCF solves are module-scoped fixtures because each one costs seconds to
minutes; recomputing helium once per test made this file the slowest in the
suite for no added coverage.
"""

import numpy as np
import pytest

from atomsim.numerics.hartree_fock import (
    HFConvergenceError,
    kinetic_and_potential,
    orbital_energy,
    scf,
    total_energy_direct,
    total_energy_from_orbitals,
)
from atomsim.numerics.hf_terms import Subshell
from atomsim.numerics.mesh import uniform_mesh


def coulomb(z):
    return lambda r: -z / r


def start(mesh, z, shells):
    """Hydrogenic warm start: P_1s at effective charge z."""
    p = 2.0 * z**1.5 * mesh.r * np.exp(-z * mesh.r)
    p /= np.sqrt(np.trapezoid(p**2, mesh.r))
    return tuple(Subshell(n=n, l=l, q=q, p=p.copy()) for n, l, q in shells)


@pytest.fixture(scope="module")
def mesh():
    """A uniform mesh, so these results stay comparable to the ones this
    module was originally validated against."""
    return uniform_mesh(30.0, 30000)


@pytest.fixture(scope="module")
def hydrogen(mesh):
    return scf(1, start(mesh, 1.0, [(1, 0, 1)]), coulomb(1.0), mesh)


@pytest.fixture(scope="module")
def helium(mesh):
    return scf(2, start(mesh, 1.7, [(1, 0, 2)]), coulomb(2.0), mesh)


@pytest.fixture(scope="module")
def beryllium(mesh):
    return scf(4, start(mesh, 3.0, [(1, 0, 2), (2, 0, 2)]), coulomb(4.0), mesh)


def quadrature_energies(sol, z, mesh):
    """eps_a from the same quadrature as I(a); see orbital_energy."""
    return tuple(
        orbital_energy(sol.subshells, i, z, mesh)
        for i in range(len(sol.subshells))
    )


def test_hydrogen_is_exactly_minus_one_half(hydrogen, mesh):
    """One electron: HF must reduce to the bare Coulomb problem with no
    self-interaction whatsoever. This is the phase's sharpest anchor and it
    costs nothing."""
    assert total_energy_direct(1, hydrogen.subshells, mesh) == pytest.approx(
        -0.5, rel=2e-4
    )
    assert hydrogen.energies[0] == pytest.approx(-0.5, rel=2e-4)


def test_helium_total_energy_is_physical(helium, mesh):
    """No vendored number needed to catch a gross error: helium must sit
    between the non-interacting limit (-4) and the single-ion limit (-2)."""
    e = total_energy_direct(2, helium.subshells, mesh)
    assert -4.0 < e < -2.0


def test_the_two_energy_routes_agree(helium, mesh):
    """Direct assembly and the orbital identity E = 1/2 sum q (I + eps) are
    algebraically identical, so any disagreement is a coding error, not a
    numerical one. Tolerance is tight on purpose.

    The identity holds exactly only when eps and I are quadratured the same
    way, so this is fed orbital_energy rather than the finite-difference
    eigenvalue. Feeding it the eigenvalue instead tests the discretization,
    which is what test_eigenvalue_and_quadrature_orbital_energies_agree does.
    """
    direct = total_energy_direct(2, helium.subshells, mesh)
    identity = total_energy_from_orbitals(
        helium.subshells, quadrature_energies(helium, 2, mesh), 2, mesh
    )
    assert direct == pytest.approx(identity, abs=1e-8)


def test_the_two_energy_routes_agree_for_beryllium(beryllium, mesh):
    """Two subshells, so cross-shell direct and exchange coefficients are in
    play. Helium alone would not exercise them."""
    direct = total_energy_direct(4, beryllium.subshells, mesh)
    identity = total_energy_from_orbitals(
        beryllium.subshells, quadrature_energies(beryllium, 4, mesh), 4, mesh
    )
    assert direct == pytest.approx(identity, abs=1e-8)


def test_eigenvalue_and_quadrature_orbital_energies_agree(helium, mesh):
    """The two ways of getting eps_a differ only by discretization: the
    eigenvalue comes from the finite-difference operator, orbital_energy from
    the trapezoid quadrature. They must agree to O(h^2), which on this mesh is
    a few times 1e-5 - large enough to break an abs=1e-8 energy identity, small
    enough to be irrelevant physically. Pinned so neither drifts unnoticed.
    """
    eig = helium.energies[0]
    quad = orbital_energy(helium.subshells, 0, 2, mesh)
    assert quad == pytest.approx(eig, abs=1e-4)
    assert quad != eig  # they are genuinely different quadratures


def test_virial_ratio_is_two(helium, mesh):
    """At a converged HF solution in a pure Coulomb field, -V/T = 2 exactly.
    Departure measures mesh and box error, not model error."""
    t, v = kinetic_and_potential(2, helium.subshells, mesh)
    assert -v / t == pytest.approx(2.0, rel=1e-3)


def test_energy_equals_minus_kinetic_at_convergence(helium, mesh):
    t, _ = kinetic_and_potential(2, helium.subshells, mesh)
    e = total_energy_direct(2, helium.subshells, mesh)
    assert e == pytest.approx(-t, rel=1e-3)


def test_beryllium_converges_and_orders_its_shells(beryllium):
    assert beryllium.energies[0] < beryllium.energies[1] < 0.0


def test_beryllium_orbital_energies_match_the_published_ones(beryllium):
    """Bunge's tabulated Be orbital energies are -4.7326699 and -0.3092695.
    They are not the total energy, so this is an independent check on the
    solver that does not go through the energy functional at all.
    """
    assert beryllium.energies[0] == pytest.approx(-4.7326699, rel=1e-4)
    assert beryllium.energies[1] == pytest.approx(-0.3092695, rel=1e-4)


def test_beryllium_shells_come_out_orthogonal(beryllium, mesh):
    """1s and 2s are eigenvectors of DIFFERENT operators - each subshell has
    its own Fock operator in this scheme - so their orthogonality is a result,
    not a construction. If it degraded, the energy functional would silently
    double-count.
    """
    overlap = np.trapezoid(
        beryllium.subshells[0].p * beryllium.subshells[1].p, mesh.r
    )
    assert abs(overlap) < 1e-5


def test_residual_history_is_monotone_enough_to_show_convergence(helium):
    assert helium.residual_history[-1] < helium.residual_history[0]
    assert helium.residual_history[-1] < 1e-8


def test_non_convergence_raises_rather_than_returning(mesh):
    with pytest.raises(HFConvergenceError, match="SCF did not converge"):
        scf(2, start(mesh, 1.7, [(1, 0, 2)]), coulomb(2.0), mesh,
            max_iterations=1, tol=1e-14)


def test_mixing_parameter_is_validated(mesh):
    with pytest.raises(ValueError, match="mixing parameter"):
        scf(2, start(mesh, 1.7, [(1, 0, 2)]), coulomb(2.0), mesh, alpha=0.0)
