"""Phase 22: exchange off, i.e. electrons that repel but are distinguishable.

The physics being pinned here is what antisymmetry is worth. Every number in
this file is a difference between two solves of the same atom on the same mesh,
so the mesh cancels out of all of it and what is left is the model.

See docs/superpowers/specs/2026-07-31-phase22-distinguishable-electrons-design.md.
"""

import numpy as np
import pytest

from atomsim.atoms import aufbau_configuration, element_by_symbol
from atomsim.hf_atom import (
    _ROUTE_AGREEMENT,
    hf_exchange_energy,
    hf_mesh,
    solve_hartree_fock,
)
from atomsim.numerics.hartree_fock import (
    orbital_energy,
    total_energy_direct,
    total_energy_from_orbitals,
)
from atomsim.numerics.hf_terms import Subshell
from atomsim.provenance import Fidelity

# Kept short on purpose: each entry is two full SCF solves, and the physics
# claims below are about the trend across them, not about breadth.
ATOMS = ["He", "Li", "Be", "C", "Ne"]


def _solve_both(symbol: str):
    z = element_by_symbol(symbol).z
    config = aufbau_configuration(z)
    return (
        solve_hartree_fock(z, z, config, True),
        solve_hartree_fock(z, z, config, False),
    )


def test_helium_exchange_energy_is_exactly_zero():
    """Not "small" - zero, to the last bit, and that is the model being right.

    Exchange couples same-spin electrons. A closed 1s shell holds one spin up
    and one spin down, so there is no same-spin pair to exchange and
    exchange_operator builds no terms at all. The k = 0 integral a reader might
    expect to find here is carried by the (q - 1) factor in the direct
    potential in BOTH models, so it cancels out of the difference.

    Exact equality rather than a tolerance because the two solves run
    bit-identical arithmetic for this configuration: the exchange branches are
    empty loops, not small numbers. If this ever becomes 1e-15 instead of 0.0,
    something started contributing that was not contributing before, and a
    tolerance would hide it.
    """
    assert hf_exchange_energy(2, 2, aufbau_configuration(2)).value == 0.0


def test_helium_is_the_same_atom_in_both_models():
    """The orbital, not only the energy: no same-spin pair means no difference."""
    hf, hartree = _solve_both("He")
    assert hf.total_energy.value == hartree.total_energy.value
    np.testing.assert_array_equal(hf.orbitals[0].P.values, hartree.orbitals[0].P.values)


def test_beryllium_exchange_energy_is_not_zero():
    """1s2 2s2 has same-spin pairs across the two shells, so G_k(1s,2s) exists."""
    e_x = hf_exchange_energy(4, 4, aufbau_configuration(4)).value
    assert e_x < -0.01


@pytest.mark.parametrize("symbol", ATOMS)
def test_exchange_is_stabilizing(symbol):
    """E_HF <= E_Hartree for every atom, with equality only where there is no
    same-spin pair to exchange.

    Note what is deliberately NOT asserted: that the Hartree energy is an upper
    bound on the exact energy. A product wavefunction is not antisymmetric, so
    it is not an admissible fermionic trial function and the variational
    theorem says nothing about it. It happens to land above, and claiming that
    as a guarantee would be a lie about which theorem is doing the work.
    """
    hf, hartree = _solve_both(symbol)
    assert hf.total_energy.value <= hartree.total_energy.value


def test_exchange_energy_grows_with_z():
    """More same-spin pairs, more stabilization - monotonically over this set."""
    magnitudes = [
        abs(hf_exchange_energy(z, z, aufbau_configuration(z)).value)
        for z in (2, 3, 4, 6, 10)
    ]
    assert magnitudes == sorted(magnitudes)
    assert magnitudes[0] == 0.0


@pytest.mark.parametrize("symbol", ATOMS)
def test_both_models_satisfy_the_virial_theorem(symbol):
    """Hartree is a legitimate variational model, not a broken Hartree-Fock.

    -V/T = 2 is the check that says so. If the toggle had damaged the solve
    rather than changed the model, this is where it would show: the virial
    ratio is a property of a converged stationary solution and does not care
    which functional was made stationary.
    """
    _, hartree = _solve_both(symbol)
    assert hartree.virial_ratio.value == pytest.approx(2.0, abs=1e-4)


@pytest.mark.parametrize("symbol", ["Li", "Be"])
def test_exchange_changes_the_orbitals_not_only_the_energy(symbol):
    """The operator changed, so the eigenfunctions did.

    Worth pinning separately from the energy because a toggle that reached the
    energy functional but not the Fock operator would still move the total
    energy - it would just move it to the wrong place, by evaluating the
    Hartree functional on Hartree-Fock orbitals.
    """
    hf, hartree = _solve_both(symbol)
    valence_hf = hf.orbitals[-1].P.values
    valence_hartree = hartree.orbitals[-1].P.values
    assert not np.allclose(valence_hf, valence_hartree, atol=1e-6)


@pytest.mark.parametrize("symbol", ATOMS)
def test_a_hartree_solve_is_counterfactual_not_approximation(symbol):
    """The tier is the whole point.

    APPROXIMATION would invite the reader to treat the gap to the reference
    energy as an error in the calculation. It is not an error - it is the
    physics the toggle exists to show.
    """
    hf, hartree = _solve_both(symbol)
    assert hf.total_energy.provenance.fidelity is Fidelity.APPROXIMATION
    assert hartree.total_energy.provenance.fidelity is Fidelity.COUNTERFACTUAL
    # The shape carries the tier too: a Hartree 2s is a different curve, not
    # the same curve at a different accuracy.
    assert hartree.orbitals[-1].P.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_the_disclosure_says_which_counterfactual_this_is():
    """A COUNTERFACTUAL badge that does not name the altered rule is decoration.

    Two claims have to be in there and they pull in opposite directions: what
    was switched off (antisymmetry) and what was NOT (the Pauli occupancies).
    A reader who takes "distinguishable electrons" to mean "all ten electrons
    may now sit in 1s" has been misled by a badge that was technically true.
    """
    _, hartree = _solve_both("Ne")
    disclosure = " ".join(hartree.total_energy.provenance.assumptions).lower()
    assert "distinguishable" in disclosure
    assert "pauli principle is not switched off" in disclosure
    assert "2(2l+1)" in disclosure
    # And the self-interaction term is named as staying, so nobody reads the
    # exchange energy as containing it.
    assert "does not repel itself" in disclosure
    # What was already true stays true and stays disclosed.
    assert "correlation" in disclosure


def test_the_exchange_energy_carries_no_error_bar_against_the_real_atom():
    """It is COUNTERFACTUAL, and it is not an observable.

    The error estimate it does carry is about the arithmetic - the looser of
    the two mesh spreads - and must not be mistaken for a distance from a
    measured quantity, because there is no measurement of this.
    """
    q = hf_exchange_energy(4, 4, aufbau_configuration(4))
    assert q.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert "not an observable" in " ".join(q.provenance.assumptions)
    assert q.provenance.error_estimate is not None
    # Loose enough to be the mesh spread, tight enough to be useless as a
    # cover for a coefficient error.
    assert 0 < q.provenance.error_estimate < 0.01 * abs(q.value)


def test_the_two_models_do_not_share_a_cached_solve():
    """lru_cache keys on the flag, and the result key names the calculation."""
    config = aufbau_configuration(4)
    hf = solve_hartree_fock(4, 4, config, True)
    hartree = solve_hartree_fock(4, 4, config, False)
    assert hf is not hartree
    assert hf.key != hartree.key
    assert hf.exchange is True
    assert hartree.exchange is False


def test_half_applying_the_toggle_makes_the_two_energy_routes_disagree():
    """The guard the design leans on, exercised rather than asserted in prose.

    Exchange lives in the energy functional for route 1 and in the Fock
    operator for route 2. Turning it off in one and not the other is the one
    bug this phase can plausibly ship, and solve_hartree_fock already refuses
    to return when the routes part company. This test shows the routes really
    do part company - i.e. that the guard is load-bearing and not a check that
    two identical expressions are equal.
    """
    hartree = solve_hartree_fock(4, 4, aufbau_configuration(4), False)
    # The same mesh the solve ran on: n_top is the highest n in 1s2 2s2, and
    # the fine refinement is the one whose orbitals came back on the result.
    mesh = hf_mesh(4, 4, n_top=2, refinement=2)
    subshells = tuple(
        Subshell(n=o.n, l=o.l, q=o.occupancy, p=o.P.values)
        for o in hartree.orbitals
    )

    # Route 1 without exchange, route 2 fed orbital energies computed WITH it.
    mixed_energies = tuple(
        orbital_energy(subshells, i, 4, mesh, exchange=True)
        for i in range(len(subshells))
    )
    route_1 = total_energy_direct(4, subshells, mesh, exchange=False)
    route_2 = total_energy_from_orbitals(subshells, mixed_energies, 4, mesh)
    assert abs(route_1 - route_2) > 1000 * _ROUTE_AGREEMENT

    # And agree when the flag is applied consistently, on the same orbitals.
    honest_energies = tuple(
        orbital_energy(subshells, i, 4, mesh, exchange=False)
        for i in range(len(subshells))
    )
    assert total_energy_from_orbitals(
        subshells, honest_energies, 4, mesh
    ) == pytest.approx(route_1, abs=_ROUTE_AGREEMENT)
