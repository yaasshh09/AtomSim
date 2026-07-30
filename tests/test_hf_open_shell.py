"""Hartree-Fock on the atoms that are not closed shells (Phase 21, Task 9).

Two claims are under test. First, that the solver handles a partially filled
subshell at all, sulfur and chlorine included, which the GSZ model cannot touch
because Szydlik and Green never published parameters for them. Second, that the
provenance says what an open shell costs and does not say what it does not:
"average of configuration" is a real limitation for carbon and an empty one for
lithium, whose configuration spans a single term.

Every atom from helium to chlorine is solved once, in a module-scoped fixture,
and shared. The monotonicity test needs each atom's lighter neighbour as well,
so solving on demand would run some atoms two and three times over; at roughly
a second for the light ones and several for chlorine, that is the difference
between a slow test file and an unusable one.
"""

import pytest

from atomsim.atoms import (
    NO_GSZ_PARAMETERS,
    aufbau_configuration,
    is_single_term,
)
from atomsim.hf_atom import solve_hartree_fock

# Open shells whose configuration spans one term only: the average over the
# configuration IS that term's energy, so there is nothing to disclose.
SINGLE_TERM = [("Li", 3), ("B", 5), ("F", 9), ("Na", 11), ("Al", 13), ("Cl", 17)]
# Open shells that really do split. Carbon's 2p2 spans 3P, 1D and 1S, and the
# configuration average is none of the three.
MULTI_TERM = [("C", 6), ("N", 7), ("O", 8), ("Si", 14), ("P", 15), ("S", 16)]
OPEN_SHELL = SINGLE_TERM + MULTI_TERM
# The two atoms GSZ has no parameters for. Hartree-Fock needs no table.
NO_GSZ = [("S", 16), ("Cl", 17)]

_NEEDED = sorted({z for _, z in OPEN_SHELL} | {z - 1 for _, z in OPEN_SHELL} | {10})


@pytest.fixture(scope="module")
def solved():
    return {z: solve_hartree_fock(z, z, aufbau_configuration(z)) for z in _NEEDED}


@pytest.mark.parametrize("symbol,z", OPEN_SHELL)
def test_open_shell_atoms_converge(solved, symbol, z):
    assert solved[z].converged
    assert solved[z].total_energy.value < 0.0


@pytest.mark.parametrize("symbol,z", NO_GSZ)
def test_atoms_gsz_cannot_do_now_work(solved, symbol, z):
    """S and Cl have no Szydlik-Green parameters and the screened model refuses
    them. Hartree-Fock builds its potential out of the orbitals it is solving
    for, so it needs no table, and this is the visible payoff."""
    assert z in NO_GSZ_PARAMETERS
    assert solved[z].total_energy.value < 0.0


@pytest.mark.parametrize("symbol,z", OPEN_SHELL)
def test_total_energy_decreases_monotonically_with_z(solved, symbol, z):
    """A heavier atom binds more tightly. Catches a configuration mis-build."""
    assert solved[z].total_energy.value < solved[z - 1].total_energy.value


@pytest.mark.parametrize("symbol,z", MULTI_TERM)
def test_multi_term_atoms_disclose_the_configuration_average(solved, symbol, z):
    joined = " ".join(solved[z].total_energy.provenance.assumptions)
    assert "average of configuration" in joined


@pytest.mark.parametrize("symbol,z", OPEN_SHELL)
def test_open_shells_disclose_the_spin_restriction(solved, symbol, z):
    """Every open shell pays for restricted Hartree-Fock, single term or not:
    both spins share one radial function, so the core cannot polarize around
    the unpaired electrons."""
    joined = " ".join(solved[z].total_energy.provenance.assumptions)
    assert "spin-polarize" in joined


@pytest.mark.parametrize("symbol,z", SINGLE_TERM)
def test_single_term_atoms_do_not_claim_a_term_average_they_do_not_make(
    solved, symbol, z
):
    """Lithium's 2s1 spans 2S alone, fluorine's 2p5 spans 2P alone.

    The average-of-configuration energy is by construction the degeneracy-
    weighted mean of the configuration's term energies, sum_T (2L+1)(2S+1) E_T
    over sum_T (2L+1)(2S+1). With one term in the sum that mean IS the term
    energy, so there is no averaging error to disclose here, and claiming one
    would inflate the uncertainty a reader thinks the number carries.
    """
    assert is_single_term(aufbau_configuration(z))
    joined = " ".join(solved[z].total_energy.provenance.assumptions)
    assert "not per term" not in joined


def test_closed_shell_does_not_claim_a_term_limitation_it_does_not_have(solved):
    """Neon has no partially filled subshell, so there is nothing to average
    over and nothing to spin-polarize; either disclosure would be misleading."""
    joined = " ".join(solved[10].total_energy.provenance.assumptions)
    assert "not per term" not in joined
    assert "spin-polarize" not in joined


def test_the_orbital_shape_carries_the_same_disclosure_as_the_energy(solved):
    """A caller who plots the orbital and never reads total_energy still needs
    to know the configuration was averaged."""
    for orbital in solved[6].orbitals:
        joined = " ".join(orbital.P.provenance.assumptions)
        assert "average of configuration" in joined


def test_virial_ratio_holds_for_open_shells(solved):
    assert solved[7].virial_ratio.value == pytest.approx(2.0, rel=5e-3)


def test_half_filled_shell_energies_are_ordered_by_shell(solved):
    """Nitrogen's 2p3 must sit above its 2s2, which sits above the 1s2. A sign
    error in the open-shell exchange term shows up here first, as a valence
    level that has fallen below the core."""
    energies = {(o.n, o.l): o.energy.value for o in solved[7].orbitals}
    assert energies[(1, 0)] < energies[(2, 0)] < energies[(2, 1)] < 0.0
