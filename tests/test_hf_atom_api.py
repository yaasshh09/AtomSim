"""The Hartree-Fock API surface, mirroring screened_atom.py one for one.

The point of mirroring is that a caller switches models by swapping a function,
not by rewriting a call. So these tests check the shapes agree as much as they
check the physics.
"""

import numpy as np
import pytest

from atomsim.atoms import aufbau_configuration
from atomsim.hf_atom import (
    evaluate_hf_state,
    hf_radial,
    hf_valence_ionization_energy,
    solve_hartree_fock,
)
from atomsim.provenance import Fidelity
from atomsim.screened_atom import solve_screened_atom, valence_ionization_energy

HARTREE_TO_EV = 27.211386245981

# NIST ASD first ionization energies, retrieved 2026-07-18, already vendored
# for the screened-atom tests.
NIST_IE_EV = {"He": (2, 24.587), "Li": (3, 5.392), "Na": (11, 5.139)}

ALKALIS = ["Li", "Na"]


def koopmans_ev(symbol):
    z, _ = NIST_IE_EV[symbol]
    result = solve_hartree_fock(z, z, aufbau_configuration(z))
    return hf_valence_ionization_energy(result).value * HARTREE_TO_EV


def gsz_ev(symbol):
    z, _ = NIST_IE_EV[symbol]
    result = solve_screened_atom(z, z, aufbau_configuration(z))
    return valence_ionization_energy(result).value * HARTREE_TO_EV


@pytest.mark.parametrize("symbol", list(NIST_IE_EV))
def test_ionization_energies_land_near_nist(symbol):
    """Absolute accuracy against the one column with outside authority.

    0.4 eV is not a tight bound and is not meant to be. Koopmans' theorem
    freezes the orbitals of the ion, and that error does not shrink with a
    better grid, so this pins the model where it actually sits rather than
    where a converged solver could reach.
    """
    assert abs(koopmans_ev(symbol) - NIST_IE_EV[symbol][1]) < 0.4


@pytest.mark.parametrize("symbol", ALKALIS)
def test_hartree_fock_beats_gsz_on_the_alkalis(symbol):
    """The cross-model comparison, on the atoms where it means something.

    Note what the GSZ leg does NOT mean: Szydlik and Green fitted their (d, K)
    parameters TO Hartree-Fock, so agreement between the two models is a check
    on this implementation, not independent confirmation of the physics. The
    NIST column is the one with outside authority.

    Alkalis are the fair ground for Koopmans. Removing the lone valence s
    electron barely disturbs the closed core, so the frozen-orbital error is
    small and what remains is genuine model quality. Helium is the opposite
    case and is pinned separately below rather than quietly dropped.
    """
    reference = NIST_IE_EV[symbol][1]
    assert abs(koopmans_ev(symbol) - reference) < abs(gsz_ev(symbol) - reference)


def test_helium_is_the_case_where_gsz_wins_and_that_is_expected():
    """Pinned because omitting it would be the dishonest way to pass a suite.

    Helium's Koopmans ionization energy is 24.98 eV against NIST's 24.587, an
    overestimate of 0.39 eV, while GSZ's fitted 24.94 lands 0.35 eV out. So the
    fitted model beats the ab-initio one here, and the reason is understood
    rather than mysterious: ionizing helium removes one of only two electrons,
    so the remaining orbital contracts hard and freezing it costs more than it
    does for an alkali. A fitted parameter that was tuned on real atoms can
    absorb some of that; Koopmans cannot.

    Two errors of opposite sign make this legible. Relaxing the ion properly
    (Delta-SCF, E(He+) - E(He)) gives 23.45 eV, UNDERshooting by 1.14 eV, which
    is helium's correlation energy of about 0.042 hartree almost exactly.
    Koopmans wins on helium by cancelling half of one error against none of the
    other, which is luck, not accuracy.
    """
    reference = NIST_IE_EV["He"][1]
    koopmans_error = abs(koopmans_ev("He") - reference)
    gsz_error = abs(gsz_ev("He") - reference)
    assert koopmans_error > gsz_error
    assert koopmans_ev("He") > reference  # frozen orbitals overestimate
    assert koopmans_error < 0.5


def test_the_ionization_energy_discloses_that_it_froze_the_orbitals():
    """IE = -epsilon is an approximation ON TOP of Hartree-Fock, and a reader
    who is handed a number in eV has no way to know that unless it says so."""
    result = solve_hartree_fock(2, 2, aufbau_configuration(2))
    ie = hf_valence_ionization_energy(result)
    assert ie.provenance.fidelity is Fidelity.APPROXIMATION
    joined = " ".join(ie.provenance.assumptions) + " " + ie.provenance.method
    assert "Koopmans" in joined
    assert "relax" in joined


def test_hf_radial_returns_fields_with_matching_grids():
    r_field, p_field = hf_radial(2, 2, 1, 0)
    assert r_field.grid.shape == p_field.grid.shape
    assert np.allclose(p_field.values, r_field.grid**2 * r_field.values**2)


def test_radial_density_integrates_to_one():
    r_field, _ = hf_radial(2, 2, 1, 0)
    norm = np.trapezoid((r_field.grid * r_field.values) ** 2, r_field.grid)
    assert norm == pytest.approx(1.0, rel=1e-3)


def test_hf_radial_rejects_n_not_greater_than_l():
    with pytest.raises(ValueError):
        hf_radial(10, 10, 1, 1)


def test_hf_radial_rejects_a_subshell_the_configuration_does_not_hold():
    """Unlike the screened model, Hartree-Fock has no potential to solve an
    arbitrary channel in: the Fock operator is built FROM the occupied
    orbitals, so an unoccupied subshell has no operator of its own here."""
    with pytest.raises(ValueError, match="not occupied"):
        hf_radial(2, 2, 3, 1)


def test_hf_radial_matches_the_solvers_own_orbital():
    """The field has to be the function the SCF converged, not a resampling
    that quietly drifted. Beryllium's 2s because it has a node, so a sign or
    normalization error cannot hide.

    The tolerance is 1e-5 rather than machine precision for a reason worth
    naming, since it looks like slop otherwise. Interpolating P/r and then
    multiplying by r is not the same operation as interpolating P: both are
    linear, but r varies across an interval, so the two routes differ at second
    order in the SOLVER mesh spacing. Measured, that difference is 6.6e-6 and
    it does NOT shrink when the display grid is refined - identical at 400,
    4096 and 40000 points - which is how you can tell it comes from the mesh
    the orbital was solved on and not from this resampling. Anything actually
    wrong here (wrong subshell, lost normalization, wrong model) would be off
    by order one, not by 6.6e-6.
    """
    result = solve_hartree_fock(4, 4, aufbau_configuration(4))
    orbital = next(o for o in result.orbitals if (o.n, o.l) == (2, 0))
    r_field, _ = hf_radial(4, 4, 2, 0)
    direct = np.interp(r_field.grid, orbital.P.grid, orbital.P.values)
    assert np.allclose(r_field.values * r_field.grid, direct, atol=1e-5)
    assert np.any(r_field.values > 0) and np.any(r_field.values < 0)  # the node


def test_evaluate_hf_state_shape_and_provenance():
    positions = np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0]])
    values = evaluate_hf_state(10, 10, 2, 1, 0, positions)
    assert values.values.shape == (2,)
    assert "Hartree-Fock" in values.provenance.method


def test_evaluate_hf_state_rejects_bad_positions():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        evaluate_hf_state(2, 2, 1, 0, 0, np.array([0.5, 0.0, 0.0]))


def test_evaluate_hf_state_is_finite_at_the_origin():
    """r = 0 is where R = P/r divides by zero, so it is the one point worth
    checking explicitly."""
    values = evaluate_hf_state(2, 2, 1, 0, 0, np.zeros((1, 3)))
    assert np.all(np.isfinite(values.values))


def test_screening_refinement_now_points_at_the_implementation():
    from atomsim.numerics.screening import screening_provenance

    refinement = screening_provenance(10, 10).refinement
    assert "hf_atom" in refinement
    assert "a later phase" not in refinement
