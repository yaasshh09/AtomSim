"""Phase 24: the Pauli exclusion principle switched off, and the atom collapses.

Phase 22 removed exchange and deliberately kept the occupancy cap, and said so
in its disclosure. This removes the root of both. What is pinned here is that
the collapsed atom is a real solve of a stated model rather than a number that
appeared because nothing raised: it is checked against a closed-form
variational energy that comes from outside this codebase, and against the
inequality that formula must satisfy.

See docs/specs/2026-07-31-phase24-pauli-off-design.md.
"""

import pytest

from atomsim.atoms import (
    aufbau_configuration,
    element_by_symbol,
    is_ground,
    subshell_terms,
    validate_config,
)
from atomsim.hf_atom import (
    collapsed_variational_energy,
    hf_mean_radius,
    pauli_collapse,
    solve_hartree_fock,
)
from atomsim.provenance import Fidelity

# Two full SCF solves each, so this stays short. Helium is the calibration case
# (collapse changes nothing there), beryllium is the first atom the cap
# actually binds, and neon is where the effect is loud.
COLLAPSE_ATOMS = ["He", "Be", "Ne"]


def _collapsed_config(n_electrons: int):
    return aufbau_configuration(n_electrons, pauli=False)


# --------------------------------------------------------------------------
# The configuration combinatorics, with and without the cap
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_electrons", [1, 2, 3, 10, 18])
def test_aufbau_without_pauli_is_one_orbital(n_electrons):
    """The whole of "why chemistry exists", as one assertion.

    The Madelung walk exists only because the cap forces it. Remove the cap and
    the ground configuration has no structure left to have.
    """
    assert _collapsed_config(n_electrons) == (((1, 0), n_electrons),)


def test_collapsed_configuration_is_ground_only_under_its_own_rule():
    """1s^10 is the ground state of the collapsed world and illegal in ours.

    is_ground has to be asked which rule it is judging under, because the same
    configuration gets opposite answers, and defaulting silently to one of them
    would make the other unrepresentable.
    """
    collapsed = _collapsed_config(10)
    assert is_ground(collapsed, pauli=False)
    assert not is_ground(collapsed, pauli=True)


def test_validate_drops_the_cap_but_keeps_n_greater_than_l():
    """Only one of the two checks is the exclusion principle.

    n > l is not Pauli - it is what makes (n, l) name a radial function at all.
    A subshell with n <= l has no orbital to put an electron in whether or not
    electrons exclude each other, so lifting the cap must not lift that.
    """
    validate_config((((1, 0), 10),), pauli=False)
    with pytest.raises(ValueError, match="exceeds capacity"):
        validate_config((((1, 0), 10),), pauli=True)
    with pytest.raises(ValueError, match="n must be > l"):
        validate_config((((1, 1), 4),), pauli=False)


def test_term_structure_raises_above_capacity_rather_than_answering():
    """A term symbol for 1s^10 is not a hard question, it is a meaningless one.

    subshell_terms counts microstates by enumerating distinct spin-orbital
    assignments, which is the exclusion principle from top to bottom. Above
    capacity there are no such assignments, and returning an empty tuple would
    read as "this configuration spans no terms" rather than "the question does
    not apply".
    """
    with pytest.raises(ValueError, match="out of range"):
        subshell_terms(0, 10)


# --------------------------------------------------------------------------
# The refusal
# --------------------------------------------------------------------------


def test_pauli_off_with_exchange_on_is_refused_not_silently_corrected():
    """Not a model, so not a number.

    A Slater determinant with two electrons in the same spin-orbital is
    identically zero, so there is no wavefunction for an exchange integral to
    act on. Quietly flipping the caller's exchange flag would hide that they
    asked for a state that does not exist.
    """
    with pytest.raises(ValueError, match="not a model"):
        solve_hartree_fock(4, 4, _collapsed_config(4), exchange=True, pauli=False)


def test_the_refusal_says_why_and_says_what_to_pass():
    message = str(
        pytest.raises(
            ValueError,
            solve_hartree_fock,
            4, 4, _collapsed_config(4), True, False,
        ).value
    )
    assert "antisymmetry" in message
    assert "exchange=False" in message


# --------------------------------------------------------------------------
# The closed-form ground truth, and the inequality it has to satisfy
# --------------------------------------------------------------------------


def test_variational_formula_reproduces_the_textbook_helium_number():
    """A number nobody in this repo chose.

    N electrons in one hydrogenic 1s of exponent zeta with no exchange
    minimize at zeta* = Z - (5/16)(N - 1). At Z = N = 2 that is 27/16 = 1.6875
    and -2.8477 hartree, which is the variational helium result out of any
    quantum mechanics textbook. If this drifts, the check on every collapsed
    solve below has stopped being external.
    """
    zeta, energy = collapsed_variational_energy(2, 2)
    assert zeta.value == pytest.approx(1.6875, abs=1e-12)
    assert energy.value == pytest.approx(-2.84765625, abs=1e-12)


def test_variational_formula_is_counterfactual_despite_being_closed_form():
    """The tier is truth-distance, not arithmetic precision.

    This is an exact statement about an atom that does not exist. EXACT would
    say the opposite of what is true about it.
    """
    zeta, energy = collapsed_variational_energy(10, 10)
    assert zeta.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert energy.provenance.fidelity is Fidelity.COUNTERFACTUAL
    # Z = 10: zeta* = 10 - 45/16 = 7.1875, and the spec's -258 hartree.
    assert zeta.value == pytest.approx(7.1875, abs=1e-12)
    assert energy.value == pytest.approx(-258.30078125, abs=1e-9)


@pytest.mark.parametrize("symbol", COLLAPSE_ATOMS)
def test_collapsed_scf_sits_below_the_exponential_bound_and_near_it(symbol):
    """E_SCF <= E(zeta*), and within a few percent.

    The SCF optimizes the 1s radial FUNCTION; the formula optimizes only the
    exponent of an exponential. The SCF therefore searches a strictly larger
    space and cannot come out above. This is a real check rather than a
    tautology: a wrong angular coefficient in the collapsed branch would still
    converge to something smooth and would break the inequality or the
    closeness, because the pair count it multiplies is what the formula's
    N(N-1)/2 also counts.
    """
    z = element_by_symbol(symbol).z
    collapsed = solve_hartree_fock(
        z, z, _collapsed_config(z), exchange=False, pauli=False
    )
    _, bound = collapsed_variational_energy(z, z)
    assert collapsed.total_energy.value <= bound.value
    assert collapsed.total_energy.value == pytest.approx(bound.value, rel=0.05)


# --------------------------------------------------------------------------
# What the collapsed atom looks like
# --------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", COLLAPSE_ATOMS)
def test_the_collapsed_ladder_has_exactly_one_rung(symbol):
    """No shells, so nothing for a Levels view to draw a structure out of."""
    z = element_by_symbol(symbol).z
    collapsed = solve_hartree_fock(
        z, z, _collapsed_config(z), exchange=False, pauli=False
    )
    assert len(collapsed.orbitals) == 1
    only = collapsed.orbitals[0]
    assert (only.n, only.l, only.occupancy) == (1, 0, z)


def test_helium_is_the_case_where_switching_pauli_off_changes_nothing():
    """1s^2 is already the ground configuration, so no cap needed lifting.

    Bit-exact rather than approximate, and that matters: the collapsed branch
    must reach the identical arithmetic when the collapse is a no-op, not
    merely a nearby number. A tolerance here would hide a branch that quietly
    took a different path for the same physics.
    """
    hartree = solve_hartree_fock(2, 2, aufbau_configuration(2), exchange=False)
    collapsed = solve_hartree_fock(
        2, 2, _collapsed_config(2), exchange=False, pauli=False
    )
    assert collapsed.total_energy.value == hartree.total_energy.value
    assert hf_mean_radius(collapsed).value == hf_mean_radius(hartree).value


def test_hydrogen_mean_radius_is_the_analytic_three_halves():
    """The one atom where <r> has a closed form, used to check the machinery.

    A one-electron Hartree-Fock solve has no repulsion to build, so its 1s is
    the exact hydrogenic one and <r> = 3a/2Z = 1.5 bohr. This tests the
    quadrature and the occupancy weighting, not the model.
    """
    hydrogen = solve_hartree_fock(1, 1, aufbau_configuration(1))
    assert hf_mean_radius(hydrogen).value == pytest.approx(1.5, rel=2e-4)


def test_mean_radius_carries_no_error_bar_in_the_wrong_dimension():
    """The solve estimates its spread in hartree. A length is not that.

    Inheriting the energy's error estimate onto a radius would not be a loose
    bar, it would be a number in the wrong unit, which is the exact shape of a
    quantity that lies about itself.
    """
    radius = hf_mean_radius(solve_hartree_fock(4, 4, aufbau_configuration(4)))
    assert radius.unit == "bohr"
    assert radius.provenance.error_estimate is None


# --------------------------------------------------------------------------
# The teaching payoff, as inequalities
# --------------------------------------------------------------------------


def test_the_collapsed_atom_is_far_more_bound():
    """Nothing holds the electrons out of the deep well any more.

    Neon's ten electrons in one 1s land near -264 hartree against the real
    -128.5. The assertion is the sign and the scale, not the digits: the point
    is that the cap is what costs the real atom half its binding.
    """
    collapse = pauli_collapse(10)
    assert collapse.binding_change.value < 0
    assert collapse.collapsed.total_energy.value < 2 * collapse.real.total_energy.value


def test_collapsed_size_falls_monotonically_while_the_real_one_does_not():
    """This inequality IS the periodic table, stated as its own absence.

    With the cap on, a new shell opens every period and the atom jumps back
    out, so size oscillates across Z - that oscillation is chemistry. With the
    cap off, zeta* = (11/16)Z + 5/16 for a neutral atom, so <r> falls like 1/Z
    forever and no period ever starts.
    """
    collapses = [pauli_collapse(z) for z in (2, 4, 10)]
    collapsed_radii = [c.collapsed_radius.value for c in collapses]
    real_radii = [c.real_radius.value for c in collapses]

    assert collapsed_radii == sorted(collapsed_radii, reverse=True)
    # Beryllium opens the 2s and is bigger than helium; neon has filled the
    # same shell and pulled back in. Sorted order would mean no periodicity.
    assert real_radii != sorted(real_radii, reverse=True)
    for collapse in collapses[1:]:
        assert collapse.radius_ratio.value < 1.0


def test_collapse_compares_two_solves_of_the_same_atom():
    collapse = pauli_collapse(4)
    assert collapse.real.z == collapse.collapsed.z == 4
    assert collapse.real.n_electrons == collapse.collapsed.n_electrons == 4
    assert collapse.real.exchange and collapse.real.pauli
    assert not collapse.collapsed.exchange and not collapse.collapsed.pauli


def test_the_three_models_never_share_a_cache_key():
    """A Hartree-Fock, a Hartree and a collapsed solve of the same atom.

    The key names the calculation rather than the atom, so nothing downstream
    that caches or labels by key can mistake one for another.
    """
    config = aufbau_configuration(4)
    keys = {
        solve_hartree_fock(4, 4, config).key,
        solve_hartree_fock(4, 4, config, exchange=False).key,
        solve_hartree_fock(4, 4, _collapsed_config(4), False, False).key,
    }
    assert len(keys) == 3


# --------------------------------------------------------------------------
# Provenance: what the reader is told
# --------------------------------------------------------------------------


def test_collapsed_solve_is_counterfactual_everywhere_it_reports():
    z = 4
    collapsed = solve_hartree_fock(
        z, z, _collapsed_config(z), exchange=False, pauli=False
    )
    assert collapsed.total_energy.provenance.fidelity is Fidelity.COUNTERFACTUAL
    for orbital in collapsed.orbitals:
        assert orbital.energy.provenance.fidelity is Fidelity.COUNTERFACTUAL
        assert orbital.P.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_the_alteration_leads_the_assumption_list():
    """A badge reading COUNTERFACTUAL without naming the altered rule is decor."""
    collapsed = solve_hartree_fock(4, 4, _collapsed_config(4), False, False)
    first = collapsed.total_energy.provenance.assumptions[0]
    assert first.startswith("COUNTERFACTUAL:")
    assert "Pauli exclusion principle is switched off" in first


def test_disclosure_contradicts_the_weaker_counterfactual_it_supersedes():
    """Phase 22 promises the cap is still on, in those words.

    A reader who learned that disclosure and then meets this one has to be told
    which of the two they are looking at, or the earlier sentence keeps
    applying in their head.
    """
    hartree = solve_hartree_fock(4, 4, aufbau_configuration(4), exchange=False)
    collapsed = solve_hartree_fock(4, 4, _collapsed_config(4), False, False)

    hartree_text = " ".join(hartree.total_energy.provenance.assumptions)
    collapsed_text = " ".join(collapsed.total_energy.provenance.assumptions)

    assert "the Pauli principle is NOT switched off" in hartree_text
    assert "the Pauli principle is NOT switched off" not in collapsed_text
    assert "occupancy cap is gone" in collapsed_text


def test_disclosure_says_exchange_was_forced_off_rather_than_chosen():
    collapsed = solve_hartree_fock(4, 4, _collapsed_config(4), False, False)
    text = " ".join(collapsed.total_energy.provenance.assumptions)
    assert "not a separate choice" in text


def test_term_structure_is_replaced_rather_than_omitted():
    """Silence would read as "this configuration happens to be a single term".

    That is a different and false claim, so the undefined-ness is stated.
    """
    collapsed = solve_hartree_fock(4, 4, _collapsed_config(4), False, False)
    text = " ".join(collapsed.total_energy.provenance.assumptions)
    assert "term structure is undefined" in text
    assert "average of configuration" not in text


def test_refinement_does_not_promise_a_better_calculation():
    """There is no calculation that makes this the real atom."""
    collapsed = solve_hartree_fock(4, 4, _collapsed_config(4), False, False)
    refinement = collapsed.total_energy.provenance.refinement
    assert "turn the exclusion principle back on" in refinement


def test_the_comparison_itself_is_counterfactual_and_says_it_is_not_observable():
    collapse = pauli_collapse(4)
    prov = collapse.binding_change.provenance
    assert prov.fidelity is Fidelity.COUNTERFACTUAL
    assert any("not an observable" in a for a in prov.assumptions)
    # Two independent solves, so the difference carries both mesh spreads.
    assert prov.error_estimate > 0
    assert collapse.radius_ratio.provenance.error_estimate is None
