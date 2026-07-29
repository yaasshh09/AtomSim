"""End-to-end Hartree-Fock atoms against the vendored reference energies.

All five vendored closed-shell atoms are benchmarked here. They used to be one
- helium - because a UNIFORM grid could not afford the rest. Measured then, for
a single SCF step from a central-field guess: Be 4.1s, Ne 12.4s, Mg 42.4s,
Ar 88.5s, times about thirty steps on each of two grids, so argon cost the
better part of an hour and warm starts did not help.

The exponential mesh puts the points where the core actually is, and it is what
makes these affordable: argon now solves in about 8 seconds on roughly 2800
points rather than 72000. That is the whole reason numerics/mesh.py exists, so
these atoms are the test that it worked.
"""

import pytest

from atomsim.atoms import aufbau_configuration
from atomsim.hf_atom import solve_hartree_fock
from atomsim.hf_reference import load_hf_reference
from atomsim.provenance import Fidelity

CLOSED_SHELL = [("He", 2), ("Be", 4), ("Ne", 10), ("Mg", 12), ("Ar", 18)]


@pytest.fixture(scope="module")
def solved():
    return {
        symbol: solve_hartree_fock(z, z, aufbau_configuration(z))
        for symbol, z in CLOSED_SHELL
    }


@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_total_energy_matches_the_vendored_reference(solved, symbol, z):
    reference = load_hf_reference(symbol)["total_energy_hartree"]
    if reference is None:
        pytest.skip("reference energies not yet transcribed from the source")
    got = solved[symbol].total_energy.value
    assert got == pytest.approx(reference, rel=1e-4)


@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_energy_is_a_variational_upper_bound(solved, symbol, z):
    """HF sits above the exact non-relativistic energy, never below. If the
    computed energy drops below the reference by more than the tolerance, the
    functional is wrong, not merely inaccurate.

    The tolerance scales with Z because the numerical error does: argon's
    energy is 180x helium's, so a fixed absolute slack would be a far stricter
    demand on argon than on helium for no physical reason.
    """
    reference = load_hf_reference(symbol)["total_energy_hartree"]
    if reference is None:
        pytest.skip("reference energies not yet transcribed from the source")
    assert solved[symbol].total_energy.value > reference * (1.0 + 1e-4)


@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_the_quoted_error_estimate_actually_brackets_the_error(solved, symbol, z):
    """The provenance error bar has to be a bound, not a decoration.

    This is the test the old Richardson one should always have been. A
    refinement pair is structurally blind to everything that does not scale
    with the step - the inner-wall truncation and the eigensolver conditioning
    both cancel, because both meshes share r_min - so the spread alone
    understates the error and does so by more as the mesh gets finer. Beryllium
    is the case that exposed it: spread 6.5e-5 against a true deviation of
    7.7e-5, an error bar smaller than the error it described.

    Measured margins with the floor term included run 1.6x (Be) to 2.2x (Ar).
    """
    reference = load_hf_reference(symbol)["total_energy_hartree"]
    if reference is None:
        pytest.skip("reference energies not yet transcribed from the source")
    result = solved[symbol]
    deviation = abs(result.total_energy.value - reference)
    assert deviation < result.total_energy.provenance.error_estimate


@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_virial_ratio_is_near_two(solved, symbol, z):
    assert solved[symbol].virial_ratio.value == pytest.approx(2.0, rel=2e-3)


@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_orbital_energies_are_ordered_by_binding(solved, symbol, z):
    """The core must come out below the valence, on every atom. Nothing in the
    solve enforces this - each subshell is an eigenvector of its own Fock
    operator - so an ordering inversion would mean the channels had been
    matched to the wrong subshells.
    """
    energies = [orbital.energy.value for orbital in solved[symbol].orbitals]
    assert energies == sorted(energies)
    assert all(e < 0.0 for e in energies)


@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_total_energy_is_approximation_with_a_numerical_sub_scale(solved, symbol, z):
    energy = solved[symbol].total_energy
    assert energy.provenance.fidelity is Fidelity.APPROXIMATION
    assert energy.provenance.error_estimate is not None
    joined = " ".join(energy.provenance.assumptions)
    assert "correlation" in joined
    assert "variational" in joined


@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_virial_ratio_is_numerical_not_approximation(solved, symbol, z):
    """A convergence diagnostic is a statement about the solve, not the atom."""
    assert solved[symbol].virial_ratio.provenance.fidelity is Fidelity.NUMERICAL


@pytest.mark.parametrize("symbol,z", [(s, z) for s, z in CLOSED_SHELL if z > 2])
def test_each_orbital_carries_its_own_error_bar_not_the_totals(solved, symbol, z):
    """An orbital energy is not the total energy and must not borrow its error.

    Argon's 3p sits at about -0.59 hartree against a total of -527, so handing
    every orbital the total's bar overstated the valence uncertainty by about
    nineteen times. Each bar is now that orbital's own coarse-to-fine spread.
    """
    result = solved[symbol]
    bars = [orbital.energy.provenance.error_estimate for orbital in result.orbitals]
    assert all(bar is not None and bar > 0.0 for bar in bars)
    assert len(set(bars)) == len(bars)  # genuinely per-orbital, not one number
    assert all(bar < result.total_energy.provenance.error_estimate for bar in bars)


def test_beryllium_orbital_error_bars_bracket_the_published_energies(solved):
    """Checked against Bunge's tabulated orbital energies rather than against
    another run of this code, so the bar is measured against an outside number.
    """
    got = [orbital.energy.value for orbital in solved["Be"].orbitals]
    bars = [orbital.energy.provenance.error_estimate for orbital in solved["Be"].orbitals]
    for value, bar, published in zip(got, bars, [-4.7326699, -0.3092695], strict=True):
        assert abs(value - published) < bar


@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_the_amplitude_field_carries_no_energy_error_bar(solved, symbol, z):
    """Provenance.error_estimate is documented as being in the unit of the
    quantity it describes. P is in bohr^-1/2, so an error bar in hartree
    attached to it would not be loose, it would be dimensionally meaningless.
    This solve does not estimate the orbital SHAPE's error, and the honest way
    to say that is to carry no number at all.
    """
    for orbital in solved[symbol].orbitals:
        assert orbital.P.unit == "bohr^-1/2"
        assert orbital.P.provenance.error_estimate is None
        assert orbital.P.provenance.fidelity is Fidelity.APPROXIMATION


def test_hydrogen_is_exact_to_the_grid():
    result = solve_hartree_fock(1, 1, aufbau_configuration(1))
    assert result.total_energy.value == pytest.approx(-0.5, rel=1e-4)


def test_hydrogen_error_estimate_brackets_the_exact_answer():
    """Hydrogen is the one case with no reference uncertainty at all: HF on one
    electron is the bare Coulomb problem and the answer is exactly -1/2. So the
    error bar is checked against truth here, not against another calculation.
    """
    result = solve_hartree_fock(1, 1, aufbau_configuration(1))
    residual = abs(result.total_energy.value - (-0.5))
    assert residual < result.total_energy.provenance.error_estimate


def test_result_records_its_convergence(solved):
    result = solved["He"]
    assert result.converged is True
    assert result.iterations > 1
    assert len(result.residual_history) == result.iterations


def test_configuration_must_match_the_electron_count():
    with pytest.raises(ValueError, match="configuration holds"):
        solve_hartree_fock(4, 4, aufbau_configuration(2))
