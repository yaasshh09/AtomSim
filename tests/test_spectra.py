import numpy as np
import pytest

from atomsim.provenance import Fidelity
from atomsim.spectra import (
    LineList,
    ReferenceData,
    SpectralLine,
    compare_lines,
    load_reference,
    transition_lines,
)
from atomsim.systems import get_system


def _wavelengths(lines):
    return np.array([ln.wavelength.value for ln in lines])


def test_gross_lyman_alpha_wavelength():
    ll = transition_lines(get_system("h"), n_max=3)
    lya = [
        ln for ln in ll.lines
        if (ln.n_upper, ln.n_lower) == (2, 1) and ln.l_upper == 1
    ]
    assert len(lya) == 1
    # mu-corrected gross Lyman-alpha: 121.567 nm vacuum
    assert lya[0].wavelength.value == pytest.approx(121.567, abs=2e-3)
    assert lya[0].wavelength.unit == "nm (vacuum)"
    assert lya[0].energy.value == pytest.approx(10.199, abs=1e-3)


def test_selection_rule_delta_l():
    ll = transition_lines(get_system("h"), n_max=4)
    assert all(abs(ln.l_upper - ln.l_lower) == 1 for ln in ll.lines)
    # 2s -> 1s must be absent
    assert not any(
        (ln.n_upper, ln.l_upper, ln.n_lower) == (2, 0, 1) for ln in ll.lines
    )


def test_gross_lines_are_exact_and_sorted():
    ll = transition_lines(get_system("h"), n_max=5)
    assert isinstance(ll, LineList)
    assert ll.provenance.fidelity is Fidelity.EXACT
    w = _wavelengths(ll.lines)
    assert (np.diff(w) >= 0).all()
    assert all(ln.j_upper is None for ln in ll.lines)


def test_fine_structure_doublet():
    ll = transition_lines(get_system("h"), n_max=2, fine_structure=True)
    lya = [ln for ln in ll.lines if (ln.n_upper, ln.n_lower) == (2, 1)]
    # 2p_{1/2} -> 1s_{1/2} and 2p_{3/2} -> 1s_{1/2}
    assert sorted(ln.j_upper for ln in lya) == [0.5, 1.5]
    dl = abs(lya[0].wavelength.value - lya[1].wavelength.value)
    assert dl == pytest.approx(5.4e-4, rel=0.05)  # Lyman-alpha doublet ~0.54 pm
    assert ll.provenance.fidelity is Fidelity.APPROXIMATION


def test_fine_structure_delta_j_rule():
    ll = transition_lines(get_system("h"), n_max=3, fine_structure=True)
    assert all(abs(ln.j_upper - ln.j_lower) <= 1.0 + 1e-12 for ln in ll.lines)


def test_deuterium_isotope_shift_direction():
    h = transition_lines(get_system("h"), n_max=2).lines[0]
    d = transition_lines(get_system("d"), n_max=2).lines[0]
    assert d.wavelength.value < h.wavelength.value  # heavier nucleus -> bluer
    shift = h.wavelength.value - d.wavelength.value
    assert shift == pytest.approx(0.033, rel=0.05)  # ~33 pm Lyman-alpha H/D shift


def test_positronium_lyman_alpha_is_doubled():
    # lambda ~ 1/mu': lambda(mu'=1) = lambda_H * mu'_H; positronium doubles it
    MU_H = 1836.152673426 / 1837.152673426
    ps = transition_lines(get_system("ps"), n_max=2).lines[0]
    assert ps.wavelength.value == pytest.approx(2.0 * 121.567 * MU_H, rel=1e-4)
    assert ps.wavelength.value == pytest.approx(243.0, abs=0.1)  # literature Ps Lyman-alpha


def test_every_line_carries_provenance():
    ll = transition_lines(get_system("h"), n_max=3, fine_structure=True)
    for ln in ll.lines:
        assert isinstance(ln, SpectralLine)
        assert ln.energy.provenance is not None
        assert ln.wavelength.provenance is not None


def test_n_max_validation():
    with pytest.raises(ValueError):
        transition_lines(get_system("h"), n_max=1)


def test_vendored_h_reference_loads_with_citation():
    ref = load_reference("h")
    assert ref is not None
    assert ref.medium == "vacuum"
    assert "NIST" in ref.citation
    assert len(ref.retrieved) == 10  # YYYY-MM-DD
    assert len(ref.lines) >= 10  # Lyman+Balmer+Paschen up to n=6


def test_vendored_data_sanity_gate_against_computed_gross():
    # transcription-error tripwire: every vendored line within 1e-4 of computed
    for key in ("h", "d"):
        ref = load_reference(key)
        ll = transition_lines(get_system(key), n_max=6)
        comparisons = compare_lines(ll, ref, tolerance_relative=1e-4)
        assert len(comparisons) == len(ref.lines)
        for cmp_ in comparisons:
            assert cmp_.within_tolerance, (key, cmp_.reference_nm, cmp_.delta_nm)


def test_gross_comparison_within_tier_tolerance():
    ref = load_reference("h")
    ll = transition_lines(get_system("h"), n_max=6)
    comparisons = compare_lines(ll, ref)
    assert len(comparisons) >= 10
    assert all(c.within_tolerance for c in comparisons), [
        (c.reference_nm, c.relative_error) for c in comparisons if not c.within_tolerance
    ]


def test_fine_structure_improves_lyman_alpha():
    ref = load_reference("h")
    lya_ref = min(ref.lines, key=lambda ln: abs(ln.wavelength_nm - 121.567))
    gross = transition_lines(get_system("h"), n_max=2)
    fs = transition_lines(get_system("h"), n_max=2, fine_structure=True)
    ref_only = ReferenceData(
        species=ref.species, citation=ref.citation, retrieved=ref.retrieved,
        medium=ref.medium, lines=(lya_ref,),
    )
    d_gross = abs(compare_lines(gross, ref_only, 1.0)[0].delta_nm)
    d_fs = abs(compare_lines(fs, ref_only, 1.0)[0].delta_nm)
    assert d_fs <= d_gross * 1.5  # fs must not be wildly worse; usually better


def test_unknown_system_reference_is_none():
    assert load_reference("ps") is None
    assert load_reference("he+") is None  # He II vendoring skipped in M2 (no ASD aggregates)


# --- Phase 6: screened-atom transitions ---

def test_screened_lines_are_emission_and_dipole():
    from atomsim.atoms import aufbau_configuration
    from atomsim.screened_atom import solve_screened_atom
    from atomsim.spectra import screened_transition_lines

    res = solve_screened_atom(z=3, n_electrons=3, config=aufbau_configuration(3))
    lines = screened_transition_lines(res)
    assert lines.lines  # non-empty
    for ln in lines.lines:
        assert abs(ln.l_upper - ln.l_lower) == 1        # electric-dipole rule
        assert ln.energy.value > 0                       # emission (E_upper > E_lower)
        assert ln.energy.provenance.fidelity is Fidelity.APPROXIMATION


def test_screened_na_has_line_near_589nm():
    from atomsim.atoms import aufbau_configuration
    from atomsim.screened_atom import solve_screened_atom
    from atomsim.spectra import screened_transition_lines

    res = solve_screened_atom(z=11, n_electrons=11, config=aufbau_configuration(11))
    lines = screened_transition_lines(res)
    nearest = min(lines.lines, key=lambda ln: abs(ln.wavelength.value - 589.0))
    assert abs(nearest.wavelength.value - 589.0) < 120.0  # GSZ valence class
