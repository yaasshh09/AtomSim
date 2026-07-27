"""Validation for the blended absorption spectrum (Phase 20).

The claims worth testing are the ones that separate this from Phase 19:

- a whole line list still reduces to the single-line chain when there is only
  one line, computed down an entirely independent path;
- the thin limit is recovered, so nothing in the summing machinery has
  invented or destroyed absorption;
- the two ways the whole is less than the sum of its parts, saturation and
  blending, are both present and both in the right direction;
- the window is wide enough that the equivalent width is not silently short,
  which is the failure Phase 19 shipped and had to come back for.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy import constants as sc

from atomsim.broadening import (
    doppler_sigma_nm,
    level_decay_rates,
    natural_gamma_nm,
    voigt,
)
from atomsim.populations import ThermalConditions
from atomsim.provenance import Fidelity
from atomsim.spectra import transition_lines
from atomsim.systems import emitter_mass, get_system
from atomsim.transfer import (
    SIGMA_INTEGRAL,
    absorb,
    cross_section,
    equivalent_width,
    optical_depth,
)

HOT = ThermalConditions(temperature_k=8000.0, electron_density_cm3=1e12)


def _hydrogen(n_max: int = 4, thermal: ThermalConditions | None = HOT):
    return transition_lines(
        get_system("h"), n_max=n_max, intensities=True, thermal=thermal
    )


def _mass():
    return emitter_mass(get_system("h"))


def _only(line_list, wavelength_nm: float):
    """Cut a list down to the single line nearest a wavelength."""
    pick = min(line_list.lines, key=lambda ln: abs(ln.wavelength.value - wavelength_nm))
    return replace(line_list, lines=(pick,))


# --------------------------------------------------------------------------
# The thin limit: the anchor everything else is measured against.
# --------------------------------------------------------------------------

def test_thin_limit_recovers_the_analytic_sum():
    """At low column every line is linear, so W is the closed-form sum.

    This is the strongest single check on the summing machinery: the analytic
    total depends only on f, lambda and N, and knows nothing about the grid,
    the Voigt kernel or the wing cuts. Agreeing with it means none of those
    created or lost absorption.
    """
    result = absorb(_hydrogen(), 1e16, emitter_mass=_mass())
    assert result.equivalent_width.value == pytest.approx(
        result.thin_limit_width.value, rel=0.01
    )
    assert result.saturation.value == pytest.approx(1.0, abs=0.01)
    assert all(d.regime == "linear" for d in result.lines)


def test_thin_limit_is_linear_in_the_column():
    """W doubles when N doubles, while nothing is saturated."""
    a = absorb(_hydrogen(), 1e15, emitter_mass=_mass())
    b = absorb(_hydrogen(), 2e15, emitter_mass=_mass())
    assert b.equivalent_width.value == pytest.approx(
        2.0 * a.equivalent_width.value, rel=0.01
    )


def test_single_line_matches_the_phase_19_chain():
    """One line, summed here, equals the same line built cross-section-first.

    The two paths share only the Voigt function. This one weights an
    area-normalized profile by an integrated optical depth on an adaptive
    grid; the other builds sigma(lambda) explicitly, multiplies by N and
    integrates on a uniform grid. Widths are computed here from the public
    broadening helpers rather than read out of the result, so a bug in how
    `absorb` assembles its profiles cannot hide by being used twice.
    """
    line_list = _only(_hydrogen(), 121.567)  # Lyman-alpha
    line = line_list.lines[0]
    lam = line.wavelength.value
    column = 1e19

    rates = level_decay_rates(line_list.lines)
    gamma = natural_gamma_nm(
        rates.get((line.n_upper, line.l_upper, line.j_upper), 0.0)
        + rates.get((line.n_lower, line.l_lower, line.j_lower), 0.0),
        lam,
    )
    sigma = doppler_sigma_nm(lam, HOT.temperature_k, _mass().value)

    # Phase 19 path, on its own grid, wide enough for this column.
    span = 200.0 * max(sigma, gamma)
    grid = np.linspace(lam - span, lam + span, 400_001)
    tau = optical_depth(
        cross_section(line.oscillator_strength.value, lam, voigt(grid - lam, sigma, gamma)),
        column * line.lower_fraction.value,
    )
    reference = equivalent_width(tau, grid)

    result = absorb(line_list, column, emitter_mass=_mass())
    assert result.equivalent_width.value == pytest.approx(reference.value, rel=0.02)


# --------------------------------------------------------------------------
# Saturation and blending: why the whole is less than the sum.
# --------------------------------------------------------------------------

def test_saturation_drives_the_width_below_the_thin_sum():
    """Once cores go black, W falls far short of the summed thin widths."""
    result = absorb(_hydrogen(), 1e21, emitter_mass=_mass())
    assert result.saturation.value < 0.5
    assert result.equivalent_width.value < result.thin_limit_width.value
    assert any(d.regime != "linear" for d in result.lines)
    # And the shortfall is disclosed, not merely present in the numbers.
    assert any(
        "black core" in a for a in result.transmission.provenance.assumptions
    )


def test_equivalent_width_never_decreases_with_column():
    """More gas can never remove less light, at any point on the curve."""
    widths = [
        absorb(_hydrogen(), n, emitter_mass=_mass()).equivalent_width.value
        for n in np.geomspace(1e15, 1e24, 12)
    ]
    assert all(b >= a for a, b in zip(widths, widths[1:], strict=False))


def test_blended_lines_absorb_less_than_the_sum_of_the_parts():
    """Transmissions multiply where lines overlap, so absorptions sub-add.

    Built as a controlled pair: one real line, and a copy of it displaced by
    a fraction of its own width. Their optical depths add exactly, so the
    combined absorption must be strictly less than twice the single one and
    strictly more than the single one.
    """
    single = _only(_hydrogen(), 121.567)
    line = single.lines[0]
    shifted = replace(
        line,
        wavelength=replace(line.wavelength, value=line.wavelength.value * (1 + 2e-6)),
    )
    pair = replace(single, lines=(line, shifted))

    column = 3e19
    one = absorb(single, column, emitter_mass=_mass())
    two = absorb(pair, column, emitter_mass=_mass())

    assert one.equivalent_width.value < two.equivalent_width.value
    assert two.equivalent_width.value < 2.0 * one.equivalent_width.value
    assert two.blends


def test_transmission_stays_inside_zero_and_one():
    """exp(-tau) with tau >= 0 can approach zero but never leave the interval."""
    for column in (1e14, 1e20, 1e25):
        values = absorb(_hydrogen(), column, emitter_mass=_mass()).transmission.values
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0 + 1e-12)


# --------------------------------------------------------------------------
# The physics the phase exists to show.
# --------------------------------------------------------------------------

def test_lyman_is_opaque_where_balmer_is_transparent():
    """One gas, one column, two lines differing by orders of magnitude in tau.

    Only the lower-level population explains it: at 8000 K essentially every
    neutral atom is in n = 1, so Lyman-alpha absorbs out of a full level and
    Balmer-alpha out of an almost empty one. This is the single fact the
    emission path could not represent.
    """
    result = absorb(_hydrogen(), 1e20, emitter_mass=_mass())
    lyman = min(result.lines, key=lambda d: abs(d.wavelength_nm - 121.567))
    balmer = min(result.lines, key=lambda d: abs(d.wavelength_nm - 656.3))

    assert lyman.tau_centre > 1.0
    assert balmer.tau_centre < 1.0
    assert lyman.lower_column_m2 > 1e3 * balmer.lower_column_m2


def test_regimes_are_classified_by_optical_depth():
    """Every line's label agrees with its own tau, on the Phase 19 criteria."""
    result = absorb(_hydrogen(), 1e23, emitter_mass=_mass())
    assert {d.regime for d in result.lines} & {"saturated", "damping"}
    for d in result.lines:
        if d.regime == "linear":
            assert d.tau_centre < 1.0
        else:
            assert d.tau_centre >= 1.0


def test_ionizing_the_gas_makes_it_transparent():
    """Absorption follows the bound population down, and Saha never hits zero.

    An earlier version of this test demanded the equivalent width be exactly
    zero at 200,000 K. It is not, and should not be: Saha leaves a neutral
    fraction of ~1e-15, so the honest claim is not "no absorption" but
    "absorption suppressed in step with the population that does it". Asserting
    the literal zero would have been asserting a bug.
    """
    thin = ThermalConditions(temperature_k=8_000.0, electron_density_cm3=1e8)
    ionized = ThermalConditions(temperature_k=200_000.0, electron_density_cm3=1e8)
    cool = absorb(_hydrogen(thermal=thin), 1e20, emitter_mass=_mass())
    hot = absorb(_hydrogen(thermal=ionized), 1e20, emitter_mass=_mass())

    assert hot.equivalent_width.value < 1e-6 * cool.equivalent_width.value
    # Transparent to the eye, and to a part in a million.
    assert np.all(hot.transmission.values > 1.0 - 1e-6)
    # Both are far too thin to saturate, so what is left is purely the census.
    assert hot.saturation.value == pytest.approx(1.0, abs=0.01)
    assert all(d.regime == "linear" for d in hot.lines)


# --------------------------------------------------------------------------
# The window, and the failure it is defended against.
# --------------------------------------------------------------------------

def test_window_grows_with_the_column():
    """A saturated line needs a wider window than its FWHM, and gets one."""
    narrow = absorb(_hydrogen(), 1e16, emitter_mass=_mass())
    wide = absorb(_hydrogen(), 1e24, emitter_mass=_mass())
    narrow_span = narrow.transmission.grid[-1] - narrow.transmission.grid[0]
    wide_span = wide.transmission.grid[-1] - wide.transmission.grid[0]
    assert wide_span > narrow_span


def test_edge_absorption_is_measured_and_disclosed():
    """Forcing a too-narrow window makes the result say so rather than lie."""
    result = absorb(
        _hydrogen(), 1e23, emitter_mass=_mass(), window_nm=(121.0, 122.0)
    )
    assert any(
        "still absorbing" in a for a in result.transmission.provenance.assumptions
    )


def test_a_wide_enough_window_does_not_claim_edge_absorption():
    """The self-check is not a permanent warning: it clears when it should."""
    result = absorb(_hydrogen(), 1e18, emitter_mass=_mass())
    assert not any(
        "still absorbing" in a for a in result.transmission.provenance.assumptions
    )
    assert result.flux_closure == pytest.approx(1.0, abs=0.02)


def test_degenerate_lines_keep_their_own_oscillator_strengths():
    """Lines sharing a wavelength must not share each other's properties.

    In the gross-structure model 3s->2p, 3p->2s and 3d->2p all sit at exactly
    656.4696 nm with f of 0.0136, 0.4351 and 0.6962 and two different lower
    levels. Anything that identifies a line by its wavelength collapses the
    three into one and silently reports one line's f against another's column.
    Fourteen hydrogen lines to n=4 occupy only six distinct wavelengths, so
    this is the common case, not an edge case.
    """
    result = absorb(_hydrogen(), 1e20, emitter_mass=_mass())
    assert len(result.lines) == len(_hydrogen().lines)

    balmer_alpha = [d for d in result.lines if abs(d.wavelength_nm - 656.4696) < 1e-3]
    assert len(balmer_alpha) == 3
    # Three distinct strengths, not one value repeated three times.
    assert len({round(d.oscillator_strength, 6) for d in balmer_alpha}) == 3
    # And the labels tell them apart, so a blend report can name them.
    assert len({d.label for d in balmer_alpha}) == 3

    # Every line's own numbers are self-consistent: the thin width it reports
    # is the closed form built from the f and column it reports.
    for d in result.lines:
        assert d.thin_width_nm == pytest.approx(
            SIGMA_INTEGRAL / sc.c
            * d.lower_column_m2
            * d.oscillator_strength
            * (d.wavelength_nm * 1e-9) ** 2
            * 1e9,
            rel=1e-9,
        )


def test_a_p_lower_level_and_an_s_lower_level_differ_in_column():
    """The degeneracy split is physical: 2s and 2p hold different numbers.

    3p->2s absorbs out of 2s and 3d->2p out of 2p, and those levels have
    different statistical weights, so one gas gives the two lines different
    columns. Getting this right is the whole point of pairing a profile with
    its line rather than its wavelength.
    """
    result = absorb(_hydrogen(), 1e20, emitter_mass=_mass())
    balmer = {
        d.label: d for d in result.lines if abs(d.wavelength_nm - 656.4696) < 1e-3
    }
    from_2s = next(d for k, d in balmer.items() if k.endswith("(p->s)"))
    from_2p = next(d for k, d in balmer.items() if k.endswith("(d->p)"))
    # 2p holds three times the 2s population: g = 6 against g = 2, degenerate
    # in energy in the gross-structure model.
    assert from_2p.lower_column_m2 == pytest.approx(3.0 * from_2s.lower_column_m2, rel=1e-6)


# --------------------------------------------------------------------------
# Provenance and refusals.
# --------------------------------------------------------------------------

def test_absorption_is_labelled_approximation():
    result = absorb(_hydrogen(), 1e19, emitter_mass=_mass())
    for field in (result.transmission, result.optical_depth):
        assert field.provenance.fidelity is Fidelity.APPROXIMATION
    assert result.column_density.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_the_missing_pieces_are_named():
    """The slab assumptions ride along on every field this phase produces."""
    text = " ".join(absorb(_hydrogen(), 1e19, emitter_mass=_mass())
                    .transmission.provenance.assumptions)
    assert "stimulated emission" in text
    assert "continuous opacity" in text
    assert "never reverses" in text


def test_absorption_refuses_a_list_with_no_populations():
    """Without a lower-level population there is nothing to absorb with."""
    with pytest.raises(ValueError, match="lower-level population"):
        absorb(_hydrogen(thermal=None), 1e19, emitter_mass=_mass())


def test_absorption_refuses_a_negative_column():
    with pytest.raises(ValueError, match="column density"):
        absorb(_hydrogen(), -1.0, emitter_mass=_mass())


def _tau_unity_half_width(result) -> float:
    """Where the line actually stops being opaque, measured off the grid."""
    grid, tau = result.optical_depth.grid, result.optical_depth.values
    above = grid[tau >= 1.0]
    return float(above[-1] - above[0]) / 2.0


@pytest.mark.parametrize("column", [1e22, 1e23, 1e24, 1e25])
def test_the_window_rule_predicts_where_the_line_stops_absorbing(column):
    """`_window_for`'s sizing rule agrees with where tau really crosses 1.

    The rule takes the larger of two half-widths: the Doppler core blacked out
    to `sigma sqrt(2 ln tau_0)`, and the Lorentzian wing falling to
    `sqrt(W gamma / pi)`. Measuring the crossing against that maximum checks it
    is the right physics rather than a factor that merely happens to be big
    enough. Both widths are rebuilt here from the public broadening helpers, so
    the rule cannot check itself.
    """
    line_list = _only(_hydrogen(), 121.567)
    line = line_list.lines[0]
    lam = line.wavelength.value
    rates = level_decay_rates(line_list.lines)
    gamma = natural_gamma_nm(
        rates.get((line.n_upper, line.l_upper, line.j_upper), 0.0)
        + rates.get((line.n_lower, line.l_lower, line.j_lower), 0.0),
        lam,
    )
    sigma = doppler_sigma_nm(lam, HOT.temperature_k, _mass().value)

    result = absorb(line_list, column, emitter_mass=_mass())
    detail = result.lines[0]
    lorentz = math.sqrt(detail.thin_width_nm * gamma / math.pi)
    doppler = sigma * math.sqrt(2.0 * math.log(detail.tau_centre))

    assert _tau_unity_half_width(result) == pytest.approx(
        max(lorentz, doppler), rel=0.10
    )


def test_the_damping_wing_overtakes_the_doppler_core():
    """Which of the two window terms is in charge changes with the column.

    The blacked-out core grows only as `sqrt(ln N)` while the wing grows as
    `sqrt(N)`, so a line that is core-limited at one column is wing-limited a
    decade later. The window has to track that changeover; the test above
    would pass on the Doppler term alone if it never happened.
    """
    narrow = _tau_unity_half_width(
        absorb(_only(_hydrogen(), 121.567), 1e22, emitter_mass=_mass())
    )
    wide = _tau_unity_half_width(
        absorb(_only(_hydrogen(), 121.567), 1e25, emitter_mass=_mass())
    )
    # Three decades of column widen the core by ~1.3x if Doppler still ruled;
    # the measured growth is an order of magnitude, which only the wing gives.
    assert wide > 10.0 * narrow
