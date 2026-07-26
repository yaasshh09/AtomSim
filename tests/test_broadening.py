"""Validation for line profiles: widths against closed forms, Voigt against limits.

Each width mechanism has a textbook anchor and is checked against it, not
against itself. The Voigt profile is checked against both of its analytic
limits and against its own normalization by numerical integration.
"""

import math

import numpy as np
import pytest
from scipy import constants as _sc
from scipy import integrate

from atomsim.analytic.transitions import einstein_A
from atomsim.broadening import (
    doppler_sigma_nm,
    instrumental_sigma_nm,
    level_decay_rates,
    natural_gamma_nm,
    stark_span_estimate,
    synthesize,
    voigt,
    voigt_fwhm,
)
from atomsim.populations import ThermalConditions
from atomsim.spectra import transition_lines
from atomsim.systems import emitter_mass, get_system, hydrogen_like

M_E = _sc.m_e


# --- emitter mass: the factor-of-1836 trap --------------------------------

def test_hydrogen_atom_mass_is_proton_plus_electron():
    """M_atom must come out as the proton plus one electron, not the proton.

    The Doppler width goes as 1/sqrt(m), so a nucleus-only mass would be wrong
    by 0.03 percent and an electron mass would be wrong by a factor of 43.
    """
    m = emitter_mass(get_system("h"))
    ratio = m.value / M_E
    proton_ratio = _sc.physical_constants["proton-electron mass ratio"][0]
    assert ratio == pytest.approx(proton_ratio + 1.0, rel=1e-9)
    assert ratio == pytest.approx(1837.15, rel=1e-4)


def test_positronium_mass_is_exactly_two_electrons():
    m = emitter_mass(get_system("ps"))
    assert m.value / M_E == pytest.approx(2.0, rel=1e-12)


def test_muonic_hydrogen_mass_is_muon_plus_proton():
    m = emitter_mass(get_system("mu-h"))
    muon = _sc.physical_constants["muon-electron mass ratio"][0]
    proton = _sc.physical_constants["proton-electron mass ratio"][0]
    assert m.value / M_E == pytest.approx(muon + proton, rel=1e-9)


def test_infinite_nucleus_gives_infinite_mass_and_says_so():
    """The generic Z preset cannot recoil. That must be stated, not hidden."""
    m = emitter_mass(hydrogen_like(3))
    assert math.isinf(m.value)
    assert any("infinit" in a.lower() for a in m.provenance.assumptions)
    assert doppler_sigma_nm(500.0, 1e4, m.value) == 0.0


# --- natural width --------------------------------------------------------

def test_lyman_alpha_natural_width_is_the_textbook_100_mhz():
    """Gamma/2pi for 2p -> 1s is 99.7 MHz, the standard quoted value."""
    a = einstein_A(2, 1, 1, 0).value
    assert a == pytest.approx(6.2649e8, rel=2e-3)
    fwhm_hz = a / (2.0 * math.pi)
    assert fwhm_hz == pytest.approx(99.7e6, rel=5e-3)


def test_lyman_alpha_natural_width_in_nm():
    """The same width in wavelength: 4.92e-6 nm at 121.567 nm."""
    gamma = natural_gamma_nm(6.2649e8, 121.567)
    assert 2.0 * gamma == pytest.approx(4.915e-6, rel=1e-3)


def test_natural_width_is_zero_without_a_decay_channel():
    assert natural_gamma_nm(0.0, 121.567) == 0.0


def test_natural_width_scales_as_lambda_squared():
    """Same rate, twice the wavelength, four times the width in nm."""
    a = natural_gamma_nm(1e8, 200.0)
    b = natural_gamma_nm(1e8, 400.0)
    assert b / a == pytest.approx(4.0, rel=1e-12)


# --- Doppler width --------------------------------------------------------

def test_h_alpha_doppler_width_at_10000_k():
    """Textbook: FWHM 0.047 nm, the classic half-angstrom."""
    m = emitter_mass(get_system("h"))
    sigma = doppler_sigma_nm(656.28, 1e4, m.value)
    fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma
    assert fwhm == pytest.approx(0.0468, rel=5e-3)


def test_doppler_width_grows_as_sqrt_t():
    m = emitter_mass(get_system("h")).value
    assert doppler_sigma_nm(500.0, 4e4, m) / doppler_sigma_nm(
        500.0, 1e4, m
    ) == pytest.approx(2.0, rel=1e-12)


def test_positronium_doppler_is_thirty_times_hydrogen():
    """sqrt(1837/2) = 30.3. The exotic presets get this for free."""
    h = doppler_sigma_nm(500.0, 1e4, emitter_mass(get_system("h")).value)
    ps = doppler_sigma_nm(500.0, 1e4, emitter_mass(get_system("ps")).value)
    assert ps / h == pytest.approx(math.sqrt(1837.15 / 2.0), rel=1e-3)


def test_doppler_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="temperature"):
        doppler_sigma_nm(500.0, 0.0, 1e-27)


# --- instrumental width ---------------------------------------------------

def test_resolving_power_gives_the_width_it_promises():
    """R = lambda/FWHM, by definition. Check the round trip."""
    lam, r = 500.0, 20000.0
    fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * instrumental_sigma_nm(lam, r)
    assert lam / fwhm == pytest.approx(r, rel=1e-12)


# --- Voigt ----------------------------------------------------------------

def test_voigt_reduces_to_a_gaussian_when_gamma_is_zero():
    x = np.linspace(-5.0, 5.0, 201)
    sigma = 1.3
    expected = np.exp(-(x**2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    assert np.allclose(voigt(x, sigma, 0.0), expected, rtol=1e-12, atol=1e-15)


def test_voigt_reduces_to_a_lorentzian_when_sigma_is_zero():
    x = np.linspace(-5.0, 5.0, 201)
    gamma = 0.7
    expected = gamma / (math.pi * (x**2 + gamma**2))
    assert np.allclose(voigt(x, 0.0, gamma), expected, rtol=1e-12)


def test_voigt_approaches_the_lorentzian_continuously():
    """A tiny but nonzero sigma must not jump: the analytic branch is a limit,
    not a special case, and a discontinuity there would show up as a visible
    kink in the rendered curve."""
    x = np.linspace(-5.0, 5.0, 101)
    exact = voigt(x, 0.0, 0.7)
    nearly = voigt(x, 1e-7, 0.7)
    assert np.allclose(nearly, exact, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(
    ("sigma", "gamma"), [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.2, 3.0), (3.0, 0.2)]
)
def test_voigt_area_is_one(sigma, gamma):
    """Area normalization is what makes the weight mean something: the area
    under a line is its emissivity, so the profile itself must integrate to 1."""
    area, _ = integrate.quad(
        lambda x: float(voigt(np.array([x]), sigma, gamma)[0]),
        -np.inf, np.inf, limit=400,
    )
    assert area == pytest.approx(1.0, rel=1e-6)


def test_voigt_refuses_a_line_with_no_width():
    with pytest.raises(ValueError, match="no width"):
        voigt(np.array([0.0]), 0.0, 0.0)


def test_voigt_fwhm_matches_a_direct_measurement():
    """Check the Olivero-Longbothum formula against the actual half-maximum
    of the evaluated profile, which is the thing it claims to approximate."""
    for sigma, gamma in [(1.0, 0.3), (0.3, 1.0), (1.0, 1.0), (2.0, 0.05)]:
        peak = float(voigt(np.array([0.0]), sigma, gamma)[0])
        x = np.linspace(0.0, 40.0 * (sigma + gamma), 400001)
        y = voigt(x, sigma, gamma)
        half = x[np.argmin(np.abs(y - peak / 2.0))]
        assert 2.0 * half == pytest.approx(voigt_fwhm(sigma, gamma), rel=2e-3)


def test_gaussian_and_lorentzian_fwhm_limits():
    assert voigt_fwhm(1.0, 0.0) == pytest.approx(2.3548, rel=1e-4)
    assert voigt_fwhm(0.0, 1.0) == pytest.approx(2.0, rel=1e-3)


# --- decay rates from the line list ---------------------------------------

def test_level_decay_rate_matches_the_analytic_lifetime():
    """Summing A over the line list must reproduce 1/tau from the engine's own
    lifetime function. If it did not, the width of a line and the rate printed
    beside it would be describing different physics."""
    from atomsim.analytic.transitions import lifetime

    sys_ = get_system("h")
    mu = sys_.mu_ratio.value
    lines = transition_lines(sys_, n_max=6, intensities=True)
    rates = level_decay_rates(lines.lines)
    for (n, l) in [(2, 1), (3, 1), (3, 2), (4, 0), (5, 3)]:
        assert rates[(n, l, None)] == pytest.approx(
            1.0 / lifetime(n, l, Z=sys_.Z, mu_ratio=mu).value, rel=1e-9
        )


def test_decay_rates_carry_the_reduced_mass():
    """The rates summed off the line list are the system's own, not the
    infinite-mass defaults: hydrogen's reduced mass moves A by 5e-4, which is
    small, real, and exactly the kind of thing that goes missing when two
    routes to the same number stop being compared."""
    from atomsim.analytic.transitions import lifetime

    sys_ = get_system("h")
    rates = level_decay_rates(
        transition_lines(sys_, n_max=3, intensities=True).lines
    )
    infinite_mass = 1.0 / lifetime(2, 1).value
    assert rates[(2, 1, None)] / infinite_mass == pytest.approx(
        sys_.mu_ratio.value, rel=1e-6
    )


def test_ground_state_has_no_decay_rate():
    lines = transition_lines(get_system("h"), n_max=4, intensities=True)
    rates = level_decay_rates(lines.lines)
    assert (1, 0, None) not in rates


def test_2s_has_no_e1_channel_so_no_natural_width():
    """The metastable 2s: no dipole decay at all, so this model gives it zero
    width. The real level decays by two-photon emission, and the synthesized
    spectrum has to say so rather than draw an infinitely sharp line."""
    lines = transition_lines(get_system("h"), n_max=4, intensities=True)
    rates = level_decay_rates(lines.lines)
    assert (2, 0, None) not in rates


# --- the missing mechanism, sized ----------------------------------------

def test_stark_estimate_matches_the_griem_scaling_for_h_beta():
    """Independent cross-check of the whole estimate.

    Griem's empirical scaling puts the H-beta Stark FWHM near 2 nm at
    n_e = 1e17 cm^-3, and it goes as n_e^(2/3), so 1e14 cm^-3 extrapolates to
    about 0.02 nm. The first-principles route here (Holtsmark field, then the
    linear Stark manifold) must land within a factor of two of that, which is
    the accuracy an order-of-magnitude flag needs.
    """
    griem = 2.0 * (1e14 / 1e17) ** (2.0 / 3.0)
    est = stark_span_estimate(4, 2, 486.1, 1e14)
    assert 0.5 < est.value / griem < 2.5
    assert est.value == pytest.approx(0.034, rel=0.1)


def test_stark_estimate_scales_as_density_to_the_two_thirds():
    a = stark_span_estimate(4, 2, 486.1, 1e14).value
    b = stark_span_estimate(4, 2, 486.1, 1e17).value
    assert b / a == pytest.approx(1000.0 ** (2.0 / 3.0), rel=1e-9)


def test_stark_estimate_declares_it_is_not_a_fwhm():
    est = stark_span_estimate(4, 2, 486.1, 1e14)
    assert any("NOT a FWHM" in a for a in est.provenance.assumptions)
    assert any("hydrogenic only" in a for a in est.provenance.assumptions)


# --- synthesis ------------------------------------------------------------

def _hydrogen_thermal(n_max=5, t=1e4, ne=1e12):
    return transition_lines(
        get_system("h"), n_max=n_max, intensities=True,
        thermal=ThermalConditions(temperature_k=t, electron_density_cm3=ne),
    )


def test_synthesis_conserves_the_line_strengths():
    """The integral of the curve must equal the sum of the line emissivities.

    This is the whole contract of an area-normalized profile: broadening moves
    flux around in wavelength, it does not create or destroy it. A failure
    here would mean the curve and the bars disagree about how bright the gas
    is, which is the exact class of quiet lie this project exists to prevent.
    """
    lines = _hydrogen_thermal()
    syn = synthesize(lines, emitter_mass=emitter_mass(get_system("h")))
    total = sum(p.weight for p in syn.profiles)
    integral = np.trapezoid(syn.spectrum.values, syn.spectrum.grid)
    assert integral == pytest.approx(total, rel=2e-3)
    # And the engine must have measured that for itself, not been told.
    assert syn.flux_closure == pytest.approx(integral / total, rel=1e-12)
    assert syn.flux_closure == pytest.approx(1.0, rel=2e-3)


def test_flux_closure_survives_a_lorentzian_dominated_spectrum():
    """The slow 1/x^2 tail is the hard case for the grid: with no thermal
    width at all, most of the area sits in wings the sampling has to earn."""
    lines = transition_lines(get_system("h"), n_max=5, intensities=True)
    syn = synthesize(lines)
    assert syn.weight_kind == "rate"
    assert all(p.sigma_nm == 0.0 for p in syn.profiles)
    assert syn.flux_closure == pytest.approx(1.0, rel=5e-3)


def test_a_long_line_list_stays_accurate_and_quick():
    """The stress case: fine structure at n_max = 10 is 855 lines.

    Summing every line at every grid point would be 1e8 profile evaluations,
    and the first version of this that coarsened the grid instead lost 20
    percent of the flux. Cutting each line's wings at a measured distance is
    what makes both the accuracy and the runtime survive, so both are pinned.
    """
    import time

    lines = transition_lines(
        get_system("h"), n_max=10, fine_structure=True, intensities=True,
        thermal=ThermalConditions(temperature_k=1e4, electron_density_cm3=1e12),
    )
    assert len(lines.lines) > 800
    start = time.perf_counter()
    syn = synthesize(lines, emitter_mass=emitter_mass(get_system("h")))
    elapsed = time.perf_counter() - start
    assert syn.flux_closure == pytest.approx(1.0, rel=5e-3)
    assert syn.spectrum.grid.size <= 24_000
    assert elapsed < 5.0


def test_fine_structure_window_never_goes_negative():
    """A within-n component out at metre wavelengths has a thermal width of
    kilometres. Padding the window by the widest line then walks the blue end
    past zero, and geomspace answers a NaN grid."""
    lines = transition_lines(
        get_system("h"), n_max=6, fine_structure=True, intensities=True,
        thermal=ThermalConditions(temperature_k=1e4, electron_density_cm3=1e12),
    )
    syn = synthesize(lines, emitter_mass=emitter_mass(get_system("h")))
    assert syn.spectrum.grid.min() > 0.0
    assert np.all(np.isfinite(syn.spectrum.grid))
    assert np.all(np.isfinite(syn.spectrum.values))


def test_dropped_wing_flux_is_measured_and_tiny():
    syn = synthesize(
        _hydrogen_thermal(), emitter_mass=emitter_mass(get_system("h"))
    )
    note = [a for a in syn.spectrum.provenance.assumptions if "wings dropped" in a]
    assert note, "the wing cut must be disclosed, not assumed harmless"
    assert "not\nassumed" in note[0] or "not assumed" in note[0].replace("\n", " ")


def test_flux_closure_is_reported_in_the_provenance():
    syn = synthesize(
        _hydrogen_thermal(), emitter_mass=emitter_mass(get_system("h"))
    )
    assert any(
        "integrates to" in a for a in syn.spectrum.provenance.assumptions
    )


def test_every_line_centre_is_a_grid_point():
    """The stated guarantee: no peak is ever undersampled."""
    lines = _hydrogen_thermal()
    syn = synthesize(lines, emitter_mass=emitter_mass(get_system("h")))
    grid = syn.spectrum.grid
    for p in syn.profiles:
        assert np.min(np.abs(grid - p.wavelength_nm)) < 1e-12 * p.wavelength_nm


def test_hotter_gas_gives_wider_lines():
    m = emitter_mass(get_system("h"))
    cool = synthesize(_hydrogen_thermal(t=5e3), emitter_mass=m)
    hot = synthesize(_hydrogen_thermal(t=2e4), emitter_mass=m)
    ratio = hot.profiles[0].fwhm_nm / cool.profiles[0].fwhm_nm
    assert ratio == pytest.approx(2.0, rel=0.05)


def test_doppler_dominates_natural_at_10000_k():
    """Sanity on the relative sizes: thermal width beats natural width by
    orders of magnitude for an optical line, which is why nobody measures
    lifetimes with a grating spectrograph."""
    syn = synthesize(
        _hydrogen_thermal(), emitter_mass=emitter_mass(get_system("h"))
    )
    optical = [p for p in syn.profiles if 400 < p.wavelength_nm < 700]
    assert optical
    for p in optical:
        assert p.sigma_nm > 100.0 * p.gamma_nm


def test_instrument_widens_every_line():
    m = emitter_mass(get_system("h"))
    lines = _hydrogen_thermal()
    sharp = synthesize(lines, emitter_mass=m)
    blurred = synthesize(lines, emitter_mass=m, resolving_power=2000.0)
    for a, b in zip(sharp.profiles, blurred.profiles, strict=True):
        assert b.fwhm_nm > a.fwhm_nm
        assert "instrumental" in b.terms


def test_no_width_source_refuses_to_draw():
    """Without a rate, a temperature or an instrument there is no width, and
    the honest output is a message rather than an invented one."""
    lines = transition_lines(get_system("h"), n_max=3, intensities=False)
    with pytest.raises(ValueError, match="zero width"):
        synthesize(lines)


def test_uniform_weighting_when_no_strengths_are_available():
    lines = transition_lines(get_system("h"), n_max=3, intensities=False)
    syn = synthesize(lines, resolving_power=5000.0)
    assert syn.weight_kind == "uniform"
    assert all(p.weight == 1.0 for p in syn.profiles)
    assert syn.spectrum.unit == "per nm"


def test_rate_weighting_without_thermal_conditions():
    lines = transition_lines(get_system("h"), n_max=4, intensities=True)
    syn = synthesize(lines, resolving_power=5000.0)
    assert syn.weight_kind == "rate"
    assert syn.spectrum.unit == "s^-1 per nm"


def test_stark_note_fires_at_high_density_and_not_at_low():
    m = emitter_mass(get_system("h"))
    thin = synthesize(_hydrogen_thermal(ne=1e8), emitter_mass=m)
    dense = synthesize(_hydrogen_thermal(ne=1e17), emitter_mass=m)
    assert thin.stark_note is None
    assert dense.stark_note is not None
    assert "Stark" in dense.stark_note


def test_synthesis_discloses_what_it_leaves_out():
    syn = synthesize(
        _hydrogen_thermal(), emitter_mass=emitter_mass(get_system("h"))
    )
    text = " ".join(syn.spectrum.provenance.assumptions)
    assert "collisional" in text
    assert "self-absorption" in text
    assert "two-photon" in text  # the 2s levels in the list


def test_curve_is_finite_and_nonnegative():
    syn = synthesize(
        _hydrogen_thermal(), emitter_mass=emitter_mass(get_system("h"))
    )
    v = syn.spectrum.values
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0.0)
    assert v.max() > 0.0


def test_fully_ionized_gas_gives_a_flat_zero_curve():
    """Not a failure: a gas with no neutrals left emits no bound-bound line at
    all, so zero everywhere is the answer. It used to divide by the summed
    strength and raise."""
    lines = _hydrogen_thermal(t=3e5, ne=1e4)
    assert lines.thermal.ionized_fraction.value == pytest.approx(1.0)
    assert sum(ln.emissivity.value for ln in lines.lines) == 0.0
    syn = synthesize(lines, emitter_mass=emitter_mass(get_system("h")))
    assert np.all(syn.spectrum.values == 0.0)
    assert syn.flux_closure == 1.0
    assert any(
        "fully ionized" in a for a in syn.spectrum.provenance.assumptions
    )


def test_window_restricts_the_lines_and_the_grid():
    syn = synthesize(
        _hydrogen_thermal(), emitter_mass=emitter_mass(get_system("h")),
        window_nm=(400.0, 700.0),
    )
    assert syn.spectrum.grid.min() >= 400.0
    assert syn.spectrum.grid.max() <= 700.0
    assert all(400.0 <= p.wavelength_nm <= 700.0 for p in syn.profiles)
