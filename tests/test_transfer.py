"""Validation for optical depth and the curve of growth.

The anchors are closed forms: the integrated cross-section is fixed by the
oscillator strength alone, the thin-limit equivalent width has an analytic
value, and the three branches of the curve of growth have known slopes.
"""

import math

import numpy as np
import pytest
from scipy import constants as _sc

from atomsim.broadening import voigt
from atomsim.transfer import (
    SIGMA_INTEGRAL,
    _thin_limit_width,
    absorption_spectrum,
    cross_section,
    curve_of_growth,
    equivalent_width,
    optical_depth,
    transmission,
)

# A representative optical line: H-alpha at 10,000 K, Doppler dominated.
LAM = 656.28
SIGMA = 0.0199      # nm, Doppler sigma
GAMMA = 1.0e-5      # nm, natural HWHM
F = 0.6407          # H-alpha oscillator strength


def _grid(span_fwhm=400.0, points=6001):
    scale = max(2.3548 * SIGMA, 2.0 * GAMMA)
    core = np.linspace(-3 * scale, 3 * scale, points // 2)
    wings = np.geomspace(3 * scale, span_fwhm * scale, points // 4)
    off = np.unique(np.concatenate([core, wings, -wings]))
    return off, LAM + off


def test_integrated_cross_section_constant():
    """The classical electron oscillator, exact: 2.654e-6 m^2 Hz."""
    assert SIGMA_INTEGRAL == pytest.approx(2.654e-6, rel=1e-3)
    # And it is the SI form of the familiar cgs pi e^2 / (m_e c) = 0.02654 cm^2 Hz.
    assert SIGMA_INTEGRAL * 1e4 == pytest.approx(0.02654, rel=1e-3)


def test_cross_section_integral_depends_only_on_f():
    """Broadening moves absorption around in wavelength; it does not create or
    destroy any. The integral must come out at f lambda^2/c times the constant
    for any width at all."""
    expected = SIGMA_INTEGRAL * F * (LAM * 1e-9) ** 2 / _sc.c
    for sig, gam in [(SIGMA, GAMMA), (SIGMA / 10, GAMMA), (0.0, 1e-3), (0.1, 1e-2)]:
        off, grid = _grid()
        phi = voigt(off, sig, gam)
        s = cross_section(F, LAM, phi)
        # integral in nm, converted to m to match the m^2 Hz constant
        integral = float(np.trapezoid(s, grid)) * 1e-9
        assert integral == pytest.approx(expected, rel=5e-3), (sig, gam)


def test_cross_section_rejects_nonsense():
    with pytest.raises(ValueError, match="f must be"):
        cross_section(-1.0, LAM, np.array([1.0]))
    with pytest.raises(ValueError, match="wavelength"):
        cross_section(F, 0.0, np.array([1.0]))


def test_optical_depth_is_linear_in_column():
    off, _ = _grid()
    s = cross_section(F, LAM, voigt(off, SIGMA, GAMMA))
    a = optical_depth(s, 1e17)
    b = optical_depth(s, 2e17)
    assert np.allclose(b, 2.0 * a, rtol=1e-12)


def test_transmission_bounds():
    tau = np.array([0.0, 1.0, 50.0, 500.0])
    t = transmission(tau)
    assert t[0] == 1.0
    assert np.all(t >= 0.0)
    assert np.all(t <= 1.0)
    assert t[-1] < 1e-100  # a truly black core


def test_thin_limit_equivalent_width_matches_the_closed_form():
    """The headline anchor: for tau << 1 the numeric W must reproduce
    (pi e^2 / m_e c^2) N f lambda^2, which contains no width at all."""
    off, grid = _grid()
    s = cross_section(F, LAM, voigt(off, SIGMA, GAMMA))
    n = 1e14  # low enough to stay thin
    tau = optical_depth(s, n)
    assert tau.max() < 0.05
    w = equivalent_width(tau, grid)
    assert w.value == pytest.approx(_thin_limit_width(F, LAM, n), rel=5e-3)


def test_thin_limit_width_is_independent_of_broadening():
    """Same column, same f, wildly different widths: identical equivalent
    width. This is the fact that makes W worth measuring."""
    n = 1e14
    got = []
    for sig, gam in [(SIGMA, GAMMA), (SIGMA * 5, GAMMA), (SIGMA, GAMMA * 100)]:
        off, grid = _grid()
        s = cross_section(F, LAM, voigt(off, sig, gam))
        got.append(equivalent_width(optical_depth(s, n), grid).value)
    assert got[1] == pytest.approx(got[0], rel=1e-2)
    assert got[2] == pytest.approx(got[0], rel=1e-2)


def test_equivalent_width_survives_an_instrument():
    """W is instrument-independent by construction, which is exactly why
    spectroscopists use it. Convolving with a slit widens sigma; the area
    removed from the continuum must not move."""
    n = 1e14
    off, grid = _grid()
    sharp = equivalent_width(
        optical_depth(cross_section(F, LAM, voigt(off, SIGMA, GAMMA)), n), grid
    ).value
    # Instrument in quadrature with the thermal width.
    blurred_sigma = math.sqrt(SIGMA**2 + (3 * SIGMA) ** 2)
    blurred = equivalent_width(
        optical_depth(cross_section(F, LAM, voigt(off, blurred_sigma, GAMMA)), n), grid
    ).value
    assert blurred == pytest.approx(sharp, rel=1e-2)


def test_saturation_breaks_the_proportionality():
    """A hundred times the gas must NOT give a hundred times the width once the
    core is black. This is the whole reason the phase exists.

    tau at line centre reaches 1 near N = 2e16 for this line, so 1e16 -> 1e18
    straddles the knee: proportionality holds at the bottom and fails badly by
    the top."""
    off, grid = _grid()
    s = cross_section(F, LAM, voigt(off, SIGMA, GAMMA))
    thin = equivalent_width(optical_depth(s, 1e16), grid).value
    thick = equivalent_width(optical_depth(s, 1e18), grid).value
    assert thick / thin < 20.0  # far short of the naive 100


def test_saturated_core_goes_black():
    off, grid = _grid()
    s = cross_section(F, LAM, voigt(off, SIGMA, GAMMA))
    t = transmission(optical_depth(s, 1e18))
    assert t.min() < 1e-6


# --- the curve of growth ---------------------------------------------------

@pytest.fixture(scope="module")
def cog():
    # Twelve decades: enough to contain all three branches with room at both
    # ends, since the knees sit near 2e16 and 6e19 for this line.
    return curve_of_growth(
        F, LAM, SIGMA, GAMMA, np.geomspace(1e12, 1e24, 73)
    )


def test_curve_of_growth_rises_monotonically(cog):
    assert np.all(np.diff(cog.equivalent_width) > 0)


def test_linear_branch_has_slope_one(cog):
    linear = cog.slope[cog.column_density < 1e14]
    assert linear.size > 0
    assert np.allclose(linear, 1.0, atol=0.05)


def test_saturated_branch_flattens(cog):
    """The middle regime: the core is black, so growth is only through the
    Doppler shoulders and W creeps as sqrt(ln N).

    Bounded by the physics rather than by hand-picked columns: well past
    tau_centre = 1 and well short of a * tau_centre = 1."""
    deep = cog.tau_centre > 30.0
    shallow = cog.damping_parameter * cog.tau_centre < 0.3
    flat = cog.slope[deep & shallow]
    assert flat.size > 0
    assert flat.max() < 0.2


def test_damping_branch_approaches_slope_one_half(cog):
    """The Lorentzian wings never end, so growth resumes as sqrt(N gamma).

    The asymptote is approached, not jumped to, so this asks for it well inside
    the branch: a * tau_centre of 100 rather than the 1 that starts it."""
    damping = cog.slope[cog.damping_parameter * cog.tau_centre > 100.0]
    assert damping.size > 0
    assert np.allclose(damping, 0.5, atol=0.03)


def test_the_window_widens_instead_of_bending_the_damping_slope(cog):
    """A fixed window clips the wings at extreme columns and quietly drags the
    slope below 0.5. The window must have grown past its default instead."""
    assert cog.equivalent_width.max() < 0.05 * 2 * cog.window_nm
    deep = cog.slope[cog.damping_parameter * cog.tau_centre > 1e3]
    assert deep.size > 0
    assert deep.min() > 0.47


def test_all_three_regimes_are_named(cog):
    assert set(cog.regime) == {"linear", "saturated", "damping"}


def test_regime_is_decided_by_optical_depth_not_by_slope(cog):
    """The trap this classifier exists to avoid: coming off the linear branch
    the slope passes through 0.5, which looks exactly like the damping branch.
    Every point there must still be called saturated or linear."""
    descending = (cog.slope > 0.4) & (cog.slope < 0.6) & (cog.tau_centre < 100)
    assert descending.any()
    for r, d in zip(cog.regime, descending, strict=True):
        if d:
            assert r != "damping"


def test_regimes_appear_in_order(cog):
    """Linear, then saturated, then damping: a curve that reported them out of
    order would be labelling noise rather than physics."""
    first = {r: list(cog.regime).index(r) for r in ("linear", "saturated", "damping")}
    assert first["linear"] < first["saturated"] < first["damping"]


def test_curve_matches_the_closed_form_where_it_should(cog):
    thin = cog.column_density < 1e14
    expected = np.array([_thin_limit_width(F, LAM, n) for n in cog.column_density[thin]])
    assert np.allclose(cog.equivalent_width[thin], expected, rtol=1e-2)


def test_wider_doppler_moves_the_knee_to_higher_columns():
    """A broader line spreads the same absorption further, so it takes more gas
    to blacken the core. The knee is what a real curve-of-growth fit measures."""
    def knee(sig):
        c = curve_of_growth(F, LAM, sig, GAMMA, np.geomspace(1e12, 1e20, 80))
        # first column where the slope has dropped out of the linear branch
        return c.column_density[np.argmax(c.slope < 0.7)]

    assert knee(4 * SIGMA) > 2.0 * knee(SIGMA)


def test_curve_of_growth_rejects_a_line_with_no_width():
    with pytest.raises(ValueError, match="no width"):
        curve_of_growth(F, LAM, 0.0, 0.0, np.geomspace(1e12, 1e18, 10))


def test_curve_of_growth_rejects_unusable_columns():
    with pytest.raises(ValueError, match="at least two"):
        curve_of_growth(F, LAM, SIGMA, GAMMA, np.array([1e14]))
    with pytest.raises(ValueError, match="must be > 0"):
        curve_of_growth(F, LAM, SIGMA, GAMMA, np.array([0.0, 1e14]))


def test_curve_of_growth_names_what_it_cannot_do(cog):
    text = " ".join(cog.provenance.assumptions)
    assert "never reverses" in text
    assert "stimulated emission" in text


# --- the absorption spectrum Field -----------------------------------------

def test_absorption_field_reports_whether_it_is_saturated():
    off, grid = _grid()
    s = cross_section(F, LAM, voigt(off, SIGMA, GAMMA))
    thin = absorption_spectrum(grid, s, 1e13)
    thick = absorption_spectrum(grid, s, 1e18)
    assert any("optically thin" in a for a in thin.provenance.assumptions)
    assert any("saturated" in a for a in thick.provenance.assumptions)
    assert thin.values.min() > 0.9
    assert thick.values.min() < 1e-6


def test_absorption_field_axes_line_up():
    off, grid = _grid()
    s = cross_section(F, LAM, voigt(off, SIGMA, GAMMA))
    f = absorption_spectrum(grid, s, 1e15)
    assert f.values.shape == grid.shape
    assert f.unit.startswith("I/I_0")
    assert np.all(f.values <= 1.0)
