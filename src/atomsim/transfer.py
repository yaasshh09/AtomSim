"""Optical depth, absorption, and the curve of growth.

Two phases ended with the same confession: the gas is optically thin, so a
strong line never saturates and never eats its own light. This is the layer
that stops assuming it.

The ingredients were already here and only needed multiplying together. An
oscillator strength (Phase 13) says how strongly a transition couples to light,
a level population (Phase 17) says how many atoms are in the lower level, and a
line profile (Phase 18) says how that coupling is spread over wavelength. Their
product is an absorption cross-section, and a cross-section times a column
density is an optical depth.

What comes out is the curve of growth: how a line's measured strength responds
to adding more gas. It has three regimes, and the middle one is the reason
this phase exists. Once the core goes black, a hundred times more gas barely
changes the line, so every phase before this one was overstating what a strong
line tells you about how much gas there is.

See docs/superpowers/specs/2026-07-26-phase19-optical-depth-design.md.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import constants as _sc

from atomsim.provenance import Fidelity, Field, Provenance, Quantity

__all__ = [
    "CurveOfGrowth",
    "SIGMA_INTEGRAL",
    "absorption_spectrum",
    "cross_section",
    "curve_of_growth",
    "default_columns",
    "equivalent_width",
    "optical_depth",
    "transmission",
]

#: Integrated absorption cross-section per unit oscillator strength, in m^2 Hz:
#:
#:     integral sigma dnu = (e^2 / (4 eps_0 m_e c)) f
#:
#: The classical electron oscillator. Exact, and the anchor the whole phase
#: hangs on: 2.654e-6 m^2 Hz, the SI form of the familiar cgs pi e^2 / m_e c.
SIGMA_INTEGRAL: float = _sc.e**2 / (4.0 * _sc.epsilon_0 * _sc.m_e * _sc.c)

_SLAB = (
    "uniform absorbing slab: one temperature, one density, no depth structure",
    "pure absorption. There is no source function and no re-emission into the "
    "beam, so a line saturates but never reverses: a self-absorbed core needs "
    "a temperature gradient through a stratified atmosphere, which this is not",
    "no stimulated emission correction (a factor 1 - g_l N_u / g_u N_l), "
    "negligible unless the populations approach inversion",
    "no continuous opacity: the continuum is taken as flat and unabsorbed",
)


@dataclass(frozen=True)
class CurveOfGrowth:
    """Equivalent width against column density, with the regimes labelled.

    The regimes are not decoration. Which branch a line sits on decides whether
    its strength measures the amount of gas at all, and that is the single most
    important thing to know before reading a column density off a spectrum.
    """

    #: Column densities of absorbers in the lower level, m^-2.
    column_density: np.ndarray
    #: Equivalent width at each column density, nm.
    equivalent_width: np.ndarray
    #: "linear" | "saturated" | "damping", per point.
    regime: tuple[str, ...]
    #: Local log-log slope. Reported because it is the visible signature of a
    #: regime (1, then ~0, then 1/2), but NOT what the regime is decided by.
    slope: np.ndarray
    #: Optical depth at line centre, per point. This is what decides the
    #: regime, and it is the number that says whether the line's strength
    #: still measures how much gas there is.
    tau_centre: np.ndarray
    #: Voigt damping parameter a = gamma / (sigma sqrt2). Fixes where the
    #: third branch starts: damping takes over once a * tau_centre exceeds 1.
    damping_parameter: float
    #: Half-width of the integration window actually used, nm.
    window_nm: float
    wavelength_nm: float
    oscillator_strength: float
    provenance: Provenance


def cross_section(
    oscillator_strength: float,
    wavelength_nm: float,
    profile: np.ndarray,
) -> np.ndarray:
    """Absorption cross-section in m^2, from f and an area-normalized profile.

        sigma(lambda) = (e^2 / (4 eps_0 m_e c)) f phi(lambda) lambda^2 / c

    `profile` is the Phase 18 Voigt, normalized to unit area **in nm**, so the
    lambda^2/c factor converts the integrated cross-section from per-frequency
    (where it is fixed by f alone) into per-wavelength.

    The integral of the result is `SIGMA_INTEGRAL * f * lambda^2 / c` no matter
    what shape the profile has, which is the point: broadening moves the
    absorption around in wavelength without changing how much of it there is.
    """
    if oscillator_strength < 0.0:
        raise ValueError(f"f must be >= 0, got {oscillator_strength}")
    if wavelength_nm <= 0.0:
        raise ValueError(f"wavelength must be > 0, got {wavelength_nm}")
    lam_m = wavelength_nm * 1e-9
    # profile is per nm; 1e9 converts it to per m so the result is a real area.
    return (
        SIGMA_INTEGRAL * oscillator_strength
        * np.asarray(profile, dtype=float) * 1e9
        * lam_m**2 / _sc.c
    )


def optical_depth(sigma: np.ndarray, column_density_m2: float) -> np.ndarray:
    """tau = N sigma, dimensionless.

    `column_density_m2` counts absorbers **in the lower level of this
    transition**, per square metre of sight line. Not the total gas: a line
    only absorbs from the level it starts in, which is why the same cloud is
    opaque in Lyman-alpha and transparent in Balmer-alpha.
    """
    if column_density_m2 < 0.0:
        raise ValueError(f"column density must be >= 0, got {column_density_m2}")
    return column_density_m2 * np.asarray(sigma, dtype=float)


def transmission(tau: np.ndarray) -> np.ndarray:
    """Beer-Lambert: I/I_0 = exp(-tau). Goes to zero, never below it."""
    return np.exp(-np.asarray(tau, dtype=float))


def equivalent_width(tau: np.ndarray, grid_nm: np.ndarray) -> Quantity:
    """W = integral (1 - exp(-tau)) dlambda, in nm.

    The width of a perfectly black rectangle that removes the same light. It is
    what a spectroscopist measures, because it survives the instrument: a slit
    function moves flux around inside the line without changing the area taken
    out of the continuum. That invariance is asserted in the tests.
    """
    absorbed = 1.0 - transmission(tau)
    value = float(np.trapezoid(absorbed, grid_nm))
    return Quantity(
        value=value,
        unit="nm",
        label="equivalent width",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="W = integral (1 - exp(-tau)) dlambda over the profile grid",
            assumptions=_SLAB + (
                "W is instrument-independent by construction: convolving with a "
                "slit function redistributes flux inside the line and leaves "
                "the area removed from the continuum unchanged",
            ),
            refinement=(
                "a source function and a depth-stratified atmosphere would turn "
                "this into a line that can reverse as well as saturate"
            ),
        ),
    )


def _thin_limit_width(
    oscillator_strength: float, wavelength_nm: float, column_density_m2: float
) -> float:
    """W in the optically thin limit, closed form, in nm.

        W = (e^2 / (4 eps_0 m_e c^2)) N f lambda^2

    For tau << 1, `1 - exp(-tau)` is `tau`, so W is just the integral of tau,
    which is fixed by f alone. Independent of every width in the problem: in
    this regime broadening cannot change a line's strength, only its shape.
    """
    lam_m = wavelength_nm * 1e-9
    return (
        SIGMA_INTEGRAL / _sc.c * column_density_m2 * oscillator_strength * lam_m**2
    ) * 1e9


def _classify(tau_centre: float, damping_parameter: float) -> str:
    """Name the branch from the physics that produces it, not from the slope.

    Classifying by slope alone is wrong, and wrong in a way that looks right:
    coming off the linear branch the slope falls from 1 to nearly 0 and passes
    straight through 0.5 on the way, so the descent gets labelled "damping"
    before saturation has even started.

    The two standard criteria instead ask what is doing the absorbing:

    - `tau_centre < 1`: the core is still transparent, every atom added
      absorbs as much as the last. Linear.
    - `a * tau_centre > 1`, with `a = gamma / (sigma sqrt2)` the Voigt damping
      parameter: the optical depth carried by the Lorentzian wings has reached
      unity, so growth has moved out into wings that never end. Damping.
    - In between the core is black and only the Doppler shoulders are still
      growing. Saturated.

    These cross in the right order for any line, because `a < 1` whenever the
    profile has a Gaussian core at all.
    """
    if tau_centre < 1.0:
        return "linear"
    if damping_parameter * tau_centre > 1.0:
        return "damping"
    return "saturated"


def default_columns(
    oscillator_strength: float,
    wavelength_nm: float,
    sigma_nm: float,
    gamma_nm: float,
    points: int = 70,
) -> np.ndarray:
    """Column densities spanning all three branches, for this particular line.

    A fixed range cannot do it. The knees sit where `tau_centre = 1` and where
    `a tau_centre = 1`, and both move by orders of magnitude with the line's
    strength and width, so a range that shows all three branches for H-alpha
    shows one branch for a weak infrared line. This anchors the range on the
    line's own knees and pads a few decades either side.
    """
    from atomsim.broadening import voigt  # circular at module scope

    peak = float(cross_section(
        oscillator_strength, wavelength_nm,
        voigt(np.array([0.0]), sigma_nm, gamma_nm),
    )[0])
    if peak <= 0.0:
        raise ValueError("this line has no absorption cross-section")
    a = gamma_nm / (sigma_nm * math.sqrt(2.0)) if sigma_nm > 0 else 1.0
    n_thin = 1.0 / peak                      # tau_centre = 1
    n_damp = n_thin / a if a > 0 else n_thin  # a tau_centre = 1
    return np.geomspace(n_thin * 1e-5, n_damp * 1e4, points)


def curve_of_growth(
    oscillator_strength: float,
    wavelength_nm: float,
    sigma_nm: float,
    gamma_nm: float,
    columns_m2: np.ndarray,
    points: int = 4001,
    span_fwhm: float = 400.0,
) -> CurveOfGrowth:
    """Equivalent width against column density, across all three regimes.

    Needs the two widths rather than a whole line list: a curve of growth is a
    property of one line, and it is the widths that place the knees.

    The integration window has to be wide. In the damping regime the growth is
    carried entirely by Lorentzian wings falling as 1/x^2, so a window clipped
    at a few widths would flatten the third branch into the second and hide the
    physics the curve exists to show.
    """
    if sigma_nm <= 0.0 and gamma_nm <= 0.0:
        raise ValueError("a line with no width has no curve of growth")
    columns = np.asarray(columns_m2, dtype=float)
    if columns.ndim != 1 or columns.size < 2:
        raise ValueError("need at least two column densities")
    if np.any(columns <= 0.0):
        raise ValueError("column densities must be > 0 for a log-log curve")

    from atomsim.broadening import voigt  # circular at module scope

    scale = max(2.3548 * sigma_nm, 2.0 * gamma_nm)

    def _sample(half: float):
        """Grid, cross-section and widths for a window of half-width `half`."""
        core = np.linspace(-3.0 * scale, 3.0 * scale, points // 2)
        wings = np.geomspace(3.0 * scale, half, points // 4)
        offs = np.unique(np.concatenate([core, wings, -wings]))
        phi = voigt(offs, sigma_nm, gamma_nm)
        s = cross_section(oscillator_strength, wavelength_nm, phi)
        g = wavelength_nm + offs
        w = np.array([
            equivalent_width(optical_depth(s, n), g).value for n in columns
        ])
        return offs, s, w

    # The window has to grow with the largest column, not sit at a fixed
    # multiple of the width. On the damping branch the line eats its way out
    # into wings that fall only as 1/x^2, so a window that comfortably held the
    # line at 1e20 absorbers per m^2 clips it at 1e24 — and a clipped line does
    # not announce itself, it just quietly bends the slope down from 0.5. The
    # test is direct: if the equivalent width is a noticeable fraction of the
    # window, the window is part of the answer, so widen it and redo.
    half = span_fwhm * scale
    offsets, sigma_lambda, widths = _sample(half)
    for _ in range(12):
        if widths.max() <= 0.05 * (2.0 * half):
            break
        half *= 4.0
        offsets, sigma_lambda, widths = _sample(half)
    # Local log-log slope by central differences. Reported as the visible
    # signature of each branch; the branch itself is decided by the physics.
    log_n, log_w = np.log10(columns), np.log10(np.maximum(widths, 1e-300))
    slope = np.gradient(log_w, log_n)
    sigma_peak = float(np.max(sigma_lambda))
    tau_centre = columns * sigma_peak
    a = gamma_nm / (sigma_nm * math.sqrt(2.0)) if sigma_nm > 0 else math.inf
    return CurveOfGrowth(
        column_density=columns,
        equivalent_width=widths,
        regime=tuple(_classify(t, a) for t in tau_centre),
        slope=slope,
        tau_centre=tau_centre,
        damping_parameter=a,
        window_nm=half,
        wavelength_nm=wavelength_nm,
        oscillator_strength=oscillator_strength,
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                "W(N) from tau = N sigma with sigma = (e^2/4 eps_0 m_e c) f phi "
                "lambda^2/c; regimes named from the local log-log slope"
            ),
            assumptions=_SLAB + (
                f"integrated over +/-{half:.4g} nm ({half / scale:.0f} line "
                "widths), widened until the largest equivalent width was under "
                "5 percent of the window, so the damping branch is carried by "
                "real Lorentzian wings and not by the edge of the integration",
                "the three branches are linear (slope 1, thin), saturated "
                "(slope ~0, black core, W grows only as sqrt(ln N)) and damping "
                "(slope 1/2, growth carried by the natural-width wings)",
                f"branch decided by the physics, not the slope: linear while "
                f"tau_centre < 1, damping once a tau_centre > 1 with the Voigt "
                f"damping parameter a = {a:.3g}, saturated in between",
            ),
            refinement=(
                "a real curve of growth is fitted to many lines of one species "
                "at once, which is how the Doppler width is measured"
            ),
        ),
    )


def absorption_spectrum(
    grid_nm: np.ndarray,
    sigma: np.ndarray,
    column_density_m2: float,
    label: str = "transmission",
) -> Field:
    """I/I_0 against wavelength, as a Field carrying its own disclosure."""
    tau = optical_depth(sigma, column_density_m2)
    peak = float(np.max(tau)) if tau.size else 0.0
    return Field(
        values=transmission(tau),
        grid=np.asarray(grid_nm, dtype=float),
        unit="I/I_0 (dimensionless)",
        grid_unit="nm (vacuum)",
        label=label,
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="Beer-Lambert: I/I_0 = exp(-N sigma(lambda))",
            assumptions=_SLAB + (
                f"peak optical depth tau = {peak:.4g}"
                + (
                    ": optically thin, so the line depth is proportional to the "
                    "column and the strength still measures the amount of gas"
                    if peak < 0.5
                    else ": the core is saturated, so adding gas barely deepens "
                    "the line and its strength no longer measures the column"
                ),
            ),
        ),
    )
