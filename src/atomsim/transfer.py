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
    "AbsorbingLine",
    "AbsorptionSpectrum",
    "CurveOfGrowth",
    "SIGMA_INTEGRAL",
    "absorb",
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

#: How far past the tau = 1 half-width a saturated line is still integrated.
#: The Lorentzian wing falls as 1/d^2, so tau is down to 1e-2 by 10 and 1e-4
#: by 100; 40 puts the residual absorption at the window edge below 1e-3,
#: which the edge self-check then confirms rather than trusts.
_SATURATED_PAD: float = 40.0


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


def _orbital(l: int) -> str:
    """Spectroscopic letter for an orbital angular momentum."""
    return "spdfghi"[l] if l < 7 else f"l={l}"


@dataclass(frozen=True)
class AbsorbingLine:
    """One line's contribution to a blended absorption spectrum."""

    wavelength_nm: float
    label: str
    oscillator_strength: float
    #: Column of absorbers in *this line's* lower level, m^-2. The whole
    #: reason one gas gives every line a different optical depth.
    lower_column_m2: float
    tau_centre: float
    #: "linear" | "saturated" | "damping", by the same physics as Phase 19.
    regime: str
    #: What this line would remove on its own with nothing saturating, nm.
    thin_width_nm: float
    fwhm_nm: float


@dataclass(frozen=True)
class AbsorptionSpectrum:
    """A whole line list absorbing at once against a flat continuum."""

    #: I/I_0 against vacuum wavelength.
    transmission: Field
    #: tau on the same grid, kept because a transmission of 1e-9 and one of
    #: 1e-30 look identical and are not.
    optical_depth: Field
    lines: tuple[AbsorbingLine, ...]
    #: Column density of the element along the sight line, m^-2.
    column_density: Quantity
    #: What the spectrum actually removes: integral (1 - I/I_0) dlambda.
    equivalent_width: Quantity
    #: What it would remove if no line saturated and none overlapped.
    thin_limit_width: Quantity
    #: measured / thin. Below 1 by exactly the amount the naive sum overstates
    #: the absorption.
    saturation: Quantity
    #: Pairs of lines close enough that their profiles overlap, so their
    #: absorption is jointly less than the sum of the parts.
    blends: tuple[tuple[str, str], ...]
    #: The synthesis grid's own quadrature error, measured not assumed.
    flux_closure: float


def _window_for(profiles, lo: float, hi: float) -> tuple[float, float]:
    """Widen a window until every line's absorption has actually ended in it.

    Phase 19 learned this the expensive way on a single line: a window sized
    by the line's FWHM silently returns a plausible wrong equivalent width,
    because a saturated line is far wider than its FWHM and the missing part
    is simply never integrated. Nothing about a list of lines makes that
    safer, so the window is sized from the same physics.

    Two half-widths, per line, whichever is larger:

    - Doppler core, black out to where the Gaussian exponent eats tau_centre:
      `sigma sqrt(2 ln tau_c)`.
    - Lorentzian wing, where `tau = W gamma / (pi d^2)` falls to 1:
      `d = sqrt(W gamma / pi)`. This is the one that grows without limit as
      the column grows, and the one a FWHM-sized window misses entirely.
    """
    from atomsim.broadening import voigt  # circular at module scope

    reach = 0.0
    for p in profiles:
        peak = p.weight * float(voigt(np.zeros(1), p.sigma_nm, p.gamma_nm)[0])
        if peak <= 1.0:
            continue
        if p.sigma_nm > 0.0:
            reach = max(reach, p.sigma_nm * math.sqrt(2.0 * math.log(peak)))
        if p.gamma_nm > 0.0:
            reach = max(reach, math.sqrt(p.weight * p.gamma_nm / math.pi))
    # A factor on top, because these are the half-widths at tau = 1 and the
    # absorption is still appreciable well past that.
    pad = _SATURATED_PAD * reach
    return max(lo - pad, 0.5 * lo), hi + pad


def absorb(
    line_list,
    column_density_m2: float,
    emitter_mass: Quantity | None = None,
    hydrogenic: bool = True,
    resolving_power: float | None = None,
    window_nm: tuple[float, float] | None = None,
    max_points: int = 24_000,
) -> AbsorptionSpectrum:
    """Put a whole line list in front of a flat continuum and see what survives.

    This is the phase every previous one deferred. Phase 19 made one line
    absorb; the thing a spectrum actually does is absorb in every line at
    once, out of levels that hold wildly different numbers of atoms, and the
    result is not the sum of the parts in two separate ways:

    - **Saturation.** Once a core is black, more gas cannot remove more light
      there, so the total absorbed falls below the sum of the thin-limit
      widths. This is the Phase 19 curve of growth, now happening to every
      line simultaneously and at a different point on its own curve.
    - **Blending.** Where two lines overlap, the transmissions multiply
      (`exp(-tau_1 - tau_2)`) rather than the absorptions adding. Two lines
      that each remove 60 percent of the light remove 84 percent together,
      not 120. The naive sum is not merely inaccurate, it is impossible.

    One `column_density_m2` is given for the *element*, and each line's own
    lower-level fraction turns it into that line's absorbers. That is why a
    single number can serve the whole list, and why the Lyman lines go black
    while the Balmer lines are invisible in the same gas.

    The sum is done by `broadening.synthesize` with the area under each line
    set to its integrated optical depth, so the grid, the wing accounting and
    the flux-closure check are the same ones the emission spectrum uses.
    """
    from atomsim.broadening import synthesize, voigt  # circular at module scope

    if column_density_m2 < 0.0:
        raise ValueError(f"column density must be >= 0, got {column_density_m2}")
    usable = [
        ln for ln in line_list.lines
        if ln.oscillator_strength is not None and ln.lower_fraction is not None
    ]
    if not usable:
        raise ValueError(
            "absorption needs an oscillator strength and a lower-level "
            "population for every line: turn on intensities and give thermal "
            "conditions. Without both there is nothing to absorb with, and a "
            "flat continuum would be drawn as if that were an answer"
        )

    def weight_fn(ln) -> float:
        if ln.oscillator_strength is None or ln.lower_fraction is None:
            return 0.0
        # The integral of tau over wavelength, nm: fixed by f and the column
        # alone, whatever the profile does with it.
        return _thin_limit_width(
            ln.oscillator_strength.value,
            ln.wavelength.value,
            column_density_m2 * ln.lower_fraction.value,
        )

    kwargs = dict(
        emitter_mass=emitter_mass,
        hydrogenic=hydrogenic,
        resolving_power=resolving_power,
        max_points=max_points,
        weight_fn=weight_fn,
        weight_label=("optical depth", "tau per nm"),
    )
    synth = synthesize(line_list, window_nm=window_nm, **kwargs)
    if window_nm is None:
        # Re-run only if the default window is too tight for how saturated
        # these lines turned out to be, which cannot be known until the
        # widths and weights exist.
        grid = synth.spectrum.grid
        wide = _window_for(synth.profiles, float(grid[0]), float(grid[-1]))
        if wide[0] < grid[0] or wide[1] > grid[-1]:
            synth = synthesize(line_list, window_nm=wide, **kwargs)

    grid = synth.spectrum.grid
    tau = np.clip(synth.spectrum.values, 0.0, None)
    trans = transmission(tau)
    measured = float(np.trapezoid(1.0 - trans, grid))

    # f and the lower column belong to the line, not the profile, and a profile
    # cannot be matched back to its line by wavelength: 3d->2p, 3p->2s and
    # 3s->2p are three lines at one wavelength, with three different oscillator
    # strengths and three different lower levels. `synthesize` hands back the
    # pairing so it does not have to be guessed.
    detail: list[AbsorbingLine] = []
    for p, ln in zip(synth.profiles, synth.lines, strict=True):
        peak = p.weight * float(voigt(np.zeros(1), p.sigma_nm, p.gamma_nm)[0])
        a = (
            p.gamma_nm / (p.sigma_nm * math.sqrt(2.0))
            if p.sigma_nm > 0.0 else 1.0
        )
        detail.append(AbsorbingLine(
            wavelength_nm=p.wavelength_nm,
            label=f"{p.label} ({_orbital(ln.l_upper)}->{_orbital(ln.l_lower)})",
            oscillator_strength=(
                ln.oscillator_strength.value
                if ln.oscillator_strength is not None else 0.0
            ),
            lower_column_m2=(
                column_density_m2 * ln.lower_fraction.value
                if ln.lower_fraction is not None else 0.0
            ),
            tau_centre=peak,
            regime=_classify(peak, a),
            thin_width_nm=p.weight,
            fwhm_nm=p.fwhm_nm,
        ))
    detail.sort(key=lambda d: d.wavelength_nm)

    thin_total = sum(d.thin_width_nm for d in detail)
    ratio = measured / thin_total if thin_total > 0.0 else 1.0

    blends: list[tuple[str, str]] = []
    for first, second in zip(detail, detail[1:], strict=False):
        gap = second.wavelength_nm - first.wavelength_nm
        if gap < first.fwhm_nm + second.fwhm_nm and min(
            first.tau_centre, second.tau_centre
        ) > 1e-3:
            blends.append((first.label, second.label))

    # The window's own self-check: if the spectrum is still absorbing at the
    # edge, the equivalent width is an underestimate by an amount nobody
    # measured. Phase 19's lesson was that this fails silently, so it is asked
    # rather than assumed.
    edge = float(max(1.0 - trans[0], 1.0 - trans[-1])) if trans.size else 0.0
    notes: list[str] = []
    if edge > 1e-3:
        notes.append(
            f"the spectrum is still absorbing {edge:.2%} of the continuum at "
            "the edge of the window, so the equivalent width below is an "
            "underestimate: widen the window or lower the column"
        )
    saturated = [d for d in detail if d.regime != "linear"]
    if saturated:
        notes.append(
            f"{len(saturated)} of {len(detail)} lines have a black core "
            "(tau at centre above 1), so their strengths no longer measure "
            "how much gas there is"
        )
    if blends:
        notes.append(
            f"{len(blends)} pair(s) of lines overlap within their own widths, "
            "so their transmissions multiply rather than their absorptions "
            "adding: the total is less than the sum of the parts by "
            "construction, not by approximation"
        )

    common = _SLAB + tuple(notes)
    return AbsorptionSpectrum(
        transmission=Field(
            values=trans,
            grid=grid,
            unit="I/I_0 (dimensionless)",
            grid_unit="nm (vacuum)",
            label="transmission",
            provenance=Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method=(
                    "Beer-Lambert through a summed line list: "
                    "I/I_0 = exp(-sum_i N_i sigma_i(lambda))"
                ),
                assumptions=common + (
                    "one column density for the element; each line's own "
                    "lower-level fraction (LTE) selects its absorbers",
                    f"grid closure {synth.flux_closure:.4f}: the summed "
                    "optical depth integrates to this times the analytic "
                    "total, measured on the grid actually used",
                ),
                refinement=(
                    "a source function and a stratified atmosphere would let "
                    "these lines re-emit and reverse instead of only darkening"
                ),
            ),
        ),
        optical_depth=Field(
            values=tau,
            grid=grid,
            unit="dimensionless",
            grid_unit="nm (vacuum)",
            label="optical depth",
            provenance=Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method="tau(lambda) = sum_i N_i sigma_i(lambda), Voigt profiles",
                assumptions=common,
            ),
        ),
        lines=tuple(detail),
        column_density=Quantity(
            column_density_m2, "m^-2", "column density of the element",
            Provenance(
                fidelity=Fidelity.COUNTERFACTUAL,
                method="chosen by the user; a knob, not a measurement",
                assumptions=(
                    "counts atoms of the element per square metre of sight "
                    "line, neutral and ionized together",
                ),
            ),
        ),
        equivalent_width=Quantity(
            measured, "nm", "equivalent width of the whole spectrum",
            Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method="W = integral (1 - I/I_0) dlambda over the full grid",
                assumptions=common,
            ),
        ),
        thin_limit_width=Quantity(
            thin_total, "nm", "summed thin-limit widths",
            Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method="sum_i (e^2/4 eps_0 m_e c^2) N_i f_i lambda_i^2",
                assumptions=_SLAB + (
                    "what the lines would remove if none saturated and none "
                    "overlapped; an upper bound, not a prediction",
                ),
            ),
        ),
        saturation=Quantity(
            ratio, "dimensionless", "measured / thin-limit width",
            Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method="W_measured / sum_i W_thin,i",
                assumptions=_SLAB + (
                    "1 means every line is optically thin and the spectrum is "
                    "a faithful census of the gas; below 1 means it is not, "
                    "and this is how much of the census is being lost",
                    "folds saturation and blending together: both make the "
                    "whole absorb less than the sum of its lines, and on a "
                    "single grid they are not separable",
                ),
            ),
        ),
        blends=tuple(blends),
        flux_closure=synth.flux_closure,
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
