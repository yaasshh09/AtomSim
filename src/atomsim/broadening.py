"""How I give a spectral line its width, and what shape that makes.

Every spectrum I drew before this module was a picket fence: a line was a bar of
zero width at one wavelength. That is right about where my lines sit and silent
about what a line *is*. Real lines have width, and the width is not noise, it is
the densest piece of information in observational spectroscopy. One profile
carries the temperature (Doppler), the density (collisions), the lifetime of the
upper level (natural), and the instrument.

I model three mechanisms here, and I leave one loud one out:

    natural       Lorentzian; from the total decay rate of both levels
    Doppler       Gaussian; from the Maxwellian at T
    instrumental  Gaussian; from a resolving power, my model of the machine
    collisional   NOT modelled; I size it and report it instead

I add the Gaussian terms in quadrature and the Lorentzian terms linearly, and
the convolution of the two families is the Voigt profile. Given the widths, my
profile is exact to machine precision, so it carries the fidelity of its inputs
rather than claiming a tier of its own.

See docs/specs/2026-07-26-phase18-line-profiles-design.md.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import constants as _sc
from scipy.special import wofz

from atomsim.provenance import Fidelity, Field, Provenance, Quantity
from atomsim.spectra import SpectralLine

__all__ = [
    "LineProfile",
    "SyntheticSpectrum",
    "doppler_sigma_nm",
    "instrumental_sigma_nm",
    "level_decay_rates",
    "natural_gamma_nm",
    "stark_span_estimate",
    "synthesize",
    "voigt",
    "voigt_fwhm",
]

#: FWHM = this times sigma, for a Gaussian.
_FWHM_PER_SIGMA: float = 2.0 * math.sqrt(2.0 * math.log(2.0))

#: Holtsmark normal field constant: F_0 = _HOLTSMARK * e * n_e^(2/3) / (4 pi eps_0)
#: with n_e in m^-3. The field a singly charged perturber produces at the mean
#: inter-particle distance, which is the natural scale of an ion microfield.
_HOLTSMARK: float = 2.603

#: Cap on profile evaluations (lines times grid points). You watch a spectrum
#: appear, so I would rather coarsen my grid than let a request hang; when I do,
#: my provenance says so.
_WORK_BUDGET: int = 4_000_000

#: Hard cap on grid points, so a short line list cannot make me mint an enormous
#: array that has to cross the wire for no visible gain.
_MAX_POINTS: int = 24_000

#: How far, in FWHM, I sum a line before I drop its wings. Generous on purpose:
#: at this distance a Lorentzian tail is about 1e-6 of the line.
_CUT_FWHM: float = 1e5

#: Cap on the grid points I evaluate for one line, so a very broad line in a
#: long list cannot bring back the quadratic cost my cut exists to avoid.
_MAX_EVAL_PER_LINE: int = 4_000

_NOT_MODELLED = (
    "I do NOT include collisional (pressure) broadening; for hydrogen in a "
    "plasma the linear Stark effect of the ion microfield overtakes Doppler "
    "at attainable densities, so I report its size separately rather than "
    "folding it in",
    "no self-absorption: an optically thick line develops a flat or reversed "
    "core that my profile can never show",
    "no bulk motion, rotation, Zeeman splitting, or hyperfine components",
)


@dataclass(frozen=True)
class LineProfile:
    """My width budget for one line, and the area I put under it."""

    wavelength_nm: float
    n_upper: int
    n_lower: int
    #: Gaussian standard deviation, nm (Doppler and instrument, in quadrature).
    sigma_nm: float
    #: Lorentzian half-width at half maximum, nm (natural).
    gamma_nm: float
    #: Total Voigt FWHM, nm.
    fwhm_nm: float
    #: The area I put under this line, in the weight's unit.
    weight: float
    #: Which mechanisms actually contributed, in the order I applied them.
    terms: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class SyntheticSpectrum:
    """A continuous spectral emissivity of mine, and the widths that shaped it."""

    #: values = spectral emissivity per nm, grid = vacuum wavelength in nm.
    spectrum: Field
    profiles: tuple[LineProfile, ...]
    #: The lines I built the profiles from, one per profile and in the same
    #: order. I keep them because a profile does not identify its line:
    #: degenerate transitions share a wavelength exactly (all of 3d->2p, 3p->2s,
    #: 3s->2p sit at 656.47 nm), so I hand over the pairing rather than let
    #: anything try to reconstruct it from what it can see.
    lines: tuple[SpectralLine, ...]
    #: "emissivity" | "rate" | "uniform", what the area under a line means.
    weight_kind: str
    resolving_power: float | None
    #: My curve's integral divided by the summed line strengths. An
    #: area-normalized profile moves flux around in wavelength without creating
    #: or destroying it, so this is 1 for a perfect grid, and the distance from
    #: 1 is my grid's own quadrature error, measured rather than assumed. Below
    #: 1 also picks up flux that fell outside the window.
    flux_closure: float = 1.0
    #: The size of the collisional broadening I leave out, for the strongest
    #: line in the list. None when it does not apply (you gave me no plasma
    #: conditions, or a non-hydrogenic atom has no linear Stark effect).
    stark_span: Quantity | None = None
    #: I set this only when that missing width is comparable to or larger than
    #: the one I model, which is when my curve stops being trustworthy.
    stark_note: str | None = None


def natural_gamma_nm(gamma_total_s: float, wavelength_nm: float) -> float:
    """My Lorentzian HWHM in nm, from a total decay rate in s^-1.

        FWHM_lambda = lambda^2 (Gamma_u + Gamma_l) / (2 pi c)

    An excited level with total decay rate Gamma has an energy uncertain by
    hbar Gamma, and the Weisskopf-Wigner treatment makes the resulting line
    exactly Lorentzian. I count both levels: a transition between two
    short-lived levels is broader than either level alone.

    My anchor: Lyman-alpha, Gamma = 6.2649e8 s^-1, gives 99.7 MHz, which is the
    textbook natural linewidth, and 4.92e-6 nm.
    """
    if gamma_total_s <= 0.0:
        return 0.0
    fwhm = wavelength_nm**2 * 1e-9 * gamma_total_s / (2.0 * math.pi * _sc.c)
    return 0.5 * fwhm


def doppler_sigma_nm(
    wavelength_nm: float, temperature_k: float, mass_kg: float
) -> float:
    """My Gaussian sigma in nm, from thermal motion of the emitting atom.

        sigma_lambda = lambda_0 sqrt(k T / (m c^2))

    `mass_kg` is the mass of the whole radiating atom. Handing me the electron
    mass would widen every line by a factor of 43, so I expect you to have gone
    through `systems.emitter_mass`, which derives it rather than guessing.

    I return exactly zero for an infinite mass: a nucleus that cannot recoil
    cannot shift its own photon. That is the correct answer for an idealized
    preset and the wrong answer about any real ion, which is why `emitter_mass`
    says so on the mass itself.

    My anchor: H-alpha at 10,000 K gives FWHM 0.0468 nm.
    """
    if temperature_k <= 0.0:
        raise ValueError(f"temperature must be > 0 K, got {temperature_k}")
    if not math.isfinite(mass_kg):
        return 0.0
    if mass_kg <= 0.0:
        raise ValueError(f"emitter mass must be > 0 kg, got {mass_kg}")
    return wavelength_nm * math.sqrt(_sc.k * temperature_k / (mass_kg * _sc.c**2))


def instrumental_sigma_nm(wavelength_nm: float, resolving_power: float) -> float:
    """My Gaussian sigma in nm for a slit function of resolving power R.

    R = lambda/dlambda, and this is not the atom. It is my model of the
    spectrograph, and it belongs in the profile only because every real spectrum
    has been through one. Turning R down until a resolved doublet merges is the
    whole reason I carry it.
    """
    if resolving_power <= 0.0:
        raise ValueError(f"resolving power must be > 0, got {resolving_power}")
    return wavelength_nm / (resolving_power * _FWHM_PER_SIGMA)


def voigt(delta_nm: np.ndarray, sigma_nm: float, gamma_nm: float) -> np.ndarray:
    """My area-normalized Voigt profile: a Gaussian convolved with a Lorentzian.

        V(x) = Re[w(z)] / (sigma sqrt(2 pi)),   z = (x + i gamma) / (sigma sqrt 2)

    with w the Faddeeva function. I hit both limits exactly rather than
    approaching them: gamma = 0 gives the Gaussian (w of a real argument is real
    and equals exp(-x^2)), and for sigma = 0 I take the analytic Lorentzian
    branch, since z would otherwise diverge.
    """
    x = np.asarray(delta_nm, dtype=float)
    if sigma_nm < 0.0 or gamma_nm < 0.0:
        raise ValueError(f"widths must be >= 0, got sigma={sigma_nm}, gamma={gamma_nm}")
    if sigma_nm == 0.0 and gamma_nm == 0.0:
        raise ValueError(
            "a line with no width has no profile; give me a temperature, a "
            "decay rate, or a resolving power"
        )
    if sigma_nm == 0.0:
        return gamma_nm / (math.pi * (x**2 + gamma_nm**2))
    z = (x + 1j * gamma_nm) / (sigma_nm * math.sqrt(2.0))
    return np.real(wofz(z)) / (sigma_nm * math.sqrt(2.0 * math.pi))


def voigt_fwhm(sigma_nm: float, gamma_nm: float) -> float:
    """The total FWHM I quote for a Voigt profile, Olivero & Longbothum (1977).

        f_V = 0.5346 f_L + sqrt(0.2166 f_L^2 + f_G^2)

    Accurate to 0.02 percent, which is far inside anything I use it for (grid
    spacing and a width I report). The Voigt FWHM has no closed form, so my
    alternative would be a root-find per line for no gain.
    """
    f_l = 2.0 * gamma_nm
    f_g = _FWHM_PER_SIGMA * sigma_nm
    return 0.5346 * f_l + math.sqrt(0.2166 * f_l**2 + f_g**2)


def level_decay_rates(lines) -> dict[tuple, float]:
    """The total E1 decay rate I find out of each upper level, keyed (n, l, j).

    In s^-1, and I sum it over your own line list rather than recomputing it, so
    the width I give a line and the rates shown beside it cannot disagree.

    For a hydrogen-like list my sum is **complete, not truncated**: decay only
    goes downward, so every channel out of a level with n <= n_max lands on a
    level with n' < n that is already in the list. I am cutting nothing off
    here, unlike my partition function in Phase 17.

    What I do cut off is the multipole order. E1 only means the 2s_1/2 level,
    which has no dipole-allowed decay at all, comes out with Gamma = 0 and an
    infinitely sharp line. The real 2s is metastable and decays by two-photon
    emission at about 8.2 s^-1, a lifetime of 0.12 s rather than infinity. I
    name that in my provenance wherever a zero width is used.
    """
    rates: dict[tuple, float] = {}
    for ln in lines:
        if ln.einstein_a is None:
            continue
        key = (ln.n_upper, ln.l_upper, ln.j_upper)
        rates[key] = rates.get(key, 0.0) + ln.einstein_a.value
    return rates


def stark_span_estimate(
    n_upper: int, n_lower: int, wavelength_nm: float, electron_density_cm3: float
) -> Quantity:
    """How big the collisional broadening I leave out would be, in nm.

    Not a profile: a number, so you can see how wrong my modelled width is
    instead of hearing only that something is missing. Two steps, both from
    parts I already have:

    1. The Holtsmark normal field of the ion microfield,
       F_0 = 2.603 e n_e^(2/3) / (4 pi eps_0), the field a singly charged
       perturber makes at the mean inter-particle distance.
    2. The linear Stark splitting of the hydrogenic levels at that field
       (Phase 11's parabolic manifold): the extreme component of level n sits
       at (3/2) n (n-1) e a_0 F, so the transition's components span
       3 e a_0 F_0 [n_u(n_u - 1) + n_l(n_l - 1)] peak to peak.

    What I return is that peak-to-peak span, which is a well-defined quantity
    and which I deliberately do not call a FWHM: most emitters see a field below
    F_0, so the real Holtsmark-averaged FWHM is smaller, by roughly a factor of
    two for Balmer lines.

    I cross-checked it independently: for H-beta at n_e = 1e14 cm^-3 I get
    0.034 nm, while Griem's empirical n_e^(2/3) scaling anchored on the standard
    ~2 nm at 1e17 cm^-3 extrapolates to 0.02 nm. Agreement to within a factor of
    two is all an order-of-magnitude flag needs.

    Linear Stark is a hydrogenic effect: it exists because the l levels of a
    shell are degenerate. A screened atom has no such manifold and shifts
    quadratically, so I must never apply this to one.
    """
    if electron_density_cm3 <= 0.0:
        raise ValueError(f"electron density must be > 0, got {electron_density_cm3}")
    n_e_m3 = electron_density_cm3 * 1e6
    f_0 = (
        _HOLTSMARK * _sc.e * n_e_m3 ** (2.0 / 3.0)
        / (4.0 * math.pi * _sc.epsilon_0)
    )  # V/m
    manifold = n_upper * (n_upper - 1) + n_lower * (n_lower - 1)
    de_j = 3.0 * _sc.e * _sc.physical_constants["Bohr radius"][0] * f_0 * manifold
    # dlambda = lambda dE / E_photon, with E_photon = hc/lambda.
    photon_j = _sc.h * _sc.c / (wavelength_nm * 1e-9)
    span_nm = wavelength_nm * de_j / photon_j
    return Quantity(
        value=span_nm,
        unit="nm",
        label=f"linear Stark span at n_e={electron_density_cm3:g} cm^-3",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                "I took the peak-to-peak span of the linear Stark components at "
                "the Holtsmark normal field "
                "F_0 = 2.603 e n_e^(2/3) / (4 pi eps_0); "
                f"F_0 = {f_0:.3g} V/m here"
            ),
            assumptions=(
                "quasi-static approximation: I freeze the perturbing ions while "
                "the atom radiates",
                "I let one representative field F_0 stand in for the whole "
                "Holtsmark distribution, so this is a span, NOT a FWHM: the real "
                "averaged profile is narrower, by roughly a factor of two for "
                "Balmer lines",
                "hydrogenic only: linear Stark needs the degenerate l manifold of "
                "a shell, and a screened atom shifts quadratically instead",
                "I take the electron and ion densities as equal (singly ionized "
                "gas)",
            ),
            refinement=(
                "a Holtsmark field distribution, or Griem's tabulated line "
                "shapes, would turn my estimate into an actual profile"
            ),
        ),
    )


def _tail_fraction(span_nm: float, sigma_nm: float, gamma_nm: float) -> float:
    """How much of a line's area lies further than `span_nm` from its centre.

    This is the price of cutting a line's wings, which I compute rather than
    wave at. Both families have closed-form tails, and adding them is a bound
    rather than an identity: the Voigt is their convolution, not their sum, so I
    cannot understate what I dropped.

        Lorentzian:  2 gamma / (pi d)   for d >> gamma
        Gaussian:    erfc(d / (sigma sqrt 2))
    """
    if span_nm <= 0.0:
        return 1.0
    lorentz = 2.0 * gamma_nm / (math.pi * span_nm) if gamma_nm > 0.0 else 0.0
    gauss = (
        math.erfc(span_nm / (sigma_nm * math.sqrt(2.0))) if sigma_nm > 0.0 else 0.0
    )
    return min(1.0, lorentz + gauss)


def _offsets(n_core: int, n_wing: int) -> np.ndarray:
    """Where I sample one line, in units of its own FWHM.

    Uniform through the core, geometric through the wings. I chose both spacings
    against the quadrature error, not by eye.

    A uniform grid over a function that decays to zero at both ends is
    spectrally accurate (the Euler-Maclaurin correction terms vanish), so I
    sample the core uniformly out to 2.5 FWHM, where a Gaussian is down to 3e-8
    of its peak and finished. I once stopped the uniform region at 1.0 FWHM and
    jumped straight to sparse wing points; the chord across that gap sat above a
    still-steep Gaussian and put 0.8 percent of extra flux into every line in my
    spectrum.

    Past the core only the Lorentzian tail survives, falling as 1/x^2, for which
    geometric spacing gives equal relative error per interval: a ratio r costs
    (r + 1/r)/2 - 1 per step, so r near 1.2 holds the tail integral inside 0.2
    percent.

    My wings run to 1e5 FWHM, which looks absurd until you price the
    alternative. A natural width can be 2e-6 nm while the gap to the next
    background sample is several nm. Stopping the cluster at a few hundred FWHM
    leaves a chord from a still-appreciable wing value straight across that gap,
    which is not merely a quadrature error: a polyline renderer draws it, so a
    3e-3 nm spike acquires a multi-nm tent at its foot. Going out to 1e5 FWHM
    costs me only 60 points, because geometric spacing is cheap.
    """
    return np.concatenate([
        np.linspace(0.0, 2.5, n_core), np.geomspace(2.8, 1e5, n_wing)
    ])


def _grid(
    centres: np.ndarray,
    widths: np.ndarray,
    lo: float,
    hi: float,
    n_background: int,
    n_core: int,
    n_wing: int,
) -> np.ndarray:
    """My adaptive wavelength grid: coarse background plus a cluster per line.

    A uniform grid fine enough for a 1e-6 nm natural width across an 800 nm
    window would need 1e9 points. Clustering buys me the same peak fidelity for
    a few thousand, and gives one guarantee worth stating in the caption: every
    line's own centre is a grid point of mine, so I never underestimate a peak
    because my sampling missed it.

    The clustering only decides *where* my samples are. I still evaluate every
    line at every point, so there is no wing cutoff and overlapping wings add up
    exactly.
    """
    half = _offsets(n_core, n_wing)
    offsets = np.concatenate([-half[::-1], half[1:]])
    clustered = (centres[:, None] + widths[:, None] * offsets[None, :]).ravel()
    background = np.geomspace(lo, hi, max(n_background, 2))
    grid = np.concatenate([clustered, background, [lo, hi]])
    grid = grid[(grid >= lo) & (grid <= hi)]
    return np.unique(grid)


def synthesize(
    line_list,
    emitter_mass: Quantity | None = None,
    hydrogenic: bool = True,
    resolving_power: float | None = None,
    window_nm: tuple[float, float] | None = None,
    n_background: int = 600,
    max_points: int = _MAX_POINTS,
    weight_fn: Callable[[Any], float] | None = None,
    weight_label: tuple[str, str] | None = None,
) -> SyntheticSpectrum:
    """I sum a LineList's profiles into one continuous spectral emissivity.

    I put the same quantity under each line that my bars already use, so the two
    renderings of one spectrum cannot disagree:

        thermal state present -> LTE emissivity   [eV/s per atom per nm]
        Einstein A present    -> rate             [s^-1 per nm]
        neither               -> uniform          [per nm], equal area each

    `emitter_mass` is the mass of the whole radiating atom (`systems.emitter_mass`
    for a preset, the element's standard atomic weight for a screened atom). I
    take it as a Quantity rather than looking it up here so that both kinds of
    caller can hand me one, and so an infinite mass carries its own explanation.
    Without it, or without a temperature, I have no thermal width and my profile
    is natural plus instrumental only.

    `hydrogenic` gates my collisional-broadening estimate, which is a linear
    Stark effect and therefore exists only for a degenerate l manifold.

    `max_points` is a transport limit, not a physics one: my curve has to cross
    a wire. Lowering it coarsens my grid, and whatever that costs shows up in
    `flux_closure` rather than hiding.

    `weight_fn` replaces the area rule above with your own, taking a line and
    returning the area I should put under it; `weight_label` names the result as
    (kind, unit). It exists so that absorption can be summed by exactly this
    routine: an optical depth is the same superposition of area-normalized
    Voigts as an emissivity, differing only in what each area means, and
    duplicating my grid, my wing accounting and my closure check for the sake of
    one line of arithmetic would be two things to keep true instead of one.

    I raise if no mechanism gives any line a width, because my honest output
    then is not a curve but a message saying which knob is missing.
    """
    lines = [
        ln for ln in line_list.lines
        if window_nm is None
        or window_nm[0] <= ln.wavelength.value <= window_nm[1]
    ]
    if not lines:
        raise ValueError("no lines in the requested window")

    thermal = line_list.thermal
    if weight_fn is not None:
        weight_kind, unit = weight_label or ("custom", "per nm")
    elif thermal is not None and any(ln.emissivity is not None for ln in lines):
        weight_kind = "emissivity"
        unit = "eV/s per atom per nm"
    elif any(ln.einstein_a is not None for ln in lines):
        weight_kind = "rate"
        unit = "s^-1 per nm"
    else:
        weight_kind = "uniform"
        unit = "per nm"

    mass = emitter_mass
    temperature = thermal.conditions.temperature_k if thermal is not None else None
    rates = level_decay_rates(lines)
    # I look the lower level's decay rate up in the same table: it is an upper
    # level for some shorter transition unless it is the ground state.
    profiles: list[LineProfile] = []
    metastable: set[str] = set()
    for ln in lines:
        lam = ln.wavelength.value
        terms: list[str] = []
        lower_key = (ln.n_lower, ln.l_lower, ln.j_lower)
        # A lower level with no E1 channel out of it, that is not the ground
        # state, is metastable. I give it exactly zero natural width.
        if ln.n_lower > 1 and lower_key not in rates:
            metastable.add(f"{ln.n_lower}{'spdfgh'[ln.l_lower]}")
        gamma_s = (
            rates.get((ln.n_upper, ln.l_upper, ln.j_upper), 0.0)
            + rates.get(lower_key, 0.0)
        )
        gamma = natural_gamma_nm(gamma_s, lam)
        if gamma > 0.0:
            terms.append("natural")
        sigma_sq = 0.0
        if temperature is not None and mass is not None:
            sd = doppler_sigma_nm(lam, temperature, mass.value)
            if sd > 0.0:
                sigma_sq += sd**2
                terms.append("Doppler")
        if resolving_power is not None:
            sigma_sq += instrumental_sigma_nm(lam, resolving_power) ** 2
            terms.append("instrumental")
        sigma = math.sqrt(sigma_sq)
        if weight_fn is not None:
            weight = float(weight_fn(ln))
        elif weight_kind == "emissivity":
            weight = ln.emissivity.value if ln.emissivity else 0.0
        elif weight_kind == "rate":
            weight = ln.einstein_a.value if ln.einstein_a else 0.0
        else:
            weight = 1.0
        profiles.append(LineProfile(
            wavelength_nm=lam,
            n_upper=ln.n_upper,
            n_lower=ln.n_lower,
            sigma_nm=sigma,
            gamma_nm=gamma,
            fwhm_nm=voigt_fwhm(sigma, gamma),
            weight=weight,
            terms=tuple(terms),
            label=f"{ln.n_upper}->{ln.n_lower}",
        ))

    if all(p.fwhm_nm <= 0.0 for p in profiles):
        raise ValueError(
            "every line in this list has zero width: I have no decay rate "
            "(turn on intensities), no temperature (turn on LTE weighting), "
            "and no instrument (set a resolving power). I would have to invent "
            "a profile, so I draw none"
        )

    lo = min(p.wavelength_nm for p in profiles)
    hi = max(p.wavelength_nm for p in profiles)
    if window_nm is not None:
        lo, hi = window_nm
    else:
        pad = max(3.0 * max(p.fwhm_nm for p in profiles), 0.01 * (hi - lo), 1e-6)
        # The widest line in the list sets my pad, and with fine structure that
        # is a within-n component out at metre wavelengths whose thermal width
        # alone is kilometres. Subtracting that from the bluest line walks my
        # window past zero, and a wavelength grid does not have a negative end.
        # So I floor it at half the bluest line, which keeps a real margin
        # without inventing negative light.
        lo, hi = max(lo - pad, 0.5 * lo), hi + pad
    # Zero-width lines (a 2s lower level with no E1 channel, at T = 0) would
    # collapse their own cluster onto one point; I give them the list's median
    # width for *sampling* only, never for the profile itself.
    positive = [p.fwhm_nm for p in profiles if p.fwhm_nm > 0.0]
    fallback = float(np.median(positive))
    widths = np.array([p.fwhm_nm if p.fwhm_nm > 0 else fallback for p in profiles])
    centres = np.array([p.wavelength_nm for p in profiles])

    # I cap the grid for the payload's sake, not the arithmetic's: my sampling
    # per line is fixed, so a long line list gets a longer grid, and only a very
    # long one has to give any resolution back.
    n_wing, n_core = 60, 21
    background = n_background
    grid = _grid(centres, widths, lo, hi, background, n_core, n_wing)
    coarsened = False
    # I give up the cheapest thing first: the baseline between lines, then the
    # far wings, and only last the core, which is the only part that decides
    # what the peak looks like.
    while grid.size > max_points and (background > 2 or n_wing > 4 or n_core > 5):
        if background > 2:
            background = max(background // 4, 2)
        elif n_wing > 4:
            n_wing = max(n_wing // 2, 4)
        else:
            n_core = max(n_core // 2, 5)
        grid = _grid(centres, widths, lo, hi, background, n_core, n_wing)
        coarsened = True

    # I sum each line only over the span where it has anything to say.
    #
    # My alternative, evaluating every line at every grid point, is exact and
    # quadratic, and a fine-structure list at n_max = 10 makes that 1e8 profile
    # evaluations. Cutting the wings is my compromise, so I pay for the cut
    # honestly: I compute the neglected flux analytically from the tail
    # integrals and report it, rather than assuming it is small.
    #
    # A Lorentzian tail beyond d carries 2 gamma / (pi d) of the line, and a
    # Gaussian one carries erfc(d / (sigma sqrt2)). At my default cut of 1e5
    # FWHM that is around 1e-6 of a line, but I measure the number for the cut I
    # actually used, which may be tighter for a broad line.
    values = np.zeros_like(grid)
    neglected = 0.0
    for p in profiles:
        if p.weight == 0.0 or (p.sigma_nm == 0.0 and p.gamma_nm == 0.0):
            continue
        span = _CUT_FWHM * p.fwhm_nm
        i0, i1 = np.searchsorted(grid, [p.wavelength_nm - span, p.wavelength_nm + span])
        if i1 - i0 > _MAX_EVAL_PER_LINE:
            centre_i = int(np.searchsorted(grid, p.wavelength_nm))
            i0 = max(i0, centre_i - _MAX_EVAL_PER_LINE // 2)
            i1 = min(i1, i0 + _MAX_EVAL_PER_LINE)
            span = min(p.wavelength_nm - grid[i0], grid[i1 - 1] - p.wavelength_nm)
        if i1 <= i0:
            continue
        chunk = grid[i0:i1]
        values[i0:i1] += p.weight * voigt(
            chunk - p.wavelength_nm, p.sigma_nm, p.gamma_nm
        )
        neglected += p.weight * _tail_fraction(span, p.sigma_nm, p.gamma_nm)

    # A gas hot and dense enough to be fully ionized has no bound-bound
    # emission at all, so every weight is legitimately zero. My curve is then
    # flat zero, which is the answer rather than a failure, and I have no total
    # to normalize the closure or the dropped flux against.
    total_weight = sum(p.weight for p in profiles)
    closure = (
        float(np.trapezoid(values, grid)) / total_weight if total_weight > 0 else 1.0
    )
    lost = neglected / total_weight if total_weight > 0 else 0.0
    assumptions = [
        "I used a Voigt profile (Gaussian widths in quadrature, Lorentzian "
        "widths added) evaluated with the Faddeeva function; exact given the "
        "widths",
        "my grid is adaptive (a cluster per line over a coarse background); "
        "each line's own centre is a grid point, so I undersample no peak",
        f"I summed each line out to {_CUT_FWHM:.0e} of its own FWHM, with its "
        f"wings dropped beyond that, which loses {lost:.2e} "
        "of the total flux (computed from the analytic tail integrals, not "
        "assumed); I sum overlapping wings inside that span exactly",
    ]
    if total_weight <= 0.0:
        assumptions.append(
            "every line in this spectrum has zero strength, so my curve is "
            "flat zero: at these conditions the gas is fully ionized and there "
            "are no bound electrons left to make a bound-bound line"
        )
    if window_nm is not None:
        assumptions.append(
            "I drop lines outside the requested window whole, including the "
            "wing flux they would have contributed inside it"
        )
    if "Doppler" not in {t for p in profiles for t in p.terms} and temperature:
        assumptions.append(
            "no Doppler width: this system has no finite emitter mass, so my "
            "thermal term is exactly zero (see the mass provenance)"
        )
    assumptions.append(
        f"my curve integrates to {closure:.4f} times the summed line "
        "strengths on this grid; the gap is my quadrature error plus any "
        "flux whose wings fall outside the window, measured here rather "
        "than assumed"
    )
    if metastable:
        assumptions.append(
            "the lower level(s) " + ", ".join(sorted(metastable)) + " have no "
            "E1 decay channel at all, so I give them exactly zero natural "
            "width. The real 2s is metastable rather than stable: it decays by "
            "two-photon emission at about 8.2 s^-1, a 0.12 s lifetime. That "
            "rate is negligible beside the upper level's, so my width is right, "
            "but the level is not the infinitely sharp thing I imply"
        )
    if coarsened:
        assumptions.append(
            f"I coarsened my grid to stay inside the evaluation budget "
            f"({background} background points, {n_wing} wing samples per line); "
            "I kept the uniform core of every line, so peak heights are "
            "unaffected and it is the wings and the baseline I sample more "
            "sparsely"
        )
    if resolving_power is not None:
        assumptions.append(
            f"I included a Gaussian instrument profile at R = {resolving_power:g}, "
            "which is a model of a spectrograph and not a property of the atom"
        )
    # What I am missing, sized. I report it for the strongest line, since that
    # is the one you are looking at, and I flag it only when it is big enough to
    # change the shape you are reading.
    stark_span = stark_note = None
    if thermal is not None and hydrogenic:
        strongest = max(profiles, key=lambda p: p.weight)
        stark_span = stark_span_estimate(
            strongest.n_upper, strongest.n_lower, strongest.wavelength_nm,
            thermal.conditions.electron_density_cm3,
        )
        if stark_span.value >= 0.3 * strongest.fwhm_nm:
            stark_note = (
                f"At n_e = {thermal.conditions.electron_density_cm3:.1e} cm^-3, "
                f"collisional (linear Stark) broadening of the {strongest.label} "
                f"line would span about {stark_span.value:.3g} nm, against the "
                f"{strongest.fwhm_nm:.3g} nm I model. My curve is therefore too "
                "narrow: pressure broadening is the mechanism I am missing, and "
                "at these densities it is not a correction."
            )
            assumptions.insert(0, stark_note)

    return SyntheticSpectrum(
        spectrum=Field(
            values=values,
            grid=grid,
            unit=unit,
            grid_unit="nm (vacuum)",
            label=f"spectral emissivity ({weight_kind}-weighted)",
            provenance=Provenance(
                # My Voigt evaluation is exact; every uncertainty here rides in
                # on the widths (E1-only rates, a Maxwellian, LTE conditions).
                fidelity=Fidelity.APPROXIMATION,
                method=(
                    "I summed area-normalized Voigt line profiles; the area I "
                    f"gave each line = its {weight_kind}"
                ),
                assumptions=tuple(assumptions) + _NOT_MODELLED,
                refinement=(
                    "collisional broadening, self-absorption, and a velocity "
                    "field would each add width my curve does not have"
                ),
            ),
        ),
        profiles=tuple(profiles),
        lines=tuple(lines),
        weight_kind=weight_kind,
        resolving_power=resolving_power,
        flux_closure=closure,
        stark_span=stark_span,
        stark_note=stark_note,
    )
