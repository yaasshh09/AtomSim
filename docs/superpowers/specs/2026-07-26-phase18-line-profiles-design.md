# Phase 18: Line profiles and the synthetic spectrum

Status: design, 2026-07-26.

Every spectrum this app has drawn so far is a picket fence. A line is a
zero-width bar at one wavelength, and its height is whatever quantity Phase 13
through 17 could justify: an Einstein A, then an LTE emissivity. That is a
correct statement about *where* the lines are and *how strong* they are, and a
silent lie about *what a line is*. Real lines have width, the width is set by
physics we already have every ingredient for, and the shape it makes is the
single most information-dense object in observational spectroscopy: from one
line profile you read the temperature, the density, the bulk motion, and the
instrument.

This phase gives every line a width and a shape, and sums them into a
continuous spectral emissivity curve.

## What this adds

    natural       Lorentzian, width = total decay rate of both levels
    Doppler       Gaussian, width from the Maxwellian at T
    instrumental  Gaussian, width from a resolving power R
    Voigt         the convolution of the two families, evaluated exactly
    synthesis     sum over lines on an adaptive grid -> eps_lambda(lambda)

The Spectrum view stops being a bar chart and becomes a spectrograph trace,
with the bars still available underneath.

## The three widths, stated

### Natural width: the upper level will not sit still

An excited level with total decay rate `Gamma = 1/tau` has an energy uncertain
by `hbar Gamma`, and the emitted line is Lorentzian with

    FWHM_omega = Gamma_upper + Gamma_lower       [rad/s]
    FWHM_lambda = lambda^2 (Gamma_u + Gamma_l) / (2 pi c)

`Gamma` for each level is already in the engine: `analytic.transitions.lifetime`
sums `A(n l -> n' l')` over every dipole-allowed lower level, and
`lifetime_fine` does it j-resolved. **This sum is complete, not truncated.**
Decay only goes downward, so every channel out of a level with `n <= n_max`
has `n' < n <= n_max` and is already in the list. Unlike the partition function
of Phase 17, nothing is being cut off here, and the spec says so rather than
attaching a truncation warning out of reflex.

What *is* cut off: E1 only. That has one loud consequence, and it is a feature.
The `2s_1/2` level has no dipole-allowed decay at all, so this model gives it
`Gamma = 0` and an infinitely sharp line. The real 2s state is metastable and
decays by **two-photon emission** at about 8.2 s^-1, giving it a lifetime of
0.12 s rather than infinity. That is a genuine, famous piece of physics sitting
one selection rule outside the model, and the provenance names it.

Anchor: Lyman-alpha. `A(2p -> 1s) = 6.2649e8 s^-1`, `Gamma_1s = 0`, so
`FWHM_nu = Gamma/(2 pi) = 99.7 MHz`, the textbook natural linewidth, and
`FWHM_lambda = 4.92e-6 nm` at 121.567 nm.

### Doppler width: the atom is moving

Thermal motion along the line of sight shifts each emitter, and a Maxwellian at
`T` gives a Gaussian profile:

    sigma_lambda = lambda_0 sqrt(k T / (m c^2))
    FWHM_lambda  = lambda_0 sqrt(8 ln2 k T / (m c^2))

`m` is the mass of the **whole radiating atom**, not the electron. This is the
easiest thing in the phase to get wrong by a factor of 1836, so it gets a test
of its own.

For a hydrogen-like preset the atom mass is not new data: it falls out of what
`System` already carries. With `x = m_orb / M_nuc` (the stored `m_over_M`) and
the stored reduced mass `mu = m_orb / (1 + x)`,

    M_atom / m_e = mu_ratio * (1 + x)^2 / x

Checked three ways: hydrogen gives 1837.15 m_e (proton 1836.15 plus one
electron), positronium gives exactly 2 m_e, muonic hydrogen gives 2042.8 m_e
(muon 206.77 plus proton 1836.15). The exotic presets therefore get their real
Doppler widths for free, and positronium's is dramatic: at the same
temperature it is `sqrt(1837/2) = 30` times wider than hydrogen's.

The generic `hydrogen_like(Z)` preset has `m_over_M = 0`, an infinitely massive
nucleus that cannot recoil. Its Doppler width is therefore exactly zero, which
is the correct answer *for the model* and a wrong answer about any real ion.
That gets stated in the provenance rather than passed off as a sharp line.

Screened atoms need a real mass, so `Element` gains a standard atomic weight
(IUPAC/CIAAW). A natural isotope mixture is itself a source of width through
isotope shifts, which this does not model; noted in the assumptions.

Anchor: H-alpha at 10,000 K gives FWHM = 0.0468 nm, the textbook half-angstrom.

### Instrumental width: the spectrograph is not perfect

A Gaussian slit function at resolving power `R = lambda / FWHM`:

    sigma_inst = lambda / (R * 2 sqrt(2 ln 2))

This is not the atom. It is a model of the machine, it is under the user's
control, and it is what makes the view teach: crank R down and a resolved
fine-structure doublet merges into one line in front of you, which is the exact
reason 19th-century spectroscopy did not see fine structure. It is labelled as
an instrument model everywhere it appears, and R = off is the default.

### Combining them

Gaussian terms add in quadrature, Lorentzian terms add linearly:

    sigma^2 = sigma_Doppler^2 + sigma_instrument^2
    gamma   = gamma_natural   (+ collisional, not modelled: see below)

and the line shape is the Voigt profile, the convolution of the two:

    V(x; sigma, gamma) = Re[w(z)] / (sigma sqrt(2 pi)),  z = (x + i gamma)/(sigma sqrt2)

evaluated with `scipy.special.wofz` (the Faddeeva function). Given the widths,
this is exact to machine precision, so the profile carries the fidelity of its
*inputs* rather than claiming a tier of its own. Both limits are checked in the
tests: `gamma -> 0` reproduces the Gaussian, `sigma -> 0` reproduces the
Lorentzian, and the area is 1 in every case.

## What is NOT modelled, quantified

**Pressure (collisional) broadening is absent.** For hydrogen in a plasma the
dominant mechanism is the linear Stark effect of the ion microfield, and it is
not a small correction: it overtakes Doppler at electron densities that the
Phase 17 slider reaches with room to spare.

Rather than write "pressure broadening is not included" and leave the user to
guess when that matters, this phase *computes the size of the thing it is
missing* and shows it. The estimate is first-principles from parts already in
the engine:

1. The Holtsmark normal field of the ion microfield,
   `F_0 = 2.603 e n_e^(2/3) / (4 pi eps_0)`.
2. The linear Stark splitting of the hydrogenic levels at that field, which is
   Phase 11's parabolic manifold: extreme component at
   `dE = (3/2) n (n - 1) e a_0 F_0`.

For H-beta at `n_e = 1e14 cm^-3` this gives about 0.029 nm, against a Doppler
FWHM of 0.035 nm at 10,000 K: comparable, exactly as claimed. The estimate was
checked against an independent empirical route (Griem's `n_e^(2/3)` scaling
anchored on the standard H-beta value of ~2 nm at `1e17 cm^-3`, which
extrapolates to 0.02 nm), and the two agree within 50 percent, which is all an
order-of-magnitude flag needs.

So the view can say: *at these conditions the width you are looking at is an
underestimate by roughly this much, for this reason*. That is worth more than
either faking a Stark profile or staying quiet.

The estimate is hydrogenic only. A screened atom has no degenerate manifold and
broadens quadratically, so for those the honest output is the plain statement
that the mechanism is missing.

Also absent, and listed rather than modelled: self-absorption and radiative
transfer (inherited from Phase 17's optically-thin assumption, and the reason a
real strong line develops a *flat* or reversed core that this will never show),
bulk velocity fields and rotation, Zeeman splitting of the profile in a
magnetic field (Phase 10 has the splitting; folding it into the profile is a
later phase), and hyperfine components.

## The grid: a resolution problem that must not be silent

A synthesized spectrum on a grid coarser than its narrowest line is a lie about
peak heights, and a plausible-looking one. Natural widths here run to `1e-6 nm`
while the window is hundreds of nm, so a uniform grid resolving everything
would need `1e9` points.

The grid is therefore adaptive: a coarse background grid across the window, plus
a cluster of points around every line centre spanning its own total width. Two
guarantees follow, and both are stated in the caption:

- **Every line's own centre is on the grid**, so no peak height is ever
  underestimated by sampling.
- **Every line is evaluated at every grid point.** The clustering decides where
  the samples are, not which lines contribute; there is no wing cutoff, so the
  overlapping wings of neighbouring lines are summed exactly.

Point count is budgeted (`n_lines * n_points`). When the budget bites, the
background grid thins first and the per-line clusters survive, because they are
what carries the physics.

## Weighting

The area under each line is its strength, using the same quantity the bars
already use, so the two renderings cannot disagree:

    thermal on   -> LTE emissivity   [eV/s per atom per nm]
    intensities  -> Einstein A       [s^-1 per nm], a rate, not a brightness
    neither      -> uniform          [per nm], every line given equal area

The uniform case is honest and useful (it shows where lines crowd), and its
axis label says exactly what it is.

## Interfaces

New module `src/atomsim/broadening.py`:

    natural_width(gamma_upper_s, gamma_lower_s, wavelength_nm) -> Quantity
    doppler_sigma(wavelength_nm, temperature_k, mass_kg)       -> Quantity
    instrumental_sigma(wavelength_nm, resolving_power)         -> Quantity
    voigt(delta_nm, sigma_nm, gamma_nm)                        -> ndarray
    line_widths(line, system, thermal, resolving_power)        -> LineProfile
    synthesize(line_list, ...)                                 -> SyntheticSpectrum
    stark_width_estimate(n_upper, n_lower, wavelength_nm, n_e) -> Quantity

`SyntheticSpectrum` carries a `Field` of wavelengths and a `Field` of spectral
emissivity, plus the per-line widths, so the view can report what set the width
of any given line.

Server: `/api/spectrum` gains `resolving_power` and `profile` parameters and an
optional `profile` block on the response (wavelength and intensity arrays as
JSON; a few thousand points, well under the size that would need the binary
job path).

Frontend: `SpectrumView` gains a profile trace drawn over the bars, a resolving
power control, and captions for the width breakdown and the Stark warning.

## Validation anchors

| Quantity | Expected | Source |
|---|---|---|
| Lyman-alpha natural FWHM | 99.7 MHz | `Gamma/2pi` from the engine's own A |
| Lyman-alpha natural FWHM | 4.92e-6 nm | same, converted |
| H-alpha Doppler FWHM at 1e4 K | 0.0468 nm | closed form, textbook value |
| H atom mass from `System` | 1837.15 m_e | proton + electron |
| Positronium mass from `System` | 2.000 m_e | exactly two leptons |
| Voigt area | 1.000 | numerical integration |
| Voigt, gamma -> 0 | Gaussian | analytic limit |
| Voigt, sigma -> 0 | Lorentzian | analytic limit |
| 2s natural width | 0 (with the two-photon note) | no E1 channel |
| H-beta Stark estimate at 1e14 cm^-3 | ~0.03 nm | Griem scaling, independent |

## Deferred

- Collisional broadening folded into the profile (needs a Holtsmark or
  Griem-tabulated treatment; this phase quantifies its absence instead).
- Self-absorption and optical depth, still inherited from Phase 17.
- Zeeman and hyperfine components inside the profile.
- Absorption spectra and the curve of growth.
