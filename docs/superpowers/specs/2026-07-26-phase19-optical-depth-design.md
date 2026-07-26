# Phase 19: Optical depth, absorption, and the curve of growth

Status: implemented 2026-07-26. Three corrections from the build are in
"What the build changed" at the end. The absorption *spectrum* (a whole
line list eating a continuum) is deferred; the per-line machinery,
the endpoint and the curve of growth all landed.

Two phases in a row have ended with the same confession. Phase 17 said the gas
is **optically thin**: no radiative transfer, no self-absorption, no escape
probability. Phase 18 repeated it and sharpened it: an optically thick line
develops a flat or reversed core that the profile can never show. Both named
the same cure. This is it.

The ingredients are already in the engine and need only to be multiplied
together. An oscillator strength `f` (Phase 13), a level population (Phase 17)
and a line profile `phi(lambda)` (Phase 18) are exactly what an absorption
cross-section is made of, and a cross-section times a column density is an
optical depth.

## What this adds

    sigma(lambda)   absorption cross-section: f, times the Phase 18 profile
    tau(lambda)     optical depth: cross-section times a column density
    transmission    I/I_0 = exp(-tau), the Beer-Lambert law
    W_lambda        equivalent width: the area the line removes
    curve of growth W against column density, over the three regimes

The gas stops being a lamp and becomes something light passes through, which
is what almost every real spectrum actually is: the Fraunhofer lines are
absorption, stellar spectra are absorption, quasar forests are absorption.

## The chain, stated

The integrated cross-section of a line is fixed by its oscillator strength and
nothing else:

    integral sigma dnu = (e^2 / (4 eps_0 m_e c)) f = 2.654e-6 f   [m^2 Hz]

That constant is the classical electron oscillator, and it is exact. Spreading
it over the line profile in wavelength (`phi_lambda` normalized to 1 in nm,
which is what Phase 18 already returns):

    sigma(lambda) = (e^2 / (4 eps_0 m_e c)) f phi(lambda) lambda^2 / c   [m^2]

Then, for a column of `N` absorbers per m^2 in the lower level,

    tau(lambda) = N sigma(lambda)
    I / I_0     = exp(-tau(lambda))
    W_lambda    = integral (1 - exp(-tau)) dlambda        [nm]

`W_lambda` is the width a totally black rectangle would need to remove the same
amount of light. It is the quantity spectroscopists actually measure, because
it is independent of the instrument: convolving with a slit function moves
flux around inside the line without changing the area removed.

## The curve of growth is the payoff

Plot `W` against `N` on log-log axes and the line traces three regimes with
different slopes, each for a different physical reason. This is one of the
most-used diagrams in observational astrophysics and it falls straight out of
the arithmetic above.

1. **Linear**, slope 1. The line is optically thin, every atom added absorbs
   as much as the last, `W = (e^2 / (4 eps_0 m_e c^2)) N f lambda^2`. This is
   the regime where Phase 17's emissivity was honest.
2. **Flat (saturated)**, slope near zero. The core has gone black and cannot
   absorb any more; growth continues only through the Doppler shoulders, so
   `W` creeps up as `sqrt(ln N)`. Adding a hundred times more gas barely
   widens the line. **This is the regime where the previous phases were
   lying**, and it is why a strong line's strength is a terrible measure of
   how much gas there is.
3. **Damping**, slope 1/2. The Gaussian shoulders are exhausted too, and the
   growth is carried by the Lorentzian wings from the natural width, which
   fall as `1/x^2` and never truly end. `W` goes as `sqrt(N gamma)`.

The transition points depend on the Doppler width, so dragging the temperature
moves the knee, and the shape of the curve reveals the width parameters. That
is a real technique, not a demo.

## What this closes, and what it does not

**Closed:** the optically-thin assumption, as a *choice* rather than a
limitation. Emission stays optically thin and says so; absorption now models
the depth explicitly.

**Not closed, and stated:** this is a pure absorbing slab. There is no source
function, no re-emission into the beam, and therefore no line *reversal* (a
self-absorbed emission line with a dip in its core needs a temperature
gradient, which needs a stratified atmosphere). Saturation is modelled; the
reversed core is still out of reach, and the spec says which extra ingredient
it would take rather than implying the phase covers it.

Also absent: stimulated emission's correction to the cross-section (a factor
`1 - g_l N_u / g_u N_l`, negligible unless populations approach inversion),
continuous opacity, and any geometry beyond a uniform slab.

## Interfaces

New module `src/atomsim/transfer.py`:

    cross_section(f, wavelength_nm, profile_values)  -> Field    [m^2]
    optical_depth(sigma, column_density_m2)          -> Field
    transmission(tau)                                -> Field
    equivalent_width(tau, grid)                      -> Quantity [nm]
    curve_of_growth(line, widths, columns)           -> CurveOfGrowth

`CurveOfGrowth` carries the column densities, the equivalent widths, and the
regime each point falls in, so the view can label the three branches rather
than leaving the user to infer them from a slope.

Server: the `/api/spectrum` profile block gains an optional absorption mode
(`column_density_m2`) returning transmission alongside emission, and a
`/api/curve-of-growth` endpoint.

Frontend: the zoom panel gains an absorption view (the line eating a
continuum) and a curve-of-growth plot with the three regimes marked.

## Validation anchors

| Quantity | Expected | Source |
|---|---|---|
| Integrated cross-section constant | 2.654e-6 m^2 Hz | e^2/(4 eps_0 m_e c), closed form |
| W in the thin limit | `(pi e^2/m_e c^2) N f lambda^2` | analytic, must match numeric |
| Thin-limit slope on log-log | 1.000 | curve of growth |
| Saturated-branch slope | < 0.2 | curve of growth |
| Damping-branch slope | 0.5 | curve of growth |
| Transmission at line centre, large tau | -> 0 | Beer-Lambert |
| W independent of instrument R | invariant | equivalent width's defining property |

## What the build changed

**Classifying the branches by slope was wrong, and wrong invisibly.** Coming
off the linear branch the log-log slope falls from 1 to nearly 0 and passes
straight through 0.5 on the way, so every point on that descent got labelled
"damping" before saturation had even started. The shipped classifier asks the
physics instead: linear while `tau_centre < 1`, damping once `a tau_centre > 1`
with `a = gamma / (sigma sqrt2)` the Voigt damping parameter, saturated in
between. Those two cross in the right order for any line, because `a < 1`
whenever the profile has a Gaussian core at all. The slope is still reported,
as the visible signature of a branch rather than its definition.

**A fixed integration window silently bends the damping slope.** On the third
branch the line eats outward into wings falling only as `1/x^2`, so a window
that comfortably held the line at `1e20` absorbers per m^2 clipped it at
`1e24` — and a clipped line does not announce itself, it just returns a slope
of 0.46 instead of 0.49. The window now grows until the largest equivalent
width is under 5 percent of it, and reports the width it settled on.

**The column range has to be built from the line.** The knees sit at
`tau_centre = 1` and `a tau_centre = 1`, and both move by orders of magnitude
with `f` and the widths, so a fixed range that shows three branches for
H-alpha shows one branch for a weak infrared line. `default_columns` anchors on
the line's own knees and pads either side.

Measured on H-alpha at 10,000 K, the three branches come out at slope **1.000**,
**0.089** and **0.492** against the textbook 1, ~0 and 1/2.

## Deferred

- The absorption *spectrum*: a whole line list absorbing against a
  continuum. Needs a per-line lower-level column threaded through the line
  list, which the emission path does not currently carry.
- Line reversal and source functions (needs a stratified atmosphere).
- Continuous opacity and the true continuum.
- Stimulated emission correction to the cross-section.
