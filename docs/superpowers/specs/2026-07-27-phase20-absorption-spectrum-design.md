# Phase 20: The absorption spectrum

Status: implemented 2026-07-27. Three corrections from the build are in "What
the build changed" at the end.

Phase 19 ended with a confession of its own. It made **one line** absorb, and
deferred the thing a spectrum actually does: absorb in every line at once,
against a real continuum, out of levels holding wildly different numbers of
atoms. The reason it was deferred is that the emission path had nowhere to put
a per-line lower-level population. That gap is now closed, and this phase is
what it was closed for.

## What this adds

`transfer.absorb(line_list, column_density_m2, ...)`: a whole `LineList` in
front of a flat continuum, returning transmission, the optical depth behind it,
and every line's own share of the result.

One column density is given for the **element**. Each line's own
`lower_fraction` (Phase 17) turns it into that line's absorbers. That single
choice is the whole physics of the phase: it is why the Lyman lines go black
while the Balmer lines stay invisible in the same gas, and it is the one fact
an emission spectrum cannot represent.

## The whole is less than the sum, twice over

Adding lines together is not adding their absorptions, and it fails in two
independent ways that this phase has to keep distinct in the code and merge
honestly in the reporting.

**Saturation.** Once a core is black it cannot remove more light no matter how
much gas arrives. Every line sits somewhere on its own Phase 19 curve of
growth, and at one column density they sit at different places on different
curves. The total falls below the summed thin-limit widths by however much of
the census is being lost.

**Blending.** Where profiles overlap the transmissions multiply,
`exp(-tau_1 - tau_2)`, rather than the absorptions adding. Two lines each
removing 60 percent of the light remove 84 percent together, not 120. The naive
sum is not merely inaccurate; it is impossible.

`saturation = W_measured / sum_i W_thin,i` reports both together. They are
deliberately not separated, because on a single grid they are not separable:
the same integral is short for both reasons at once. The provenance says so
rather than implying a decomposition that does not exist.

## Summing is not a second implementation

The sum runs through `broadening.synthesize`, with the area under each line set
to its integrated optical depth instead of its emissivity. An optical depth
spectrum and an emissivity spectrum are the same superposition of
area-normalized Voigts, differing only in what an area means, so `synthesize`
gained a `weight_fn` rather than gaining a twin. The adaptive grid, the wing
accounting and the flux-closure self-check are then the same ones the emission
spectrum uses, and there is one of each to keep true.

## The window has to grow with the column

Phase 19's expensive lesson, restated for a list. A window sized from a line's
FWHM silently returns a plausible wrong equivalent width, because a saturated
line is far wider than its FWHM and the missing part is simply never
integrated. Nothing about having more lines makes that safer.

`_window_for` sizes from the physics, taking the larger of two half-widths per
line:

- the Doppler core blacked out to `sigma sqrt(2 ln tau_c)`, which grows only as
  `sqrt(ln N)`;
- the Lorentzian wing falling to `tau = 1` at `d = sqrt(W gamma / pi)`, which
  grows as `sqrt(N)` and therefore takes over at high column.

Then the result is **checked** at the edge rather than trusted: if the spectrum
is still absorbing where the window ends, the equivalent width says so.

## Interfaces

Engine (`src/atomsim/transfer.py`):

- `absorb(line_list, column_density_m2, emitter_mass=, hydrogenic=,
  resolving_power=, window_nm=, max_points=) -> AbsorptionSpectrum`
- `AbsorptionSpectrum`: `transmission` and `optical_depth` as `Field`s on one
  grid; `lines` as `AbsorbingLine`s; `column_density` (`COUNTERFACTUAL`),
  `equivalent_width`, `thin_limit_width`, `saturation`; `blends`;
  `flux_closure`.
- `AbsorbingLine`: wavelength, label, `f`, `lower_column_m2`, `tau_centre`,
  `regime`, `thin_width_nm`, `fwhm_nm`.

Supporting:

- `broadening.synthesize(..., weight_fn=, weight_label=)`.
- `broadening.SyntheticSpectrum.lines`: the source lines, one per profile and
  in the same order.
- `spectra.orbital_label` / `spectra.subshell_label`.

Server: `GET /api/absorption`, `AbsorptionSpectrumModel`.

Frontend: `AbsorptionView`, store slice `absorption` / `logColumn` /
`absorptionData`, URL keys `abs` and `col`.

## Validation anchors

- **Thin limit.** At low column the summed equivalent width equals the closed
  form `sum_i (e^2/4 eps_0 m_e c^2) N_i f_i lambda_i^2`, which knows nothing
  about the grid, the Voigt kernel or the wing cuts. Agreeing with it means
  none of those created or destroyed absorption. Linear in `N`, to 1%.
- **Independent path.** A single line summed here matches the same line built
  cross-section-first on a uniform grid, to 2%.
- **Window rule.** Where `tau` actually crosses 1 matches
  `max(sigma sqrt(2 ln tau_c), sqrt(W gamma / pi))` within 6% across four
  decades of column, and the crossover from core-limited to wing-limited is its
  own test.
- **Monotonicity.** `W` never decreases with column, anywhere on the curve.
- **Bounds.** `0 <= I/I_0 <= 1` at every column from 1e14 to 1e25.
- **Blending.** A line and a displaced copy of itself absorb strictly more than
  one and strictly less than two.

## What the build changed

**Matching a profile to its line by wavelength is wrong, and quietly so.**
`absorb` first paired the two with a dict keyed on wavelength. Hydrogen to
n = 4 has fourteen lines on **six** distinct wavelengths: 3s->2p, 3p->2s and
3d->2p all sit at exactly 656.4696 nm with oscillator strengths of 0.014, 0.435
and 0.696, and two different lower levels. The dict kept one line per
wavelength, so eight of fourteen lines were reported with another line's `f` and
another line's column. Every array had the right length and every number looked
reasonable. The fix was to stop guessing: `synthesize` now hands back the
pairing. `/api/curve-of-growth` had the same latent flaw and was drawing
H-alpha's curve for the weakest of its three components, putting the knee more
than an order of magnitude off in column density.

**Two tests were wrong before the code was.** One demanded exactly zero
absorption from a 200,000 K gas; Saha leaves a neutral fraction near 1e-15, so
the literal zero would have been a bug to satisfy. The claim worth testing is
suppression in step with the population. The other sized the damping wing off
the Voigt FWHM instead of the Lorentzian half-width, inflating the prediction
56-fold, which made a wrong test pass for a wrong reason.

**The plus sign in `1e+22`.** JavaScript stringifies from 1e21 up in
exponential form and a raw `+` decodes as a space, so the API received "1e 22".
This was already reachable: the electron density slider ends at 1e22 cm^-3, so
dragging it to the end broke the spectrum endpoint before this phase existed.

## Deferred

- Line reversal and source functions (needs a stratified atmosphere). Until
  then these lines darken the continuum and never re-emit into it.
- Continuous opacity and a real continuum shape.
- The stimulated emission correction to the cross-section, which matters once
  the upper level is appreciably populated.
- Separating the saturation and blending contributions, which needs more than
  one grid.
