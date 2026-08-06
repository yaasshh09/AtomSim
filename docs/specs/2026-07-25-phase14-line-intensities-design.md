# Phase 14: Line intensities in the Spectrum view

Status: design approved 2026-07-25. Takes the Phase 13 engine
(`analytic/transitions.py`) and surfaces it: engine line lists carry strengths,
`/api/spectrum` serves them, and the Spectrum view stops drawing every line at
the same brightness.

## The honesty problem this fixes

`SpectrumView` currently draws every line with `strokeWidth 1.5` and
`opacity 0.9`. A spectrum plot where all bars are equally bright *asserts* that
all lines are equally strong, which is false by orders of magnitude: Lyman-alpha
and the faint high-n lines differ by ~10^4 in Einstein A. That assertion is
undisclosed, so by the prime directive it is a bug, not a styling choice. This
phase either encodes the real strength or says plainly that it is not modelled.

## What this adds

### Engine (`spectra.py`)

`SpectralLine` gains two optional fields:

    einstein_a:          Quantity | None   # s^-1, spontaneous emission rate
    oscillator_strength: Quantity | None   # dimensionless, absorption f

`transition_lines(system, n_max, fine_structure=False, intensities=False)`. When
`intensities=True` the two fields are filled from `transitions.einstein_A` and
`transitions.oscillator_strength`; otherwise they stay `None`. Default off, so
every existing caller is untouched.

### Where intensities are available, and where they are not

| spectrum | intensities | why |
|----------|-------------|-----|
| hydrogen-like, gross structure | **yes**, NUMERICAL | Phase 13 engine applies directly |
| hydrogen-like, fine structure  | no (`None`) | needs the 6j j-branching factor |
| screened atoms (GSZ)           | no (`None`) | needs dipole integrals over numerically solved radials |

The two gaps are **disclosed in the response**, not silently empty. A
`intensity_note` string on the line list states which case applies and what is
missing, and the view renders that sentence rather than quietly falling back to
uniform bars.

Splitting a gross-structure rate across its fine-structure components requires

    A(n'l'j' -> nlj) = (4/3) alpha^3 dE^3 (2j+1) {j 1 j'; l' 1/2 l}^2 l_max |R|^2

and the project has no 6j implementation (no sympy dependency). Attributing the
unsplit multiplet rate to each j component would misstate every fine-structure
line, so the honest move is to withhold it. That is Phase 15.

### Server

`/api/spectrum?intensities=true`. `LineModel` gains `einstein_a_s` and
`oscillator_strength` (`QuantityModel | None`), `SpectrumResponse` gains
`intensity_note: str | None`. Cost is ~0.24 s for the full n_max=10 list after
the Gauss-Laguerre change, so this stays on the synchronous endpoint; no job
protocol.

### Client

Store flag `intensities` (default **on** for hydrogen-like gross structure),
URL-addressable as `&intensity=1`, in the `INVALIDATED` set for the spectrum
payload. `SpectrumView` maps A to bar height and opacity.

## The rendering rule, and the liberty in it

A spans ~4 decades, so a linear map would render everything but the top few
lines invisible. Bars therefore encode

    t = (log10 A - log10 A_min) / (log10 A_max - log10 A_min)   in [0, 1]

with height and opacity both rising with t, and the decade range printed in the
caption so the compression is readable off the plot.

**The liberty that must be stated:** bar brightness encodes the Einstein A
coefficient, which is a *rate*, not a predicted brightness. What a real
spectrograph records is `N_upper * A * h nu`, it depends on level populations
(temperature, density, optical depth), and this project models none of those.
So the caption and a `SPECTRUM_INTENSITY_LIBERTY` entry in `lib/liberties.ts`
say: *bar height/opacity is log-scaled spontaneous emission rate A, not a
population-weighted line intensity.* Getting this wrong would be the most
misleading thing in the phase, since the plot looks exactly like an observed
spectrum.

## Validation anchors (locked in tests)

Engine and server:

- Balmer-alpha (3->2) is the strongest Balmer line in the list.
- Lyman-alpha (2p->1s) carries A = 6.27e8 s^-1 and is the strongest line overall.
- Every line with intensities has A > 0 (a listed line is dipole-allowed by
  construction, so a zero A would mean a bug, not a selection rule).
- `intensities=False` leaves both fields `None` and changes no wavelength.
- Fine-structure and screened responses return `None` intensities **and** a
  non-empty `intensity_note`.
- Sum check: for a given upper (n, l), the A values of its lines in the list sum
  to `1 / lifetime(n, l)` from the Phase 13 engine, when n_max covers every
  lower level. This ties the spectrum back to the validated engine.

Client: URL round-trip for the flag, and a test that unequal A produces unequal
rendered opacity.

## Deferred

- j-resolved fine-structure line strengths (needs 6j), Phase 15.
- Dipole integrals over screened/numerical radial functions, later.
- Population modelling (Boltzmann/Saha) to turn rates into predicted observed
  intensities. This is a genuinely different physics layer and would need its
  own temperature control; noting it here so the omission stays visible.
