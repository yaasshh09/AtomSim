# Phase 17: Population modelling

Status: design approved 2026-07-26.

Closes the loudest disclosed liberty in the app. `SPECTRUM_INTENSITY_LIBERTY`
currently says, in as many words, that bar height "is a log-compressed rate,
NOT a predicted observed intensity", that "no level populations are modelled:
no temperature, density or optical depth", and names its own cure:
"population modelling (Boltzmann/Saha) would turn rates into observable
intensities". This is that.

## What this adds

A temperature and an electron density, and the two-step LTE chain that turns
an Einstein A into an emissivity:

    Saha       -> how much of the element is still neutral at (T, n_e)
    Boltzmann  -> what fraction of the neutrals sit in the upper level
    emissivity -> that population, times A, times the photon energy

The bars in the Spectrum view then mean something a spectroscopist would
recognise, and the reason Balmer lines dominate a 10,000 K spectrum while
Lyman dominates a cool one becomes something you can produce by dragging a
slider rather than something the caption asserts.

## The chain, stated

Level populations, LTE, energies measured from the ground level:

    N_i / N_neutral = g_i exp(-(E_i - E_1) / kT) / U(T)
    U(T) = sum_i g_i exp(-(E_i - E_1) / kT)

Degeneracy comes from whichever level scheme is in play: `2(2l+1)` for a gross
`(n, l)` sublevel, `2j+1` for a fine-structure `(n, l, j)` one. Both sum to
`2n^2` per shell, which is the arithmetic check.

Ionization, with the ion being a bare nucleus so `U_II = 1`:

    n_II / n_I = (2 U_II / U_I) (2 pi m_e k T / h^2)^(3/2) exp(-chi / kT) / n_e
    x = n_II / (n_I + n_II)

Emissivity per line, per atom of the element (neutral and ionized together):

    eps = (1 - x) * (N_u / N_neutral) * A * h nu     [eV/s per atom]

Dimensioned on purpose. A bare 0-to-1 "relative intensity" would hide that the
whole spectrum dims as the gas ionizes, which is half of what the density
slider is for.

## The partition function diverges, and that is the interesting part

`U(T) = sum_n 2n^2 exp(-13.6(1 - 1/n^2)/kT)` does not converge. The terms tend
to `2n^2 exp(-13.6/kT)`, which grows without bound: an isolated atom in
thermodynamic equilibrium has infinitely many bound states crowding the
ionization limit, and they carry infinite statistical weight.

The sum is therefore truncated at the same `n_max` the line list uses, and the
truncation is disclosed in the provenance rather than buried. This is not a
numerical convenience dressed up as physics: the real resolution is that a
plasma at finite density has no `n = 100` states to occupy, because neighbouring
ions blur them into the continuum (pressure ionization, and the occupation
probability formalisms that model it). `n_max` is a crude stand-in for a real
cutoff, and the provenance says so, with the refinement naming what would
replace it.

Practically this matters most at high T and small `n_max`, where U is
noticeably sensitive to where the sum stops. The assumption string carries the
`n_max` used so the number is never quoted without its cutoff.

## Fidelity

`APPROXIMATION` throughout, with these assumptions carried on every quantity:

- **LTE.** One temperature describes both the level populations and the
  ionization. Real nebulae are not in LTE, and neither is a discharge lamp.
- **Optically thin.** No radiative transfer, no self-absorption, no escape
  probability. A strong line in a thick medium is not this bright.
- **The partition function is truncated at `n_max`**, as above.
- **`n_e` is an independent control, not solved self-consistently.** In a pure
  hydrogen plasma the electron density is *produced by* the ionization it
  drives; here you set both, so you can dial in combinations that no single
  equilibrium gas would hold. This is a deliberate lab-knob choice and it has
  to be said out loud.
- **No continuum.** Only bound-bound lines; no recombination continuum, no
  free-free.

The arithmetic given the model is exact, so no `error_estimate` is invented.
What would be dishonest here is a tight error bar on a model this schematic;
the assumptions are the error bar.

## Module boundaries

New `src/atomsim/populations.py`, which knows nothing about spectra:

- `partition_function(levels, T)` -> `Quantity`, carrying the truncation
- `boltzmann_fractions(levels, T)` -> per-level occupation fractions
- `saha_ionization_fraction(T, n_e, chi, u_neutral)` -> `Quantity` in [0, 1]
- `line_emissivity(population, einstein_a, photon_energy, neutral_fraction)`

`spectra.py` gains `SpectralLine.emissivity` and a `thermal=ThermalConditions(...)`
argument, following exactly the shape `intensities` already has. Thermal implies
intensities, since emissivity is built on A.

Scope: hydrogen-like systems are first-class. Screened atoms need a first
ionization potential, and the honest source for one here is Koopmans applied to
the outermost occupied GSZ orbital, which is a further approximation on top of
an already crude model. Included if it falls out cleanly, deferred with a note
if it does not. It must not be half-wired.

## Surfacing

Two sliders in the Spectrum view, matching the pattern the B and F sliders set
in Phases 10 and 11: temperature, and `log10(n_e / cm^-3)` with ticks marking
nebula, stellar photosphere and lab discharge. Both go into URL state next to
`bField` and `eField`. Bars scale by emissivity when thermal is on and by A when
it is not, so the existing behaviour stays reachable.

`SPECTRUM_INTENSITY_LIBERTY` gets rewritten rather than deleted. Log compression
of the bar height is still a presentational choice and stays disclosed; what
changes is that the quantity being compressed is now a modelled emissivity, and
the assumption list points at the LTE model instead of saying no populations
exist.

## The wavelength axis, folded in

Carried from Phase 15 through Phase 16. A fine-structure line list puts
within-`n` microwave components (27 mm to 11 m) beside optical lines (94 nm to
7.5 um), so the log axis runs to 11 m and squeezes the optical lines into a
sliver.

Thermal weighting fixes the bar half of this by itself and for the right
reason: those lines have `A ~ 1e-12 s^-1`, so their emissivity is negligible
and they draw as nothing. It does not fix the axis. So the view gets a
wavelength window with a disclosed default covering the emitting band, a
"full range" control, and a line saying how many lines fall outside and that
they remain in the data. Nothing is dropped, and the disclosure is a plain
sentence of fact rather than a new liberty.

## Validation anchors

The temperature dependence is the load-bearing claim, so it is checked by
behaviour that has a known direction, not by remembered numbers:

| check | target |
|-------|--------|
| Boltzmann fractions sum to 1 | exactly, at every T |
| level degeneracies sum to `2n^2` per shell | exact, both schemes |
| high-T limit | populations approach the degeneracy ratio `g_i / sum g` |
| low-T limit | ground state takes essentially everything |
| Balmer / Lyman emissivity ratio | rises monotonically with T |
| Saha `x` | monotone up in T, monotone down in `n_e` |
| Saha at fixed `x = 0.5` | the T it happens at rises with `n_e` |
| `U(T)` sensitivity to `n_max` | grows with T, which is why it is disclosed |
| emissivity with thermal off vs on | ordering by A alone vs by A weighted |

No literature intensity ratio is asserted to a digit. The one external anchor
worth stating is qualitative and robust: hydrogen in a stellar photosphere is
about half ionized in the neighbourhood of 10^4 K, and the test asserts the
order of magnitude of that crossover, not its value.

## Deferred

- Doppler and pressure line profiles; lines stay as bars.
- Radiative transfer and optical depth.
- Non-LTE level kinetics.
- Recombination and free-free continuum.
- Screened-atom thermal spectra, if the Koopmans route does not land cleanly.
- Hyperfine-resolved line strengths, unchanged from Phase 15.
