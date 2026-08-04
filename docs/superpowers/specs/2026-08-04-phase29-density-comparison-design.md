# Phase 29: The two densities, on one axis

Status: implemented. See the plan's closeout for what building it changed.
Predecessors: Phase 21 (Hartree-Fock atoms), Phase 27 (total density under HF),
Phase 28 (total density under GSZ, in
`plans/2026-08-04-sulfur-chlorine-and-the-screened-density.md`).

## 1. What this is

Both many-electron models now draw the total radial density D(r). Phase 27 gave
it to Hartree-Fock, Phase 28 gave it to the screened model on a box that can
actually resolve it. They answer the same question about the same observable and
they do not give the same answer, and the app currently asks the reader to hold
one curve in their head while they fetch the other. It says so out loud, in
`web/src/components/RadialView.tsx`:

> Switch to Hartree-Fock, where each subshell gets its own field and nothing is
> fitted, and compare.

That is an instruction to do arithmetic by memory. This phase draws both curves
on one axis and puts a number on the gap.

Scope is the density only. The orbital plots above it stay on one model,
because R(r) and P(r) are a basis choice and two models are under no obligation
to agree on them; a gap between two R(r) curves would be a difference of
convention presented with the visual grammar of a result.

## 2. What was measured before designing

Every number in this section came from running the two engine functions
directly, at 800 display points, before any of the design below was fixed.

**The disagreement is small.** Half the L1 norm of the difference, which is
exactly the number of electrons the two models put in different places:

| element | Z | displaced (e) | as % of N |
|---|---|---|---|
| He | 2 | 0.0003 | 0.02 |
| Li | 3 | 0.0434 | 1.45 |
| Be | 4 | 0.0295 | 0.74 |
| C | 6 | 0.0484 | 0.81 |
| Ne | 10 | 0.0232 | 0.23 |
| Na | 11 | 0.1218 | 1.11 |
| Al | 13 | 0.1274 | 0.98 |
| Si | 14 | 0.1272 | 0.91 |
| Ar | 18 | 0.0600 | 0.33 |

So the two models agree about where the charge sits to roughly one percent,
while GSZ's valence ionization energies sit 2 to 24 percent off NIST. That is
the lesson this view teaches, and it is not the one the current captions
suggest. GSZ was fitted to reproduce Hartree-Fock potentials, so a close density
is what the fit was for; the place the model gives out is the energy.

**Which puts an existing claim under review.** The GSZ branch of the density
caption tells the reader:

> Measurable, and this model is further from it than usual: GSZ was fitted to
> reproduce a potential, not a density, and every shell here sees the same one.

The one measurement available does not support "further from it than usual."
The comparison is against Hartree-Fock rather than against experiment, and HF is
not truth, so this is not yet a refutation. But a claim the app makes on screen
that the app's own new number appears to contradict cannot stay unexamined. The
implementation revisits that caption once the displaced-charge figure is
computed in a test, and rewrites it to say what was measured. It does not get
rewritten in advance of the measurement.

**The grids overlap almost entirely.** Both densities land on
`np.geomspace(solver_r[0], solver_r[-1], points)`, but off different solver
meshes: for argon HF spans [5.6e-5, 60] and GSZ spans [1.4e-3, 64]. Restricting
to the intersection loses under 6e-4 electrons for every element in He..Ar,
which is smaller than either model's own closure residual. The loss is measured
and added to the reported bar rather than assumed negligible.

**Sodium and magnesium have no third peak under GSZ.** Not a shallow one, none:
there is no local maximum and no minimum past the L shell, and the valence
charge rides out on the tail as a shoulder. Hartree-Fock does resolve a third
maximum, marginally.

| | GSZ maxima (bohr) | HF maxima (bohr) | HF outer dimple depth |
|---|---|---|---|
| Na | 0.095, 0.567 | 0.095, 0.569, 3.163 | 0.3% (min 0.2662, max 0.2670) |
| Mg | 0.087, 0.502 | 0.087, 0.502, 2.434 | 1.5% (min 0.6971, max 0.7075) |

Al, Si, P and Ar give three clean peaks under both. Two consequences for the
design, both in section 4: "this model does not resolve this shell as a peak" is
a first-class answer, and a marginal peak ships with its depth beside it so
nobody reads a 0.3 percent dimple as a crisp shell.

## 3. The engine

A new module, `src/atomsim/density_compare.py`. New rather than folded into
either existing one, because it is the first thing in the codebase that has to
import both `hf_atom` and `screened_atom`, and neither should learn about the
other. `hf_atom.py` mirrors the `screened_atom.py` API surface precisely so the
two are swappable, and that property survives only while neither imports the
other.

```python
@dataclass(frozen=True)
class ShellPeak:
    label: str                # "K", "L", "M"
    gsz_radius: float | None  # None: this model resolves no peak here
    hf_radius: float | None
    gsz_depth: float | None   # relative depth of the preceding minimum
    hf_depth: float | None

@dataclass(frozen=True)
class DensityComparison:
    grid: np.ndarray          # the common log grid, bohr
    gsz: Field                # resampled onto grid
    hf: Field                 # resampled onto grid
    displaced_charge: Quantity  # electrons
    shells: tuple[ShellPeak, ...]
    provenance: Provenance
```

`compare_total_densities(z, n_electrons, *, config, exchange, pauli, points)`
builds it. It keeps `n_electrons` separate from `z` to match the signature both
density functions already take, and guards N != Z with a `ValueError`, because
the Szydlik-Green (d, K) parameters are fitted to neutral atoms and comparing
Hartree-Fock against a model run outside its own fit is not a comparison. The
guard is on the engine's claim rather than on a request: `/api/radial` cannot
produce N != Z in the first place, since `_hf_view_target` sets
`n_electrons = element.z` (`app.py:880`) and `_resolve_config` rejects any
configuration that does not hold exactly `element.z` electrons (`app.py:850`).
An HTTP refusal for it would be a branch no request can reach.

**With Pauli off, both models take the same configuration.** The Hartree-Fock
side resolves to `1s^N`, and the GSZ side is handed that same configuration
object rather than being resolved separately, so the overlay answers one
counterfactual question instead of two. This also routes around
`_resolve_config`, which validates against the occupancy cap and would reject
`1s^18` on the way past. The common grid is
`np.geomspace(max(lo_hf, lo_gsz), min(hi_hf, hi_gsz), points)`; both densities
are interpolated onto it; `displaced_charge` is `0.5 * trapezoid(|D_hf - D_gsz|)`
over that grid.

**The error bar has three terms and states all three**: both models' own closure
residuals, plus the charge that falls outside the common window. Measured for
argon that is 0.0036 + 0.0019 + 0.0006, against a displaced charge of 0.0600. It
is a tenth of the signal, which is worth knowing and worth printing.

**When the bar exceeds the number, the readout says so instead of printing a
figure.** That branch is designed in now because it is cheap now and a retrofit
later.

Helium was expected to be the case that fires it, and it is not. Measured, its
displaced charge is 0.000343 against a bar of 0.000179, so the number is 1.9
times its own bar: thin, but resolved, and the readout is right to state it.
What the thin margin actually needed was the other half of the same idea, since
a fixed three decimals renders that measurement "0.000 plus or minus 0.000",
which is a zero standing in for something that is not zero. The decimals follow
the bar instead, carrying it to two significant figures, so no atom in He..Ar
can print a resolved disagreement as nothing.

**Provenance takes the weaker of the two tiers.** Both models are
`APPROXIMATION` at rest, so the comparison is too. With exchange or Pauli off
the Hartree-Fock density is already `COUNTERFACTUAL` and the comparison inherits
it, its method string names which curve was altered, and the caption says the
overlay is now a fitted model against a deliberately broken one. This is what
makes it a legitimate second question rather than a trap: how much of the gap
between the models is exchange.

**Shell labelling.** Peaks are found on the common grid as local maxima above a
noise floor, matched to shells in order of increasing radius, and labelled K, L,
M by position.

The floor is not in the original design and was added under measurement. Both
solvers produce sign-flipping jitter out in the tail where the orbital amplitude
has decayed past what a float64 eigensolve can represent, at about 1e-34 of the
peak for argon under Hartree-Fock and 1e-60 for neon under the screened model
with the occupancy cap off, and every one of those wiggles is a local maximum.
The floor sits at 1e-8 of the tallest value: six orders below the faintest real
shell in He..Ar, which is sodium's outermost Hartree-Fock peak at 2.2e-2, and
twenty-six orders above the loudest noise. It applies to the maxima only. A deep
valley is what "well separated" means, so flooring the minima would discard the
measurement the depth number exists to make, and would discard it hardest in the
clearest cases. Where
one model has fewer maxima than the other, the missing entries are the outer
ones (that is what Na and Mg do, and it is what physically happens: a valence
shell that fails to separate merges into the tail, not into the core), so
matching is by index from the inside out and the shorter list is padded at the
outer end with `None`. The label count comes from the configuration's occupied
principal quantum numbers, not from either peak list, so the table cannot claim
sodium has two shells.

## 4. The server

`/api/radial/{n}/{l}` gains `compare: bool = False`, and `RadialResponse` gains
`density_comparison: DensityComparisonModel | None`. The model carries both
resampled curves as `FieldModel`, the displaced charge as a `QuantityModel` in
electrons, the shell table, and the provenance.

No job pipeline. Timed cold and warm on this machine:

| | cold | warm |
|---|---|---|
| HF density, Ne | 0.40s | 0.000s (cached) |
| HF density, Ar | 1.33s | 0.000s (cached) |
| GSZ density, Ne | 0.02s | 0.015s |
| GSZ density, Ar | 0.07s | 0.072s |

Turning compare on adds at most one cold HF solve, which is a cost this GET
already pays whenever `model=hf`. The Hartree-Fock solve is cached and the GSZ
solve is cheap, so the steady-state cost of the toggle is about 70 ms.

**Refusals, two of them, and neither is new.** `compare=true` needs both models
to be able to speak, so it calls the two resolvers that already say when they
cannot, and inherits their status codes and their wording:

- `_gsz_element` refuses sulfur and chlorine with 400, naming Szydlik and Green,
  exactly as the model radio already does;
- the hydrogen-like refusal (422) that `_hf_view_target` carries today, since a
  one-electron system has no total density for either model to draw.

**One refusal has to be split out rather than reused.** `_hf_view_target` also
refuses an unoccupied (n, l), and that check belongs to the orbital plots, not
to the density, which does not depend on (n, l) at all. Reusing it whole would
refuse a legitimate comparison because of an orbital the reader is not asking
Hartree-Fock to draw. So the shared part, the hydrogenic refusal plus the
configuration resolution plus `_validate_hf_request`, moves into
`_many_electron_target(system, config, pauli)`, and `_hf_view_target` becomes
that call plus the occupancy check it owns. One resolver for the shared
refusals, no duplication, and the two cannot drift apart.

## 5. The frontend

- **`state/store.ts`**: `compare: boolean`, false by default, added to the set
  that invalidates the radial response. It changes what gets fetched, so it is
  physics state and not a presentational toggle.
- **`lib/urlState.ts`**: `compare=1`, round-trip tested. This is a tour hook:
  `?system=na&view=radial&compare=1` lands directly on the sodium finding.
- **`components/Controls.tsx`**: a "Compare models" checkbox, disabled with the
  reason beside it in each refusal case, and absent for hydrogen-like systems.
  The client derives the availability rather than being handed it, matching the
  ATOM_KEYS split from Phase 28.
- **`components/RadialView.tsx`**: `FieldPlot` takes an optional second field
  and a legend. The second curve is dashed, not a second hue, so the comparison
  survives the density colour map and greyscale printing. Below the plot: the
  displaced-charge readout with its `Badge`, and the shell table with an
  explicit "no separate peak" cell where a model resolves none, and the dimple
  depth beside a marginal one.
- Captions: what the number means, that both curves integrate to N so the signed
  difference is zero and the L1 half-norm is the whole story, and the section 2
  claim once it has been measured.

## 6. Tests

Engine:

- displaced charge between two analytic curves with a hand-computable answer;
- zero against itself, exactly;
- symmetric under swapping the two models;
- window loss measured across He..Ar and asserted below the reported bar;
- Na and Mg pinned as the no-third-peak case under GSZ and the marginal-peak
  case under HF, with the dimple depth asserted so a solver change that flattens
  it fails loudly rather than silently dropping a shell;
- Al, Si, P, Ar pinned at three peaks under both;
- the displaced-charge figures in section 2 pinned at the tolerance the bar
  supports, so the claim the caption ends up making is checkable.

Server: `compare=true` returns the comparison; each of the three refusals
returns 422 with its reason; the counterfactual switches produce a
`COUNTERFACTUAL` tier on the comparison.

Web: legend and table render; the toggle disables with its reason for sulfur;
`compare=1` round-trips through the URL; the radial response invalidates when
the toggle flips.

## 7. Out of scope

- Comparing the plane or the cloud. Point clouds do not difference by eye and
  the 2-D comparison wants its own spec.
- A chooseable second curve (GSZ, or HF with exchange off). The inherited-switch
  behaviour in section 3 already reaches the exchange question without a second
  selector; if the selector is wanted later it is additive.
- The screened solver's box, still open and still the maintainer's call, written
  up in `plans/2026-08-04-sulfur-chlorine-and-the-screened-density.md`.
