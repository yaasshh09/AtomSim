# Phase 29: The two densities, on one axis

Status: designed, not implemented.
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
density functions already take, and does not itself refuse N != Z; that refusal
is a statement about where the GSZ parameters are valid and belongs at the
server boundary with the other two, in section 4. The common grid is
`np.geomspace(max(lo_hf, lo_gsz), min(hi_hf, hi_gsz), points)`; both densities
are interpolated onto it; `displaced_charge` is `0.5 * trapezoid(|D_hf - D_gsz|)`
over that grid.

**The error bar has three terms and states all three**: both models' own closure
residuals, plus the charge that falls outside the common window. Measured for
argon that is 0.0036 + 0.0019 + 0.0006, against a displaced charge of 0.0600. It
is a tenth of the signal, which is worth knowing and worth printing.

**When the bar exceeds the number, the readout says so instead of printing a
figure.** Helium's displaced charge is 0.0003 against a bar of about 0.0001,
which is a real measurement, but the margin is thin enough that the interface has
to be able to report "the two models agree to within the resolution of this
comparison" rather than a spuriously precise decimal. That branch is designed in
now because it is cheap now and a retrofit later.

**Provenance takes the weaker of the two tiers.** Both models are
`APPROXIMATION` at rest, so the comparison is too. With exchange or Pauli off
the Hartree-Fock density is already `COUNTERFACTUAL` and the comparison inherits
it, its method string names which curve was altered, and the caption says the
overlay is now a fitted model against a deliberately broken one. This is what
makes it a legitimate second question rather than a trap: how much of the gap
between the models is exchange.

**Shell labelling.** Peaks are found on the common grid as local maxima, matched
to shells in order of increasing radius, and labelled K, L, M by position. Where
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

**Refusals, each by name.** `compare=true` is rejected with 422 and a reason for:

- systems with no GSZ parameters (sulfur, chlorine), naming Szydlik-Green as the
  model radio already does;
- hydrogen-like systems, which have no total density at all;
- ions and any `config` with N != Z, because the GSZ (d, K) parameters are
  fitted to neutral atoms and running them at N != Z compares Hartree-Fock
  against a model outside its own fit. This one is the least obvious and gets
  the longest reason.

The refusals live in one resolver so the endpoint and the client cannot drift
apart, the way `_hf_view_target` already holds the HF refusals for four views.

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
