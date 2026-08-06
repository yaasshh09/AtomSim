# Phase 2: What-If Lab: Constants Lab (α + Z): Design

**Date:** 2026-07-15 · **Status:** Approved (design interview 2026-07-15) · **Author:** Yash Gupta
**Parent spec:** `2026-07-04-atom-sim-requirements-design.md` (§4 What-If Lab, §9 Phase 2) · **Builds on:** Phase 1 complete (analytic engine, provenance, five views, deep links)

---

## 1. Goal and success criteria

Open the What-If Lab with its first, smallest, most honest increment: a **constants lab** that lets the user alter the **fine-structure constant α** (continuous) and the **nuclear charge Z** (integer stepper), and watch the *real consequences* under an unmissable `COUNTERFACTUAL` banner. The flagship lesson: **α is the lever that lifts hydrogen's accidental l-degeneracy; α→0 re-fuses it, and the gross Bohr ladder never moves because α does not enter it.**

This is the "α-first, 5 behind" hybrid: ship the α/Z lab now using the existing analytic machinery, and leave a clean seam so the full five-constant panel (ℏ, e, mₑ, ε₀, c → derived α/a₀/E_h) drops in later without rework.

**Done when:**

1. A new sixth view, **What-If**, sits alongside cloud/plane/radial/levels/spectrum.
2. It shows the **real-universe** level diagram beside the **altered** one, with a `COUNTERFACTUAL` banner shown only when α ≠ the real value.
3. An **α slider** (real value marked) and an **integer-Z stepper** drive the altered column live; the gross ladder is visibly identical on both sides, only the fine splitting moves.
4. Every altered-α quantity carries **COUNTERFACTUAL** provenance to the browser; real-α quantities stay **APPROXIMATION** as in Phase 1.
5. The view surfaces the perturbative fine-structure **error** and warns when it exceeds validity, cranking α up lands honestly in "don't trust this number" territory.
6. The What-If state (view, α, lab Z) is **addressable by deep link**, the Phase 2 guided-tour hook.
7. Validation suite green in CI: engine α-parametrization, server endpoint, and web logic/URL round-trip.

## 2. Decisions locked in the 2026-07-15 interview

| Question | Decision |
|---|---|
| Which thread of Phase 2 first | **Constants lab** (What-If Lab entry point) |
| What the lab varies | **α (continuous) + Z (integer stepper)**; reduced mass already exact. Hybrid "α-first, 5 constants behind a seam" |
| Where the lesson lives | **Dedicated sixth "What-If" view**; rest of the app stays real and untouched |
| Presentation | **Real vs. altered** level diagrams side by side under a `COUNTERFACTUAL` banner |
| Z representation | **Integer** stepper via the existing `hydrogen_like`/int-Z engine path, no `Z: int→float` change |
| Architecture | **Approach A**, thin `alpha` parameter through the existing analytic path + one new view. Rejected: (B) thread a full constants object now (premature); (C) client-side α² rescaling (frontend physics with no provenance, violates the prime directive) |
| Altered-α fidelity | **COUNTERFACTUAL** is the headline fidelity; the α² Pauli approximation + `(Zα)²` error are disclosed in the provenance body |

## 3. Physics grounding (why this design is honest)

The engine computes in Hartree atomic units. The dimensionless level structure depends only on **Z**, the **reduced-mass ratio μ/mₑ**, and **α** (entering fine structure as α²):

- **Gross Bohr energy** `E_n = −μ·Z²/(2n²)` is **α-independent**. Real and altered columns share an identical gross ladder, captioned explicitly. This is the honest core of the degeneracy lesson.
- **Fine-structure shift** `ΔE = −(μ·Z⁴·α²/2n⁴)(n/(j+½) − ¾)` scales as **α²** and **Z⁴**. Turning α up lifts the accidental l-degeneracy (SO(4) symmetry of 1/r); turning α→0 restores it.
- The shift is **perturbative (order α²)**. Its own `(Zα)²` error term already quantifies where it stops being trustworthy, precisely the regime the slider reaches into. Surfacing that error *is* the lesson, not a bug to hide.
- **Lamb-shift honesty hook:** even with fine structure on, states of equal (n, j) but different l (e.g. 2s₁/₂, 2p₁/₂) stay degenerate here, the model has no Lamb shift. Disclosed in assumptions and captioned.

Changing Z to an integer > 1 selects a *real* hydrogen-like ion (Li²⁺, …), not a counterfactual, so the `COUNTERFACTUAL` banner keys on **α ≠ real α only**.

## 4. Engine: `analytic/fine_structure.py`

Add `alpha: float = ALPHA` to `fine_structure_shift(...)` and `level_energy(...)`. The value/error formulas already reference α; they read the parameter instead of the module global.

- `value = -(mu_ratio * Z**4 * alpha**2 / (2 n**4)) * (n/(j+0.5) - 0.75)`
- `error = abs(value) * ((Z*alpha)**2 + m_over_M + _G2)`
- **Provenance branch:**
  - `math.isclose(alpha, ALPHA)` (real) → `Fidelity.APPROXIMATION`, byte-for-byte as Phase 1 (regression-preserving).
  - otherwise → `Fidelity.COUNTERFACTUAL`; method appended with `"; altered fine-structure constant α = {alpha:g} (real {ALPHA:g})"`; the Pauli-approximation `assumptions`, `error_estimate`, and Dirac `refinement` carried unchanged.
- `level_energy` combines the α-independent EXACT Bohr energy with the (possibly counterfactual) shift; when α is altered its combined headline fidelity follows the shift → COUNTERFACTUAL.

**The seam:** `ALPHA` remains the real anchor and the parameter default. A comment documents that the `alpha` argument is the dimensionless value a future `FundamentalConstants.alpha` will supply, the five-constant panel plugs in here with no signature change.

## 5. Server: `server/app.py`

- `/api/levels` gains optional `alpha: float | None`. Validated `0 < alpha <= 0.5` (the upper bound deliberately reaches into the broken regime so the error readout can teach; out-of-range → HTTP 422). Flows into the `level_energy`/`fine_structure_shift` calls; inert when `fine_structure=False`.
- `_resolve_system` learns to parse a `z{N}` key (N ≥ 1) → `systems.hydrogen_like(N)`, expressing the integer-Z stepper as a real generic hydrogen-like ion. Reuses existing code. (Alternative, a raw `Z` query param, rejected as less consistent with the system-keyed API.)
- `LevelsResponse` echoes the applied `alpha` (float | None) so the response is self-describing for the banner. Per-level provenance already rides through `QuantityModel`, so COUNTERFACTUAL survives to the browser with no schema work.

Scope note: only `/api/levels` is α-aware in this increment. `/api/state` and `/api/spectrum` gain the same optional `alpha` later (same one-line parameter threading) if the tour needs them.

## 6. Web: the sixth view

- **Store (`state/store.ts`)**, a lab slice isolated from the main physics, so it never triggers `INVALIDATED`:
  - `labAlpha: number` (default = real α), `labZ: number` (int ≥ 1, default 1), their setters.
  - `whatif: { real: LevelsResponse; altered: LevelsResponse } | null`, status/progress, `loadWhatIf()` which fetches `/api/levels` twice, real (no α) and altered (α = labAlpha), both at `system=z{labZ}`, `fine_structure=true`, shared `n_max = N_MAX_DIAGRAM` (6, reused from the Levels view for a readable diagram).
  - `ViewMode` gains `"whatif"`.
- **`lib/whatif.ts`** (pure, unit-tested): builds the two level-column layouts (reusing `lib/levels.ts`), detects whether the accidental degeneracy is lifted, formats α as `1/N`, and computes the max fractional fine-structure error across altered levels plus the warn-threshold decision.
- **`components/WhatIfView.tsx`**: real-vs-altered columns; `COUNTERFACTUAL` banner (only when α ≠ real); α slider with the real value marked; integer-Z stepper; a provenance `Badge`; and the error readout with its warning band.
- **`api/client.ts` + `api/types.ts`**: `getLevels` gains optional `alpha`; types mirror the `LevelsResponse.alpha` echo.
- **`lib/urlState.ts`**: deep-link schema gains `view=whatif`, `alpha`, and lab `Z`. α is clamped to `(0, 0.5]`, lab Z to integer `[1, 10]`; junk drops (mirrors existing hard-validation style). Round-trip tested. This is the guided-tour hook, a tour can link straight to "α = 1/50."
- **`App.tsx` / tabs**: add the What-If tab.

## 7. Honesty / provenance (the heart)

- **Gross levels**: identical both sides, stay `EXACT`, captioned "α does not enter the gross Bohr energy."
- **Altered-α fine levels**: `COUNTERFACTUAL` headline (you changed the rule), with the α² Pauli approximation and `(Zα)²` error disclosed in the provenance body, a value that is honestly *both* counterfactual and approximate, the more important disclosure winning the single fidelity slot.
- **Breakdown as lesson**: the view shows the max fractional fine-structure error; past ~10% it renders a warning band ("perturbative fine structure is past its validity here, exact Dirac would differ"). The `refinement` field already names the fix (Phase 3 Dirac).
- **Lamb-shift caption**: equal-(n,j) states stay degenerate; the model has no Lamb shift, disclosed.
- **No frontend physics**: every number originates in the engine and crosses the boundary as a provenance-carrying `Quantity`. The client only lays out and labels.

## 8. Testing

- **Engine (`tests/test_fine_structure.py`)**: real α reproduces current values (regression); altered α scales the shift by `(alpha/ALPHA)²`; fidelity flips COUNTERFACTUAL↔APPROXIMATION on the `isclose` boundary; error grows with `(Zα)²`.
- **Server (`tests/test_server.py`)**: `/api/levels?alpha=…` returns COUNTERFACTUAL fine levels; omitted/real α unchanged; `system=z3` resolves via `hydrogen_like`; `alpha<=0` or `>0.5` → 422; `LevelsResponse.alpha` echo correct.
- **Web**: `lib/whatif.test.ts` (column diff, degeneracy-lifted detection, α→`1/N` formatting, warn-threshold logic); `state/store.test.ts` (lab slice setters do **not** invalidate main physics; `loadWhatIf` wiring); `lib/urlState.test.ts` (alpha/Z round-trip + invalid clamping).

## 9. Unit boundaries (for isolation and clarity)

| Unit | One job | Depends on |
|---|---|---|
| `fine_structure.alpha` param | make α an injectable dimensionless input | `ALPHA` anchor only |
| `/api/levels?alpha` + `z{N}` resolve | expose altered levels + generic-Z over HTTP | engine, `hydrogen_like` |
| `lib/whatif.ts` | derive the comparison/degeneracy/error readout from two responses | `lib/levels.ts`, response types |
| `WhatIfView` | render + drive the two columns and the knobs | store slice, `whatif.ts`, `Badge` |
| store lab slice | hold α/Z lab state + fetch, isolated from main physics | `api/client` |
| `urlState` additions | make the lab state addressable | existing validators |

## 10. Out of scope (YAGNI / deferred)

- The full five-constant panel (ℏ, e, mₑ, ε₀, c), seam left open in §4; not built now.
- α on `/api/state`, `/api/spectrum`, the 3D cloud, or the 2D plane, α does not change orbital *shape* (only fine-structure energies), so the cloud/plane would not move; deferred until a concrete need.
- Continuous (float) Z.
- Force-law morphing, screening/multi-electron, and the guided tour itself, separate Phase 2 threads with their own specs.
