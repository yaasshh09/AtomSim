# Five-Constant Panel (What-If Lab): Design Spec

**Date:** 2026-07-15 · **Status:** Approved · **Phase:** 2 (What-If Lab), item 1 completion
**Author:** Yash Gupta

---

## 1. What this is

Phase 2's first item (requirements §4, §9) is the *fundamental-constants lab with the
degeneracy lesson*. Phase 2's shipped increment covered only a direct **α** slider plus a
**Z** stepper. This spec completes that item: expose the **raw five constants**, ℏ, e, mₑ,
ε₀, c, as inputs, derive α and the atomic scales from them, and surface the headline
teaching moment: **many different constant-tuples produce the identical atom**, and the
honesty layer names exactly which dimensionless combination actually moved.

α stops being a direct input. It becomes a *derived* readout computed from the raw five, 
you can only reach α values that some real choice of constants produces. Letting the user
drag both α and the constants it is made of would permit physically inconsistent states,
which the prime directive forbids.

This builds on the seam already in the code: `constants.py` defines
`FundamentalConstants(hbar, e, m_e, eps0, c)` with derived `alpha`, `bohr_radius`, and
`hartree_energy` properties, whose docstring already states "A FundamentalConstants instance
with altered fields IS a counterfactual universe (What-If Lab, later phase)."

## 2. Physics: what is actually observable

The observable content of a one-electron atom collapses to three numbers:

| Observable | Formula | Role |
|---|---|---|
| **α** (dimensionless) | e² / (4π ε₀ ℏ c) | drives fine-structure splitting |
| **a₀** (length) | 4π ε₀ ℏ² / (mₑ e²) | the atom's physical size (shown in pm) |
| **E_h** (energy) | ℏ² / (mₑ a₀²) | its binding energy (shown in eV) |

Everything else the user can vary is either redundant or unobservable against fixed SI
rulers. As multiplier ratios (altered / real), with multipliers (h, q, m, ε, γ) on
(ℏ, e, mₑ, ε₀, c):

- ratio(α) = q² / (ε · h · γ)
- ratio(a₀) = ε · h² / (m · q²)
- ratio(E_h) = m · q⁴ / (ε² · h²)

**The degeneracy lesson.** These ratios are what make the teaching moment concrete:

- **e×2, ε₀×4** → ratio(α)=1, ratio(a₀)=1, ratio(E_h)=1 → *identical atom, nothing
  observable changed* despite two altered constants. This is the headline case.
- **mₑ×½** → ratio(a₀)=2, ratio(E_h)=½, ratio(α)=1 → atom doubles in size, half as deeply
  bound, fine structure unchanged.

The honesty layer reports, per observable, whether it `changed` and by what ratio. The
"identical atom" case is when non-unit multipliers still leave all three flags false.

### Honesty: structure vs. scale (resolved decision)

The reused level diagram shows fine-structure **structure**, driven by the derived α, and is
labeled in **units of E_h** (normalized), *not* real eV. Absolute scale (a₀ in pm, E_h in
eV) lives only in the readouts, where the numbers carry that story directly. This keeps the
diagram honest without threading an altered eV-conversion through the level reuse: fine
structure still visibly responds because the ratio of splitting to E_h scales as α².

### Honesty: the model-breakdown edge case

Multipliers in [¼, 4] can drive derived α well past 0.5, where the perturbative fine
structure is meaningless (`/api/levels` validates α ∈ (0, 0.5]). The lab handles this
honestly rather than hiding it:

- The readouts always show the **true** derived α, even when large, the displayed number is
  never clamped.
- The altered level diagram is requested **only when derived α ≤ 0.5**. Past that, the view
  shows the existing "beyond the perturbative model's validity" notice in place of the
  altered column, we refuse to draw a diagram we cannot honestly compute.

## 3. Architecture

Physics, deriving α, a₀, E_h from raw constants, happens **server-side** and crosses the
boundary as provenance-carrying `Quantity`s. No physics in the frontend.

**Flow when a slider moves:**

1. Client sends the five multipliers to the new **`GET /api/constants`** endpoint.
2. Server builds an altered `FundamentalConstants`, returns derived **α, a₀ (pm), E_h (eV)**
   as `Quantity`s (`COUNTERFACTUAL` when any multiplier ≠ 1, else `EXACT`), plus a
   **degeneracy analysis** (per-observable ratio + `changed`).
3. Client feeds the **derived α** into the existing `/api/levels?system=z{Z}&alpha=…`, 
   the fine-structure diagram is reused unchanged (subject to the α ≤ 0.5 guard above).

*Alternative considered and rejected:* folding the five multipliers into `/api/levels` as
extra params. It overloads a stable endpoint and couples scale-derivation to level
computation. A dedicated `/api/constants` keeps each endpoint to one job.

### Module split

| Layer | File | Responsibility |
|---|---|---|
| Engine | `src/atomsim/constants_lab.py` *(new)* | `analyze_constants(...) -> ConstantsReport`: pure derivation of the three observables + degeneracy flags. Keeps `constants.py` as pure anchors. |
| Server | `src/atomsim/server/app.py`, `schemas.py` | `/api/constants` endpoint + `ConstantsReportModel` / `DerivedObservableModel`. |
| Web (logic) | `web/src/lib/whatif.ts` | Multiplier constants (`CONST_MIN`, `CONST_MAX`, `CONSTANT_KEYS`), `formatRatio`, `deriveAlphaValid`; keep existing `formatAlpha`/error helpers. |
| Web (state) | `web/src/state/store.ts`, `api/client.ts`, `api/types.ts` | `labConst` five-multiplier object replaces `labAlpha`; `getConstants()` client; `loadWhatIf` orchestrates both fetches. |
| Web (view) | `web/src/components/WhatIfView.tsx` | Five sliders + readout table + degeneracy sentence; keep Z stepper and diagram. |
| Web (URL) | `web/src/lib/urlState.ts`, `main.tsx` | `?alpha=` replaced by `?hbar=&e=&me=&eps0=&c=`; `?z=` stays. |

## 4. Data structures

**Engine `ConstantsReport`** (frozen dataclass):

- `alpha: DerivedObservable`
- `bohr_radius_pm: DerivedObservable`
- `hartree_ev: DerivedObservable`
- `altered: bool`, any multiplier ≠ 1

where `DerivedObservable` holds `quantity: Quantity`, `ratio: float` (altered / real), and
`changed: bool` (`not isclose(ratio, 1)`).

- Fidelity: `EXACT` at all-ones (closed-form from CODATA); `COUNTERFACTUAL` when altered.
- `method` cites the derivation formula and the applied multipliers (real values named).

**Server response** mirrors the report via `QuantityModel.from_quantity`, so provenance
survives to the browser (existing pattern).

**Store `labConst`:** `{ hbar: number; e: number; m_e: number; eps0: number; c: number }`,
all default 1.0. `whatif` state becomes
`{ report: ConstantsReport; real: LevelsResponse; altered: LevelsResponse | null }`, 
`altered` is null when derived α > 0.5 (beyond-validity).

## 5. Validation and ranges

- Each multiplier: slider range **[¼, 4]** (log scale); server validates **[0.25, 4]**,
  returning 422 outside it. Deep-link params clamp to the same range.
- `Z` stepper unchanged: [1, 10], integer.
- Derived α: reported truthfully at any value; only the *diagram request* is gated at ≤ 0.5.

## 6. URL schema change

`?alpha=` (from the shipped increment) is **removed**, α is no longer a direct input. It is
replaced by five optional multiplier params, each omitted when equal to 1 and clamped to
[0.25, 4] on parse:

    ?view=whatif&e=2&eps0=4&z=3

`?z=` is unchanged. This breaks old `?alpha=` deep-links; acceptable pre-v1 for a showcase.
Round-trip tested, as the existing URL contract is.

## 7. Testing (TDD, per repo convention)

**Engine, `tests/test_constants_lab.py`:**

- all-ones → `EXACT`, every ratio 1.0, every `changed` false, `altered` false.
- **degeneracy case:** e×2 & ε₀×4 → α, a₀, E_h all `changed=false` (identical atom).
- mₑ×½ → ratio(a₀)=2, ratio(E_h)=½, ratio(α)=1.
- altered → `COUNTERFACTUAL`; `method` names the applied multipliers.
- derived α matches `FundamentalConstants(...).alpha` for a spot check.

**Server, `tests/test_server.py`:**

- `/api/constants` default → α ≈ real CODATA, `altered=false`.
- degenerate pair (`e=2&eps0=4`) → `alpha.changed` and `hartree_ev.changed` both false.
- out-of-range multiplier → 422.

**Web:**

- `lib/whatif.test.ts`, `formatRatio`, constant-key helpers, α-validity guard.
- `state/store.test.ts`, `setLabConst` clears only lab data, never `positions`/`levels`
  (the store invariant already guarded for the α slider).
- `lib/urlState.test.ts`, five-multiplier round-trip; clamp out-of-range; junk dropped.

`WhatIfView` follows the repo pattern of no component unit test, verified by
`tsc --noEmit` / build + manual QA.

## 8. Out of scope (YAGNI)

- To-scale size visualization (the atom growing/shrinking), readouts carry the size story.
- Rescaling the diagram's eV axis with altered E_h, resolved via structure/scale split.
- Mass-ratio (μ/mₑ) as a What-If input, separate from the five raw constants.
- A "quick α" convenience slider, add only if the guided tour later needs it.
- Back-compat parsing of the old `?alpha=` param.

## 9. Success criteria

- Dragging any raw slider updates α, a₀, E_h readouts live, each with a ratio and provenance
  Badge, `COUNTERFACTUAL` when altered.
- The e×2 / ε₀×4 degeneracy renders as "nothing observable changed" with all three flags
  false, the headline lesson lands.
- Driving derived α past 0.5 shows the honest beyond-validity notice, not a wrong diagram or
  a crash.
- `?view=whatif&e=2&eps0=4&z=3` deep-links into that exact lab state.
- Full `pytest`, `npm test`, `npm run build`, and `ruff check .` green.
