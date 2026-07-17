# Phase 5 — What-If: Force-Law Preset Library (design)

**Date:** 2026-07-17
**Branch:** `phase5-force-law-presets`
**Status:** approved, ready for planning

## 1. Purpose

Phase 4 shipped one counterfactual force law — the power-law `V(r) = -Z/rᵖ` —
against an EXACT hydrogen baseline. This phase completes the **curated preset
library** the requirements doc calls for (§4, "Force laws: v1 ships a curated
preset library — Yukawa/screened, power-law 1/rᵖ, harmonic, finite well,
Coulomb-plus-core"). Four presets join the existing one, each a different central
potential the *same* numerical radial solver can handle.

The engine already solves arbitrary `V(r)` (`numerics/radial_solver.py`); this
phase generalizes the phase-4 driver from one hardcoded shape into a **preset
registry** and teaches the view to draw the potential itself, not just an energy
ladder. It invents no new fidelity machinery.

## 2. Physics (the honest core)

### 2.1 The five presets

All potentials are the **bare radial** `V(r)` in Hartree atomic units; the solver
adds the centrifugal barrier `l(l+1)/(2μr²)` internally
(`radial_solver.py`). `Z` and reduced-mass ratio `μ` come from the selected
system, exactly as elsewhere.

| Preset key | `V(r)` | Params (range) | Honest reference | Uses `Z`? | All states bound? |
|------------|--------|----------------|------------------|-----------|-------------------|
| `powerlaw` *(exists)* | `-Z / rᵖ` | `p ∈ [0.5, 1.5]` | EXACT hydrogen, `n = l+1+k` | yes | yes (clamped range) |
| `yukawa` | `-(Z/r)·e^{-r/λ}` | `λ ∈ [0.5, 20]` a₀ | EXACT hydrogen (λ→∞ limit) | yes | **no** (finite count) |
| `harmonic` | `½ k r²`, `k = μω²` | `ω ∈ [0.05, 1.0]` | **EXACT 3-D isotropic QHO**: `E = ω(2k + l + 3/2)` | no | yes (infinite tower) |
| `finitewell` | `-V₀` for `r<a`, else `0` | `V₀ ∈ [0.1, 5]`, `a ∈ [0.5, 10]` a₀ | structural markers: floor `-V₀`, threshold `0` | no | **no** (finite count, maybe 0) |
| `coulombcore` | `-Z/r + c/r²` | `c ∈ [0, 1]` | EXACT hydrogen (c→0 limit) | yes | yes |

**Parameter conventions.**
- `harmonic` is parametrized by angular frequency `ω` (intuitive, sets the level
  spacing directly); `k = μω²` is derived so `E = ω(2k_r + l + 3/2)` holds with the
  reduced mass folded in.
- `coulombcore` uses `+c/r²` deliberately: it adds to the centrifugal term, so it
  maps to an effective `l_eff(l, c)` and reproduces a **quantum-defect / core-
  penetration** splitting — the alkali story, the on-ramp to Tier-2 screening.
  (A Gaussian core was considered and rejected as less analytically transparent.)

### 2.2 Bound-state filtering — a correctness requirement

A finite-difference solver on a finite box returns `n_states` eigenvalues
**whether or not** they are physical bound states. For `yukawa` and `finitewell`
the true bound spectrum is finite (possibly empty), and the box also manufactures
spurious **positive-energy** "particle-in-a-box" eigenstates.

Rule: **keep only genuinely bound states, `E < 0`.** The engine returns the bound
subset (0..n_states levels); the response records how many were found, and the
view discloses shortfalls ("only 2 bound states at this λ", "no bound states —
well too shallow"). This is a physics-honesty invariant, not a UI nicety: drawing
a box artifact as a bound level would be a quiet lie.

`powerlaw` (clamped away from fall-to-center), `harmonic` (positive-definite,
infinite tower), and `coulombcore` (Coulomb tail) always yield `n_states` bound
levels, so filtering is a no-op there but still applied uniformly.

### 2.3 Per-preset honest reference

The reference generalizes from "always hydrogen" to a small tagged shape:

```
Reference = { kind: "levels" | "markers", items: [ ReferenceItem, ... ] }
ReferenceItem = { label: str, energy: Quantity, ... }
```

- **`kind: "levels"`** — a list of reference energies, each EXACT:
  - `powerlaw` / `yukawa` / `coulombcore` → closed-form hydrogen `E_n = -(Z²μ)/(2n²)`
    for `n = l+1+k`, paired index-for-index with the counterfactual (the Coulomb
    limit each preset deforms away from).
  - `harmonic` → closed-form 3-D isotropic QHO `E = ω(2k + l + 3/2)`, `k = 0..n_states-1`.
    A **new EXACT ground truth**, independent of hydrogen — see §6.
- **`kind: "markers"`** — horizontal reference lines that are structural, not a
  level ladder:
  - `finitewell` → well floor `-V₀` and continuum threshold `0`, both EXACT
    (definitional given the params).

Every reference item carries a `Provenance` (EXACT). The asymmetry — NUMERICAL
counterfactual vs EXACT reference — is disclosed per item, exactly as in phase 4.

**Reference count under shortfall.** For `kind: "levels"` the reference always
lists the full `n_states`-entry ideal ladder (hydrogen or QHO), even when
`yukawa`/`finitewell` return `bound_count < n_states` counterfactual levels.
Pairing is by radial index `k` where both exist; the *unpaired* upper reference
levels are exactly the states screening/shallowness removed, and drawing them is
the point — the view shows the Coulomb ladder with its top rungs unmatched.

### 2.4 The potential curve

So the view can draw the actual shape (§4), the engine samples `V(r)` on a curve
(~256 points) over a preset-appropriate `r` range and returns it as a `Field`.
`V(r)` is exact arithmetic given the params, so the curve is **EXACT**. Returning
the solver's own potential (rather than recomputing in TypeScript) keeps a single
source of truth and prevents Python↔TS drift.

## 3. Engine — `force_law.py` → preset registry

Refactor the hardcoded power-law into a registry. Each preset is a small record:

```python
@dataclass(frozen=True)
class ForcePreset:
    key: str
    build_potential: Callable[[Params, int], Callable[[np.ndarray], np.ndarray]]
    params: tuple[ParamSpec, ...]      # name, min, max, default, unit
    reference: Callable[..., Reference]
    uses_Z: bool
    r_max: Callable[[Params], float]   # curve/box range hint per preset
```

`force_law_levels(preset_key, params, l, system, n_states)`:
1. look up the preset, validate params against each `ParamSpec` (raise `ValueError`
   out of range),
2. `build_potential(params, Z)` → `V(r)`,
3. drive the **unchanged** `solve_radial_with_error(V, l, mu_ratio=μ, n_states=...)`,
4. **filter to bound states** (§2.2),
5. build the per-preset `Reference` (§2.3),
6. sample the potential curve (§2.4),
7. stamp preset-specific provenance notes ("Yukawa screened potential λ=…").

The existing `powerlaw` path is preserved exactly (same numbers, same clamping)
as one registry entry — phase-4 tests must stay green.

Result type generalizes `ForceLawResult`: `preset_key`, `params`, `l`, `z`,
`system_key`, `counterfactual: tuple[ForceLawLevel, ...]`, `bound_count: int`,
`requested_count: int`, `reference: Reference`, `potential_curve: Field`.

## 4. Server — generalized `GET /api/forcelaw`

Stays synchronous (scalar-small payload plus a ~256-point curve; no async job).

```
GET /api/forcelaw?preset=<key>&l=<int≥0>&system=<key>&n_states=<int>
                  &p=… | &lambda=… | &omega=… | &v0=…&a=… | &core=…
```

- **Back-compat:** `preset` absent ⇒ `powerlaw`, reading `p` — every existing
  phase-4 deep link keeps working unchanged.
- Only the active preset's params are read; unknown/irrelevant params ignored.
- **Validation → 422** on unknown `preset`, out-of-range param, or `l < 0`
  (same pattern as the constants endpoint).
- `system` via the existing `_resolve_system`.
- eV conversion at the boundary, appended to provenance `method` (existing
  `_to_ev`).

Response (Pydantic, mapping `Quantity`/`Field`/`Provenance` straight through):

```jsonc
{
  "preset": "yukawa",
  "params": { "lambda": 3.0 },
  "l": 0,
  "system": { ...SystemModel... },
  "counterfactual": [ { "energy": QuantityModel, "energy_ev": QuantityModel,
                        "radial_index": 0, "provenance": ProvenanceModel }, ... ],
  "bound_count": 2,
  "requested_count": 4,
  "reference": { "kind": "levels",
                 "items": [ { "label": "n=1", "energy": QuantityModel,
                              "energy_ev": QuantityModel, "provenance": ProvenanceModel }, ... ] },
  "potential_curve": { "r": FieldModel, "v": FieldModel }   // eV / a₀, EXACT
}
```

## 5. Frontend

### 5.1 State & URL

- Store slice extends the flat phase-4 style (`forceP`, `forceL`):
  `forcePreset` + one field per param (`forceLambda`, `forceOmega`, `forceV0`,
  `forceA`, `forceCore`), each **clamped** to its `ParamSpec`. Plus a presentational
  `forceViz: "well" | "ladder"`.
- `INVALIDATED`: `forcePreset`, `forceL`, the active param fields, and `system`
  change the physics and invalidate `forceLaw`. `forceViz` is presentational and
  **invalidates nothing** (store invariant, matching the other view/color toggles).
- **URL:** add `preset` (default `powerlaw`) and serialize the active preset's
  params (`lambda`, `omega`, `v0`, `a`, `core`); `p` and `fl` unchanged. `forceViz`
  stays **out of the URL** (presentational). Round-trip tested per preset; treated
  as a stable query-schema contract.

### 5.2 View — `ForceLawView`

- **Preset picker** + **dynamic parameter controls**: the visible sliders change
  with the preset (λ for Yukawa, ω for harmonic, V₀+a for finite well, c for
  Coulomb-core, p for power-law); `l` selector persists (all central potentials).
- **Default — potential-well diagram:** draw the returned `V(r)` curve; each bound
  level as a horizontal line across its classically-allowed span (`E > V(r)`);
  per-preset reference overlaid — ghosted hydrogen/QHO levels (`kind: "levels"`) or
  floor/threshold markers (`kind: "markers"`).
- **Toggle — energy ladder:** the phase-4 two-column diagram (reference | counter-
  factual), reference column swapping per preset.
- **Shortfall disclosure:** when `bound_count < requested_count`, a plain-language
  note ("only N bound states at this λ", "no bound states — well too shallow").
- **Provenance:** `Badge` per side (NUMERICAL counterfactual vs EXACT reference),
  read straight off each item's `provenance`. The curve badge is EXACT. No invented
  visual-liberty entry — energies and the curve are plotted faithfully on labeled
  linear axes, so the data badges are the whole disclosure.
- **Color/scale:** reuse the single color authority (`lib/luts.ts`) and existing
  level-diagram styling.

## 6. Testing (the honesty checks)

New physics gets a validation test, not a smoke test:

1. **Power-law regression:** every phase-4 `force_law` and endpoint test stays
   green through the registry refactor (same numbers).
2. **Harmonic vs EXACT QHO (headline):** numerical levels match
   `E = ω(2k + l + 3/2)` to solver tolerance, for `l = 0, 1` and a couple of `ω`.
   Independent validation of the solver against a non-Coulomb closed form.
3. **Yukawa:** as `λ → large` the levels approach exact hydrogen (tolerance);
   `bound_count` decreases as `λ` shrinks; only `E < 0` states are returned.
4. **Finite well:** ground state of a chosen `(V₀, a)` matches the known
   transcendental-equation value (fixture) to tolerance; every returned `E ∈ (-V₀, 0)`;
   a too-shallow well returns `bound_count = 0`.
5. **Coulomb-core:** `c → 0` recovers hydrogen (tolerance); `c > 0` splits fixed-`n`
   states by `l` in the quantum-defect direction (sign check).
6. **Bound-state filter:** a preset/param known to expose box artifacts returns no
   positive-energy level.
7. **Endpoint:** each preset round-trips through `/api/forcelaw`; `422` on unknown
   preset, out-of-range param, negative `l`; back-compat (no `preset` ⇒ `powerlaw`);
   both provenance tiers present and distinct.
8. **Frontend:** store slice (clamping, invalidation isolation of `forceViz`);
   `urlState` round-trip per preset; view renders the well diagram and the ladder
   toggle.

## 7. Module boundaries

| Unit | Purpose | Depends on |
|------|---------|-----------|
| `numerics/force_law.py` (registry) | Build each `V(r)`, solve, filter bound, build reference, sample curve | `radial_solver`, `analytic/hydrogen`, new `analytic` QHO helper, `systems` |
| `analytic/` QHO levels | Closed-form 3-D isotropic oscillator `E = ω(2k+l+3/2)` (EXACT) | `provenance` |
| `/api/forcelaw` handler | Validate preset+params, resolve system, serialize incl. curve | `force_law`, `schemas` |
| `ForceLawView` + slice | Preset picker, dynamic params, well diagram + ladder toggle, badges, URL | api client, store, `liberties`, `luts` |

Each is understandable and testable without reading the others' internals.

## 8. Explicitly out of scope (YAGNI)

- **No free-form `V(r)` expression entry** — the requirements doc defers safe
  expression parsing to a later phase; this is a *curated* preset library.
- **No async job path** — payload is scalar levels plus a ~256-point curve.
- **No `forceViz` in the URL** — presentational toggle, invalidates nothing.
- **No new fidelity tier** — every value is NUMERICAL (solver), EXACT (hydrogen /
  QHO / definitional markers / analytic curve), or the existing eV-conversion note.
- **No Tier-2 multi-electron screening** — `yukawa` and `coulombcore` are the
  single-particle *on-ramp* to that; the self-consistent screening phase is separate.
