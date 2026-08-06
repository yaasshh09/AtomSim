# Phase 4: What-If: Force Law (design)

**Date:** 2026-07-17
**Branch:** `phase4-force-law`
**Status:** approved, ready for planning

## 1. Purpose

Answer *why is hydrogen so degenerate?*, the accidental fact that a `2s` and a
`2p` electron sit at exactly the same energy, by **breaking it on purpose**. The
electrostatic force law `V(r) = -Z/r` is special: only the `1/r` shape makes the
energy depend on `n` alone. Bend the exponent away from 1 and the degeneracy
lifts, different `l` at the same radial index split apart.

This is a **What-If Lab counterfactual force law**: the user turns a single knob
`p` in `V(r) = -Z/rᵖ` and watches the level structure reorganize, always against
an **EXACT** hydrogen baseline so the contrast is honest. It reuses the existing
numerical radial solver (`numerics/radial_solver.py`), the seam the README names
for exactly this, and invents no new fidelity machinery.

## 2. Physics (the honest core)

### 2.1 The counterfactual potential: `NUMERICAL`

```
V(r) = -Z / rᵖ        p ∈ [0.5, 1.5],   p = 1 ⇒ real Coulomb
```

`Z` and the reduced-mass ratio `μ` come from the currently selected system (so H,
D, muonic H, He⁺, positronium, generic Z all scale correctly, exactly as the rest
of the engine does). The solver adds the centrifugal barrier `l(l+1)/(2μr²)`
internally (`radial_solver.py:47`), so `V(r)` here is the *bare* radial potential.

Levels come from `solve_radial_with_error(potential, l, mu_ratio=μ, n_states=4)`.
The result is the lowest **4** bound radial states at the chosen `l`, each a
`Quantity(unit="hartree", fidelity=NUMERICAL)`. The grid-halving error estimate is
attached per level, cheap here because the clamped range never approaches
fall-to-center.

**Why the range is clamped to `[0.5, 1.5]`.** As `p → 2` the `-Z/rᵖ` well competes
with the centrifugal `1/r²` term; past the critical coupling the atom "falls to the
center", bound states cease to exist, and a finite-difference solver returns
plausible-looking garbage eigenvalues **without failing**. Clamping to `[0.5, 1.5]`
keeps every returned state genuinely box-converged and physical, the honest choice
over a wider range plus failure-detection machinery.

### 2.2 The reference overlay: `EXACT`

Alongside the numerical levels, compute the closed-form hydrogen levels **for the
selected `l`**, reduced-mass-scaled exactly as `analytic/hydrogen.py` does:

```
E_n = -(Z² · μ) / (2 n²)   hartree,   for n ≥ l + 1
```

- **Gated to `n ≥ l+1`.** A given `l` has no state below `n = l+1`; the overlay
  must not draw phantom reference levels. For `l=1` the reference starts at `n=2`.
- Tier `EXACT`, the closed form, not a second numerical run. Using the analytic
  truth as the anchor is *more* honest than re-solving Coulomb numerically, because
  it keeps solver error out of the ground truth.

**Pairing convention.** The counterfactual's radial index `k` (0-based, node
count) pairs with hydrogen level `n = l + 1 + k`. Both lists carry `n_states`
entries, index-for-index, so at `p = 1` the `k`-th numerical level equals the
`k`-th reference level. (`2s = (l=0, k=1)`, `2p = (l=1, k=0)`, both `n=2`.)

**The asymmetry is disclosed, not hidden.** The response carries both provenances
distinctly (`NUMERICAL` counterfactual vs `EXACT` reference). Any visible gap is
physics **plus** solver error, and the badge must say which side is which.

**Built-in calibration.** At `p = 1` the numerical levels should land on the exact
ones to solver tolerance. That coincidence is a live, in-UI demonstration that the
solver error is small, and doubles as an automated honesty test (§6).

## 3. Server endpoint (synchronous GET)

Mirrors the other What-If GETs (`/api/constants`, `/api/classical`): synchronous,
because the payload is a handful of scalars from a partial tridiagonal eigensolve
(tens of ms even running the coarse+fine grids), not a large `float32` array that
would need the async job path.

```
GET /api/forcelaw?p=<0.5..1.5>&l=<int≥0>&system=<key>&n_states=<int>
```

Response (Pydantic model mapping `Quantity`/`Provenance` straight through, per
`server/schemas.py`):

```jsonc
{
  "p": 1.2,
  "l": 1,
  "system": { ... SystemModel ... },
  "counterfactual": [ { "energy": QuantityModel, "energy_ev": QuantityModel,
                        "radial_index": 0, "provenance": ProvenanceModel }, ... ],
  "reference":      [ { "energy": QuantityModel, "energy_ev": QuantityModel,
                        "n": 2, "provenance": ProvenanceModel }, ... ]
}
```

- eV conversion happens at the boundary and appends to the provenance `method`,
  matching the existing `_to_ev` convention.
- Validation: `422` if `p ∉ [0.5, 1.5]` or `l < 0`, following the same pattern as
  the constants endpoint's multiplier bounds.
- `system` resolved through the existing `_resolve_system` (case-sensitive,
  exact-match, a known constraint).

## 4. Frontend: a new `ForceLawView`

A **new view**, sibling to the existing five, deliberately *not* an extension of
`LevelsView`. `LevelsView` renders pure `EXACT` analytic levels; this view renders
a `NUMERICAL` counterfactual against an `EXACT` overlay. Keeping the tiers in
separate components keeps provenance boundaries clean and each component
single-purpose.

- **Controls:** a `p` slider (0.5-1.5, snapping to 1.0) and an `l` selector.
- **Diagram:** the counterfactual levels drawn against the hydrogen reference
  ghosted behind them, so the `l`-split is immediately visible; at `p=1` the two
  coincide.
- **Provenance:** a `Badge` per side (`NUMERICAL` counterfactual vs `EXACT`
  reference), reading the tier straight off each level's `provenance`. No
  visual-liberty entry is invented, the diagram plots energies faithfully on a
  labeled linear axis, so the data badges are the whole disclosure. (Inventing a
  liberty where none exists would itself violate the prime directive.)
- **Color/scale:** reuse the single color authority (`lib/luts.ts`) and existing
  level-diagram styling where applicable.

### 4.1 State & URL

- New store slice for `(p, l)`; `system` already exists.
- `p`, `l`, `system` join the `INVALIDATED` set (they change the physics);
  view/color toggles invalidate nothing, per the store invariant.
- Deep-linked: `?view=forcelaw&p=1.2&l=1&system=h`, round-trip tested, treated as a
  stable query-schema contract per `lib/urlState.ts`.

## 5. Module boundaries

| Unit | Purpose | Depends on |
|------|---------|-----------|
| `numerics/force_law.py` | Build `V(r) = -Z/rᵖ`, call solver, pair with EXACT reference | `radial_solver`, `analytic/hydrogen`, `systems` |
| `/api/forcelaw` handler | Validate, resolve system, serialize | `force_law`, `schemas` |
| `ForceLawView` + slice | Controls, diagram, badges, URL | api client, store, `liberties`, `luts` |

Each is understandable and testable without reading the others' internals.

## 6. Testing (the honesty checks)

New physics gets a validation test, not a smoke test:

1. **Identity case (calibration):** at `p=1`, `force_law` levels match
   `analytic/hydrogen.py` `E_n = -(Z²μ)/(2n²)` to solver tolerance, for `l=0,1`.
   This validates the entire numerical path against exact ground truth.
2. **Degeneracy breaking:** lowest `l=0` and `l=1` states that coincide at `p=1`
   split measurably at `p=0.8` and `p=1.2`.
3. **Direction of split:** the s-vs-p ordering flips sign across `p=1`, in the
   direction fixed by the level-ordering theorem (sign of the 3-D Laplacian
   `ΔV = V'' + (2/r)V' = Z·p·(1−p)·r^(−p−2)`):
   - `p < 1` (softer, `ΔV > 0`): `E_2s > E_2p`, the penetrating s state lies *above* p.
   - `p > 1` (harder, `ΔV < 0`): `E_2s < E_2p`, s lies *below* p (alkali ordering).
   Here `2s = (l=0, radial index 1)` and `2p = (l=1, radial index 0)`, both `n=2`.
4. **Reference gating:** the `EXACT` reference for `l` starts at `n = l+1`, never
   below.
5. **Endpoint:** `422` on out-of-range `p` and negative `l`; both provenance tiers
   present and distinct in the response.
6. **URL round-trip:** `(view, p, l, system)` survives encode→decode.

## 7. Explicitly out of scope (YAGNI)

- No async job path, the payload is scalar-small (§3).
- No wider `p` range or fall-to-center visualization, clamped and safe (§2.1).
- No coupling to the 5-multiplier constants lab, that rescales Coulomb without
  changing its `1/r` shape, so it would *not* break `l`-degeneracy and would make
  the `l` selector pointless.
- No new fidelity tier, this is `solve_radial` with one potential plus the
  existing `EXACT` hydrogen formula.
