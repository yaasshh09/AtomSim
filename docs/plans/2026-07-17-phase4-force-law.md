# What-If: Force Law Implementation Plan

**Goal:** Let the user bend the electrostatic force law to `V(r) = -Z/rᵖ` and watch hydrogen's accidental l-degeneracy break, shown as `NUMERICAL` levels against an `EXACT` hydrogen baseline.

**Architecture:** A thin engine module drives the existing numerical radial solver with a power-law potential and pairs each level with the exact closed-form hydrogen level it maps to. A synchronous GET endpoint serializes both sides with distinct provenance. A new `ForceLawView` (p slider + l selector + level diagram) renders the contrast, deep-linked in the URL.

**Tech Stack:** Python 3.12 (numpy, scipy, FastAPI, Pydantic, pytest), TypeScript/React/Zustand, Vitest, d3-scale.

## Global Constraints

- Engine math is in **Hartree atomic units**; eV conversions happen only at the server boundary and append to the provenance `method` (reuse the existing pattern).
- Every physical value crossing a boundary is a `Quantity` carrying `Provenance` with its `Fidelity` tier. Counterfactual levels are `NUMERICAL`; reference levels are `EXACT`. Never emit a bare float across a boundary.
- `p` is clamped to **[0.5, 1.5]** everywhere (engine raises `ValueError`, endpoint returns `422`).
- Pairing convention: counterfactual radial index `k` (0-based node count) ↔ hydrogen level `n = l + 1 + k`. Both lists have `n_states` entries, index-for-index.
- `l` is the orbital angular-momentum quantum number (ruff E741 is ignored project-wide, keep the name `l`).
- New physics gets a validation test against analytic ground truth, not a smoke test.
- Run backend tests with the `atomsim` conda env active (Windows needs its BLAS/LAPACK DLLs on `PATH`, or numpy's eigensolver aborts).
- Frontend: rebuild (`npm run build`) after any `web/src` change, `atomsim serve` only mounts `web/dist`.

## File Structure

| File | Responsibility | New/Modified |
|------|----------------|--------------|
| `src/atomsim/numerics/force_law.py` | Build `V(r)=-Z/rᵖ`, solve, pair with EXACT reference | Create |
| `tests/test_force_law.py` | Engine validation (identity, degeneracy, ordering, gating) | Create |
| `src/atomsim/server/schemas.py` | `ForceLawLevelModel`, `ReferenceLevelModel`, `ForceLawModel`, `_ev` | Modify |
| `src/atomsim/server/app.py` | `GET /api/forcelaw` handler | Modify |
| `tests/test_server.py` | Endpoint tests | Modify |
| `web/src/api/types.ts` | `ForceLawLevel`, `ReferenceLevel`, `ForceLawResult` types | Modify |
| `web/src/api/client.ts` | `getForceLaw()` | Modify |
| `web/src/state/store.ts` | force-law slice (`forceP`, `forceL`, data, `loadForceLaw`) | Modify |
| `web/src/lib/urlState.ts` | `forcelaw` view + `p`/`fl` params | Modify |
| `web/src/lib/urlState.test.ts` | round-trip test for new params | Modify |
| `web/src/components/ForceLawView.tsx` | slider, selector, level diagram, badges | Create |
| `web/src/components/Controls.tsx` | register the `forcelaw` view option | Modify |
| `web/src/App.tsx` | render `ForceLawView` when `view === "forcelaw"` | Modify |
| `web/src/index.css` | force-law diagram styles | Modify |

---

### Task 1: Force-law engine

**Files:**
- Create: `src/atomsim/numerics/force_law.py`
- Test: `tests/test_force_law.py`

**Interfaces:**
- Consumes: `solve_radial_with_error(potential, l, mu_ratio, r_max, n_points, n_states)` from `numerics/radial_solver.py` (returns `RadialSolution` with `.energies: tuple[Quantity, ...]`); `energy(n, Z, mu_ratio) -> Quantity` from `analytic/hydrogen.py`; `System`, `get_system` from `systems.py`.
- Produces: `force_law_levels(p: float, l: int, system: str | System = "h", n_states: int = 4) -> ForceLawResult`; dataclasses `ForceLawLevel(radial_index: int, energy: Quantity)`, `ReferenceLevel(n: int, energy: Quantity)`, `ForceLawResult(p, l, z, system_key, counterfactual: tuple[ForceLawLevel, ...], reference: tuple[ReferenceLevel, ...])`; module constants `P_MIN = 0.5`, `P_MAX = 1.5`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_force_law.py
import pytest

from atomsim.analytic.hydrogen import energy as hydrogen_energy
from atomsim.numerics.force_law import P_MAX, P_MIN, force_law_levels

def test_identity_case_matches_exact_hydrogen():
    # At p=1 the numerical levels reproduce the exact Bohr formula.
    for l in (0, 1):
        res = force_law_levels(p=1.0, l=l, system="h", n_states=2)
        assert len(res.counterfactual) == len(res.reference) == 2
        for lvl, ref in zip(res.counterfactual, res.reference):
            exact = hydrogen_energy(ref.n, Z=1, mu_ratio=1.0).value
            assert lvl.energy.value == pytest.approx(exact, rel=2e-3)

def test_provenance_tiers_are_distinct():
    res = force_law_levels(p=1.2, l=0, system="h", n_states=2)
    assert all(x.energy.provenance.fidelity.value == "numerical" for x in res.counterfactual)
    assert all(x.energy.provenance.fidelity.value == "exact" for x in res.reference)
    # numerical levels carry a grid-halving error estimate
    assert all(x.energy.provenance.error_estimate is not None for x in res.counterfactual)

def test_reference_gated_to_n_ge_l_plus_1():
    res = force_law_levels(p=1.0, l=1, system="h", n_states=3)
    assert [r.n for r in res.reference] == [2, 3, 4]
    assert [c.radial_index for c in res.counterfactual] == [0, 1, 2]

def _two_s_minus_two_p(p: float) -> float:
    # 2s = (l=0, radial index 1); 2p = (l=1, radial index 0); both n=2.
    e_2s = force_law_levels(p=p, l=0, system="h", n_states=2).counterfactual[1].energy.value
    e_2p = force_law_levels(p=p, l=1, system="h", n_states=1).counterfactual[0].energy.value
    return e_2s - e_2p

def test_degeneracy_intact_at_coulomb():
    assert _two_s_minus_two_p(1.0) == pytest.approx(0.0, abs=3e-3)

def test_degeneracy_breaks_with_correct_ordering():
    # p<1 (softer, DeltaV>0): s above p  -> E_2s > E_2p  -> positive
    # p>1 (harder, DeltaV<0): s below p  -> E_2s < E_2p  -> negative
    soft = _two_s_minus_two_p(0.8)
    hard = _two_s_minus_two_p(1.2)
    assert soft > 3e-3
    assert hard < -3e-3

def test_p_out_of_range_rejected():
    with pytest.raises(ValueError):
        force_law_levels(p=P_MAX + 0.1, l=0)
    with pytest.raises(ValueError):
        force_law_levels(p=P_MIN - 0.1, l=0)

def test_negative_l_rejected():
    with pytest.raises(ValueError):
        force_law_levels(p=1.0, l=-1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_force_law.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.numerics.force_law'`

- [ ] **Step 3: Write the implementation**

```python
# src/atomsim/numerics/force_law.py
"""What-If force law: energy levels under a counterfactual V(r) = -Z / r^p.

Only the 1/r Coulomb shape makes hydrogen's energy depend on n alone; bending the
exponent p away from 1 lifts the accidental l-degeneracy. This module drives the
numerical radial solver with the power-law potential (NUMERICAL) and pairs each
level with the EXACT closed-form hydrogen level it maps to, so the contrast is an
honest EXACT-vs-COUNTERFACTUAL one. p is clamped to [0.5, 1.5], comfortably inside
the fall-to-center threshold (p -> 2), so every returned state is box-converged.
"""

from dataclasses import dataclass

import numpy as np

from atomsim.analytic.hydrogen import energy as hydrogen_energy
from atomsim.numerics.radial_solver import solve_radial_with_error
from atomsim.provenance import Quantity
from atomsim.systems import System, get_system

P_MIN = 0.5
P_MAX = 1.5

@dataclass(frozen=True)
class ForceLawLevel:
    radial_index: int   # 0-based node count k
    energy: Quantity    # NUMERICAL, hartree, carries a grid-halving error estimate

@dataclass(frozen=True)
class ReferenceLevel:
    n: int
    energy: Quantity    # EXACT closed-form hydrogen level, hartree

@dataclass(frozen=True)
class ForceLawResult:
    p: float
    l: int
    z: int
    system_key: str
    counterfactual: tuple[ForceLawLevel, ...]
    reference: tuple[ReferenceLevel, ...]

def force_law_levels(
    p: float, l: int, system: str | System = "h", n_states: int = 4
) -> ForceLawResult:
    if not P_MIN <= p <= P_MAX:
        raise ValueError(f"p must be in [{P_MIN}, {P_MAX}], got {p}")
    if l < 0:
        raise ValueError(f"orbital quantum number l must be >= 0, got {l}")
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")

    # Accept a bare key (registered systems) or an already-resolved System, so the
    # server can hand us generic hydrogen-like ions (z{N}) that get_system omits.
    sys = system if isinstance(system, System) else get_system(system)
    z = sys.Z
    mu = sys.mu_ratio.value

    def potential(r: np.ndarray) -> np.ndarray:
        return -z / r**p

    sol = solve_radial_with_error(potential, l=l, mu_ratio=mu, n_states=n_states)
    counterfactual = tuple(
        ForceLawLevel(radial_index=k, energy=sol.energies[k]) for k in range(n_states)
    )
    # radial index k <-> hydrogen level n = l + 1 + k (since n_r = n - l - 1 = k)
    reference = tuple(
        ReferenceLevel(n=l + 1 + k, energy=hydrogen_energy(l + 1 + k, Z=z, mu_ratio=mu))
        for k in range(n_states)
    )
    return ForceLawResult(
        p=p,
        l=l,
        z=z,
        system_key=sys.key,
        counterfactual=counterfactual,
        reference=reference,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_force_law.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Lint**

Run: `ruff check src/atomsim/numerics/force_law.py tests/test_force_law.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/atomsim/numerics/force_law.py tests/test_force_law.py
git commit -m "Add the force-law engine (power-law potential vs exact hydrogen)"
```

---

### Task 2: `/api/forcelaw` endpoint

**Files:**
- Modify: `src/atomsim/server/schemas.py`
- Modify: `src/atomsim/server/app.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `force_law_levels`, `ForceLawResult`, `ForceLawLevel`, `ReferenceLevel`, `P_MIN`, `P_MAX` from Task 1; `_resolve_system(key) -> System` and the `create_app()` nested-endpoint pattern in `app.py`; `HARTREE_EV` from `constants.py`.
- Produces: `GET /api/forcelaw?p&l&system&n_states` returning `ForceLawModel { p, l, z, system: SystemModel, counterfactual: [{radial_index, energy, energy_ev}], reference: [{n, energy, energy_ev}] }`.

- [ ] **Step 1: Write the failing endpoint tests**

```python
# tests/test_server.py  (append; uses the existing `client` fixture)
def test_forcelaw_identity_matches_reference(client):
    r = client.get("/api/forcelaw?p=1.0&l=0&system=h&n_states=2")
    assert r.status_code == 200
    body = r.json()
    assert body["p"] == 1.0 and body["l"] == 0 and body["z"] == 1
    cf, ref = body["counterfactual"], body["reference"]
    assert len(cf) == len(ref) == 2
    assert cf[0]["radial_index"] == 0 and ref[0]["n"] == 1
    # at p=1 numerical ~ exact
    assert cf[0]["energy"]["value"] == pytest.approx(ref[0]["energy"]["value"], rel=2e-3)
    # provenance survives to the browser, tiers distinct
    assert cf[0]["energy"]["provenance"]["fidelity"] == "numerical"
    assert ref[0]["energy"]["provenance"]["fidelity"] == "exact"
    # eV conversion present at the boundary
    assert cf[0]["energy_ev"]["unit"] == "eV"

def test_forcelaw_reference_gated_to_n_ge_l_plus_1(client):
    body = client.get("/api/forcelaw?p=1.1&l=1&n_states=3").json()
    assert [r["n"] for r in body["reference"]] == [2, 3, 4]

def test_forcelaw_rejects_out_of_range_p(client):
    assert client.get("/api/forcelaw?p=2.5&l=0").status_code == 422
    assert client.get("/api/forcelaw?p=0.1&l=0").status_code == 422

def test_forcelaw_rejects_negative_l(client):
    assert client.get("/api/forcelaw?p=1.0&l=-1").status_code == 422
```

(If `import pytest` is not already at the top of `tests/test_server.py`, add it.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_server.py -k forcelaw -v`
Expected: FAIL, 404 responses (route not defined) / `ImportError` for `ForceLawModel`

- [ ] **Step 3: Add the schema models**

The eV conversion reuses the existing `app.py._to_ev` (DRY, do NOT add a second copy in `schemas.py`), so these models are plain data holders built by the endpoint, exactly like `FineLevelModel`/`LevelsResponse` already are. In `src/atomsim/server/schemas.py`, add, after the `ClassicalGhostModel` block:

```python
class ForceLawLevelModel(BaseModel):
    radial_index: int
    energy: QuantityModel
    energy_ev: QuantityModel

class ReferenceLevelModel(BaseModel):
    n: int
    energy: QuantityModel
    energy_ev: QuantityModel

class ForceLawModel(BaseModel):
    p: float
    l: int
    z: int
    system: SystemModel
    counterfactual: list[ForceLawLevelModel]
    reference: list[ReferenceLevelModel]
```

No new imports are needed in `schemas.py` (`BaseModel`, `QuantityModel`, `SystemModel` are already defined there).

- [ ] **Step 4: Add the endpoint**

In `src/atomsim/server/app.py`, add to the imports:

```python
from atomsim.numerics.force_law import P_MAX, P_MIN, force_law_levels
```

and add `ForceLawLevelModel`, `ForceLawModel`, `ReferenceLevelModel` to the existing `from atomsim.server.schemas import (...)` list (which already imports `QuantityModel` and `SystemModel`).

Then, immediately after the `classical_endpoint` function (around line 374), add the handler. It reuses the module-level `_to_ev` helper already defined in `app.py`, mirroring how the levels endpoint builds its eV quantities:

```python
    @app.get("/api/forcelaw", response_model=ForceLawModel)
    def forcelaw_endpoint(
        p: float = 1.0, l: int = 0, system: str = "h", n_states: int = 4
    ) -> ForceLawModel:
        if not P_MIN <= p <= P_MAX:
            raise HTTPException(
                status_code=422, detail=f"p must be in [{P_MIN}, {P_MAX}], got {p}"
            )
        if l < 0:
            raise HTTPException(status_code=422, detail=f"l must be >= 0, got {l}")
        if not 1 <= n_states <= 8:
            raise HTTPException(
                status_code=422, detail=f"n_states must be in [1, 8], got {n_states}"
            )
        sys_ = _resolve_system(system)
        result = force_law_levels(p=p, l=l, system=sys_, n_states=n_states)
        return ForceLawModel(
            p=result.p,
            l=result.l,
            z=result.z,
            system=SystemModel.from_system(sys_),
            counterfactual=[
                ForceLawLevelModel(
                    radial_index=c.radial_index,
                    energy=QuantityModel.from_quantity(c.energy),
                    energy_ev=QuantityModel.from_quantity(_to_ev(c.energy)),
                )
                for c in result.counterfactual
            ],
            reference=[
                ReferenceLevelModel(
                    n=r.n,
                    energy=QuantityModel.from_quantity(r.energy),
                    energy_ev=QuantityModel.from_quantity(_to_ev(r.energy)),
                )
                for r in result.reference
            ],
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_server.py -k forcelaw -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Full suite + lint**

Run: `pytest && ruff check .`
Expected: all pass, `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add src/atomsim/server/schemas.py src/atomsim/server/app.py tests/test_server.py
git commit -m "Add the /api/forcelaw endpoint"
```

---

### Task 3: Frontend types + API client

**Files:**
- Modify: `web/src/api/types.ts`
- Modify: `web/src/api/client.ts`

**Interfaces:**
- Consumes: existing `Quantity` type in `types.ts`; `getJson` helper in `client.ts`.
- Produces: TS types `ForceLawLevel`, `ReferenceLevel`, `ForceLawResult`; `getForceLaw(system, p, l, nStates?) => Promise<ForceLawResult>`.

- [ ] **Step 1: Add the types**

Append to `web/src/api/types.ts` (mirrors the Python `ForceLawModel` shape exactly):

```typescript
export interface ForceLawLevel {
  radial_index: number;
  energy: Quantity;
  energy_ev: Quantity;
}

export interface ReferenceLevel {
  n: number;
  energy: Quantity;
  energy_ev: Quantity;
}

export interface ForceLawResult {
  p: number;
  l: number;
  z: number;
  system: SystemInfo;
  counterfactual: ForceLawLevel[];
  reference: ReferenceLevel[];
}
```

(`SystemInfo` and `Quantity` already exist in this file; confirm they are exported.)

- [ ] **Step 2: Add the client call**

In `web/src/api/client.ts`, add `ForceLawResult` to the type import from `./types`, then add after `getClassical`:

```typescript
export function getForceLaw(
  system: string,
  p: number,
  l: number,
  nStates = 4,
): Promise<ForceLawResult> {
  return getJson(
    `/api/forcelaw?system=${encodeURIComponent(system)}&p=${p}&l=${l}&n_states=${nStates}`,
  );
}
```

- [ ] **Step 3: Typecheck**

Run: `cd web && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/api/types.ts web/src/api/client.ts
git commit -m "Add force-law frontend types and API client"
```

---

### Task 4: Store slice

**Files:**
- Modify: `web/src/state/store.ts`

**Interfaces:**
- Consumes: `getForceLaw` (Task 3), `ForceLawResult` type.
- Produces: store fields `forceP: number`, `forceL: number`, `forceLaw: ForceLawResult | null`, `forceStatus: SampleStatus`; actions `setForceP(p)`, `setForceL(l)`, `loadForceLaw()`. `ViewMode` gains `"forcelaw"`.

- [ ] **Step 1: Write the failing store test**

```typescript
// web/src/state/store.test.ts  (create if absent; else append)
import { describe, expect, it } from "vitest";
import { useAppStore } from "./store";

describe("force-law slice", () => {
  it("changing p or l clears stale force-law data", () => {
    useAppStore.setState({ forceLaw: { p: 1, l: 0 } as never, forceStatus: "ready" });
    useAppStore.getState().setForceP(1.2);
    expect(useAppStore.getState().forceLaw).toBeNull();
    expect(useAppStore.getState().forceStatus).toBe("idle");

    useAppStore.setState({ forceLaw: { p: 1 } as never, forceStatus: "ready" });
    useAppStore.getState().setForceL(1);
    expect(useAppStore.getState().forceLaw).toBeNull();
  });

  it("setForceP clamps into [0.5, 1.5]", () => {
    useAppStore.getState().setForceP(9);
    expect(useAppStore.getState().forceP).toBe(1.5);
    useAppStore.getState().setForceP(0);
    expect(useAppStore.getState().forceP).toBe(0.5);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/state/store.test.ts`
Expected: FAIL, `setForceP is not a function`.

- [ ] **Step 3: Extend the store**

In `web/src/state/store.ts`:

1. Add `"forcelaw"` to the `ViewMode` union:

```typescript
export type ViewMode = "cloud" | "plane" | "radial" | "levels" | "spectrum" | "whatif" | "forcelaw";
```

2. Add the import:

```typescript
import type { ForceLawResult } from "../api/types";
```

3. Add fields to the `AppState` interface (near the `ghost`/`classicalGhost` fields):

```typescript
  forceP: number;
  forceL: number;
  forceLaw: ForceLawResult | null;
  forceStatus: SampleStatus;
  setForceP: (p: number) => void;
  setForceL: (l: number) => void;
  loadForceLaw: () => Promise<void>;
```

4. Add initial values (near `ghost: false`):

```typescript
  forceP: 1.0,
  forceL: 0,
  forceLaw: null,
  forceStatus: "idle",
```

5. Add the actions (near `loadClassical`). The force-law slice is its own axis (`forceP`, `forceL`), independent of the main `(n,l,m,system)` physics, so it is never in `INVALIDATED`; changing `forceP`/`forceL` clears only the force-law data:

```typescript
  setForceP: (p) =>
    set({
      forceP: Math.min(Math.max(p, 0.5), 1.5),
      forceLaw: null,
      forceStatus: "idle",
    }),
  setForceL: (l) =>
    set({ forceL: Math.max(0, Math.round(l)), forceLaw: null, forceStatus: "idle" }),
  loadForceLaw: async () => {
    const { forceP, forceL, system } = get();
    set({ forceStatus: "sampling", error: null });
    try {
      const forceLaw = await client.getForceLaw(system, forceP, forceL);
      set({ forceLaw, forceStatus: "ready" });
    } catch (err) {
      set({ forceStatus: "error", error: err instanceof Error ? err.message : String(err) });
    }
  },
```

6. Changing the **system** must also drop stale force-law data (it depends on Z/μ). Update `setSystem` to clear it:

```typescript
  setSystem: (system) =>
    set({
      system,
      ...INVALIDATED,
      classicalGhost: null,
      classicalStatus: "idle",
      forceLaw: null,
      forceStatus: "idle",
    }),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run src/state/store.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/state/store.ts web/src/state/store.test.ts
git commit -m "Add the force-law store slice"
```

---

### Task 5: URL deep-linking

**Files:**
- Modify: `web/src/lib/urlState.ts`
- Test: `web/src/lib/urlState.test.ts`

**Interfaces:**
- Consumes: `UrlState`, `parseAppUrl`, `serializeAppUrl`; `ViewMode` now includes `"forcelaw"`.
- Produces: URL params `p` (force-law exponent) and `fl` (force-law l); `forcelaw` is a valid `view`. `forceP`/`forceL` join `UrlState`.

Note on naming: the force-law l uses the param `fl`, NOT `l`, `l` is already the main orbital quantum number. The force-law exponent uses `p`.

- [ ] **Step 1: Write the failing round-trip test**

```typescript
// web/src/lib/urlState.test.ts  (append)
import { describe, expect, it } from "vitest";
import { parseAppUrl, serializeAppUrl, URL_DEFAULTS } from "./urlState";

describe("force-law url state", () => {
  it("round-trips forcelaw view with p and fl", () => {
    const state = { ...URL_DEFAULTS, view: "forcelaw" as const, forceP: 1.2, forceL: 1 };
    const round = { ...URL_DEFAULTS, ...parseAppUrl(serializeAppUrl(state)) };
    expect(round.view).toBe("forcelaw");
    expect(round.forceP).toBeCloseTo(1.2);
    expect(round.forceL).toBe(1);
  });

  it("drops an out-of-range p and a negative fl", () => {
    const out = parseAppUrl("?p=9&fl=-2");
    expect(out.forceP).toBeUndefined();
    expect(out.forceL).toBeUndefined();
  });

  it("omits p and fl when at defaults", () => {
    expect(serializeAppUrl({ ...URL_DEFAULTS })).not.toContain("fl=");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/lib/urlState.test.ts`
Expected: FAIL, `forceP` undefined after round-trip.

- [ ] **Step 3: Extend urlState**

In `web/src/lib/urlState.ts`:

1. Add to the `UrlState` interface and `URL_DEFAULTS`:

```typescript
  forceP: number;
  forceL: number;
```
```typescript
  forceP: 1.0,
  forceL: 0,
```

2. Add `"forcelaw"` to the `VIEWS` array:

```typescript
const VIEWS: ViewMode[] = ["cloud", "plane", "radial", "levels", "spectrum", "whatif", "forcelaw"];
```

3. In `parseAppUrl`, before `return out;`, add:

```typescript
  const fp = pickFloat(q.get("p"));
  if (fp !== undefined && fp >= 0.5 && fp <= 1.5) out.forceP = fp;

  const fl = pickInt(q.get("fl"));
  if (fl !== undefined && fl >= 0) out.forceL = fl;
```

4. In `serializeAppUrl`, before the final `const s = q.toString();`, add:

```typescript
  if (Math.abs(state.forceP - URL_DEFAULTS.forceP) > 1e-9) q.set("p", String(state.forceP));
  if (state.forceL !== URL_DEFAULTS.forceL) q.set("fl", String(state.forceL));
```

- [ ] **Step 4: Wire URL <-> store in `web/src/main.tsx`**

The hydration side is already automatic: `main.tsx` calls `useAppStore.setState(parseAppUrl(...))`, and because the `UrlState` keys `forceP`/`forceL` match the store field names, they apply with no extra code. Only the serialize side needs the two fields. In the `serializeAppUrl({ ... })` object inside `useAppStore.subscribe(...)`, add:

```typescript
    forceP: s.forceP,
    forceL: s.forceL,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd web && npx vitest run src/lib/urlState.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add web/src/lib/urlState.ts web/src/lib/urlState.test.ts web/src/main.tsx
git commit -m "Deep-link the force-law exponent and l in the URL"
```

---

### Task 6: ForceLawView + registration

**Files:**
- Create: `web/src/components/ForceLawView.tsx`
- Modify: `web/src/components/Controls.tsx`
- Modify: `web/src/App.tsx`
- Modify: `web/src/index.css`

**Interfaces:**
- Consumes: store fields/actions from Task 4 (`forceP`, `forceL`, `forceLaw`, `forceStatus`, `error`, `setForceP`, `setForceL`, `loadForceLaw`); `Badge` (`{ provenance }`); `scaleLinear` from `d3-scale`.
- Produces: `ForceLawView` React component; a `{ value: "forcelaw", label: "What-If: force law" }` option in `Controls`.

- [ ] **Step 1: Create the view component**

```tsx
// web/src/components/ForceLawView.tsx
import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 640;
const H = 460;
const PAD = { top: 32, right: 24, bottom: 40, left: 64 };
const L_CHOICES = [0, 1, 2, 3];

export function ForceLawView() {
  const { forceP, forceL, forceLaw, forceStatus, error, setForceP, setForceL, loadForceLaw } =
    useAppStore();

  useEffect(() => {
    if (forceLaw === null && forceStatus === "idle") void loadForceLaw();
  }, [forceLaw, forceStatus, loadForceLaw]);

  const cfProv = forceLaw?.counterfactual[0]?.energy.provenance ?? null;
  const refProv = forceLaw?.reference[0]?.energy.provenance ?? null;

  // Energy axis spans both sides (eV). Guard against an empty payload.
  const evs =
    forceLaw === null
      ? []
      : [
          ...forceLaw.counterfactual.map((x) => x.energy_ev.value),
          ...forceLaw.reference.map((x) => x.energy_ev.value),
        ];
  const emin = evs.length ? Math.min(...evs) : -14;
  const emax = evs.length ? Math.max(...evs, -0.1) : 0;
  const y = scaleLinear([emin, emax], [H - PAD.bottom, PAD.top]);

  return (
    <div className="forcelaw">
      <div className="whatif-controls">
        <label>
          Force-law exponent p = {forceP.toFixed(2)}
          <input
            type="range"
            min={0.5}
            max={1.5}
            step={0.05}
            value={forceP}
            onChange={(e) => setForceP(Number(e.target.value))}
          />
          <span className="hint-block">V(r) = −Z / r^p, p = 1 is real hydrogen</span>
        </label>
        <label>
          Orbital l
          <select value={forceL} onChange={(e) => setForceL(Number(e.target.value))}>
            {L_CHOICES.map((l) => (
              <option key={l} value={l}>
                {l} ({"spdf"[l]})
              </option>
            ))}
          </select>
        </label>
      </div>

      {forceStatus === "error" && <p className="error">{error}</p>}
      {forceLaw === null ? (
        <p className="hint-block">solving force law…</p>
      ) : (
        <>
          <div className="forcelaw-legend">
            {cfProv && (
              <span>
                counterfactual V=−Z/r^{forceP.toFixed(2)} <Badge provenance={cfProv} />
              </span>
            )}
            {refProv && (
              <span>
                real hydrogen (reference) <Badge provenance={refProv} />
              </span>
            )}
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} className="forcelaw-svg" role="img"
               aria-label="energy levels under the counterfactual force law versus real hydrogen">
            {/* reference (exact hydrogen), ghosted, left column */}
            {forceLaw.reference.map((r) => (
              <g key={`ref-${r.n}`}>
                <line
                  x1={PAD.left}
                  x2={W / 2 - 8}
                  y1={y(r.energy_ev.value)}
                  y2={y(r.energy_ev.value)}
                  className="forcelaw-ref"
                />
                <text x={PAD.left} y={y(r.energy_ev.value) - 4} className="forcelaw-label">
                  n={r.n}
                </text>
              </g>
            ))}
            {/* counterfactual (numerical), solid, right column */}
            {forceLaw.counterfactual.map((c) => (
              <g key={`cf-${c.radial_index}`}>
                <line
                  x1={W / 2 + 8}
                  x2={W - PAD.right}
                  y1={y(c.energy_ev.value)}
                  y2={y(c.energy_ev.value)}
                  className="forcelaw-cf"
                />
                <text x={W - PAD.right} y={y(c.energy_ev.value) - 4} textAnchor="end"
                      className="forcelaw-label">
                  {c.energy_ev.value.toFixed(2)} eV
                </text>
              </g>
            ))}
            <text x={W / 4} y={PAD.top - 12} textAnchor="middle" className="forcelaw-col">
              real hydrogen
            </text>
            <text x={(3 * W) / 4} y={PAD.top - 12} textAnchor="middle" className="forcelaw-col">
              V = −Z / r^{forceP.toFixed(2)}
            </text>
          </svg>
          <p className="hint-block">
            At p = 1 the numerical levels land on the exact hydrogen levels (solver
            calibration). Away from 1, states of the same n but different l split, the
            accidental Coulomb degeneracy is gone.
          </p>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Register the view**

In `web/src/components/Controls.tsx`, append to `VIEW_OPTIONS`:

```typescript
  { value: "forcelaw", label: "What-If: force law" },
```

In `web/src/App.tsx`, add the import and the render line:

```tsx
import { ForceLawView } from "./components/ForceLawView";
```
```tsx
        {view === "forcelaw" && <ForceLawView />}
```

- [ ] **Step 3: Add styles**

Append to `web/src/index.css`:

```css
.forcelaw-svg { width: 100%; max-width: 640px; height: auto; }
.forcelaw-ref { stroke: #4ade80; stroke-width: 2; stroke-dasharray: 4 3; opacity: 0.55; }
.forcelaw-cf { stroke: #60a5fa; stroke-width: 2.5; }
.forcelaw-label { fill: #cbd5e1; font-size: 12px; }
.forcelaw-col { fill: #e2e8f0; font-size: 13px; font-weight: 600; }
.forcelaw-legend { display: flex; gap: 1.5rem; margin: 0.5rem 0; align-items: center; }
```

- [ ] **Step 4: Build and typecheck**

Run: `cd web && npm run build`
Expected: `tsc --noEmit` clean, vite build writes `web/dist`.

- [ ] **Step 5: Manual verification**

Run `atomsim serve --no-browser`, open `http://127.0.0.1:8000/?view=forcelaw&p=1.3&fl=1`, confirm:
- the view loads with p=1.30 and l=1 (`p` state, `2p` split visible);
- dragging p to 1.00 collapses the two columns onto each other;
- both badges render (NUMERICAL and EXACT);
- the URL updates as you drag.

- [ ] **Step 6: Full frontend test run**

Run: `cd web && npm test`
Expected: all vitest files pass.

- [ ] **Step 7: Commit**

```bash
git add web/src/components/ForceLawView.tsx web/src/components/Controls.tsx web/src/App.tsx web/src/index.css
git commit -m "Add the What-If force-law view"
```

---

## Self-Review notes

- **Spec coverage:** §2.1 counterfactual → Task 1; §2.2 reference + pairing + gating → Task 1 (tests) + Task 2 (eV, provenance); §3 endpoint → Task 2; §4 view/controls/badges → Task 6; §4.1 state+URL → Tasks 4-5; §6 tests → Tasks 1-2 + 5 (round-trip). All covered.
- **Direction of split:** Task 1 `test_degeneracy_breaks_with_correct_ordering` asserts `soft > 0` (p<1 → E₂ₛ>E₂ₚ) and `hard < 0` (p>1 → E₂ₛ<E₂ₚ), matching the corrected level-ordering derivation in the spec.
- **Naming:** force-law l is `forceL` in the store and `fl` in the URL, deliberately distinct from the main `l`. Force-law exponent is `forceP`/`p`.
- **Type consistency:** `getForceLaw` (Task 3) ↔ `loadForceLaw` (Task 4) ↔ `ForceLawView` (Task 6) all use `forceP`/`forceL`; Python `radial_index`/`n` field names match the TS `ForceLawLevel`/`ReferenceLevel` interfaces and the endpoint tests.
