# Constants Lab (α + Z) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the What-If Lab's first increment — a sixth "What-If" view that alters the fine-structure constant α (continuous) and nuclear charge Z (integer stepper) and shows the real-vs-altered energy-level diagram under a COUNTERFACTUAL banner.

**Architecture:** Thread an `alpha` parameter through the existing analytic fine-structure functions (default = the real CODATA `ALPHA`; the seam a future five-constant panel plugs into). Expose it on `/api/levels` plus a `z{N}` generic-system resolver. Add an isolated store slice + pure `lib/whatif.ts` logic + a `WhatIfView`, and make the lab state deep-linkable. No new engine physics — only α is now injectable.

**Tech Stack:** Python 3.12 (NumPy/SciPy, FastAPI, pytest), TypeScript/React 19 + react-three-fiber, Zustand, d3-scale, Vitest.

## Global Constraints

- **Provenance boundary rule:** every physical value crossing a module boundary is a `Quantity`/`Field`/provenance-carrying container. No physics computed in the frontend — the client only lays out and labels server numbers.
- **Altered-α fidelity:** when `alpha` ≠ real `ALPHA`, the value's provenance fidelity is `COUNTERFACTUAL` (headline); the α² Pauli-approximation assumptions and `(Zα)²` error stay disclosed in the provenance body. When α is real, fidelity stays `APPROXIMATION` (byte-for-byte as Phase 1).
- **α range:** `0 < alpha ≤ 0.5` at every boundary (server validation, URL clamp, slider max).
- **Commit messages contain NO AI attribution** — no `Co-Authored-By: Claude …`, no "Generated with" line. (Repo policy as of 2026-07-15.)
- **TDD:** write the failing test first for every engine/server/web-logic change. Views (`WhatIfView`) follow the repo pattern of no component unit test — verified by `tsc --noEmit`/build + QA.
- **Windows/Miniforge:** run Python from the `atomsim` conda env; set `PYTHONUTF8=1` if matplotlib/encoding warnings appear. Web commands run from `web/`.
- **Engine unit convention:** Hartree atomic units internally; eV/pm conversions already happen at the server boundary — do not add unit conversions in the client.

---

### Task 1: Engine — inject α into fine structure

**Files:**
- Modify: `src/atomsim/analytic/fine_structure.py`
- Test: `tests/test_fine_structure.py`

**Interfaces:**
- Consumes: `atomsim.constants.ALPHA` (real CODATA value), `Fidelity`, `Provenance`, `Quantity`.
- Produces: `fine_structure_shift(n, l, j, Z=1, mu_ratio=1.0, m_over_M=0.0, alpha=ALPHA) -> Quantity` and `level_energy(n, l, j, Z=1, mu_ratio=1.0, m_over_M=0.0, alpha=ALPHA) -> Quantity`. When `alpha` differs from `ALPHA`, both carry `Fidelity.COUNTERFACTUAL`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fine_structure.py`:

```python
import math

from atomsim.constants import ALPHA


def test_alpha_defaults_to_real_and_stays_approximation():
    default = fine_structure_shift(2, 1, 1.5)
    explicit = fine_structure_shift(2, 1, 1.5, alpha=ALPHA)
    assert default.value == explicit.value
    assert default.provenance.fidelity is Fidelity.APPROXIMATION


def test_altered_alpha_scales_shift_quadratically():
    base = fine_structure_shift(2, 1, 1.5).value
    doubled = fine_structure_shift(2, 1, 1.5, alpha=2 * ALPHA).value
    assert doubled / base == pytest.approx(4.0)


def test_altered_alpha_is_counterfactual_and_disclosed():
    q = fine_structure_shift(2, 1, 1.5, alpha=0.05)
    assert q.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert "altered" in q.provenance.method.lower()
    assert f"{ALPHA:g}" in q.provenance.method          # real value cited
    # Pauli-approximation error still quantified under the altered rule
    assert q.provenance.error_estimate == pytest.approx(abs(q.value) * ((1 * 0.05) ** 2 + 2 * 0.00116))


def test_level_energy_follows_altered_fidelity():
    assert level_energy(2, 1, 1.5, alpha=0.05).provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert level_energy(2, 1, 1.5).provenance.fidelity is Fidelity.APPROXIMATION
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_fine_structure.py -k "altered or defaults_to_real or follows_altered" -v`
Expected: FAIL — `fine_structure_shift() got an unexpected keyword argument 'alpha'`.

- [ ] **Step 3: Implement the α parameter**

In `src/atomsim/analytic/fine_structure.py`, add `import math` at the top (after the module docstring). Replace `fine_structure_shift` and `level_energy` with:

```python
def fine_structure_shift(
    n: int, l: int, j: float, Z: int = 1, mu_ratio: float = 1.0,
    m_over_M: float = 0.0, alpha: float = ALPHA,
) -> Quantity:
    """Fine-structure energy shift Delta E(n, l, j) in hartree.

    APPROXIMATION at the real alpha; COUNTERFACTUAL when alpha is altered
    (the What-If constants lab). The `alpha` argument is the seam a future
    FundamentalConstants.alpha will supply — no signature change needed then.
    """
    validate_quantum_numbers(n, l)
    validate_j(l, j)
    value = -(mu_ratio * Z**4 * alpha**2 / (2.0 * n**4)) * (n / (j + 0.5) - 0.75)
    error = abs(value) * ((Z * alpha) ** 2 + m_over_M + _G2)
    altered = not math.isclose(alpha, ALPHA, rel_tol=1e-12)
    method = (
        "combined Pauli fine structure "
        "dE = -(mu' Z^4 alpha^2 / 2 n^4)(n/(j+1/2) - 3/4)"
    )
    if altered:
        method += f"; altered fine-structure constant alpha = {alpha:g} (real {ALPHA:g})"
    return Quantity(
        value=value,
        unit="hartree",
        label=f"dE_fs {n},{l},j={j:g} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.COUNTERFACTUAL if altered else Fidelity.APPROXIMATION,
            method=method,
            assumptions=_FS_ASSUMPTIONS,
            error_estimate=error,
            refinement="exact Dirac hydrogen solution (planned Phase 3 flagship)",
        ),
    )


def level_energy(
    n: int, l: int, j: float, Z: int = 1, mu_ratio: float = 1.0,
    m_over_M: float = 0.0, alpha: float = ALPHA,
) -> Quantity:
    """Bohr energy plus fine-structure shift, in hartree.

    Fidelity follows the shift: APPROXIMATION at real alpha, COUNTERFACTUAL when altered.
    """
    bohr = energy(n, Z=Z, mu_ratio=mu_ratio)
    shift = fine_structure_shift(
        n, l, j, Z=Z, mu_ratio=mu_ratio, m_over_M=m_over_M, alpha=alpha
    )
    return Quantity(
        value=bohr.value + shift.value,
        unit="hartree",
        label=f"E {n},{l},j={j:g} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=shift.provenance.fidelity,
            method=f"{bohr.provenance.method} + {shift.provenance.method}",
            assumptions=_FS_ASSUMPTIONS,
            error_estimate=shift.provenance.error_estimate,
            refinement=shift.provenance.refinement,
        ),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_fine_structure.py -v`
Expected: PASS (all existing tests still green — the real-α path is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/atomsim/analytic/fine_structure.py tests/test_fine_structure.py
git commit -m "feat(engine): inject alpha into fine structure (counterfactual when altered)"
```

---

### Task 2: Server — α-aware `/api/levels` + `z{N}` systems

**Files:**
- Modify: `src/atomsim/server/app.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `level_energy`/`fine_structure_shift` with `alpha` (Task 1); `atomsim.systems.hydrogen_like`; `atomsim.constants.ALPHA`.
- Produces: `GET /api/levels?system=&n_max=&fine_structure=&alpha=` returning `LevelsResponse` with a new `alpha: float` field (the applied α). `system=z{N}` (1 ≤ N ≤ 10) resolves to a generic hydrogen-like ion.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
from atomsim.constants import ALPHA


def test_levels_default_alpha_is_real_and_approximation(client):
    body = client.get("/api/levels?fine_structure=true").json()
    assert body["alpha"] == pytest.approx(ALPHA)
    assert body["fine"][0]["shift"]["provenance"]["fidelity"] == "approximation"


def test_levels_altered_alpha_is_counterfactual(client):
    real = client.get("/api/levels?fine_structure=true").json()
    alt = client.get("/api/levels?fine_structure=true&alpha=0.05").json()
    assert alt["alpha"] == pytest.approx(0.05)
    assert alt["fine"][0]["shift"]["provenance"]["fidelity"] == "counterfactual"
    # bigger alpha -> deeper (more negative) fine shift
    assert abs(alt["fine"][0]["shift"]["value"]) > abs(real["fine"][0]["shift"]["value"])


def test_levels_generic_z_system_resolves(client):
    body = client.get("/api/levels?system=z3&fine_structure=true").json()
    assert body["system"]["z"] == 3


def test_levels_rejects_bad_alpha_and_z(client):
    assert client.get("/api/levels?alpha=0").status_code == 422
    assert client.get("/api/levels?alpha=0.6").status_code == 422
    assert client.get("/api/levels?system=z0").status_code == 422
    assert client.get("/api/levels?system=z99").status_code == 422
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_server.py -k "levels_default_alpha or altered_alpha or generic_z or bad_alpha" -v`
Expected: FAIL — response has no `alpha` key / `z3` unknown system (422) / bad inputs return 200.

- [ ] **Step 3: Wire α and `z{N}` into the server**

In `src/atomsim/server/app.py`:

3a. Add `import re` at the top (with the other stdlib imports).

3b. Extend the constants and systems imports:
```python
from atomsim.constants import ALPHA, BOHR_RADIUS_PM, HARTREE_EV
from atomsim.systems import get_system, hydrogen_like, list_systems
```

3c. Add the `alpha` field to `LevelsResponse` (near the other response models):
```python
class LevelsResponse(BaseModel):
    system: SystemModel
    n_max: int
    fine_structure: bool
    alpha: float
    gross: list[GrossLevelModel]
    fine: list[FineLevelModel] | None
```

3d. Replace the `_resolve_system` closure inside `create_app` so it recognizes `z{N}`:
```python
    _Z_KEY = re.compile(r"^z(\d+)$")

    def _resolve_system(key: str):
        zmatch = _Z_KEY.match(key)
        if zmatch:
            Z = int(zmatch.group(1))
            if not 1 <= Z <= 10:
                raise HTTPException(
                    status_code=422,
                    detail=f"generic hydrogen-like Z must be in [1, 10], got {Z}",
                )
            return hydrogen_like(Z)
        try:
            return get_system(key)
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
```

3e. Replace the `levels_endpoint` signature and body to thread `alpha`:
```python
    @app.get("/api/levels", response_model=LevelsResponse)
    def levels_endpoint(system: str = "h", n_max: int = 6,
                        fine_structure: bool = False,
                        alpha: float | None = None) -> LevelsResponse:
        if not 1 <= n_max <= 10:
            raise HTTPException(status_code=422, detail="n_max must be in [1, 10]")
        if alpha is not None and not 0.0 < alpha <= 0.5:
            raise HTTPException(status_code=422, detail="alpha must be in (0, 0.5]")
        sys_ = _resolve_system(system)
        mu = sys_.mu_ratio.value
        alpha_used = ALPHA if alpha is None else alpha
        gross = []
        for n in range(1, n_max + 1):
            e = energy(n, Z=sys_.Z, mu_ratio=mu)
            gross.append(GrossLevelModel(
                n=n, degeneracy=2 * n * n,
                energy=QuantityModel.from_quantity(e),
                energy_ev=QuantityModel.from_quantity(_to_ev(e)),
            ))
        fine = None
        if fine_structure:
            fine = []
            for n in range(1, n_max + 1):
                for l in range(n):
                    for j in ([0.5] if l == 0 else [l - 0.5, l + 0.5]):
                        le = level_energy(
                            n, l, j, Z=sys_.Z, mu_ratio=mu,
                            m_over_M=sys_.m_over_M, alpha=alpha_used,
                        )
                        sh = fine_structure_shift(
                            n, l, j, Z=sys_.Z, mu_ratio=mu,
                            m_over_M=sys_.m_over_M, alpha=alpha_used,
                        )
                        fine.append(FineLevelModel(
                            n=n, l=l, j=j,
                            energy=QuantityModel.from_quantity(le),
                            energy_ev=QuantityModel.from_quantity(_to_ev(le)),
                            shift=QuantityModel.from_quantity(sh),
                            shift_ev=QuantityModel.from_quantity(_to_ev(sh)),
                        ))
        return LevelsResponse(
            system=SystemModel.from_system(sys_), n_max=n_max,
            fine_structure=fine_structure, alpha=alpha_used, gross=gross, fine=fine,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_server.py -v`
Expected: PASS (existing `/api/levels` tests still green — `alpha` defaults to the real value).

- [ ] **Step 5: Commit**

```bash
git add src/atomsim/server/app.py tests/test_server.py
git commit -m "feat(server): alpha param + z{N} systems on /api/levels"
```

---

### Task 3: Web — pure lab logic `lib/whatif.ts`

**Files:**
- Create: `web/src/lib/whatif.ts`
- Test: `web/src/lib/whatif.test.ts`

**Interfaces:**
- Consumes: `FineLevel` from `../api/types`.
- Produces: `REAL_ALPHA`, `ALPHA_MAX`, `FINE_WARN_FRACTION` constants; `formatAlpha(alpha)`, `isAltered(alpha, realAlpha)`, `fineErrorFraction(fine)`, `isBeyondValidity(fine)`, `shellSplitting(fine, n)`.

- [ ] **Step 1: Write the failing tests**

Create `web/src/lib/whatif.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { FineLevel, Provenance, Quantity } from "../api/types";
import {
  FINE_WARN_FRACTION, REAL_ALPHA, fineErrorFraction, formatAlpha,
  isAltered, isBeyondValidity, shellSplitting,
} from "./whatif";

const prov: Provenance = {
  fidelity: "counterfactual", method: "", assumptions: [],
  error_estimate: null, refinement: null,
};
function q(value: number, error_estimate: number | null): Quantity {
  return { value, unit: "", label: "", provenance: { ...prov, error_estimate } };
}
function mkFine(n: number, l: number, j: number, shiftEv: number, err: number | null): FineLevel {
  return {
    n, l, j,
    energy: q(0, null), energy_ev: q(0, null),
    shift: q(shiftEv, err), shift_ev: q(shiftEv, err),
  };
}

describe("formatAlpha", () => {
  it("renders reciprocal form", () => {
    expect(formatAlpha(REAL_ALPHA)).toBe("1/137");
    expect(formatAlpha(0.02)).toBe("1/50");
    expect(formatAlpha(0)).toBe("0");
  });
});

describe("isAltered", () => {
  it("is false at the real value, true otherwise", () => {
    expect(isAltered(REAL_ALPHA, REAL_ALPHA)).toBe(false);
    expect(isAltered(0.02, REAL_ALPHA)).toBe(true);
  });
});

describe("fineErrorFraction / isBeyondValidity", () => {
  it("takes the max error_estimate/|shift| and thresholds it", () => {
    const fine = [mkFine(2, 1, 1.5, -1e-4, 2e-5), mkFine(2, 1, 0.5, -2e-4, 6e-5)];
    expect(fineErrorFraction(fine)).toBeCloseTo(0.3, 6);
    expect(isBeyondValidity(fine)).toBe(0.3 > FINE_WARN_FRACTION);
    expect(fineErrorFraction(null)).toBe(0);
  });
});

describe("shellSplitting", () => {
  it("returns the eV span of a shell's fine shifts", () => {
    const fine = [mkFine(2, 1, 1.5, 3e-6, null), mkFine(2, 1, 0.5, -1e-6, null)];
    expect(shellSplitting(fine, 2)).toBeCloseTo(4e-6, 12);
    expect(shellSplitting(fine, 3)).toBe(0);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd web && npx vitest run src/lib/whatif.test.ts`
Expected: FAIL — cannot resolve `./whatif`.

- [ ] **Step 3: Implement the pure logic**

Create `web/src/lib/whatif.ts`:

```ts
import type { FineLevel } from "../api/types";

/** Mirror of the engine's CODATA fine-structure constant (src/atomsim/constants.py
 *  ALPHA), used only to position the α control and its default. The COUNTERFACTUAL
 *  banner compares the server-echoed α values, never this constant. */
export const REAL_ALPHA = 0.0072973525643;

/** α slider/URL upper bound — matches the server's (0, 0.5] validation. */
export const ALPHA_MAX = 0.5;

/** Fine-structure fractional error past which the perturbative model is untrustworthy. */
export const FINE_WARN_FRACTION = 0.1;

/** Human form of α as a reciprocal, e.g. 0.0073 -> "1/137". */
export function formatAlpha(alpha: number): string {
  if (alpha <= 0) return "0";
  return `1/${Math.round(1 / alpha)}`;
}

/** True when α departs from the real (server-echoed) value. */
export function isAltered(alpha: number, realAlpha: number): boolean {
  return Math.abs(alpha - realAlpha) > 1e-12 * realAlpha;
}

/** Max fractional fine-structure error (error_estimate / |shift|) across levels; 0 if none. */
export function fineErrorFraction(fine: FineLevel[] | null): number {
  if (!fine || fine.length === 0) return 0;
  let max = 0;
  for (const f of fine) {
    const err = f.shift.provenance.error_estimate;
    const mag = Math.abs(f.shift.value);
    if (err !== null && mag > 0) max = Math.max(max, err / mag);
  }
  return max;
}

/** Is the perturbative fine structure past its stated validity? */
export function isBeyondValidity(fine: FineLevel[] | null): boolean {
  return fineErrorFraction(fine) > FINE_WARN_FRACTION;
}

/** eV span of the fine shifts within shell n (0 if fewer than two sub-levels). */
export function shellSplitting(fine: FineLevel[] | null, n: number): number {
  const s = (fine ?? []).filter((f) => f.n === n).map((f) => f.shift_ev.value);
  if (s.length < 2) return 0;
  return Math.max(...s) - Math.min(...s);
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd web && npx vitest run src/lib/whatif.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/whatif.ts web/src/lib/whatif.test.ts
git commit -m "feat(web): pure constants-lab logic (alpha formatting, error, splitting)"
```

---

### Task 4: Web — client/types + isolated store slice

**Files:**
- Modify: `web/src/api/types.ts` (add `alpha` to `LevelsResponse`)
- Modify: `web/src/api/client.ts` (optional `alpha` on `getLevels`)
- Modify: `web/src/state/store.ts` (lab slice + `loadWhatIf`, `whatif` in `ViewMode`)
- Test: `web/src/state/store.test.ts`

**Interfaces:**
- Consumes: `client.getLevels(system, nMax, fineStructure, alpha?)`, `REAL_ALPHA` (Task 3), `N_MAX_DIAGRAM` (existing in store).
- Produces store additions: `labAlpha: number`, `labZ: number`, `whatif: { real: LevelsResponse; altered: LevelsResponse } | null`, `whatifStatus: SampleStatus`, `setLabAlpha(n)`, `setLabZ(n)`, `loadWhatIf()`, and `"whatif"` added to `ViewMode`.

- [ ] **Step 1: Write the failing tests**

Append to `web/src/state/store.test.ts`:

```ts
it("lab alpha change clears only the what-if data, not main physics", () => {
  pretendLoaded();
  useAppStore.setState({ whatif: {} as never, whatifStatus: "ready" });
  useAppStore.getState().setLabAlpha(0.05);
  const s = useAppStore.getState();
  expect(s.labAlpha).toBe(0.05);
  expect(s.whatif).toBeNull();
  expect(s.whatifStatus).toBe("idle");
  expect(s.positions).not.toBeNull();
  expect(s.levels).not.toBeNull();
});

it("lab Z change clears only the what-if data", () => {
  pretendLoaded();
  useAppStore.setState({ whatif: {} as never, whatifStatus: "ready" });
  useAppStore.getState().setLabZ(3);
  const s = useAppStore.getState();
  expect(s.labZ).toBe(3);
  expect(s.whatif).toBeNull();
  expect(s.positions).not.toBeNull();
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd web && npx vitest run src/state/store.test.ts`
Expected: FAIL — `setLabAlpha` / `setLabZ` are not functions.

- [ ] **Step 3a: Add `alpha` to the `LevelsResponse` type**

In `web/src/api/types.ts`, add the field to `LevelsResponse`:

```ts
export interface LevelsResponse {
  system: SystemInfo;
  n_max: number;
  fine_structure: boolean;
  alpha: number;
  gross: GrossLevel[];
  fine: FineLevel[] | null;
}
```

- [ ] **Step 3b: Add the optional `alpha` argument to `getLevels`**

In `web/src/api/client.ts`, replace `getLevels`:

```ts
export function getLevels(
  system: string,
  nMax: number,
  fineStructure: boolean,
  alpha?: number,
): Promise<LevelsResponse> {
  const a = alpha === undefined ? "" : `&alpha=${alpha}`;
  return getJson(
    `/api/levels?system=${system}&n_max=${nMax}&fine_structure=${fineStructure}${a}`,
  );
}
```

- [ ] **Step 3c: Add the lab slice to the store**

In `web/src/state/store.ts`:

Add the import near the other lib imports:
```ts
import { REAL_ALPHA } from "../lib/whatif";
```

Extend the `ViewMode` union:
```ts
export type ViewMode = "cloud" | "plane" | "radial" | "levels" | "spectrum" | "whatif";
```

Add these fields to the `AppState` interface (after `spectrum: SpectrumResponse | null;`):
```ts
  labAlpha: number;
  labZ: number;
  whatif: { real: LevelsResponse; altered: LevelsResponse } | null;
  whatifStatus: SampleStatus;
  setLabAlpha: (labAlpha: number) => void;
  setLabZ: (labZ: number) => void;
  loadWhatIf: () => Promise<void>;
```

Add the initial values in the `create<AppState>` object (after `planeQuantity: "density",`):
```ts
  labAlpha: REAL_ALPHA,
  labZ: 1,
  whatif: null,
  whatifStatus: "idle",
```

Add the actions (after `setFps`):
```ts
  // lab slice: independent of the main (n,l,m,system) physics — never in INVALIDATED
  setLabAlpha: (labAlpha) => set({ labAlpha, whatif: null, whatifStatus: "idle" }),
  setLabZ: (labZ) => set({ labZ, whatif: null, whatifStatus: "idle" }),
  loadWhatIf: async () => {
    const { labAlpha, labZ } = get();
    const sys = `z${labZ}`;
    set({ whatifStatus: "sampling", error: null });
    try {
      const [real, altered] = await Promise.all([
        client.getLevels(sys, N_MAX_DIAGRAM, true),
        client.getLevels(sys, N_MAX_DIAGRAM, true, labAlpha),
      ]);
      set({ whatif: { real, altered }, whatifStatus: "ready" });
    } catch (err) {
      set({
        whatifStatus: "error",
        error: err instanceof Error ? err.message : String(err),
      });
    }
  },
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd web && npx vitest run src/state/store.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/api/types.ts web/src/api/client.ts web/src/state/store.ts web/src/state/store.test.ts
git commit -m "feat(web): lab store slice (alpha/Z) + getLevels alpha arg"
```

---

### Task 5: Web — the What-If view + wiring

**Files:**
- Create: `web/src/components/WhatIfView.tsx`
- Modify: `web/src/App.tsx` (render the view)
- Modify: `web/src/components/Controls.tsx` (add the tab option)
- Modify: `web/src/index.css` (append lab styles)

**Interfaces:**
- Consumes: the store lab slice (Task 4), `lib/whatif.ts` (Task 3), `Badge`.
- Produces: `WhatIfView` React component; a `view === "whatif"` branch; a `{ value: "whatif", label: "What-If: constants" }` tab.

- [ ] **Step 1: Create the component**

Create `web/src/components/WhatIfView.tsx`:

```tsx
import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import {
  ALPHA_MAX, fineErrorFraction, formatAlpha, isAltered,
  isBeyondValidity, shellSplitting,
} from "../lib/whatif";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 720;
const H = 480;
const ZOOM_N = 2; // textbook shell: 2p3/2 - 2p1/2 split grows with alpha

export function WhatIfView() {
  const {
    labAlpha, labZ, whatif, whatifStatus, error,
    setLabAlpha, setLabZ, loadWhatIf,
  } = useAppStore();

  useEffect(() => {
    if (whatif === null && whatifStatus === "idle") void loadWhatIf();
  }, [whatif, whatifStatus, loadWhatIf]);

  if (whatifStatus === "error") return <p className="error">{error}</p>;
  if (!whatif) return <p className="hint-block">loading What-If lab…</p>;

  const { real, altered } = whatif;
  const altOn = isAltered(altered.alpha, real.alpha);
  const badgeProv = (altered.fine ?? real.fine ?? [])[0]?.shift.provenance;

  const eMin = real.gross[0].energy_ev.value;
  const y = scaleLinear([eMin, 0], [H - 40, 60]);
  const rx1 = 70;
  const rx2 = 300;

  const realFine = (real.fine ?? []).filter((f) => f.n === ZOOM_N);
  const altFine = (altered.fine ?? []).filter((f) => f.n === ZOOM_N);
  const shiftsUeV = [...realFine, ...altFine].map((f) => f.shift_ev.value * 1e6);
  const lo = Math.min(0, ...shiftsUeV);
  const hi = Math.max(0, ...shiftsUeV);
  const pad = (hi - lo || 1) * 0.2;
  const yz = scaleLinear([lo - pad, hi + pad], [H - 60, 90]);
  const columns = [
    { x: 470, rows: realFine, label: "real", cf: false },
    { x: 590, rows: altFine, label: "altered", cf: altOn },
  ];

  const errFrac = fineErrorFraction(altered.fine);
  const beyond = isBeyondValidity(altered.fine);
  const splitEv = shellSplitting(altered.fine, ZOOM_N);

  return (
    <div className="view-wrap">
      <div className="view-header">
        <span className="plot-title">
          What-If: fundamental constants{" "}
          {badgeProv && <Badge provenance={badgeProv} />}
        </span>
      </div>

      {altOn && (
        <div className="counterfactual-banner">
          COUNTERFACTUAL · α = {formatAlpha(altered.alpha)} (real {formatAlpha(real.alpha)})
        </div>
      )}

      <svg viewBox={`0 0 ${W} ${H}`} role="img" className="levels-svg">
        <text x={(rx1 + rx2) / 2} y={30} textAnchor="middle" className="tick">
          gross levels (Z={altered.system.z}) — α-independent, EXACT
        </text>
        {real.gross.map((g) => (
          <g key={g.n}>
            <line
              x1={rx1} x2={rx2}
              y1={y(g.energy_ev.value)} y2={y(g.energy_ev.value)}
              className="rung"
            />
            <text x={rx1 - 8} y={y(g.energy_ev.value)} dy="0.32em" textAnchor="end" className="tick">
              n={g.n}
            </text>
            <text x={rx2 + 8} y={y(g.energy_ev.value)} dy="0.32em" className="tick">
              2n²={g.degeneracy}
            </text>
          </g>
        ))}

        <text x={530} y={54} textAnchor="middle" className="tick">
          n={ZOOM_N} fine split [µeV] — real vs altered
        </text>
        {columns.map((col) => (
          <g key={col.label}>
            <text x={col.x + 20} y={78} textAnchor="middle" className="tick">
              {col.label}
            </text>
            {col.rows.map((f) => (
              <g key={`${col.label}-${f.l}-${f.j}`}>
                <line
                  x1={col.x} x2={col.x + 40}
                  y1={yz(f.shift_ev.value * 1e6)} y2={yz(f.shift_ev.value * 1e6)}
                  className={col.cf ? "rung rung-counterfactual" : "rung"}
                />
                <text x={col.x + 46} y={yz(f.shift_ev.value * 1e6)} dy="0.32em" className="tick">
                  j={f.j} · {(f.shift_ev.value * 1e6).toFixed(1)}
                </text>
              </g>
            ))}
          </g>
        ))}
      </svg>

      <div className="whatif-controls">
        <label>
          α = {formatAlpha(labAlpha)} ({labAlpha.toExponential(2)})
          <input
            type="range" min={0.0005} max={ALPHA_MAX} step={0.0005}
            value={labAlpha}
            onChange={(e) => setLabAlpha(Number(e.target.value))}
          />
        </label>
        <div className="stepper">
          <span>nuclear charge Z</span>
          <button type="button" onClick={() => setLabZ(Math.max(1, labZ - 1))} disabled={labZ <= 1}>
            −
          </button>
          <span>{labZ}</span>
          <button type="button" onClick={() => setLabZ(Math.min(10, labZ + 1))} disabled={labZ >= 10}>
            +
          </button>
        </div>
        <button type="button" className="primary" onClick={() => setLabAlpha(real.alpha)}>
          reset α to real
        </button>
      </div>

      <p className={beyond ? "error" : "caption"}>
        {beyond
          ? `Fine-structure error ≈ ${(errFrac * 100).toFixed(0)}% — past the perturbative model's validity. The exact Dirac solution would differ; this is the honest limit, not a glitch.`
          : `Fine-structure fractional error ≈ ${(errFrac * 100).toFixed(1)}% (grows as (Zα)²). n=${ZOOM_N} splitting: ${splitEv.toExponential(2)} eV. α never touches the gross ladder — turn it down and the accidental l-degeneracy re-fuses. Equal-j states (2s₁/₂, 2p₁/₂) stay degenerate at this order — the Lamb shift is honestly absent.`}
      </p>
    </div>
  );
}
```

- [ ] **Step 2: Render the view in `App.tsx`**

In `web/src/App.tsx`, add the import (after the `SpectrumView` import):
```ts
import { WhatIfView } from "./components/WhatIfView";
```
And the render branch (after the `spectrum` line):
```tsx
        {view === "whatif" && <WhatIfView />}
```

- [ ] **Step 3: Add the tab option in `Controls.tsx`**

In `web/src/components/Controls.tsx`, append to `VIEW_OPTIONS`:
```ts
  { value: "whatif", label: "What-If: constants" },
```

- [ ] **Step 4: Append lab styles to `index.css`**

Append to `web/src/index.css`:
```css
/* What-If constants lab */
.counterfactual-banner {
  margin: 4px 0 8px;
  padding: 6px 10px;
  border: 1px solid #f472b6;
  border-radius: 6px;
  color: #f472b6;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-size: 0.85rem;
}
.rung-counterfactual {
  stroke: #f472b6;
}
.whatif-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 16px;
  margin-top: 8px;
}
.whatif-controls label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 220px;
}
.whatif-controls .stepper {
  display: flex;
  align-items: center;
  gap: 8px;
}
.whatif-controls .stepper button {
  width: 28px;
}
```

- [ ] **Step 5: Verify it type-checks and builds**

Run: `cd web && npm run build`
Expected: PASS — `tsc --noEmit` clean, `vite build` emits `dist/`.

- [ ] **Step 6: Commit**

```bash
git add web/src/components/WhatIfView.tsx web/src/App.tsx web/src/components/Controls.tsx web/src/index.css
git commit -m "feat(web): What-If constants view (real vs altered levels)"
```

---

### Task 6: Web — deep links for the lab state

**Files:**
- Modify: `web/src/lib/urlState.ts`
- Modify: `web/src/main.tsx` (serialize the new fields)
- Test: `web/src/lib/urlState.test.ts`

**Interfaces:**
- Consumes: `REAL_ALPHA`, `ALPHA_MAX` (Task 3).
- Produces: `UrlState` gains `labAlpha: number`, `labZ: number`; URL params `alpha` (clamped to (0, 0.5]) and `z` (clamped to [1, 10]); `"whatif"` added to the `VIEWS` whitelist.

- [ ] **Step 1: Write the failing tests**

In `web/src/lib/urlState.test.ts`, add cases and update the round-trip object. Add inside `describe("parseAppUrl")`:

```ts
it("parses lab alpha and Z for the what-if view", () => {
  expect(parseAppUrl("?view=whatif&alpha=0.02&z=3")).toEqual({
    view: "whatif",
    labAlpha: 0.02,
    labZ: 3,
  });
});

it("clamps alpha to (0, 0.5] and Z to [1, 10], dropping junk", () => {
  expect(parseAppUrl("?alpha=0.9")).toEqual({ labAlpha: 0.5 });
  expect(parseAppUrl("?alpha=0")).toEqual({});
  expect(parseAppUrl("?alpha=nope")).toEqual({});
  expect(parseAppUrl("?z=0")).toEqual({ labZ: 1 });
  expect(parseAppUrl("?z=99")).toEqual({ labZ: 10 });
});
```

Replace the `serializeAppUrl` round-trip test's `state` object so it includes the new fields:

```ts
  it("round-trips through parseAppUrl", () => {
    const state = {
      n: 4,
      l: 2,
      m: 2,
      system: "he+",
      basis: "real" as const,
      view: "whatif" as const,
      colorMode: "density" as const,
      fineStructure: true,
      nucleusMode: "hidden" as const,
      planeQuantity: "psi" as const,
      labAlpha: 0.02,
      labZ: 3,
    };
    const parsed = parseAppUrl(serializeAppUrl(state));
    expect({ ...URL_DEFAULTS, ...parsed }).toEqual(state);
  });
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd web && npx vitest run src/lib/urlState.test.ts`
Expected: FAIL — `labAlpha`/`labZ` not parsed; `whatif` not an allowed view.

- [ ] **Step 3: Extend `urlState.ts`**

In `web/src/lib/urlState.ts`:

3a. Add the import:
```ts
import { ALPHA_MAX, REAL_ALPHA } from "./whatif";
```

3b. Add the fields to `UrlState`:
```ts
export interface UrlState {
  n: number;
  l: number;
  m: number;
  system: string;
  basis: Basis;
  view: ViewMode;
  colorMode: ColorMode;
  fineStructure: boolean;
  nucleusMode: NucleusMode;
  planeQuantity: PlaneQuantity;
  labAlpha: number;
  labZ: number;
}
```

3c. Add the defaults:
```ts
export const URL_DEFAULTS: UrlState = {
  n: 1,
  l: 0,
  m: 0,
  system: "h",
  basis: "complex",
  view: "cloud",
  colorMode: "solid",
  fineStructure: false,
  nucleusMode: "marker",
  planeQuantity: "density",
  labAlpha: REAL_ALPHA,
  labZ: 1,
};
```

3d. Add `"whatif"` to the `VIEWS` whitelist:
```ts
const VIEWS: ViewMode[] = ["cloud", "plane", "radial", "levels", "spectrum", "whatif"];
```

3e. Add a float parser (next to `pickInt`):
```ts
function pickFloat(raw: string | null): number | undefined {
  if (raw === null || !/^-?\d*\.?\d+(e-?\d+)?$/i.test(raw)) return undefined;
  const v = Number(raw);
  return Number.isFinite(v) ? v : undefined;
}
```

3f. In `parseAppUrl`, before `return out;`, add:
```ts
  const alpha = pickFloat(q.get("alpha"));
  if (alpha !== undefined && alpha > 0) out.labAlpha = Math.min(alpha, ALPHA_MAX);
  const z = pickInt(q.get("z"));
  if (z !== undefined) out.labZ = Math.min(Math.max(z, 1), 10);
```

3g. In `serializeAppUrl`, before the `const s = q.toString();` line, add:
```ts
  if (Math.abs(state.labAlpha - URL_DEFAULTS.labAlpha) > 1e-9) {
    q.set("alpha", String(state.labAlpha));
  }
  if (state.labZ !== URL_DEFAULTS.labZ) q.set("z", String(state.labZ));
```

- [ ] **Step 4: Serialize the new fields in `main.tsx`**

In `web/src/main.tsx`, add the two fields to the object passed to `serializeAppUrl`:
```ts
    nucleusMode: s.nucleusMode,
    planeQuantity: s.planeQuantity,
    labAlpha: s.labAlpha,
    labZ: s.labZ,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd web && npx vitest run src/lib/urlState.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add web/src/lib/urlState.ts web/src/lib/urlState.test.ts web/src/main.tsx
git commit -m "feat(web): deep-link the what-if lab (view, alpha, Z)"
```

---

### Task 7: Ship — full gates, docs, PR

**Files:**
- Modify: `README.md` (add the What-If lab to the status/feature list)

- [ ] **Step 1: Run the full Python suite**

Run: `pytest`
Expected: PASS (all green).

- [ ] **Step 2: Run the full web suite and build**

Run: `cd web && npm test && npm run build`
Expected: PASS — vitest green, `tsc --noEmit` clean, `dist/` built.

- [ ] **Step 3: Lint**

Run: `ruff check .`
Expected: PASS (no findings).

- [ ] **Step 4: Manual QA (the app the change is for)**

Run: `atomsim serve` and in the browser:
- Switch View → "What-If: constants". Confirm gross ladder + n=2 real/altered columns render.
- Drag α up: the altered column's split visibly grows, the COUNTERFACTUAL banner appears, and the error readout climbs (turns into the red warning past ~10%).
- Step Z up: the split grows faster (Z⁴).
- Load `http://127.0.0.1:8000/?view=whatif&alpha=0.02&z=3` directly and confirm the lab opens in that state.

- [ ] **Step 5: Update the README**

Add a bullet to the feature list in `README.md` describing the What-If constants lab (α slider + Z stepper, real-vs-altered levels, COUNTERFACTUAL provenance, honest breakdown readout). Keep the existing "never quietly lies" framing.

- [ ] **Step 6: Commit the docs**

```bash
git add README.md
git commit -m "docs: What-If constants lab in feature list"
```

- [ ] **Step 7: Open the PR**

```bash
git push -u origin phase2-constants-lab
gh pr create --base main --head phase2-constants-lab \
  --title "Phase 2: What-If constants lab (alpha + Z)" \
  --body "Adds the sixth What-If view: continuous alpha slider + integer-Z stepper driving a real-vs-altered level diagram under a COUNTERFACTUAL banner. alpha is threaded through the analytic fine-structure path (seam for a future five-constant panel); altered-alpha levels carry COUNTERFACTUAL provenance with the Pauli-approximation error disclosed, and the view surfaces where the perturbative model breaks down. Spec: docs/superpowers/specs/2026-07-15-phase2-constants-lab-design.md"
```

Expected: PR created on `github.com/yaasshh09/atomsim`. (No AI attribution anywhere in the title/body — repo policy.)

---

## Notes for the implementer

- **Do not** add unit conversions in the client — `shift_ev` and `energy_ev` are already server-converted; the view multiplies eV→µeV for display only (×1e6), which is a display scale, not a physics conversion.
- The two `getLevels` calls in `loadWhatIf` differ only by the `alpha` argument; the real one omits it so the server echoes the real `ALPHA`, which is the authoritative reference for the banner (never the client `REAL_ALPHA`, which only positions the slider).
- Keep the lab slice out of the store's `INVALIDATED` block — changing (n, l, m, system) must not disturb the lab, and vice versa.
