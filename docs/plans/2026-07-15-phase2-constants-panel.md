# Five-Constant Panel (What-If Lab) Implementation Plan

**Goal:** Replace the What-If lab's direct α slider with the raw five constants (ℏ, e, mₑ, ε₀, c) as multiplier inputs, deriving α + the atomic scales server-side and surfacing the degeneracy lesson (which observable combination actually changed).

**Architecture:** A new pure engine function `analyze_constants` turns five multipliers into a `ConstantsReport` (derived α, Bohr radius, Hartree energy, each a provenance-carrying `Quantity` plus a `changed`/`ratio`). A new `GET /api/constants` endpoint exposes it. The web store's single `labAlpha` becomes a five-multiplier `labConst`; α is now derived and fed into the existing `/api/levels?alpha=` reuse. The reused level diagram shows fine-structure *structure* in units of E_h (absolute scale lives only in the readouts), and refuses to draw the altered column when derived α exceeds the perturbative bound of 0.5.

**Tech Stack:** Python 3.12 (SciPy, FastAPI, pytest), TypeScript/React 19, Zustand, d3-scale, Vitest.

## Global Constraints

- **Provenance boundary rule:** every physical value crossing a module boundary is a `Quantity`/`Field`/provenance-carrying container. No physics in the frontend, deriving α, a₀, E_h happens server-side.
- **Fidelity rule:** derived observables are `EXACT` at all-ones multipliers (closed-form from CODATA), `COUNTERFACTUAL` when any multiplier ≠ 1.
- **Multiplier range:** each of the five ∈ **[0.25, 4]** at every boundary (server validation, URL clamp, slider). Z stepper stays [1, 10].
- **Honesty, structure vs. scale:** the level diagram is normalized (units of E_h); absolute a₀ (pm) and E_h (eV) live only in the readouts. eV is a **fixed real-universe** ruler (1 eV = real elementary charge in joules).
- **Honesty, model breakdown:** the readouts always show the true derived α even when large; the altered level diagram is requested only when derived α ≤ 0.5, else the beyond-validity notice replaces the altered column.
- **TDD:** failing test first for every engine/server/web-logic change. `WhatIfView` follows the repo pattern of no component unit test, verified by `npm run build` + QA.
- **Windows/Miniforge:** run Python from the `atomsim` conda env. Web commands run from `web/`.
- **Engine units:** Hartree atomic units internally; SI/eV/pm conversions at the server boundary only.

---

### Task 1: Engine: `constants_lab.py` (`analyze_constants`)

**Files:**
- Create: `src/atomsim/constants_lab.py`
- Test: `tests/test_constants_lab.py`

**Interfaces:**
- Consumes: `atomsim.constants.FundamentalConstants` (`.codata()`, `.alpha`, `.bohr_radius`, `.hartree_energy`); `Fidelity`, `Provenance`, `Quantity`.
- Produces:
  - `DerivedObservable(quantity: Quantity, ratio: float, changed: bool)` (frozen dataclass).
  - `ConstantsReport(alpha: DerivedObservable, bohr_radius_pm: DerivedObservable, hartree_ev: DerivedObservable, altered: bool)` (frozen dataclass).
  - `analyze_constants(hbar=1.0, e=1.0, m_e=1.0, eps0=1.0, c=1.0) -> ConstantsReport`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_constants_lab.py`:

```python
import math

import pytest

from atomsim.constants import ALPHA, FundamentalConstants
from atomsim.constants_lab import ConstantsReport, analyze_constants
from atomsim.provenance import Fidelity

def test_real_universe_is_exact_and_unchanged():
    r = analyze_constants()
    assert isinstance(r, ConstantsReport)
    assert r.altered is False
    assert r.alpha.quantity.value == pytest.approx(ALPHA, rel=1e-6)
    assert r.alpha.ratio == pytest.approx(1.0)
    assert r.alpha.changed is False
    assert r.bohr_radius_pm.changed is False
    assert r.hartree_ev.changed is False
    assert r.alpha.quantity.provenance.fidelity is Fidelity.EXACT
    # real readouts land on the textbook values
    assert r.bohr_radius_pm.quantity.value == pytest.approx(52.9, rel=1e-2)
    assert r.hartree_ev.quantity.value == pytest.approx(27.211, rel=1e-4)

def test_degeneracy_pair_changes_nothing_observable():
    # doubling e and quadrupling eps0 leaves alpha, a0, E_h all identical
    r = analyze_constants(e=2.0, eps0=4.0)
    assert r.altered is True
    assert r.alpha.changed is False
    assert r.bohr_radius_pm.changed is False
    assert r.hartree_ev.changed is False
    assert r.alpha.quantity.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_electron_mass_scales_size_and_binding():
    r = analyze_constants(m_e=0.5)
    assert r.bohr_radius_pm.ratio == pytest.approx(2.0)
    assert r.hartree_ev.ratio == pytest.approx(0.5)
    assert r.alpha.changed is False

def test_altered_provenance_names_the_multipliers():
    r = analyze_constants(e=2.0, eps0=4.0)
    method = r.alpha.quantity.provenance.method.lower()
    assert "altered" in method
    assert "e" in method and "eps0" in method

def test_derived_alpha_matches_fundamental_constants():
    real = FundamentalConstants.codata()
    altered = FundamentalConstants(
        hbar=real.hbar, e=real.e * 1.5, m_e=real.m_e, eps0=real.eps0, c=real.c
    )
    r = analyze_constants(e=1.5)
    assert r.alpha.quantity.value == pytest.approx(altered.alpha, rel=1e-12)
    assert r.alpha.ratio == pytest.approx(altered.alpha / real.alpha, rel=1e-12)
    assert not math.isclose(r.alpha.ratio, 1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_constants_lab.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.constants_lab'`.

- [ ] **Step 3: Implement `constants_lab.py`**

Create `src/atomsim/constants_lab.py`:

```python
"""What-If Lab: derive the observable consequences of altering the raw constants.

The five fundamental constants (hbar, e, m_e, eps0, c) are supplied as multipliers
on their real CODATA values. From an altered FundamentalConstants this module
derives the three quantities that are actually observable for a one-electron atom
against fixed SI rulers, the fine-structure constant alpha, the Bohr radius
(size), and the Hartree energy (binding), each as a Quantity whose fidelity is
COUNTERFACTUAL when any multiplier departs from 1, EXACT otherwise. The
per-observable `changed` flag IS the degeneracy lesson: many distinct multiplier
tuples leave all three observables identical.
"""

import math
from dataclasses import dataclass

from atomsim.constants import FundamentalConstants
from atomsim.provenance import Fidelity, Provenance, Quantity

_MULT_NAMES = ("hbar", "e", "m_e", "eps0", "c")

@dataclass(frozen=True)
class DerivedObservable:
    quantity: Quantity
    ratio: float   # altered / real
    changed: bool  # not isclose(ratio, 1)

@dataclass(frozen=True)
class ConstantsReport:
    alpha: DerivedObservable
    bohr_radius_pm: DerivedObservable
    hartree_ev: DerivedObservable
    altered: bool

def _observable(
    label: str, unit: str, alt_value: float, real_value: float,
    formula: str, altered: bool, applied: str,
) -> DerivedObservable:
    ratio = alt_value / real_value
    changed = not math.isclose(ratio, 1.0, rel_tol=1e-9)
    method = formula
    if altered:
        method += f"; altered raw constants ({applied})"
    return DerivedObservable(
        quantity=Quantity(
            value=alt_value,
            unit=unit,
            label=label,
            provenance=Provenance(
                fidelity=Fidelity.COUNTERFACTUAL if altered else Fidelity.EXACT,
                method=method,
                assumptions=(
                    "observable measured against fixed real-universe SI rulers",
                ),
            ),
        ),
        ratio=ratio,
        changed=changed,
    )

def analyze_constants(
    hbar: float = 1.0, e: float = 1.0, m_e: float = 1.0,
    eps0: float = 1.0, c: float = 1.0,
) -> ConstantsReport:
    """Derive alpha, Bohr radius (pm), and Hartree energy (eV) from multipliers."""
    mults = (hbar, e, m_e, eps0, c)
    altered = any(not math.isclose(v, 1.0, rel_tol=1e-12) for v in mults)
    applied = ", ".join(
        f"{n}x{v:g}" for n, v in zip(_MULT_NAMES, mults, strict=True)
        if not math.isclose(v, 1.0, rel_tol=1e-12)
    )

    real = FundamentalConstants.codata()
    alt = FundamentalConstants(
        hbar=real.hbar * hbar, e=real.e * e, m_e=real.m_e * m_e,
        eps0=real.eps0 * eps0, c=real.c * c,
    )
    # 1 eV = (real elementary charge) joules: a fixed SI ruler, never altered.
    joule_per_ev = real.e

    return ConstantsReport(
        alpha=_observable(
            "alpha", "", alt.alpha, real.alpha,
            "alpha = e^2 / (4 pi eps0 hbar c)", altered, applied,
        ),
        bohr_radius_pm=_observable(
            "Bohr radius", "pm", alt.bohr_radius * 1e12, real.bohr_radius * 1e12,
            "a0 = 4 pi eps0 hbar^2 / (m_e e^2)", altered, applied,
        ),
        hartree_ev=_observable(
            "Hartree energy", "eV",
            alt.hartree_energy / joule_per_ev, real.hartree_energy / joule_per_ev,
            "E_h = hbar^2 / (m_e a0^2)", altered, applied,
        ),
        altered=altered,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_constants_lab.py -v`
Expected: PASS (all five).

- [ ] **Step 5: Commit**

```bash
git add src/atomsim/constants_lab.py tests/test_constants_lab.py
git commit -m "feat(engine): derive observables from raw constants (What-If Lab)"
```

---

### Task 2: Server: `/api/constants` endpoint

**Files:**
- Modify: `src/atomsim/server/schemas.py`
- Modify: `src/atomsim/server/app.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `analyze_constants` (Task 1); `QuantityModel.from_quantity`; `ConstantsReport`, `DerivedObservable`.
- Produces: `GET /api/constants?hbar=&e=&m_e=&eps0=&c=` returning `ConstantsReportModel` (each field `DerivedObservableModel { quantity, ratio, changed }` + `altered: bool`). Multipliers validated to [0.25, 4].

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
def test_constants_default_is_real_and_exact(client):
    body = client.get("/api/constants").json()
    assert body["altered"] is False
    assert body["alpha"]["quantity"]["value"] == pytest.approx(ALPHA, rel=1e-6)
    assert body["alpha"]["quantity"]["provenance"]["fidelity"] == "exact"
    assert body["alpha"]["changed"] is False

def test_constants_degeneracy_pair_changes_nothing_observable(client):
    body = client.get("/api/constants?e=2&eps0=4").json()
    assert body["altered"] is True
    assert body["alpha"]["changed"] is False
    assert body["bohr_radius_pm"]["changed"] is False
    assert body["hartree_ev"]["changed"] is False
    assert body["alpha"]["quantity"]["provenance"]["fidelity"] == "counterfactual"

def test_constants_electron_mass_scales_size_and_binding(client):
    body = client.get("/api/constants?m_e=0.5").json()
    assert body["bohr_radius_pm"]["ratio"] == pytest.approx(2.0)
    assert body["hartree_ev"]["ratio"] == pytest.approx(0.5)

def test_constants_rejects_out_of_range(client):
    assert client.get("/api/constants?e=0.1").status_code == 422
    assert client.get("/api/constants?hbar=5").status_code == 422
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_server.py -k constants -v`
Expected: FAIL, `/api/constants` returns 404 (route absent).

- [ ] **Step 3: Add the response models to `schemas.py`**

In `src/atomsim/server/schemas.py`, add the import (with the other `atomsim` imports near the top):

```python
from atomsim.constants_lab import ConstantsReport, DerivedObservable
```

Add these models (after `QuantityModel`, before `FieldModel`):

```python
class DerivedObservableModel(BaseModel):
    quantity: QuantityModel
    ratio: float
    changed: bool

    @classmethod
    def from_observable(cls, o: DerivedObservable) -> "DerivedObservableModel":
        return cls(
            quantity=QuantityModel.from_quantity(o.quantity),
            ratio=o.ratio,
            changed=o.changed,
        )

class ConstantsReportModel(BaseModel):
    alpha: DerivedObservableModel
    bohr_radius_pm: DerivedObservableModel
    hartree_ev: DerivedObservableModel
    altered: bool

    @classmethod
    def from_report(cls, r: ConstantsReport) -> "ConstantsReportModel":
        return cls(
            alpha=DerivedObservableModel.from_observable(r.alpha),
            bohr_radius_pm=DerivedObservableModel.from_observable(r.bohr_radius_pm),
            hartree_ev=DerivedObservableModel.from_observable(r.hartree_ev),
            altered=r.altered,
        )
```

- [ ] **Step 4: Add the endpoint to `app.py`**

In `src/atomsim/server/app.py`, add the engine import (after the `fine_structure` import line):

```python
from atomsim.constants_lab import analyze_constants
```

Add `ConstantsReportModel` to the `from atomsim.server.schemas import (...)` block (keep alphabetical-ish with the others):

```python
    ConstantsReportModel,
```

Add the endpoint inside `create_app`, right after the `levels_endpoint` definition (before `@app.get("/api/radial/...")`):

```python
    @app.get("/api/constants", response_model=ConstantsReportModel)
    def constants_endpoint(hbar: float = 1.0, e: float = 1.0, m_e: float = 1.0,
                           eps0: float = 1.0, c: float = 1.0) -> ConstantsReportModel:
        for name, mult in (("hbar", hbar), ("e", e), ("m_e", m_e),
                           ("eps0", eps0), ("c", c)):
            if not 0.25 <= mult <= 4.0:
                raise HTTPException(
                    status_code=422,
                    detail=f"{name} multiplier must be in [0.25, 4], got {mult}",
                )
        report = analyze_constants(hbar=hbar, e=e, m_e=m_e, eps0=eps0, c=c)
        return ConstantsReportModel.from_report(report)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_server.py -k constants -v`
Expected: PASS (all four). Then `pytest tests/test_server.py -q`, existing endpoints still green.

- [ ] **Step 6: Commit**

```bash
git add src/atomsim/server/schemas.py src/atomsim/server/app.py tests/test_server.py
git commit -m "feat(server): /api/constants derives observables from raw multipliers"
```

---

### Task 3: Web: pure logic, types, and client for `/api/constants`

**Files:**
- Modify: `web/src/lib/whatif.ts`
- Modify: `web/src/api/types.ts`
- Modify: `web/src/api/client.ts`
- Test: `web/src/lib/whatif.test.ts`

**Interfaces:**
- Produces (whatif.ts): `CONST_MIN=0.25`, `CONST_MAX=4`, `CONSTANT_KEYS` (readonly tuple `["hbar","e","m_e","eps0","c"]`), `type ConstantKey`, `CONSTANT_LABELS: Record<ConstantKey, string>`, `isAlphaValid(alpha) -> boolean`, `formatRatio(ratio) -> string`. Keeps all existing exports.
- Produces (types.ts): `DerivedObservable`, `ConstantsReport`.
- Produces (client.ts): `type ConstMultipliers`, `getConstants(m: ConstMultipliers) -> Promise<ConstantsReport>`.

- [ ] **Step 1: Write the failing tests**

In `web/src/lib/whatif.test.ts`, extend the import line and append two describe blocks. Change the top import to also pull the new names:

```ts
import {
  CONST_MAX, CONST_MIN, FINE_WARN_FRACTION, REAL_ALPHA, fineErrorFraction,
  formatAlpha, formatRatio, isAlphaValid, isAltered, isBeyondValidity, shellSplitting,
} from "./whatif";
```

Append at the end of the file:

```ts
describe("formatAlpha beyond the reciprocal regime", () => {
  it("shows a decimal for α ≥ 0.5 (the reciprocal form is nonsense there)", () => {
    expect(formatAlpha(0.5)).toBe("0.50");
    expect(formatAlpha(7.5)).toBe("7.50");
  });
});

describe("formatRatio", () => {
  it("labels unchanged and scaled ratios", () => {
    expect(formatRatio(1)).toBe("unchanged");
    expect(formatRatio(2)).toBe("×2.00");
    expect(formatRatio(0.5)).toBe("×0.50");
  });
});

describe("isAlphaValid", () => {
  it("is true within (0, 0.5], false past it", () => {
    expect(isAlphaValid(REAL_ALPHA)).toBe(true);
    expect(isAlphaValid(0.5)).toBe(true);
    expect(isAlphaValid(0.9)).toBe(false);
    expect(isAlphaValid(0)).toBe(false);
  });
  it("multiplier bounds match the server range", () => {
    expect([CONST_MIN, CONST_MAX]).toEqual([0.25, 4]);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd web && npx vitest run src/lib/whatif.test.ts`
Expected: FAIL, `formatRatio`, `isAlphaValid`, `CONST_MIN`, `CONST_MAX` are not exported.

- [ ] **Step 3: Extend `whatif.ts`**

In `web/src/lib/whatif.ts`, first replace the existing `formatAlpha` function so large derived α renders as a decimal instead of a nonsensical `1/0`:

```ts
/** Human form of α: "1/137" in the reciprocal regime, a decimal once α ≥ 0.5. */
export function formatAlpha(alpha: number): string {
  if (alpha <= 0) return "0";
  if (alpha >= 0.5) return alpha.toFixed(2);
  return `1/${Math.round(1 / alpha)}`;
}
```

Then append (keep the existing exports untouched):

```ts
/** Raw-constant multiplier bounds, matches the server's [0.25, 4] validation. */
export const CONST_MIN = 0.25;
export const CONST_MAX = 4;

/** The five raw constants, in the order shown in the panel. */
export const CONSTANT_KEYS = ["hbar", "e", "m_e", "eps0", "c"] as const;
export type ConstantKey = (typeof CONSTANT_KEYS)[number];

/** Display glyphs for the five constants. */
export const CONSTANT_LABELS: Record<ConstantKey, string> = {
  hbar: "ℏ",
  e: "e",
  m_e: "mₑ",
  eps0: "ε₀",
  c: "c",
};

/** Derived α only yields a physical perturbative diagram when in (0, 0.5] (server bound). */
export function isAlphaValid(alpha: number): boolean {
  return alpha > 0 && alpha <= ALPHA_MAX;
}

/** Human form of an altered/real ratio: "unchanged" at 1, else "×2.00" / "×0.50". */
export function formatRatio(ratio: number): string {
  if (Math.abs(ratio - 1) < 1e-9) return "unchanged";
  return `×${ratio.toFixed(2)}`;
}
```

- [ ] **Step 4: Add the response types to `types.ts`**

In `web/src/api/types.ts`, append (after the `LevelsResponse` interface):

```ts
export interface DerivedObservable {
  quantity: Quantity;
  ratio: number;
  changed: boolean;
}

export interface ConstantsReport {
  alpha: DerivedObservable;
  bohr_radius_pm: DerivedObservable;
  hartree_ev: DerivedObservable;
  altered: boolean;
}
```

- [ ] **Step 5: Add the client call**

In `web/src/api/client.ts`, add `ConstantsReport` to the type import block at the top:

```ts
import type {
  ConstantsReport,
  JobInfo,
  JobMeta,
  LevelsResponse,
  RadialResponse,
  SpectrumResponse,
  StateResponse,
  SystemsResponse,
} from "./types";
```

And append after `getLevels`:

```ts
export interface ConstMultipliers {
  hbar: number;
  e: number;
  m_e: number;
  eps0: number;
  c: number;
}

export function getConstants(m: ConstMultipliers): Promise<ConstantsReport> {
  return getJson(
    `/api/constants?hbar=${m.hbar}&e=${m.e}&m_e=${m.m_e}&eps0=${m.eps0}&c=${m.c}`,
  );
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd web && npx vitest run src/lib/whatif.test.ts`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add web/src/lib/whatif.ts web/src/api/types.ts web/src/api/client.ts web/src/lib/whatif.test.ts
git commit -m "feat(web): constants-lab logic, types, and client for /api/constants"
```

---

### Task 4: Web: store slice (labConst replaces labAlpha)

**Files:**
- Modify: `web/src/state/store.ts`
- Test: `web/src/state/store.test.ts`

**Interfaces:**
- Consumes: `client.getConstants`, `type ConstMultipliers` (Task 3); `isAlphaValid` (Task 3); `type ConstantsReport` (Task 3); existing `N_MAX_DIAGRAM`, `client.getLevels`.
- Produces store changes: removes `labAlpha`/`setLabAlpha`; adds `labConst: ConstMultipliers`, `setLabConst: (partial: Partial<ConstMultipliers>) => void`; `whatif` becomes `{ report: ConstantsReport; real: LevelsResponse; altered: LevelsResponse | null } | null`. Keeps `labZ`, `setLabZ`, `whatifStatus`, `loadWhatIf`.

- [ ] **Step 1: Update the failing tests**

In `web/src/state/store.test.ts`, replace the `"lab alpha change clears only the what-if data, not main physics"` test (the whole `it(...)` block) with:

```ts
  it("lab constant change clears only the what-if data, not main physics", () => {
    pretendLoaded();
    useAppStore.setState({ whatif: {} as never, whatifStatus: "ready" });
    useAppStore.getState().setLabConst({ e: 2 });
    const s = useAppStore.getState();
    expect(s.labConst.e).toBe(2);
    expect(s.labConst.hbar).toBe(1);
    expect(s.whatif).toBeNull();
    expect(s.whatifStatus).toBe("idle");
    expect(s.positions).not.toBeNull();
    expect(s.levels).not.toBeNull();
  });
```

(Leave the `"lab Z change clears only the what-if data"` test unchanged.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd web && npx vitest run src/state/store.test.ts`
Expected: FAIL, `setLabConst` is not a function.

- [ ] **Step 3a: Fix the imports**

In `web/src/state/store.ts`, replace the whatif import line:

```ts
import { REAL_ALPHA } from "../lib/whatif";
```

with:

```ts
import { isAlphaValid } from "../lib/whatif";
```

Add `ConstantsReport` to the `../api/types` type import block, and add a `ConstMultipliers` import from the client. The top import block becomes:

```ts
import { create } from "zustand";
import * as client from "../api/client";
import type { Basis, ConstMultipliers, PlaneQuantity } from "../api/client";
import type {
  ConstantsReport,
  LevelsResponse,
  PlaneMeta,
  RadialResponse,
  SampleMeta,
  SpectrumResponse,
  StateResponse,
  SystemInfo,
} from "../api/types";
import type { NucleusMode } from "../lib/nucleus";
import { clampState } from "../lib/quantum";
import { isAlphaValid } from "../lib/whatif";
```

- [ ] **Step 3b: Update the `AppState` interface**

In the `AppState` interface, replace:

```ts
  labAlpha: number;
  labZ: number;
  whatif: { real: LevelsResponse; altered: LevelsResponse } | null;
  whatifStatus: SampleStatus;
  setLabAlpha: (labAlpha: number) => void;
  setLabZ: (labZ: number) => void;
  loadWhatIf: () => Promise<void>;
```

with:

```ts
  labConst: ConstMultipliers;
  labZ: number;
  whatif: {
    report: ConstantsReport;
    real: LevelsResponse;
    altered: LevelsResponse | null;
  } | null;
  whatifStatus: SampleStatus;
  setLabConst: (partial: Partial<ConstMultipliers>) => void;
  setLabZ: (labZ: number) => void;
  loadWhatIf: () => Promise<void>;
```

- [ ] **Step 3c: Update the initial values**

In the `create<AppState>` object, replace:

```ts
  labAlpha: REAL_ALPHA,
  labZ: 1,
```

with:

```ts
  labConst: { hbar: 1, e: 1, m_e: 1, eps0: 1, c: 1 },
  labZ: 1,
```

- [ ] **Step 3d: Update the actions**

Replace the `setLabAlpha` / `setLabZ` / `loadWhatIf` block:

```ts
  // lab slice: independent of the main (n,l,m,system) physics, never in INVALIDATED
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

with:

```ts
  // lab slice: independent of the main (n,l,m,system) physics, never in INVALIDATED
  setLabConst: (partial) =>
    set((s) => ({
      labConst: { ...s.labConst, ...partial },
      whatif: null,
      whatifStatus: "idle",
    })),
  setLabZ: (labZ) => set({ labZ, whatif: null, whatifStatus: "idle" }),
  loadWhatIf: async () => {
    const { labConst, labZ } = get();
    const sys = `z${labZ}`;
    set({ whatifStatus: "sampling", error: null });
    try {
      const report = await client.getConstants(labConst);
      const alpha = report.alpha.quantity.value;
      const real = await client.getLevels(sys, N_MAX_DIAGRAM, true);
      // altered diagram only when the derived alpha stays in the perturbative range
      const altered =
        report.altered && isAlphaValid(alpha)
          ? await client.getLevels(sys, N_MAX_DIAGRAM, true, alpha)
          : null;
      set({ whatif: { report, real, altered }, whatifStatus: "ready" });
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
Expected: PASS (both lab tests + all existing store tests).

- [ ] **Step 5: Commit**

```bash
git add web/src/state/store.ts web/src/state/store.test.ts
git commit -m "feat(web): five-multiplier labConst store slice (alpha now derived)"
```

---

### Task 5: Web: deep links (labConst + Z)

**Files:**
- Modify: `web/src/lib/urlState.ts`
- Modify: `web/src/main.tsx`
- Test: `web/src/lib/urlState.test.ts`

**Interfaces:**
- Consumes: `CONST_MIN`, `CONST_MAX`, `CONSTANT_KEYS`, `type ConstantKey` (Task 3); `type ConstMultipliers` (Task 3, from client).
- Produces: `UrlState` gains `labConst: ConstMultipliers` (replacing `labAlpha`); URL params `hbar`, `e`, `me`, `eps0`, `c` (each omitted when 1, clamped to [0.25, 4]); `z` unchanged. The old `?alpha=` param is removed.

- [ ] **Step 1: Update the failing tests**

In `web/src/lib/urlState.test.ts`, replace the two lab tests (`"parses lab alpha and Z for the what-if view"` and `"clamps alpha to (0, 0.5]..."`) with:

```ts
  it("parses lab constant multipliers and Z for the what-if view", () => {
    expect(parseAppUrl("?view=whatif&e=2&eps0=4&z=3")).toEqual({
      view: "whatif",
      labConst: { hbar: 1, e: 2, m_e: 1, eps0: 4, c: 1 },
      labZ: 3,
    });
  });

  it("clamps constant multipliers to [0.25, 4] and Z to [1, 10], dropping junk", () => {
    expect(parseAppUrl("?e=9")).toEqual({
      labConst: { hbar: 1, e: 4, m_e: 1, eps0: 1, c: 1 },
    });
    expect(parseAppUrl("?me=0.1")).toEqual({
      labConst: { hbar: 1, e: 1, m_e: 0.25, eps0: 1, c: 1 },
    });
    expect(parseAppUrl("?e=0")).toEqual({});
    expect(parseAppUrl("?e=nope")).toEqual({});
    expect(parseAppUrl("?z=0")).toEqual({ labZ: 1 });
    expect(parseAppUrl("?z=99")).toEqual({ labZ: 10 });
  });
```

And in the `"round-trips through parseAppUrl"` test, replace `labAlpha: 0.02,` with:

```ts
      labConst: { hbar: 1, e: 2, m_e: 1, eps0: 4, c: 1 },
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd web && npx vitest run src/lib/urlState.test.ts`
Expected: FAIL, `labConst` undefined; `?alpha=` still parsed instead.

- [ ] **Step 3a: Update `urlState.ts` imports and `UrlState`**

In `web/src/lib/urlState.ts`, replace the client import and whatif import:

```ts
import type { Basis, PlaneQuantity } from "../api/client";
```
```ts
import { ALPHA_MAX, REAL_ALPHA } from "./whatif";
```

with:

```ts
import type { Basis, ConstMultipliers, PlaneQuantity } from "../api/client";
```
```ts
import { CONST_MAX, CONST_MIN, CONSTANT_KEYS, type ConstantKey } from "./whatif";
```

In the `UrlState` interface, replace `labAlpha: number;` with:

```ts
  labConst: ConstMultipliers;
```

In `URL_DEFAULTS`, replace `labAlpha: REAL_ALPHA,` with:

```ts
  labConst: { hbar: 1, e: 1, m_e: 1, eps0: 1, c: 1 },
```

- [ ] **Step 3b: Add the param-name map**

After the `SYSTEM_KEY` constant, add:

```ts
// short URL names for the five constant multipliers (m_e -> "me")
const CONST_PARAMS: Record<ConstantKey, string> = {
  hbar: "hbar",
  e: "e",
  m_e: "me",
  eps0: "eps0",
  c: "c",
};
```

- [ ] **Step 3c: Replace the parse of `alpha`/`z`**

In `parseAppUrl`, replace:

```ts
  const alpha = pickFloat(q.get("alpha"));
  if (alpha !== undefined && alpha > 0) out.labAlpha = Math.min(alpha, ALPHA_MAX);
  const z = pickInt(q.get("z"));
  if (z !== undefined) out.labZ = Math.min(Math.max(z, 1), 10);
```

with:

```ts
  const lc: Partial<ConstMultipliers> = {};
  for (const k of CONSTANT_KEYS) {
    const v = pickFloat(q.get(CONST_PARAMS[k]));
    if (v !== undefined && v > 0) lc[k] = Math.min(Math.max(v, CONST_MIN), CONST_MAX);
  }
  if (Object.keys(lc).length > 0) out.labConst = { ...URL_DEFAULTS.labConst, ...lc };

  const z = pickInt(q.get("z"));
  if (z !== undefined) out.labZ = Math.min(Math.max(z, 1), 10);
```

- [ ] **Step 3d: Replace the serialize of `alpha`/`z`**

In `serializeAppUrl`, replace:

```ts
  if (Math.abs(state.labAlpha - URL_DEFAULTS.labAlpha) > 1e-9) {
    q.set("alpha", String(state.labAlpha));
  }
  if (state.labZ !== URL_DEFAULTS.labZ) q.set("z", String(state.labZ));
```

with:

```ts
  for (const k of CONSTANT_KEYS) {
    if (Math.abs(state.labConst[k] - URL_DEFAULTS.labConst[k]) > 1e-9) {
      q.set(CONST_PARAMS[k], String(state.labConst[k]));
    }
  }
  if (state.labZ !== URL_DEFAULTS.labZ) q.set("z", String(state.labZ));
```

- [ ] **Step 3e: Update `main.tsx`**

In `web/src/main.tsx`, in the object passed to `serializeAppUrl`, replace `labAlpha: s.labAlpha,` with:

```ts
    labConst: s.labConst,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd web && npx vitest run src/lib/urlState.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/urlState.ts web/src/main.tsx web/src/lib/urlState.test.ts
git commit -m "feat(web): deep-link the five constant multipliers (replaces alpha param)"
```

---

### Task 6: Web: `WhatIfView` rework (five sliders, readouts, degeneracy)

**Files:**
- Modify: `web/src/components/WhatIfView.tsx` (full rewrite)
- Modify: `web/src/index.css` (append lab styles)

**Interfaces:**
- Consumes: store lab slice (`labConst`, `setLabConst`, `labZ`, `setLabZ`, `whatif`, `whatifStatus`, `loadWhatIf`) from Task 4; `CONSTANT_KEYS`, `CONSTANT_LABELS`, `CONST_MIN`, `CONST_MAX`, `formatAlpha`, `formatRatio`, `fineErrorFraction`, `isBeyondValidity` from Task 3; `type DerivedObservable`; `Badge`.
- Produces: the reworked view. No test (component); verified by `npm run build` + QA.

- [ ] **Step 1: Rewrite `WhatIfView.tsx`**

Replace the entire contents of `web/src/components/WhatIfView.tsx` with:

```tsx
import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import type { ConstMultipliers } from "../api/client";
import type { DerivedObservable } from "../api/types";
import {
  CONST_MAX,
  CONST_MIN,
  CONSTANT_KEYS,
  CONSTANT_LABELS,
  fineErrorFraction,
  formatAlpha,
  formatRatio,
  shellSplitting,
} from "../lib/whatif";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 720;
const H = 480;
const ZOOM_N = 2; // textbook shell: 2p3/2 - 2p1/2 split grows with alpha

const REAL_ALL: ConstMultipliers = { hbar: 1, e: 1, m_e: 1, eps0: 1, c: 1 };

export function WhatIfView() {
  const { labConst, labZ, whatif, whatifStatus, error, setLabConst, setLabZ, loadWhatIf } =
    useAppStore();

  useEffect(() => {
    if (whatif === null && whatifStatus === "idle") void loadWhatIf();
  }, [whatif, whatifStatus, loadWhatIf]);

  if (whatifStatus === "error") return <p className="error">{error}</p>;
  if (!whatif) return <p className="hint-block">loading What-If lab…</p>;

  const { report, real, altered } = whatif;
  const altOn = report.altered;
  const beyondValidity = altOn && altered === null;
  const alphaValue = report.alpha.quantity.value;

  const readouts: { key: string; label: string; obs: DerivedObservable; text: string }[] = [
    {
      key: "alpha",
      label: "α, fine-structure constant",
      obs: report.alpha,
      text: `${formatAlpha(alphaValue)} (${alphaValue.toExponential(3)})`,
    },
    {
      key: "a0",
      label: "a₀, Bohr radius (atom size)",
      obs: report.bohr_radius_pm,
      text: `${report.bohr_radius_pm.quantity.value.toFixed(2)} pm`,
    },
    {
      key: "eh",
      label: "E_h, Hartree energy (binding)",
      obs: report.hartree_ev,
      text: `${report.hartree_ev.quantity.value.toFixed(3)} eV`,
    },
  ];

  // Gross ladder: STRUCTURE only, in units of E_h (hartree). Absolute scale lives
  // in the readouts above, the honest structure/scale split.
  const eMin = real.gross[0].energy.value;
  const y = scaleLinear([eMin, 0], [H - 40, 60]);
  const rx1 = 70;
  const rx2 = 300;

  // n=2 fine split, in µE_h (hartree * 1e6), normalized, not real eV.
  const realFine = (real.fine ?? []).filter((f) => f.n === ZOOM_N);
  const altFine = (altered?.fine ?? []).filter((f) => f.n === ZOOM_N);
  const shifts = [...realFine, ...altFine].map((f) => f.shift.value * 1e6);
  const lo = Math.min(0, ...shifts);
  const hi = Math.max(0, ...shifts);
  const pad = (hi - lo || 1) * 0.2;
  const yz = scaleLinear([lo - pad, hi + pad], [H - 60, 90]);
  const columns = [
    { x: 470, rows: realFine, label: "real", cf: false },
    { x: 590, rows: altFine, label: "altered", cf: true },
  ];

  const errFrac = fineErrorFraction(altered?.fine ?? null);
  const splitUeH = shellSplitting(
    (altered?.fine ?? []).map((f) => ({ ...f, shift_ev: f.shift })),
    ZOOM_N,
  ) * 1e6;

  const changed = readouts.filter((r) => r.obs.changed).map((r) => r.label.split(" ")[0]);
  const caption = (() => {
    if (beyondValidity) {
      return `Derived α = ${formatAlpha(alphaValue)} exceeds 0.5, the perturbative fine structure is meaningless here, so the altered split isn't drawn. The readouts still show the true α; this is the honest model boundary, not a glitch.`;
    }
    if (altOn && changed.length === 0) {
      return "You altered the constants, but α, a₀, and E_h are all unchanged, a different universe that is observationally identical to ours. That degeneracy is the whole lesson: only dimensionless combinations and fixed-ruler scales are observable.";
    }
    if (altOn) {
      return `Altered. Observably changed: ${changed.join(", ")}. Fine-structure fractional error ≈ ${(errFrac * 100).toFixed(1)}% (grows as (Zα)²); n=${ZOOM_N} split ≈ ${splitUeH.toFixed(1)} µE_h. The gross ladder is α-independent structure; absolute size and binding are in the readouts.`;
    }
    return "Drag any raw constant. α, a₀, and E_h are derived from all five, only these dimensionless and fixed-ruler quantities are observable. Watch which actually move: try e ×2 and ε₀ ×4 together.";
  })();

  return (
    <div className="view-wrap">
      <div className="view-header">
        <span className="plot-title">
          What-If: fundamental constants{" "}
          <Badge provenance={report.alpha.quantity.provenance} />
        </span>
      </div>

      {altOn && (
        <div className="counterfactual-banner">
          COUNTERFACTUAL · derived α = {formatAlpha(alphaValue)}
        </div>
      )}

      <dl className="readouts">
        {readouts.map((r) => (
          <div key={r.key} className="readout-row">
            <dt>
              {r.label} <Badge provenance={r.obs.quantity.provenance} />
            </dt>
            <dd>
              {r.text}{" "}
              <span className={r.obs.changed ? "readout-ratio changed" : "readout-ratio"}>
                {formatRatio(r.obs.ratio)}
              </span>
            </dd>
          </div>
        ))}
      </dl>

      <svg viewBox={`0 0 ${W} ${H}`} role="img" className="levels-svg">
        <text x={(rx1 + rx2) / 2} y={30} textAnchor="middle" className="tick">
          gross levels (Z={real.system.z}), structure in units of E_h, α-independent
        </text>
        {real.gross.map((g) => (
          <g key={g.n}>
            <line x1={rx1} x2={rx2} y1={y(g.energy.value)} y2={y(g.energy.value)} className="rung" />
            <text x={rx1 - 8} y={y(g.energy.value)} dy="0.32em" textAnchor="end" className="tick">
              n={g.n}
            </text>
            <text x={rx2 + 8} y={y(g.energy.value)} dy="0.32em" className="tick">
              2n²={g.degeneracy}
            </text>
          </g>
        ))}

        <text x={530} y={54} textAnchor="middle" className="tick">
          n={ZOOM_N} fine split [µE_h], real vs altered
        </text>
        {beyondValidity ? (
          <text x={530} y={H / 2} textAnchor="middle" className="tick">
            α &gt; 0.5, beyond perturbative validity
          </text>
        ) : (
          columns.map((col) => (
            <g key={col.label}>
              <text x={col.x + 20} y={78} textAnchor="middle" className="tick">
                {col.label}
              </text>
              {col.rows.map((f) => (
                <g key={`${col.label}-${f.l}-${f.j}`}>
                  <line
                    x1={col.x}
                    x2={col.x + 40}
                    y1={yz(f.shift.value * 1e6)}
                    y2={yz(f.shift.value * 1e6)}
                    className={col.cf && altOn ? "rung rung-counterfactual" : "rung"}
                  />
                  <text x={col.x + 46} y={yz(f.shift.value * 1e6)} dy="0.32em" className="tick">
                    j={f.j} · {(f.shift.value * 1e6).toFixed(1)}
                  </text>
                </g>
              ))}
            </g>
          ))
        )}
      </svg>

      <div className="const-sliders">
        {CONSTANT_KEYS.map((k) => (
          <label key={k}>
            <span>
              {CONSTANT_LABELS[k]} ×{labConst[k].toFixed(2)}
            </span>
            <input
              type="range"
              min={Math.log2(CONST_MIN)}
              max={Math.log2(CONST_MAX)}
              step={0.25}
              value={Math.log2(labConst[k])}
              onChange={(e) =>
                setLabConst({ [k]: 2 ** Number(e.target.value) } as Partial<ConstMultipliers>)
              }
            />
          </label>
        ))}
      </div>

      <div className="whatif-controls">
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
        <button type="button" className="primary" onClick={() => setLabConst(REAL_ALL)}>
          reset to real constants
        </button>
      </div>

      <p className={beyondValidity ? "error" : "caption"}>{caption}</p>
    </div>
  );
}
```

- [ ] **Step 2: Append the lab styles to `index.css`**

Append to `web/src/index.css`:

```css
/* What-If five-constant panel */
.const-sliders {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px 20px;
  margin-top: 10px;
}
.const-sliders label {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.const-sliders label span {
  font-variant-numeric: tabular-nums;
}
.readout-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}
.readout-ratio {
  color: #7c8798;
  font-variant-numeric: tabular-nums;
}
.readout-ratio.changed {
  color: #f472b6;
  font-weight: 600;
}
```

- [ ] **Step 3: Verify it type-checks and builds**

Run: `cd web && npm run build`
Expected: PASS, `tsc --noEmit` clean (all `labAlpha` references gone across store, urlState, main, view), `vite build` emits `dist/`.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/WhatIfView.tsx web/src/index.css
git commit -m "feat(web): five-constant What-If panel with degeneracy readouts"
```

---

### Task 7: Ship: full gates + docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Full Python suite**

Run: `pytest`
Expected: PASS (all green, incl. the NIST comparison).

- [ ] **Step 2: Full web suite + build**

Run: `cd web && npm test && npm run build`
Expected: PASS, vitest green, `tsc --noEmit` clean, `dist/` built.

- [ ] **Step 3: Lint**

Run: `ruff check .`
Expected: PASS (no findings).

- [ ] **Step 4: Manual QA**

Run: `atomsim serve` and in the browser:
- View → "What-If: constants". Confirm five sliders (ℏ, e, mₑ, ε₀, c), the readout table (α, a₀, E_h with ratios), the gross ladder, and the n=2 real/altered split render.
- Drag **e ×2**, then **ε₀ ×4**: the COUNTERFACTUAL banner appears, but all three readouts read "unchanged", the identical-universe caption shows. This is the headline lesson.
- Drag **mₑ ×0.5**: a₀ readout shows ×2.00, E_h shows ×0.50, α unchanged.
- Crank **e** and **c** to drive derived α past 0.5: the altered column is replaced by the "beyond perturbative validity" notice; the α readout still shows the true (large) value.
- Step Z up: the n=2 split grows.
- Load `http://127.0.0.1:8000/?view=whatif&e=2&eps0=4&z=3` directly and confirm the lab opens in that state.

- [ ] **Step 5: Update the README**

In `README.md`, replace the existing What-If constants-lab bullet (the α-slider description) with one describing the five-constant panel: raw ℏ, e, mₑ, ε₀, c multiplier sliders → derived α, Bohr radius, and Hartree energy with per-observable change ratios; the degeneracy lesson (e×2 & ε₀×4 → identical atom); COUNTERFACTUAL provenance; and the honest beyond-validity boundary when derived α exceeds 0.5. Keep the "never quietly lies" framing.

- [ ] **Step 6: Commit the docs**

```bash
git add README.md
git commit -m "docs: five-constant What-If panel in feature list"
```

---

## Notes for the implementer

- **No physics in the client.** α, a₀, E_h are all derived server-side in `analyze_constants`; the view only lays out and labels server numbers. The only client-side arithmetic is display scaling (hartree → µE_h via ×1e6, log₂ for the slider), never a physics conversion.
- **The banner and the beyond-validity gate both key off `report.altered`** (the authoritative server flag) and `altered === null`, never off comparing echoed α values, that avoids float-noise false positives when multipliers are all 1.
- **The diagram is normalized** (units of E_h = hartree `energy.value` / `shift.value`), per the structure/scale honesty decision. Absolute size (pm) and binding (eV) appear only in the readouts.
- **Keep the lab slice out of `INVALIDATED`**, changing (n, l, m, system) must not disturb the lab, and vice versa. The store tests guard this.
- **`?alpha=` is intentionally dropped**, α is no longer a direct input. Old alpha deep-links silently fall back to defaults; acceptable pre-v1.
```