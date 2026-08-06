# Classical Ghost Overlay Implementation Plan

**Goal:** Overlay an animated classical "ghost" (Bohr orbits + Larmor radiative collapse) on the 3D |ψ|² cloud, showing why a classical atom would die in picoseconds, under an unmissable COUNTERFACTUAL banner.

**Architecture:** New engine module `classical.py` computes Bohr radii (APPROXIMATION) and the Larmor collapse time / orbit count (COUNTERFACTUAL) as provenance-carrying `Quantity`s. A new `GET /api/classical` endpoint surfaces them via the existing Quantity→Pydantic mapping. The frontend adds a pure trajectory/format helper (`lib/classical.ts`), a `classicalGhost` store slice + `ghost` toggle (invalidates nothing physical, like `nucleusMode`), a deep-link param, and a Three.js overlay inside CloudView that animates the collapse on the existing `useFrame` loop.

**Tech Stack:** Python 3.12 (Hartree atomic units internally), FastAPI + Pydantic, React + TypeScript, Zustand, @react-three/fiber / three, vitest, pytest.

## Global Constraints

- Prime directive: every value crossing a module boundary is a `Quantity`/`Field` with `Provenance` stating its `Fidelity` tier. No bare float, silent zero, or undisclosed liberty across a boundary.
- Engine math is Hartree atomic units; SI/display conversions (ps, pm) happen at the server boundary and are appended to the provenance `method`.
- Fidelity tiers used here: Bohr orbit geometry = `APPROXIMATION`; Larmor collapse time / orbit count / period = `COUNTERFACTUAL`; animation slow-motion factor = `VISUAL_LIBERTY` (disclosed in HUD).
- Ruff line-length 100, E741 ignored. Python: `C:\Users\yashg\.conda\envs\atomsim\python.exe -m pytest` with `C:\Users\yashg\.conda\envs\atomsim\Library\bin` prepended to PATH (no conda on PATH).
- Frontend from `web/`: `npx vitest run`, `npm run build` (tsc --noEmit + vite). Rebuild `web/dist` before serving.
- Commit messages are written in plain prose, one logical change each.
- New physics gets a validation test (analytic anchor), not a smoke test.

**Physics validation anchors (hydrogen, n=1, Z=1):**
- `t_collapse = 1.556e-11 s` (Larmor radiative collapse from the Bohr radius).
- `N_orbits ≈ 2.05e5` (revolutions before collapse; formula `ω₀·t_collapse/π`).
- `r_n = n²·a₀_sys/Z` (Bohr orbit radius; `a₀_sys = a₀ / mu_ratio` respects reduced mass).

**Closed forms (SI):**
```
r_n         = n² · a0_sys / Z                         # a0_sys = a0 / mu_ratio (reduced mass)
t_collapse  = r0³ · m² · c³ / (4 · Z · k² · e⁴)       # k = 1/(4πε₀);  = r0³ / (4·Z·r_e²·c)
r(τ)        = r0 · (1 − τ)^(1/3)                       # τ = t / t_collapse ∈ [0,1]
θ(τ)        = 2π·N_orbits · (1 − √(1 − τ))             # total swept angle 2π·N_orbits
N_orbits    = ω₀ · t_collapse / π                      # ω₀ = v₀/r0, v₀ = √(Z·k·e²/(m·r0))
```
`m` is the orbiting particle's mass (`mu_ratio · m_e`), `Z = system.Z`, `mu_ratio = system.mu_ratio.value`.

---

### Task 1: Engine: `classical.py` (Bohr radii + Larmor collapse)

**Files:**
- Create: `src/atomsim/classical.py`
- Test: `tests/test_classical.py`

**Pattern to mirror:** `src/atomsim/constants_lab.py` (dataclass report of provenance-carrying quantities) and `src/atomsim/systems.py` / `src/atomsim/constants.py` for pulling `Z`, `mu_ratio`, and SI constants. Read those before writing.

**Interfaces:**
- Consumes: `atomsim.provenance.{Fidelity, Provenance, Quantity}`; `atomsim.systems.get_system(key) -> System` with `.Z: int` and `.mu_ratio: Quantity` (unit "m_e"); `atomsim.constants.FundamentalConstants.codata()` for SI `e, m_e, c, eps0, hbar` and `.bohr_radius` (meters).
- Produces:
  - `@dataclass(frozen=True) BohrOrbit: n: int; radius_bohr: Quantity; radius_pm: Quantity` (both `APPROXIMATION`)
  - `@dataclass(frozen=True) ClassicalGhost: n: int; system_key: str; z: int; orbits: tuple[BohrOrbit, ...]; r0_bohr: Quantity; collapse_time_s: Quantity; orbital_period_s: Quantity; orbit_count: Quantity` (last three `COUNTERFACTUAL`)
  - `classical_ghost(n: int, system: str = "h") -> ClassicalGhost`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_classical.py
import math
import pytest
from atomsim.classical import classical_ghost, BohrOrbit, ClassicalGhost
from atomsim.provenance import Fidelity

def test_hydrogen_ground_collapse_time_matches_literature():
    g = classical_ghost(n=1, system="h")
    # Larmor radiative collapse from the Bohr radius ~ 1.56e-11 s (textbook).
    assert g.collapse_time_s.value == pytest.approx(1.556e-11, rel=0.02)
    assert g.collapse_time_s.unit == "s"
    assert g.collapse_time_s.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_hydrogen_ground_orbit_count_matches_literature():
    g = classical_ghost(n=1, system="h")
    assert g.orbit_count.value == pytest.approx(2.05e5, rel=0.03)
    assert g.orbit_count.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_bohr_radius_scales_as_n_squared_over_z():
    g = classical_ghost(n=3, system="h")
    # orbits cover n'=1..n, current n last; radius_bohr = n'^2 / Z (H: Z=1, mu~1)
    assert tuple(o.n for o in g.orbits) == (1, 2, 3)
    r1 = g.orbits[0].radius_bohr.value
    r3 = g.orbits[2].radius_bohr.value
    assert r3 / r1 == pytest.approx(9.0, rel=1e-6)
    assert g.orbits[0].radius_bohr.provenance.fidelity is Fidelity.APPROXIMATION

def test_r0_is_current_n_radius():
    g = classical_ghost(n=2, system="h")
    assert g.r0_bohr.value == pytest.approx(g.orbits[-1].radius_bohr.value, rel=1e-12)

def test_higher_z_collapses_faster():
    h = classical_ghost(n=1, system="h")
    he = classical_ghost(n=1, system="he+")   # Z=2
    assert he.collapse_time_s.value < h.collapse_time_s.value

def test_muonic_hydrogen_smaller_and_faster():
    h = classical_ghost(n=1, system="h")
    mu = classical_ghost(n=1, system="mu-h")   # reduced mass ~186 m_e
    assert mu.r0_bohr.value < h.r0_bohr.value
    assert mu.collapse_time_s.value < h.collapse_time_s.value
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_classical.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.classical'`.
(System keys are lowercase and case-sensitive: `h`, `d`, `t`, `mu-h`, `ps`, `he+`. `get_system` does NOT lowercase input, so use these exact keys.)

- [ ] **Step 3: Implement `classical.py`**

```python
"""Classical ghost: what classical electrodynamics predicts for a one-electron atom.

A classical orbiting electron is an accelerating charge; by the Larmor formula it
radiates, loses energy, and spirals into the nucleus in picoseconds. That is the
COUNTERFACTUAL truth of classical rules, computed exactly. The circular Bohr orbits
are a semi-classical APPROXIMATION (right energy scale, wrong picture). Quantum
mechanics forbids the collapse, which is the whole lesson.
"""

import math
from dataclasses import dataclass

from atomsim.constants import FundamentalConstants
from atomsim.provenance import Fidelity, Provenance, Quantity
from atomsim.systems import get_system

_APPROX = "Bohr model r_n = n^2 a0 / Z (semi-classical circular orbit)"
_LARMOR = "Larmor radiative collapse, classical E&M, exact under classical rules"
_ASSUME = ("classical electrodynamics (deliberately non-quantum)",)

@dataclass(frozen=True)
class BohrOrbit:
    n: int
    radius_bohr: Quantity   # for drawing in the cloud's Bohr units
    radius_pm: Quantity     # for display

@dataclass(frozen=True)
class ClassicalGhost:
    n: int
    system_key: str
    z: int
    orbits: tuple[BohrOrbit, ...]
    r0_bohr: Quantity
    collapse_time_s: Quantity
    orbital_period_s: Quantity
    orbit_count: Quantity

def _bohr_orbit(n: int, a0_sys_m: float, a0_m: float, z: int) -> BohrOrbit:
    r_m = n * n * a0_sys_m / z
    prov = Provenance(fidelity=Fidelity.APPROXIMATION, method=_APPROX, assumptions=_ASSUME)
    return BohrOrbit(
        n=n,
        radius_bohr=Quantity(r_m / a0_m, "bohr", f"Bohr orbit n={n}", prov),
        radius_pm=Quantity(r_m * 1e12, "pm", f"Bohr orbit n={n}", prov),
    )

def classical_ghost(n: int, system: str = "h") -> ClassicalGhost:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    sys = get_system(system)
    z = sys.Z
    c = FundamentalConstants.codata()
    m = sys.mu_ratio.value * c.m_e                 # orbiting reduced mass (kg)
    a0_m = c.bohr_radius                           # real-electron Bohr radius (m)
    a0_sys_m = a0_m / sys.mu_ratio.value           # reduced-mass Bohr radius (m)
    k = 1.0 / (4.0 * math.pi * c.eps0)

    orbits = tuple(_bohr_orbit(i, a0_sys_m, a0_m, z) for i in range(1, n + 1))
    r0_m = n * n * a0_sys_m / z

    # Larmor closed form: t = r0^3 m^2 c^3 / (4 Z k^2 e^4)
    t_collapse = (r0_m**3 * m**2 * c.c**3) / (4.0 * z * k**2 * c.e**4)
    # initial circular speed and angular velocity
    v0 = math.sqrt(z * k * c.e**2 / (m * r0_m))
    omega0 = v0 / r0_m
    period = 2.0 * math.pi / omega0
    n_orbits = omega0 * t_collapse / math.pi

    cf = Provenance(fidelity=Fidelity.COUNTERFACTUAL, method=_LARMOR, assumptions=_ASSUME)
    return ClassicalGhost(
        n=n,
        system_key=sys.key,
        z=z,
        orbits=orbits,
        r0_bohr=Quantity(r0_m / a0_m, "bohr", "classical start radius", cf),
        collapse_time_s=Quantity(t_collapse, "s", "classical collapse time", cf),
        orbital_period_s=Quantity(period, "s", "initial orbital period", cf),
        orbit_count=Quantity(n_orbits, "", "orbits before collapse", cf),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_classical.py -v`
Expected: PASS (all 6). If an anchor is off, recheck unit consistency (SI throughout), do NOT loosen the tolerance to hide a real error.

- [ ] **Step 5: Lint**

Run: `ruff check src/atomsim/classical.py tests/test_classical.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/atomsim/classical.py tests/test_classical.py
git commit -m "Add the classical-ghost engine (Bohr orbits + Larmor collapse)"
```

---

### Task 2: Server: `GET /api/classical`

**Files:**
- Modify: `src/atomsim/server/schemas.py` (add models + mapper)
- Modify: `src/atomsim/server/app.py` (add endpoint)
- Test: `tests/test_server.py` (append)

**Pattern to mirror:** the Phase-2 `/api/constants` endpoint, find it with `grep -n "api/constants" src/atomsim/server/app.py` and its `ConstantsReportModel` / `DerivedObservableModel` in `schemas.py`. Copy that structure exactly (including how a `Quantity` is mapped to its Pydantic model with `value/unit/label/provenance`).

**Interfaces:**
- Consumes: `atomsim.classical.{classical_ghost, ClassicalGhost, BohrOrbit}`; the existing `QuantityModel` (or equivalently named Quantity→Pydantic model already in `schemas.py`, grep for how `/api/constants` maps a `Quantity`).
- Produces: `GET /api/classical?system={key}&n={int}` → JSON `ClassicalGhostModel`:
  ```
  { n, system_key, z,
    orbits: [{ n, radius_bohr: QuantityModel, radius_pm: QuantityModel }],
    r0_bohr: QuantityModel, collapse_time_s: QuantityModel,
    orbital_period_s: QuantityModel, orbit_count: QuantityModel }
  ```
  Query validation: `n >= 1` (422 otherwise), unknown `system` → 404/422 consistent with `/api/constants` or `/api/levels` behavior for a bad system (check which and match it).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_server.py`)

```python
def test_classical_hydrogen_ground(client):
    r = client.get("/api/classical?system=h&n=1")
    assert r.status_code == 200
    body = r.json()
    assert body["z"] == 1
    assert body["collapse_time_s"]["value"] == pytest.approx(1.556e-11, rel=0.02)
    assert body["collapse_time_s"]["provenance"]["fidelity"] == "counterfactual"
    assert body["orbits"][0]["radius_bohr"]["provenance"]["fidelity"] == "approximation"

def test_classical_orbits_cover_1_to_n(client):
    body = client.get("/api/classical?system=h&n=3").json()
    assert [o["n"] for o in body["orbits"]] == [1, 2, 3]

def test_classical_rejects_n_below_one(client):
    assert client.get("/api/classical?system=h&n=0").status_code == 422
```
(Match `client` fixture usage and the `pytest.approx` import to the existing patterns already in `tests/test_server.py`.)

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_server.py -k classical -v`
Expected: FAIL (404, route not defined).

- [ ] **Step 3: Add Pydantic models to `schemas.py`**

Add `BohrOrbitModel` and `ClassicalGhostModel` plus a `classical_ghost_to_model(g: ClassicalGhost) -> ClassicalGhostModel` mapper, reusing the SAME `Quantity`→model helper `/api/constants` uses (do not invent a new one). Each `Quantity` maps to the existing quantity model with `value, unit, label, provenance{fidelity, method, assumptions, error_estimate, refinement}`, mirror `constants_report_to_model` exactly.

- [ ] **Step 4: Add the endpoint to `app.py`**

Mirror the `/api/constants` route. Sketch:
```python
@app.get("/api/classical", response_model=ClassicalGhostModel)
def get_classical(system: str = "h", n: int = Query(ge=1)) -> ClassicalGhostModel:
    return classical_ghost_to_model(classical_ghost(n=n, system=system))
```
Add imports for `classical_ghost` and the new schema symbols. Use the same `Query(ge=1)` validation style already present in the file (grep `Query(` to match import + usage).

- [ ] **Step 5: Run the new tests + full server suite**

Run: `python -m pytest tests/test_server.py -k classical -v` → PASS
Run: `python -m pytest tests/test_server.py -q` → all pass (no regressions)

- [ ] **Step 6: Lint**

Run: `ruff check src/atomsim/server/schemas.py src/atomsim/server/app.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/atomsim/server/schemas.py src/atomsim/server/app.py tests/test_server.py
git commit -m "Add the /api/classical endpoint"
```

---

### Task 3: Client: `lib/classical.ts` (trajectory + format) + types + client fn

**Files:**
- Create: `web/src/lib/classical.ts`
- Create: `web/src/lib/classical.test.ts`
- Modify: `web/src/api/types.ts` (add `ClassicalGhost`, `BohrOrbit` types)
- Modify: `web/src/api/client.ts` (add `getClassical`)

**Pattern to mirror:** `web/src/lib/whatif.ts` (pure helpers + vitest), `web/src/api/types.ts` existing `Quantity`/response types, and the `getConstants` function in `web/src/api/client.ts`.

**Interfaces:**
- Consumes: existing `Quantity` TS type in `api/types.ts` (`{ value; unit; label; provenance }`). `getJson` in `client.ts`.
- Produces (types.ts):
  ```ts
  export interface BohrOrbit { n: number; radius_bohr: Quantity; radius_pm: Quantity; }
  export interface ClassicalGhost {
    n: number; system_key: string; z: number; orbits: BohrOrbit[];
    r0_bohr: Quantity; collapse_time_s: Quantity;
    orbital_period_s: Quantity; orbit_count: Quantity;
  }
  ```
- Produces (client.ts): `getClassical(system: string, n: number): Promise<ClassicalGhost>` →
  `getJson(\`/api/classical?system=${system}&n=${n}\`)`.
- Produces (classical.ts): pure functions below.

- [ ] **Step 1: Write the failing tests**

```ts
// web/src/lib/classical.test.ts
import { describe, expect, it } from "vitest";
import { ghostRadius, ghostAngle, slowMotionFactor, tauFromWall, formatSeconds } from "./classical";

describe("classical trajectory law", () => {
  it("radius follows r0*(1-tau)^(1/3), full at tau=0, zero at tau=1", () => {
    expect(ghostRadius(0, 2)).toBeCloseTo(2, 12);
    expect(ghostRadius(1, 2)).toBeCloseTo(0, 12);
    expect(ghostRadius(0.5, 1)).toBeCloseTo(Math.cbrt(0.5), 12);
  });
  it("angle sweeps 0 to 2*pi*N over the collapse", () => {
    expect(ghostAngle(0, 3)).toBeCloseTo(0, 12);
    expect(ghostAngle(1, 3)).toBeCloseTo(2 * Math.PI * 3, 12);
  });
  it("slow-motion factor is wall-duration over real collapse time", () => {
    expect(slowMotionFactor(1.556e-11, 5)).toBeCloseTo(5 / 1.556e-11, 3);
  });
  it("tau loops in [0,1) from wall time", () => {
    expect(tauFromWall(0, 5)).toBeCloseTo(0, 12);
    expect(tauFromWall(2.5, 5)).toBeCloseTo(0.5, 12);
    expect(tauFromWall(7.5, 5)).toBeCloseTo(0.5, 12); // wrapped
  });
  it("formats seconds into readable ps/fs", () => {
    expect(formatSeconds(1.556e-11)).toMatch(/ps/);
    expect(formatSeconds(1.5e-15)).toMatch(/fs/);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run (from `web/`): `npx vitest run src/lib/classical.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `lib/classical.ts`**

```ts
// Pure classical-ghost trajectory + formatting helpers. No three.js, no store.
// Trajectory law (see plan physics): r(tau)=r0*(1-tau)^(1/3),
// theta(tau)=2*pi*N*(1-sqrt(1-tau)), tau=t/t_collapse looping in [0,1).

export function ghostRadius(tau: number, r0: number): number {
  return r0 * Math.cbrt(1 - tau);
}

export function ghostAngle(tau: number, nOrbits: number): number {
  return 2 * Math.PI * nOrbits * (1 - Math.sqrt(1 - tau));
}

/** How many wall-clock seconds we stretch one real collapse over -> the slow-mo factor. */
export function slowMotionFactor(collapseSeconds: number, wallSeconds = 5): number {
  return wallSeconds / collapseSeconds;
}

/** Loop tau in [0,1) from accumulated wall time. */
export function tauFromWall(wallElapsed: number, wallSeconds = 5): number {
  return (wallElapsed % wallSeconds) / wallSeconds;
}

/** Simulated (real) elapsed time in seconds for a given tau. */
export function simSeconds(tau: number, collapseSeconds: number): number {
  return tau * collapseSeconds;
}

export function formatSeconds(s: number): string {
  if (s >= 1e-9) return `${(s * 1e12).toFixed(1)} ps`;
  if (s >= 1e-12) return `${(s * 1e12).toFixed(2)} ps`;
  if (s >= 1e-15) return `${(s * 1e15).toFixed(1)} fs`;
  return `${s.toExponential(2)} s`;
}
```

- [ ] **Step 4: Add TS types + client fn**

Add the `BohrOrbit`/`ClassicalGhost` interfaces to `api/types.ts` and `getClassical` to `api/client.ts` (signatures in Interfaces above). Import `Quantity` from wherever `types.ts` already defines it.

- [ ] **Step 5: Run tests + typecheck**

Run (from `web/`): `npx vitest run src/lib/classical.test.ts` → PASS
Run: `npm run build` → tsc clean (types compile).

- [ ] **Step 6: Commit**

```bash
git add web/src/lib/classical.ts web/src/lib/classical.test.ts web/src/api/types.ts web/src/api/client.ts
git commit -m "Add classical-ghost frontend helpers, types and API client"
```

---

### Task 4: Store: `classicalGhost` slice + `ghost` toggle

**Files:**
- Modify: `web/src/state/store.ts`
- Modify: `web/src/state/store.test.ts` (append)

**Pattern to mirror:** the `nucleusMode` presentational toggle (invalidates nothing) AND the `whatif`/`labConst` lab slice loader (`loadWhatIf`), both in `store.ts`. The classical ghost is presentational-adjacent but data-backed: the `ghost` boolean invalidates nothing physical, and `classicalGhost` data is (re)loaded on demand.

**Interfaces:**
- Consumes: `getClassical` from `api/client`, `ClassicalGhost` from `api/types`.
- Produces (store shape additions):
  ```ts
  ghost: boolean;                          // overlay on/off (default false)
  classicalGhost: ClassicalGhost | null;   // loaded data
  classicalStatus: SampleStatus;           // "idle" | "sampling" | "ready" | "error"
  setGhost: (on: boolean) => void;         // toggles overlay; if turning on and data stale, triggers load
  loadClassical: () => Promise<void>;      // fetches getClassical(system, n)
  ```
- Invariant: `ghost` and `classicalGhost` are NEVER in `INVALIDATED`. BUT the classical data depends on `(n, system)`, so `setQuantumNumbers` and `setSystem` must reset `classicalGhost: null, classicalStatus: "idle"` (so stale ghost physics can't render) WITHOUT being part of the presentational `INVALIDATED` spread, add these resets explicitly in those two setters, next to the `...INVALIDATED` spread.

- [ ] **Step 1: Write the failing tests** (append to `store.test.ts`)

```ts
it("ghost toggle is off by default and flips without touching physics fields", () => {
  const before = useAppStore.getState().positions;
  expect(useAppStore.getState().ghost).toBe(false);
  useAppStore.getState().setGhost(true);
  expect(useAppStore.getState().ghost).toBe(true);
  // presentational: sampled positions untouched by the toggle
  expect(useAppStore.getState().positions).toBe(before);
});

it("changing n or system clears loaded classical data (no stale ghost)", () => {
  useAppStore.setState({ classicalGhost: { n: 1 } as never, classicalStatus: "ready" });
  useAppStore.getState().setQuantumNumbers(2, 0, 0);
  expect(useAppStore.getState().classicalGhost).toBeNull();
  expect(useAppStore.getState().classicalStatus).toBe("idle");
});

it("changing system clears loaded classical data", () => {
  useAppStore.setState({ classicalGhost: { n: 1 } as never, classicalStatus: "ready" });
  useAppStore.getState().setSystem("he+");
  expect(useAppStore.getState().classicalGhost).toBeNull();
});
```
(Match the existing `store.test.ts` conventions, how it imports `useAppStore`, resets state between tests, and the `SampleStatus` values.)

- [ ] **Step 2: Run to verify failure**

Run (from `web/`): `npx vitest run src/state/store.test.ts`
Expected: FAIL (`ghost`/`setGhost` undefined).

- [ ] **Step 3: Implement the slice**

Add fields to the store interface and initial state (`ghost: false, classicalGhost: null, classicalStatus: "idle"`). Add:
```ts
setGhost: (on) => {
  set({ ghost: on });
  if (on && get().classicalStatus === "idle") void get().loadClassical();
},
loadClassical: async () => {
  const { n, system } = get();
  set({ classicalStatus: "sampling" });
  try {
    const classicalGhost = await client.getClassical(system, n);
    set({ classicalGhost, classicalStatus: "ready" });
  } catch (err) {
    set({ classicalStatus: "error", error: err instanceof Error ? err.message : String(err) });
  }
},
```
In `setQuantumNumbers` and `setSystem`, add `classicalGhost: null, classicalStatus: "idle"` to the `set({...})` alongside `...INVALIDATED` (explicit reset, NOT by adding to INVALIDATED).

- [ ] **Step 4: Run tests**

Run (from `web/`): `npx vitest run src/state/store.test.ts` → PASS
Run: `npm run build` → tsc clean.

- [ ] **Step 5: Commit**

```bash
git add web/src/state/store.ts web/src/state/store.test.ts
git commit -m "Add the classical-ghost store slice and toggle"
```

---

### Task 5: URL state: deep-link `?ghost=1`

**Files:**
- Modify: `web/src/lib/urlState.ts`
- Modify: `web/src/lib/urlState.test.ts` (append)

**Pattern to mirror:** how a boolean/presentational field (e.g. `nucleusMode` or `colorMode`) is read from and written to the query in `urlState.ts`. Read the file first, match its exact read/write structure and its round-trip test style.

**Interfaces:**
- Consumes: existing `stateToParams` / `paramsToState` (or equivalently named) functions in `urlState.ts`.
- Produces: `ghost` participates in the URL: `?ghost=1` when `ghost === true`, omitted/`0` when false. Round-trips through the existing encode→decode pair.

- [ ] **Step 1: Write the failing test** (append to `urlState.test.ts`)

```ts
it("round-trips the ghost toggle", () => {
  const withGhost = paramsToState(new URLSearchParams(stateToParams({ ...baseState, ghost: true })));
  expect(withGhost.ghost).toBe(true);
  const withoutGhost = paramsToState(new URLSearchParams(stateToParams({ ...baseState, ghost: false })));
  expect(withoutGhost.ghost).toBe(false);
});
```
(Use whatever the file names its encode/decode functions and its existing `baseState`/fixture, mirror the neighbouring round-trip tests exactly.)

- [ ] **Step 2: Run to verify failure**

Run (from `web/`): `npx vitest run src/lib/urlState.test.ts`
Expected: FAIL (`ghost` not in decoded state).

- [ ] **Step 3: Implement**

In the encoder: `if (state.ghost) params.set("ghost", "1");`. In the decoder: `ghost: params.get("ghost") === "1"`. Place alongside the other presentational params, matching the file's style.

- [ ] **Step 4: Run tests**

Run (from `web/`): `npx vitest run src/lib/urlState.test.ts` → PASS
Run: `npm run build` → tsc clean.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/urlState.ts web/src/lib/urlState.test.ts
git commit -m "Deep-link the classical ghost toggle in the URL"
```

---

### Task 6: CloudView: Three.js ghost overlay + HUD + toggle UI

**Files:**
- Create: `web/src/components/GhostOverlay.tsx` (the in-Canvas three.js overlay)
- Modify: `web/src/components/CloudView.tsx` (mount overlay + HUD + toggle)
- Modify: `web/src/index.css` (append ghost HUD styles)
- Modify: `web/src/lib/liberties.ts` (add a disclosed `CLASSICAL_SLOWMO` / ghost liberty, mirroring `NUCLEUS_MARKER_LIBERTY`)

**Pattern to mirror:** `CloudView.tsx` (read it fully, it uses `@react-three/fiber` `<Canvas>`, `useFrame`, `PointCloud`, and a `.canvas-overlay` HUD with `Badge`s). `lib/liberties.ts` for how a disclosed liberty `Provenance` is defined and rendered via `Badge`.

**Interfaces:**
- Consumes: store `ghost`, `classicalGhost`, `setGhost`, `loadClassical`; helpers from `lib/classical.ts` (`ghostRadius, ghostAngle, slowMotionFactor, tauFromWall, simSeconds, formatSeconds`); `Badge`; the cloud's `distance` scale (already computed in CloudView, same Bohr units as `radius_bohr`).
- Produces: `GhostOverlay` renders nothing when `ghost` is false or `classicalGhost` is null. When active: Bohr rings (n′=1..current, current highlighted), the spiral trajectory line, and one animated ghost point advancing via `useFrame`, all in the equatorial (xz) plane, radii = `orbit.radius_bohr.value`. A HUD block in `.canvas-overlay` shows: COUNTERFACTUAL banner, live simulated clock (`formatSeconds(simSeconds(tau, collapse))` counting up, resetting each loop), collapse time, orbit count, and the slow-motion disclosure `Badge`.

**Behavior spec (build against the real CloudView after reading it):**

- [ ] **Step 1: Add the ghost liberty to `lib/liberties.ts`**

Define (mirroring `NUCLEUS_MARKER_LIBERTY`):
```ts
export const CLASSICAL_SLOWMO = provenance-of(Fidelity.VISUAL_LIBERTY,
  "classical collapse shown in slow motion; the live clock shows real simulated time",
  assumptions: ["playback speed is a viewing choice, not physics"]);
```
Use the file's existing factory/shape, do not hand-roll a new Provenance format.

- [ ] **Step 2: Implement `GhostOverlay.tsx`**

A component taking `{ ghost: ClassicalGhost; distance: number }`. Compute `slowmo` and, on `useFrame`, accumulate wall time (`state.clock.elapsedTime`), derive `tau = tauFromWall(elapsed, 5)`, set the ghost point position to polar `(ghostRadius(tau, r0), ghostAngle(tau, nOrbits))` in the xz-plane. Draw:
- Rings: one `lineLoop`/`ringGeometry` per orbit at `radius_bohr.value`; highlight the last (current n) with a brighter classical color (e.g. `#8be9fd`), dimmer for inner rings.
- Spiral: a `bufferGeometry` line sampling `~256` points of `(ghostRadius(τ), ghostAngle(τ))` for τ in [0, 0.999].
- Ghost point: a small emissive sphere/points at the current position.
Expose the live `tau`/sim-time to the HUD via a store field or a `useRef` + `onFrame` callback prop (simplest: lift a `setGhostClock(simSeconds)` throttled to ~10Hz into the store, or keep a ref the HUD reads, choose the lower-churn option; do NOT re-render React every frame).

- [ ] **Step 3: Wire into `CloudView.tsx`**

Pull `ghost, classicalGhost` from the store. Inside `<Canvas>`, after `PointCloud`, render `{ghost && classicalGhost && <GhostOverlay ghost={classicalGhost} distance={distance} />}`. In `.canvas-overlay`, add: a labeled checkbox/toggle bound to `setGhost` (near the existing controls), and, when `ghost && classicalGhost`, the HUD block (COUNTERFACTUAL banner, `formatSeconds` clock, `collapse_time_s`/`orbit_count` readouts with `Badge`s, and `<Badge provenance={CLASSICAL_SLOWMO} />`). Ensure toggling `ghost` on triggers `loadClassical` (via `setGhost`) and that turning it off stops all per-frame work (component unmounts).

- [ ] **Step 4: Append CSS to `index.css`**

Add `.ghost-hud` (positioned in the overlay), `.ghost-banner` (COUNTERFACTUAL, high-contrast, mirrors `.counterfactual-banner` if present), `.ghost-clock` (tabular-nums), and `.ghost-readout` styles. Reuse existing overlay/badge variables.

- [ ] **Step 5: Build + typecheck**

Run (from `web/`): `npm run build`
Expected: tsc --noEmit clean, vite build emits `dist/`.

- [ ] **Step 6: Full frontend test suite (no regressions)**

Run (from `web/`): `npx vitest run`
Expected: all prior tests + the new classical/store/url tests pass.

- [ ] **Step 7: Commit**

```bash
git add web/src/components/GhostOverlay.tsx web/src/components/CloudView.tsx web/src/index.css web/src/lib/liberties.ts
git commit -m "Add the animated classical ghost overlay to the 3D cloud"
```

---

## Final: live QA (after Task 6)

Rebuild `web/dist` (`npm run build`), start the server (`atomsim serve --port 8011 --no-browser` via the env python), and verify end-to-end:
- `GET /api/classical?system=h&n=1` → collapse ≈ 1.556e-11 s, orbit ≈ 2.05e5, correct fidelity tiers.
- Toggle ghost on in CloudView: rings + spiral + animated point appear; clock counts real ps and loops; COUNTERFACTUAL + slow-mo badges show; toggling off removes the overlay.
- Higher Z / muonic H collapse visibly faster (smaller r0, shorter clock).
- Deep link `?ghost=1` restores the overlay on load.

## Notes for the executor
- System keys are lowercase/case-sensitive: `h`, `d`, `t`, `mu-h`, `ps`, `he+` (get_system does not lowercase). The endpoint may also receive `z{N}` generic-ion keys from app state, reuse the app's existing system resolver (grep the `get_system`/resolver used by `/api/levels` around app.py:259) so `z{N}` works, rather than calling `get_system` blindly.
- Do NOT add `ghost`/`classicalGhost` to `INVALIDATED`; reset classical data explicitly in `setQuantumNumbers`/`setSystem` only.
- Radii for drawing are in Bohr (`radius_bohr`), matching the cloud's units; ps/pm are display-only.
- Never render React every animation frame, drive the point via three.js refs, throttle any HUD clock updates.
