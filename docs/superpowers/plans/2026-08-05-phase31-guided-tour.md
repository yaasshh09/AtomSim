# Phase 31: Guided Tour Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the flagship guided tour "The Hydrogen Atom, Honestly" plus three short side tours, driven by tour content that is data shared between the browser and pytest, so a step that quotes a number fails CI the day that number stops being true.

**Architecture:** Tour content is JSON under `web/src/tours/`. TypeScript imports it and renders a docked card plus a spotlight ring; pytest reads the same files and resolves each declared claim against the real engine. A step names a partial `UrlState`, which is applied as a full reset from `URL_DEFAULTS` through a new store action that also clears every derived-physics field, so a step's picture is reproducible from its own data and can never render the previous step's physics.

**Tech Stack:** TypeScript, React, Zustand, Vite, vitest (node environment, no jsdom); Python 3.12, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-05-phase31-guided-tour-design.md`. Read it before Task 1.
- Engine-internal math is in **Hartree atomic units**. Conversions live at the boundary: `from atomsim.constants import HARTREE_EV, BOHR_RADIUS_PM`.
- **No em dashes** anywhere in tour prose, code comments, or commit messages. Use a comma, a colon, or a full stop.
- `l` is the orbital angular-momentum quantum number, never a length. ruff's E741 is ignored project-wide for this.
- Line length 100 (ruff). Run `ruff check .` from the repo root before every Python commit.
- **vitest runs in the node environment. There is no jsdom, no testing-library.** Every frontend test is a pure-function test. Never write a test that renders a component.
- Frontend tests: `npx vitest run <file>` from `web/`. Full suite: `npm test`. Build (includes `tsc --noEmit`): `npm run build`.
- Python tests: `pytest tests/test_tour_claims.py -v` from the repo root. If LAPACK crashes with `0xc06d007f`, set `MKL_THREADING_LAYER=SEQUENTIAL`.
- Commit after every task. One logical change per commit, no AI attribution trailers.
- New physics or new displayed claims get a validation test, not a smoke test.

---

## File Structure

| File | Responsibility |
|---|---|
| `web/src/tours/types.ts` | `Tour`, `TourStep`, `Claim`, `ClaimKind`. Types only, no logic. |
| `web/src/tours/step.ts` | Pure helpers: `stepState`, `clampStep`, `spotlightBox`. |
| `web/src/tours/registry.ts` | Imports the JSON files, exposes `TOURS`, `tourById`. |
| `web/src/tours/hydrogen-honestly.json` | Flagship content, 10 steps. |
| `web/src/tours/break-the-physics.json` | Side tour, 4 steps. |
| `web/src/tours/many-electrons.json` | Side tour, 4 steps. |
| `web/src/tours/a-real-spectrum.json` | Side tour, 5 steps. |
| `web/src/components/TourPanel.tsx` | The docked card: title, body, back/next, close. |
| `web/src/components/TourSpotlight.tsx` | The ring over a `data-tour` anchor. |
| `web/src/components/TourMenu.tsx` | Tour picker, opened from the top bar. |
| `web/src/state/store.ts` | Tour slice: `tourId`, `stepIndex`, `savedState`, actions. |
| `web/src/lib/urlState.ts` | `tour` and `step` parameters. |
| `src/atomsim/tour_claims.py` | Claim resolver. `CLAIM_KINDS`, `resolve_claim`, `load_tours`. |
| `tests/test_tour_claims.py` | Every claim against the engine, plus per-kind analytic tests. |

---

## Task 1: Tour types and pure step helpers

**Files:**
- Create: `web/src/tours/types.ts`
- Create: `web/src/tours/step.ts`
- Create: `web/src/tours/step.test.ts`

**Interfaces:**
- Consumes: `UrlState`, `URL_DEFAULTS` from `web/src/lib/urlState.ts`.
- Produces: `Tour`, `TourStep`, `Claim`, `ClaimKind`, `CLAIM_KINDS`; `stepState(step: TourStep): UrlState`; `clampStep(tour: Tour, i: number): number`.

- [ ] **Step 1: Write `types.ts`**

```ts
import type { UrlState } from "../lib/urlState";

/**
 * The quantities a tour step may assert about the engine.
 *
 * Deliberately narrow. A resolver that silently returns the wrong quantity
 * reports a green tick on prose that lies, which is worse than having no test
 * at all, so each kind earns its place with its own test against a value known
 * in closed form. Adding a kind is one function in the Python dispatch table
 * plus one entry here; the structural test asserts the two lists match.
 */
export const CLAIM_KINDS = [
  "energy_eV",
  "mean_r_pm",
  "wavelength_nm",
  "ionization_eV",
] as const;

export type ClaimKind = (typeof CLAIM_KINDS)[number];

/**
 * One numeric assertion a step's prose makes.
 *
 * Carries its own inputs and inherits the step's `state` for anything it does
 * not name, so a claim is checkable standing alone. `tol` is absolute, in the
 * claim's own unit, and is required: a tolerance the author had to choose is a
 * tolerance the author had to think about.
 */
export interface Claim {
  of: ClaimKind;
  is: number;
  tol: number;
  system?: string;
  n?: number;
  l?: number;
  model?: "gsz" | "hf";
  fineStructure?: boolean;
  dirac?: boolean;
  exchange?: boolean;
  pauli?: boolean;
  n_upper?: number;
  n_lower?: number;
}

export interface TourStep {
  id: string;
  title: string;
  /** Paragraphs. Rendered one <p> each; no markup is interpreted. */
  body: string[];
  /** Partial UrlState. Applied over URL_DEFAULTS, never over the previous step. */
  state: Partial<UrlState>;
  /** A `data-tour` anchor to ring, or absent for no ring. */
  spotlight?: string;
  claims?: Claim[];
}

export interface Tour {
  id: string;
  title: string;
  blurb: string;
  steps: TourStep[];
}
```

- [ ] **Step 2: Write the failing test**

```ts
// web/src/tours/step.test.ts
import { describe, expect, it } from "vitest";
import { URL_DEFAULTS } from "../lib/urlState";
import { clampStep, stepState } from "./step";
import type { Tour, TourStep } from "./types";

const step = (over: Partial<TourStep> = {}): TourStep => ({
  id: "s", title: "t", body: ["b"], state: {}, ...over,
});

const tour = (n: number): Tour => ({
  id: "t", title: "T", blurb: "b",
  steps: Array.from({ length: n }, (_, i) => step({ id: `s${i}` })),
});

describe("stepState", () => {
  it("fills every key from the defaults", () => {
    const s = stepState(step({ state: { n: 3, l: 2 } }));
    expect(s.n).toBe(3);
    expect(s.l).toBe(2);
    expect(s.system).toBe(URL_DEFAULTS.system);
    expect(Object.keys(s).sort()).toEqual(Object.keys(URL_DEFAULTS).sort());
  });

  it("is a full reset, not a patch onto the step before", () => {
    // The bug this exists to prevent: stepping back from the fine-structure
    // step leaving fine structure switched on over the step that says the
    // levels are degenerate.
    const withFs = stepState(step({ state: { fineStructure: true } }));
    const plain = stepState(step({ state: { n: 2 } }));
    expect(withFs.fineStructure).toBe(true);
    expect(plain.fineStructure).toBe(URL_DEFAULTS.fineStructure);
  });

  it("does not share structure with URL_DEFAULTS", () => {
    // A step must never be able to mutate the defaults for every later step.
    const s = stepState(step({ state: { n: 2 } }));
    expect(s).not.toBe(URL_DEFAULTS);
    expect(s.labConst).not.toBe(URL_DEFAULTS.labConst);
  });
});

describe("clampStep", () => {
  it("keeps an index inside the tour", () => {
    expect(clampStep(tour(5), 3)).toBe(3);
    expect(clampStep(tour(5), 9)).toBe(4);
    expect(clampStep(tour(5), -2)).toBe(0);
  });

  it("survives junk from a hand-edited URL", () => {
    expect(clampStep(tour(5), Number.NaN)).toBe(0);
    expect(clampStep(tour(5), 2.7)).toBe(2);
  });

  it("returns 0 for an empty tour rather than -1", () => {
    expect(clampStep(tour(0), 3)).toBe(0);
  });
});
```

- [ ] **Step 3: Run it and watch it fail**

Run: `cd web && npx vitest run src/tours/step.test.ts`
Expected: FAIL, cannot resolve `./step`.

- [ ] **Step 4: Write `step.ts`**

```ts
import { URL_DEFAULTS, type UrlState } from "../lib/urlState";
import type { Tour, TourStep } from "./types";

/**
 * The full app state a step asks for.
 *
 * A full reset from the defaults, never a patch onto whatever the previous step
 * left behind. Two reasons: a step's picture is then reproducible from that
 * step's own data, which is what makes its claims mean anything, and stepping
 * backward cannot leave a later step's physics switched on underneath an
 * earlier step's prose.
 *
 * `labConst` is copied rather than shared, so a step can never mutate the
 * defaults out from under every step after it.
 */
export function stepState(step: TourStep): UrlState {
  return {
    ...URL_DEFAULTS,
    labConst: { ...URL_DEFAULTS.labConst },
    forceParams: { ...URL_DEFAULTS.forceParams },
    ...step.state,
  };
}

/** An index inside the tour, whatever a hand-edited `?step=` supplied. */
export function clampStep(tour: Tour, i: number): number {
  if (!Number.isFinite(i)) return 0;
  const last = tour.steps.length - 1;
  if (last < 0) return 0;
  return Math.min(Math.max(Math.floor(i), 0), last);
}
```

- [ ] **Step 5: Run the tests**

Run: `cd web && npx vitest run src/tours/step.test.ts`
Expected: PASS, 6 tests.

- [ ] **Step 6: Typecheck and commit**

```bash
cd web && npm run build
cd .. && git add web/src/tours/ && git commit -m "Add tour types and the pure step helpers

A step names a partial UrlState and stepState fills it out from
URL_DEFAULTS. A full reset rather than a patch onto the step before, so a
step's picture is reproducible from its own data and stepping backward
cannot leave a later step's physics under an earlier step's prose."
```

---

## Task 2: The registry, and the flagship's first two steps

**Files:**
- Create: `web/src/tours/hydrogen-honestly.json`
- Create: `web/src/tours/registry.ts`
- Create: `web/src/tours/registry.test.ts`

**Interfaces:**
- Consumes: `Tour`, `CLAIM_KINDS` from `./types`; `parseAppUrl`, `serializeAppUrl` from `../lib/urlState`.
- Produces: `TOURS: Tour[]`, `tourById(id: string): Tour | null`.

Two steps only in this task. The remaining eight land in Task 10, after the claims resolver exists to check them.

- [ ] **Step 1: Write `hydrogen-honestly.json`**

```json
{
  "id": "hydrogen-honestly",
  "title": "The Hydrogen Atom, Honestly",
  "blurb": "One electron, one proton, and every liberty the picture takes.",
  "steps": [
    {
      "id": "the-only-one-we-can-solve",
      "title": "Start with the only atom we can solve",
      "body": [
        "One electron, one proton, and a potential that goes as 1 over r. This is the last atom in the periodic table with a closed-form answer, and everything you are about to see is that answer rather than a fit to it.",
        "The cloud is a hundred thousand points drawn from the exact probability density. Nothing here was tuned to look right."
      ],
      "state": { "n": 1, "l": 0, "m": 0, "view": "cloud" },
      "spotlight": "view-list"
    },
    {
      "id": "a-probability-not-a-shell",
      "title": "A probability, not a shell",
      "body": [
        "There is no orbit and no surface. The electron has a probability of being found anywhere, and the cloud is that probability drawn as density.",
        "The nucleus is now at true scale, which is why you cannot see it. At this magnification a proton is well under a pixel across, and the marker the app shows by default is a labelled visual liberty rather than a size."
      ],
      "state": { "n": 1, "l": 0, "m": 0, "view": "cloud", "nucleusMode": "true-scale" },
      "spotlight": "nucleus-picker"
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

```ts
// web/src/tours/registry.test.ts
import { describe, expect, it } from "vitest";
import { parseAppUrl, serializeAppUrl } from "../lib/urlState";
import { TOURS, tourById } from "./registry";
import { stepState } from "./step";
import { CLAIM_KINDS } from "./types";

describe("registry", () => {
  it("loads every tour with at least one step", () => {
    expect(TOURS.length).toBeGreaterThan(0);
    for (const t of TOURS) expect(t.steps.length).toBeGreaterThan(0);
  });

  it("finds a tour by id and drops an unknown one", () => {
    expect(tourById(TOURS[0].id)?.id).toBe(TOURS[0].id);
    expect(tourById("no-such-tour")).toBeNull();
  });

  it("has unique tour ids", () => {
    const ids = TOURS.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("has unique step ids inside each tour", () => {
    for (const t of TOURS) {
      const ids = t.steps.map((s) => s.id);
      expect(new Set(ids).size, `duplicate step id in ${t.id}`).toBe(ids.length);
    }
  });

  it("gives every step a title and a non-empty body", () => {
    for (const t of TOURS) {
      for (const s of t.steps) {
        expect(s.title.length, `${t.id}/${s.id}`).toBeGreaterThan(0);
        expect(s.body.length, `${t.id}/${s.id}`).toBeGreaterThan(0);
        for (const p of s.body) expect(p.trim().length).toBeGreaterThan(0);
      }
    }
  });

  it("round-trips every step's state through the URL", () => {
    // A step that cannot survive serialisation is a step whose deep link
    // silently shows something else, which is the whole contract broken.
    for (const t of TOURS) {
      for (const s of t.steps) {
        const want = stepState(s);
        const got = { ...want, ...parseAppUrl(serializeAppUrl(want)) };
        expect(got, `${t.id}/${s.id}`).toEqual(want);
      }
    }
  });

  it("only claims quantities the Python resolver implements", () => {
    for (const t of TOURS) {
      for (const s of t.steps) {
        for (const c of s.claims ?? []) {
          expect(CLAIM_KINDS, `${t.id}/${s.id}`).toContain(c.of);
        }
      }
    }
  });

  it("gives every claim an explicit tolerance", () => {
    for (const t of TOURS) {
      for (const s of t.steps) {
        for (const c of s.claims ?? []) {
          expect(c.tol, `${t.id}/${s.id}/${c.of}`).toBeGreaterThan(0);
        }
      }
    }
  });

  it("names Hartree-Fock on any atom the screened model cannot draw", () => {
    // Sulfur and chlorine have no GSZ parameters. A step landing on one with
    // the default model selected asks the server for a refusal.
    for (const t of TOURS) {
      for (const s of t.steps) {
        if (s.state.system === "s" || s.state.system === "cl") {
          expect(s.state.model, `${t.id}/${s.id}`).toBe("hf");
        }
      }
    }
  });

  it("never writes an em dash", () => {
    for (const t of TOURS) {
      for (const s of t.steps) {
        for (const p of [s.title, ...s.body]) {
          expect(p, `${t.id}/${s.id}`).not.toContain("—");
        }
      }
    }
  });
});
```

- [ ] **Step 3: Run it and watch it fail**

Run: `cd web && npx vitest run src/tours/registry.test.ts`
Expected: FAIL, cannot resolve `./registry`.

- [ ] **Step 4: Write `registry.ts`**

```ts
import hydrogenHonestly from "./hydrogen-honestly.json";
import type { Tour } from "./types";

/**
 * Every tour, in menu order.
 *
 * The JSON is imported rather than fetched so it is bundled and typechecked at
 * build time, and so `atomsim serve` needs no new endpoint. The same files are
 * read by `tests/test_tour_claims.py`, which is the entire reason the content
 * is data: one file, rendered by one half of the project and checked against
 * the engine by the other.
 *
 * The cast is the seam where JSON meets the type. `registry.test.ts` is what
 * makes it safe: it walks every tour and asserts the shape the cast promises.
 */
export const TOURS: Tour[] = [hydrogenHonestly as Tour];

export function tourById(id: string): Tour | null {
  return TOURS.find((t) => t.id === id) ?? null;
}
```

- [ ] **Step 5: Enable JSON module resolution if the build complains**

Run: `cd web && npm run build`

If `tsc` reports "Cannot find module './hydrogen-honestly.json'", add to `web/tsconfig.json` under `compilerOptions`: `"resolveJsonModule": true`. Re-run the build. Vite bundles JSON imports natively, so no Vite config change is needed.

- [ ] **Step 6: Run the tests**

Run: `cd web && npx vitest run src/tours/registry.test.ts`
Expected: PASS, 10 tests.

- [ ] **Step 7: Commit**

```bash
cd .. && git add web/src/tours/ web/tsconfig.json && git commit -m "Load tours from JSON, and check every step's shape

The registry casts imported JSON to Tour, and the test is what makes that
cast safe: unique ids, non-empty bodies, a tolerance on every claim, only
claim kinds the Python resolver implements, and a URL round-trip on every
step's state, because a step that cannot survive serialisation is a step
whose deep link shows something else.

Two steps of the flagship for now. The rest land once the resolver exists
to check their numbers."
```

---

## Task 3: The store's tour slice

**Files:**
- Modify: `web/src/state/store.ts`
- Create: `web/src/tours/apply.ts`
- Create: `web/src/tours/apply.test.ts`

**Interfaces:**
- Consumes: `stepState`, `clampStep`; `INVALIDATED` (module-local in the store).
- Produces: store fields `tourId: string | null`, `stepIndex: number`, `savedState: UrlState | null`; actions `startTour(id)`, `exitTour()`, `goToStep(i)`, `nextStep()`, `prevStep()`. Pure helper `tourReset(state: UrlState, systems: SystemInfo[]): Partial<AppState>`.

**This is the load-bearing task.** `INVALIDATED` in `store.ts:294` holds every derived-physics field, and it is spread inside each *action*, not on raw `setState`. A tour step that called `useAppStore.setState(stepState(step))` would change n, l, system and model while leaving the previous step's cloud, plane, levels, spectrum and Hartree-Fock solve in the store, so the views would render the previous step's physics under this step's labels. That is exactly the stale-physics render the whole invalidation design exists to prevent.

- [ ] **Step 1: Write the failing test**

```ts
// web/src/tours/apply.test.ts
import { describe, expect, it } from "vitest";
import { URL_DEFAULTS } from "../lib/urlState";
import { tourReset } from "./apply";

const systems = [
  { key: "h", name: "hydrogen", gsz: true },
  { key: "s", name: "sulfur", gsz: false },
] as never[];

describe("tourReset", () => {
  it("carries every input the step asked for", () => {
    const out = tourReset({ ...URL_DEFAULTS, n: 3, l: 2, view: "levels" }, systems);
    expect(out.n).toBe(3);
    expect(out.l).toBe(2);
    expect(out.view).toBe("levels");
  });

  it("clears every derived field, not just the level payloads", () => {
    // The failure this prevents: a step changes n and system, and the previous
    // step's cloud, plane, surface and spectrum keep rendering under the new
    // labels because raw setState never spreads INVALIDATED.
    const out = tourReset({ ...URL_DEFAULTS, n: 2 }, systems);
    for (const k of [
      "stateInfo", "positions", "density", "phase", "meta", "plane", "iso",
      "radial", "levels", "spectrum", "curveOfGrowth", "absorptionData",
    ]) {
      expect(out[k as keyof typeof out], `${k} not cleared`).toBeNull();
    }
    expect(out.status).toBe("idle");
    expect(out.planeStatus).toBe("idle");
    expect(out.isoStatus).toBe("idle");
  });

  it("clears the payloads that live outside INVALIDATED", () => {
    // A step can change n, system, config, exchange and pauli in one move, so
    // it has to clear the union of what every individual action clears, not
    // just INVALIDATED.
    const out = tourReset({ ...URL_DEFAULTS, system: "he" }, systems);
    expect(out.classicalGhost).toBeNull();
    expect(out.classicalStatus).toBe("idle");
    expect(out.hf).toBeNull();
    expect(out.hfStatus).toBe("idle");
    expect(out.forceLaw).toBeNull();
    expect(out.forceStatus).toBe("idle");
  });

  it("moves an atom with no GSZ parameters onto Hartree-Fock", () => {
    // Same guard setSystem applies. A step naming sulfur under the default
    // model asks the server for a refusal on every request.
    const out = tourReset({ ...URL_DEFAULTS, system: "s", model: "gsz" }, systems);
    expect(out.model).toBe("hf");
  });

  it("leaves a resolvable model alone", () => {
    const out = tourReset({ ...URL_DEFAULTS, system: "h", model: "gsz" }, systems);
    expect(out.model).toBe("gsz");
  });
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd web && npx vitest run src/tours/apply.test.ts`
Expected: FAIL, cannot resolve `./apply`.

- [ ] **Step 3: Export `INVALIDATED` from the store**

In `web/src/state/store.ts`, change line 294 from `const INVALIDATED = {` to `export const INVALIDATED = {`, and leave the block's contents and its comments untouched.

- [ ] **Step 4: Write `apply.ts`**

```ts
import { resolveModel } from "../lib/hfModel";
import type { SystemInfo } from "../api/client";
import type { UrlState } from "../lib/urlState";
import { INVALIDATED } from "../state/store";

/**
 * The store patch that puts the app into a tour step's state.
 *
 * A step can change n, l, m, system, config, model, exchange and pauli in one
 * move, so it must clear the union of what every individual action clears, not
 * just INVALIDATED. `INVALIDATED` is spread inside each action rather than
 * applied by `setState`, so a raw `setState(stepState(step))` would leave the
 * previous step's cloud, plane, surface, levels and spectrum in the store and
 * render them under this step's labels. `classicalGhost` (cleared by
 * setQuantumNumbers and setSystem), `hf` (setSystem, setConfig, setExchange,
 * setPauli) and `forceLaw` (setSystem) all live outside INVALIDATED, so they
 * are named here.
 *
 * The model is resolved exactly as setSystem resolves it: sulfur and chlorine
 * have no GSZ parameters, and a step landing on one under the default model
 * would ask the server for a refusal on every request. The content test in
 * registry.test.ts requires such a step to name `model: "hf"` itself; this is
 * the belt to that pair of braces.
 */
export function tourReset(state: UrlState, systems: SystemInfo[]) {
  return {
    ...state,
    model: resolveModel(systems, state.system, state.model),
    ...INVALIDATED,
    classicalGhost: null,
    classicalStatus: "idle" as const,
    hf: null,
    hfStatus: "idle" as const,
    forceLaw: null,
    forceStatus: "idle" as const,
  };
}
```

- [ ] **Step 5: Run the tests**

Run: `cd web && npx vitest run src/tours/apply.test.ts`
Expected: PASS, 5 tests.

- [ ] **Step 6: Add the tour slice to the store**

In `web/src/state/store.ts`, add to the `AppState` interface, near the other view-level fields:

```ts
  /**
   * The tour being taken, or null.
   *
   * `savedState` is the reader's own state, snapshotted on entry and restored
   * on exit. A reader three minutes into a chlorine Hartree-Fock solve should
   * not lose it by clicking a tour out of curiosity.
   */
  tourId: string | null;
  stepIndex: number;
  savedState: UrlState | null;
  startTour: (id: string, step?: number) => void;
  exitTour: () => void;
  goToStep: (i: number) => void;
```

Add to the initial state object, next to `compare: false`:

```ts
  tourId: null,
  stepIndex: 0,
  savedState: null,
```

Add the actions, after `setCompare`:

```ts
  startTour: (id, step = 0) =>
    set((s) => {
      const tour = tourById(id);
      if (!tour) return {};
      const i = clampStep(tour, step);
      return {
        ...tourReset(stepState(tour.steps[i]), s.systems),
        tourId: id,
        stepIndex: i,
        // Only on entry. Re-entering mid-tour must not overwrite the reader's
        // own state with a tour step's.
        savedState: s.savedState ?? currentUrlState(s),
      };
    }),
  goToStep: (i) =>
    set((s) => {
      const tour = s.tourId ? tourById(s.tourId) : null;
      if (!tour) return {};
      const next = clampStep(tour, i);
      return { ...tourReset(stepState(tour.steps[next]), s.systems), stepIndex: next };
    }),
  exitTour: () =>
    set((s) => ({
      ...(s.savedState ? tourReset(s.savedState, s.systems) : {}),
      tourId: null,
      stepIndex: 0,
      savedState: null,
    })),
```

Add the imports at the top of `store.ts`:

```ts
import { tourReset } from "../tours/apply";
import { clampStep, stepState } from "../tours/step";
import { tourById } from "../tours/registry";
```

- [ ] **Step 7: Extract `currentUrlState` so the snapshot and the URL agree**

`main.tsx` already builds a `UrlState` from the store by hand, listing all 38 fields. The tour snapshot needs exactly the same object. Duplicating that list is how the two fall out of sync, so move it to `web/src/lib/urlState.ts`:

```ts
/**
 * The addressable slice of a store snapshot.
 *
 * Lives here rather than in main.tsx because two callers now need the same
 * list: the subscriber that keeps the URL describing the live state, and the
 * tour, which snapshots the reader's state on entry and restores it on exit. A
 * second hand-maintained copy of 38 field names is how the two drift.
 */
export function currentUrlState(s: UrlState): UrlState {
  return {
    n: s.n, l: s.l, m: s.m, system: s.system, basis: s.basis, view: s.view,
    colorMode: s.colorMode, fineStructure: s.fineStructure, dirac: s.dirac,
    compare: s.compare, bField: s.bField, eField: s.eField,
    hyperfine: s.hyperfine, intensities: s.intensities, thermal: s.thermal,
    temperatureK: s.temperatureK, logNe: s.logNe, profile: s.profile,
    logResolvingPower: s.logResolvingPower, profileZoom: s.profileZoom,
    absorption: s.absorption, logColumn: s.logColumn, ghost: s.ghost,
    nucleusMode: s.nucleusMode, planeQuantity: s.planeQuantity,
    surfaceMode: s.surfaceMode, isoFraction: s.isoFraction,
    labConst: s.labConst, labZ: s.labZ, forcePreset: s.forcePreset,
    forceParams: s.forceParams, forceL: s.forceL, forceExpr: s.forceExpr,
    config: s.config, model: s.model, exchange: s.exchange, pauli: s.pauli,
  };
}
```

Then in `main.tsx`, replace the inline object literal inside `useAppStore.subscribe` with `serializeAppUrl(currentUrlState(s))` and import `currentUrlState`. Import it in `store.ts` too.

- [ ] **Step 8: Run the whole frontend suite and build**

Run: `cd web && npm test && npm run build`
Expected: all files pass; `tsc --noEmit` clean.

- [ ] **Step 9: Commit**

```bash
cd .. && git add web/src/ && git commit -m "Put the app into a tour step without leaving the last step's physics

INVALIDATED is spread inside each store action, not applied by setState,
so a tour step calling setState with a step's inputs would change n, l,
system and model while the previous step's cloud, plane, levels and
spectrum stayed in the store and rendered under the new labels. That is
the stale-physics render the whole invalidation design exists to prevent.

tourReset clears the union of what every individual action clears:
INVALIDATED plus classicalGhost, hf and forceLaw, which live outside it.
It resolves the model the way setSystem does, so a step naming sulfur
cannot sit on a screened model that has no parameters for it.

Entering a tour snapshots the reader's own state and exiting restores it,
so a tour never costs someone the solve they were looking at. The 38-field
UrlState projection moves out of main.tsx, because a second hand-kept copy
of that list is how the URL and the snapshot drift apart."
```

---

## Task 4: Deep-linking a tour

**Files:**
- Modify: `web/src/lib/urlState.ts`
- Modify: `web/src/lib/urlState.test.ts`
- Modify: `web/src/main.tsx`

**Interfaces:**
- Consumes: `UrlState`, `URL_DEFAULTS`.
- Produces: `tour: string | null` and `step: number` on `UrlState`, serialized as `?tour=<id>&step=<n>`.

- [ ] **Step 1: Write the failing test**

Append to `web/src/lib/urlState.test.ts`:

```ts
describe("tour deep links", () => {
  it("carries a tour and a step", () => {
    const s = parseAppUrl("?tour=hydrogen-honestly&step=4");
    expect(s.tour).toBe("hydrogen-honestly");
    expect(s.step).toBe(4);
  });

  it("defaults the step to the first one", () => {
    expect(parseAppUrl("?tour=hydrogen-honestly").step).toBe(0);
  });

  it("drops a junk step rather than throwing", () => {
    // Same contract as every other parameter here: junk never reaches the
    // store, so a typo'd link still opens the app.
    expect(parseAppUrl("?tour=x&step=banana").step).toBe(0);
    expect(parseAppUrl("?tour=x&step=-3").step).toBe(0);
  });

  it("omits the step when there is no tour", () => {
    // A bare ?step= describes nothing and would survive into a shared link as
    // noise.
    const qs = serializeAppUrl({ ...URL_DEFAULTS, tour: null, step: 5 });
    expect(qs).not.toContain("step=");
  });

  it("round-trips", () => {
    const want = { ...URL_DEFAULTS, tour: "hydrogen-honestly", step: 3 };
    expect({ ...want, ...parseAppUrl(serializeAppUrl(want)) }).toEqual(want);
  });
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd web && npx vitest run src/lib/urlState.test.ts`
Expected: FAIL, `s.tour` is undefined.

- [ ] **Step 3: Add the fields**

In `UrlState`, after `pauli`:

```ts
  /**
   * The tour being taken, and how far into it.
   *
   * Carried so the demo script is a URL: "?tour=hydrogen-honestly&step=4" is a
   * thing you can send someone. `step` is written only alongside a tour,
   * because a bare step number describes nothing.
   */
  tour: string | null;
  step: number;
```

In `URL_DEFAULTS`: `tour: null,` and `step: 0,`.

In `parseAppUrl`, before the return:

```ts
  const tour = q.get("tour");
  if (tour) {
    out.tour = tour;
    const step = pickInt(q.get("step"));
    // Clamped to the tour's real length by clampStep at apply time; here it
    // only has to be a non-negative integer, since this module does not know
    // how long any tour is.
    out.step = step !== undefined && step >= 0 ? step : 0;
  }
```

In `serializeAppUrl`, before the return:

```ts
  if (state.tour) {
    q.set("tour", state.tour);
    if (state.step !== URL_DEFAULTS.step) q.set("step", String(state.step));
  }
```

- [ ] **Step 4: Run the tests**

Run: `cd web && npx vitest run src/lib/urlState.test.ts`
Expected: PASS.

- [ ] **Step 5: Wire it in `main.tsx`**

The store's tour fields are `tourId`/`stepIndex`, and the URL's are `tour`/`step`. Bridge both directions.

After the existing `useAppStore.setState(parseAppUrl(window.location.search));` line, add:

```ts
// A tour link has to run the step's state through the store's tour action, not
// just land its id in the store: the step's own physics has to be applied and
// everything derived cleared. Deferred to a microtask so the store's initial
// state exists before startTour reads it.
const opening = parseAppUrl(window.location.search);
if (opening.tour) {
  queueMicrotask(() => useAppStore.getState().startTour(opening.tour!, opening.step ?? 0));
}
```

In the subscriber, extend `currentUrlState(s)` usage so the tour appears in the URL:

```ts
  const qs = serializeAppUrl({ ...currentUrlState(s), tour: s.tourId, step: s.stepIndex });
```

Add `tour: null, step: 0` to the `currentUrlState` return in `urlState.ts` so its output is a complete `UrlState`; the subscriber overrides both.

- [ ] **Step 6: Run the suite, build, and commit**

```bash
cd web && npm test && npm run build
cd .. && git add web/src/ && git commit -m "Make a tour a URL

?tour=hydrogen-honestly&step=4 addresses a step, so the demo script is a
link you can send someone. step is written only alongside a tour, because
a bare step number describes nothing, and junk drops rather than throwing,
matching every other parameter in this module.

Opening on a tour link runs the step through startTour rather than landing
its id in the store, so the step's physics is applied and everything
derived is cleared."
```

---

## Task 5: The tour card

**Files:**
- Create: `web/src/components/TourPanel.tsx`
- Modify: `web/src/App.tsx`
- Modify: `web/src/index.css`

**Interfaces:**
- Consumes: `tourById`, `clampStep`, store fields `tourId`, `stepIndex`, actions `goToStep`, `exitTour`.
- Produces: `<TourPanel />`, rendered by `App`.

No test file. This component holds no logic worth testing that is not already covered by `step.test.ts` and `apply.test.ts`, and this project's vitest has no DOM. Keep it thin enough that reading it is enough.

- [ ] **Step 1: Write `TourPanel.tsx`**

```tsx
import { tourById } from "../tours/registry";
import { useAppStore } from "../state/store";

/**
 * The tour's narration, docked under the stage.
 *
 * Docked rather than floating over the views: the picture is the thing the
 * prose is about, and a card on top of it would cover the evidence.
 */
export function TourPanel() {
  const { tourId, stepIndex, goToStep, exitTour } = useAppStore();
  const tour = tourId ? tourById(tourId) : null;
  if (!tour) return null;
  const step = tour.steps[stepIndex];
  if (!step) return null;
  const last = tour.steps.length - 1;
  return (
    <aside className="tour-panel" aria-label={`${tour.title}, step ${stepIndex + 1}`}>
      <div className="tour-head">
        <span className="tour-count">
          {stepIndex + 1} / {tour.steps.length}
        </span>
        <span className="tour-title">{step.title}</span>
        <button className="tour-close" type="button" onClick={exitTour} aria-label="leave the tour">
          ✕
        </button>
      </div>
      {step.body.map((p, i) => (
        <p key={i} className="tour-body">{p}</p>
      ))}
      <div className="tour-nav">
        <button
          type="button" className="link-button"
          onClick={() => goToStep(stepIndex - 1)} disabled={stepIndex === 0}
        >
          ‹ back
        </button>
        {stepIndex === last ? (
          <button type="button" className="link-button" onClick={exitTour}>
            finish ›
          </button>
        ) : (
          <button type="button" className="link-button" onClick={() => goToStep(stepIndex + 1)}>
            next ›
          </button>
        )}
      </div>
    </aside>
  );
}
```

- [ ] **Step 2: Render it in `App.tsx`**

Import `TourPanel` and place it as the last child of `<main className="center-col">`, after `<GalleryStrip />`.

- [ ] **Step 3: Style it**

Append to `web/src/index.css`, in the instrument section near the other panels:

```css
/* The tour's narration. Docked under the stage rather than floating over it:
   the picture is what the prose is about, and a card on top of it would cover
   the evidence. */
.tour-panel {
  border: 1px solid var(--edge-strong);
  border-radius: 3px;
  background: rgb(8 12 14 / 0.75);
  padding: 0.7rem 0.9rem;
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
  flex-shrink: 0;
}

.tour-head {
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
}

.tour-count {
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
  font-size: 0.68rem;
  color: var(--accent);
}

.tour-title {
  font-family: var(--sans);
  font-size: 0.9rem;
  color: var(--text);
}

.tour-close {
  margin-left: auto;
  background: transparent;
  border: 0;
  color: var(--muted);
  cursor: pointer;
  font-size: 0.8rem;
}

.tour-body {
  font-family: var(--mono);
  font-size: 0.72rem;
  line-height: 1.6;
  color: var(--muted);
  margin: 0;
  max-width: 68ch;
}

.tour-nav {
  display: flex;
  gap: 1rem;
  margin-top: 0.15rem;
}

.tour-nav button[disabled] {
  opacity: 0.35;
  cursor: default;
}
```

- [ ] **Step 4: Build and see it**

```bash
cd web && npm run build
cd .. && MKL_THREADING_LAYER=SEQUENTIAL python -m atomsim.cli serve --port 8021 --no-browser
```

Open `http://127.0.0.1:8021/?tour=hydrogen-honestly&step=0`. Confirm the card appears under the stage, `next` moves to the true-scale nucleus step and the cloud redraws, `back` returns, and `✕` restores the state you had before the link.

- [ ] **Step 5: Commit**

```bash
git add web/src/ && git commit -m "Draw the tour's narration under the stage

Docked rather than floating over the views: the picture is what the prose
is about, and a card on top of it would cover the evidence. Last step
offers finish rather than a dead next."
```

---

## Task 6: The spotlight

**Files:**
- Create: `web/src/components/TourSpotlight.tsx`
- Create: `web/src/tours/spotlight.ts`
- Create: `web/src/tours/spotlight.test.ts`
- Create: `web/src/tours/anchors.test.ts`
- Modify: `web/src/components/Controls.tsx`, `web/src/components/InfoPanel.tsx`
- Modify: `web/src/App.tsx`, `web/src/index.css`

**Interfaces:**
- Consumes: `TourStep.spotlight`.
- Produces: `spotlightBox(rect, pad): {x,y,w,h}`; `<TourSpotlight />`.

- [ ] **Step 1: Write the failing tests**

```ts
// web/src/tours/spotlight.ts is tested here as pure geometry; the DOM read
// happens in the component, which this project cannot test without jsdom.
import { describe, expect, it } from "vitest";
import { spotlightBox } from "./spotlight";

describe("spotlightBox", () => {
  it("pads the ring out from the control", () => {
    const b = spotlightBox({ left: 10, top: 20, width: 100, height: 40 }, 6);
    expect(b).toEqual({ x: 4, y: 14, w: 112, h: 52 });
  });

  it("returns null for a control with no box", () => {
    // An anchor that is display:none, unmounted, or inside a collapsed
    // <details> measures 0x0. Ringing it would draw a dot in the corner.
    expect(spotlightBox({ left: 0, top: 0, width: 0, height: 0 }, 6)).toBeNull();
  });
});
```

```ts
// web/src/tours/anchors.test.ts
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { TOURS } from "./registry";

/**
 * Every anchor a tour points at has to exist in the app.
 *
 * With no jsdom there is nothing to query, so this reads the component sources
 * and extracts the data-tour literals. A source scan is the right tool anyway:
 * it catches an anchor deleted in a refactor, which is the failure being
 * guarded, and it cannot be fooled by a component that happens not to render.
 */
function declaredAnchors(): Set<string> {
  const dir = join(__dirname, "..", "components");
  const out = new Set<string>();
  for (const f of readdirSync(dir).filter((f) => f.endsWith(".tsx"))) {
    const src = readFileSync(join(dir, f), "utf8");
    for (const m of src.matchAll(/data-tour="([^"]+)"/g)) out.add(m[1]);
  }
  return out;
}

describe("spotlight anchors", () => {
  it("every anchor a tour names exists in a component", () => {
    const have = declaredAnchors();
    for (const t of TOURS) {
      for (const s of t.steps) {
        if (s.spotlight) {
          expect(have, `${t.id}/${s.id} points at a missing anchor`).toContain(s.spotlight);
        }
      }
    }
  });

  it("finds anchors at all, so a broken scan cannot pass vacuously", () => {
    expect(declaredAnchors().size).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cd web && npx vitest run src/tours/spotlight.test.ts src/tours/anchors.test.ts`
Expected: FAIL, cannot resolve `./spotlight`; anchors test fails on the vacuity guard.

- [ ] **Step 3: Write `spotlight.ts`**

```ts
/** The rectangle a `getBoundingClientRect` gives, narrowed to what is used. */
export interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * The ring's rectangle, padded out from the control it surrounds.
 *
 * Null for a zero-sized box. An anchor that is unmounted, display:none, or
 * inside a collapsed <details> measures 0 by 0, and ringing it would draw a
 * dot in the top-left corner pointing at nothing.
 */
export function spotlightBox(box: Box, pad: number) {
  if (box.width <= 0 || box.height <= 0) return null;
  return {
    x: box.left - pad,
    y: box.top - pad,
    w: box.width + 2 * pad,
    h: box.height + 2 * pad,
  };
}
```

- [ ] **Step 4: Write `TourSpotlight.tsx`**

```tsx
import { useEffect, useState } from "react";
import { spotlightBox } from "../tours/spotlight";
import { tourById } from "../tours/registry";
import { useAppStore } from "../state/store";

const PAD = 6;

/**
 * A ring around the control a step is talking about.
 *
 * Fixed-position, pointer-events:none, so it never intercepts a click: the
 * controls stay live during a tour and ringing one must not stop it working.
 * If the anchor is missing or measures zero the ring simply does not render,
 * and the card downstairs carries the step on its own.
 */
export function TourSpotlight() {
  const { tourId, stepIndex } = useAppStore();
  const [box, setBox] = useState<ReturnType<typeof spotlightBox>>(null);
  const tour = tourId ? tourById(tourId) : null;
  const anchor = tour?.steps[stepIndex]?.spotlight ?? null;

  useEffect(() => {
    if (!anchor) {
      setBox(null);
      return;
    }
    const measure = () => {
      const el = document.querySelector(`[data-tour="${anchor}"]`);
      setBox(el ? spotlightBox(el.getBoundingClientRect(), PAD) : null);
    };
    // After paint, so the step's own re-render has settled before measuring.
    const id = requestAnimationFrame(measure);
    window.addEventListener("resize", measure);
    return () => {
      cancelAnimationFrame(id);
      window.removeEventListener("resize", measure);
    };
  }, [anchor, stepIndex, tourId]);

  if (!box) return null;
  return (
    <div
      className="tour-ring"
      style={{ left: box.x, top: box.y, width: box.w, height: box.h }}
      aria-hidden="true"
    />
  );
}
```

- [ ] **Step 5: Add the anchors the flagship needs**

Add `data-tour="..."` to these existing elements. Put it on the wrapping element, not the `<select>` or `<input>`, so the ring surrounds the label too.

In `web/src/components/Controls.tsx`:
- the view-mode button list wrapper: `data-tour="view-list"`
- the n tile: `data-tour="n-picker"`
- the l tile: `data-tour="l-picker"`
- the basis radio group wrapper: `data-tour="basis-picker"`
- the nucleus `<select>`'s wrapping `<label>`: `data-tour="nucleus-picker"`
- the fine-structure checkbox `<label>`: `data-tour="fine-structure"`
- the surface controls wrapper: `data-tour="surface-controls"`

In `web/src/components/InfoPanel.tsx`:
- the state card: `data-tour="state-card"`

If any of these elements does not exist under that description, find the control by its visible text and put the anchor on its nearest wrapping element. The anchors test will tell you if a name is missing; it will not tell you if you put one in a silly place, so check it in the browser at Step 8.

- [ ] **Step 6: Render the ring and style it**

In `App.tsx`, add `<TourSpotlight />` as the last child of `<div className="app-shell">`, after `</div>` of `app-grid`.

Append to `index.css`:

```css
/* The ring around the control a tour step is talking about. pointer-events is
   none because the controls stay live during a tour: ringing one must not stop
   it working. */
.tour-ring {
  position: fixed;
  pointer-events: none;
  border: 1px solid var(--accent);
  border-radius: 4px;
  box-shadow: 0 0 0 3px rgb(52 224 161 / 0.14);
  transition: left 140ms ease, top 140ms ease, width 140ms ease, height 140ms ease;
  z-index: 50;
}
```

- [ ] **Step 7: Point the two existing steps at real anchors**

The flagship's step 1 already names `view-list` and step 2 names `nucleus-picker`. Confirm both now resolve.

Run: `cd web && npx vitest run src/tours/`
Expected: PASS, all four tour test files.

- [ ] **Step 8: Build and check it in the browser**

```bash
cd web && npm run build
```

Open `http://127.0.0.1:8021/?tour=hydrogen-honestly&step=0`. Confirm the ring surrounds the view list, moves to the nucleus picker on `next`, does not block clicking either control, and follows a window resize.

- [ ] **Step 9: Commit**

```bash
cd .. && git add web/src/ && git commit -m "Ring the control a tour step is talking about

pointer-events:none, because the controls stay live during a tour and
ringing one must not stop it working. An anchor that is missing or
measures zero renders no ring at all rather than a dot in the corner
pointing at nothing.

The anchors test reads the component sources for data-tour literals
rather than querying a DOM, because this frontend's vitest runs in node
with no jsdom. It carries a vacuity guard, so a scan that silently stops
matching cannot pass by finding nothing."
```

---

## Task 7: The tour menu

**Files:**
- Create: `web/src/components/TourMenu.tsx`
- Modify: `web/src/components/TopBar.tsx`, `web/src/index.css`

- [ ] **Step 1: Write `TourMenu.tsx`**

```tsx
import { useState } from "react";
import { TOURS } from "../tours/registry";
import { useAppStore } from "../state/store";

/** The tour picker. Opens from the top bar, closes on pick or on Escape. */
export function TourMenu() {
  const [open, setOpen] = useState(false);
  const { tourId, startTour, exitTour } = useAppStore();
  if (tourId) {
    return (
      <button className="tour-entry" type="button" onClick={exitTour}>
        leave tour
      </button>
    );
  }
  return (
    <div className="tour-entry-wrap" onKeyDown={(e) => e.key === "Escape" && setOpen(false)}>
      <button className="tour-entry" type="button" onClick={() => setOpen(!open)}>
        guided tours
      </button>
      {open && (
        <ul className="tour-menu">
          {TOURS.map((t) => (
            <li key={t.id}>
              <button
                type="button"
                onClick={() => {
                  startTour(t.id, 0);
                  setOpen(false);
                }}
              >
                <span className="tour-menu-title">{t.title}</span>
                <span className="tour-menu-blurb">{t.blurb}</span>
                <span className="tour-menu-count">{t.steps.length} steps</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Mount it in `TopBar.tsx`**

Add `<TourMenu />` to the top bar's right-hand group, beside the frame-rate readout.

- [ ] **Step 3: Style it**

```css
.tour-entry-wrap { position: relative; }

.tour-entry {
  background: transparent;
  border: 1px solid var(--edge-strong);
  border-radius: 3px;
  color: var(--text);
  font-family: var(--mono);
  font-size: 0.68rem;
  padding: 0.2rem 0.55rem;
  cursor: pointer;
}

.tour-menu {
  position: absolute;
  right: 0;
  top: calc(100% + 0.35rem);
  z-index: 60;
  margin: 0;
  padding: 0.25rem;
  list-style: none;
  width: 22rem;
  max-width: 90vw;
  background: var(--panel);
  border: 1px solid var(--edge-strong);
  border-radius: 3px;
}

.tour-menu button {
  display: grid;
  gap: 0.15rem;
  width: 100%;
  text-align: left;
  background: transparent;
  border: 0;
  padding: 0.5rem;
  cursor: pointer;
}

.tour-menu button:hover { background: var(--edge-fill); }
.tour-menu-title { font-family: var(--sans); font-size: 0.82rem; color: var(--text); }
.tour-menu-blurb { font-family: var(--mono); font-size: 0.66rem; color: var(--muted); }
.tour-menu-count {
  font-family: var(--mono);
  font-size: 0.62rem;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}
```

- [ ] **Step 4: Build, check, commit**

```bash
cd web && npm test && npm run build
cd .. && git add web/src/ && git commit -m "Offer the tours from the top bar

One button, which becomes leave tour while a tour is running so there are
never two ways out with different meanings."
```

---

## Task 8: The claims resolver

**Files:**
- Create: `src/atomsim/tour_claims.py`
- Create: `tests/test_tour_claims.py`

**Interfaces:**
- Consumes: `atomsim.analytic.hydrogen.energy`, `.mean_radius`; `atomsim.systems.get_system`; `atomsim.spectra.transition_lines`; `atomsim.constants.HARTREE_EV`, `.BOHR_RADIUS_PM`.
- Produces: `CLAIM_KINDS: tuple[str, ...]`, `resolve_claim(claim: dict) -> float`, `load_tours() -> list[dict]`, `iter_claims() -> Iterator[tuple[str, str, dict]]`.

**This is the load-bearing piece of the phase.** A resolver that returns the wrong quantity reports a green tick on prose that lies, which is worse than having no test. Every kind gets its own test against a value known in closed form, written before any tour leans on it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tour_claims.py
"""Every number a tour step quotes, checked against the engine that draws it.

A tour is fifty-odd pieces of prose asserting things about physics, and prose
rots silently when the engine improves underneath it. A step that says
"-3.40 eV" fails here the day that stops being true.

The per-kind tests come first and matter most: a resolver that quietly returns
the wrong quantity would pass every claim in every tour while checking nothing,
so each kind is pinned to a value known in closed form before any tour uses it.
"""

import math

import pytest

from atomsim.tour_claims import CLAIM_KINDS, iter_claims, load_tours, resolve_claim


class TestResolverKinds:
    """Each kind against an independently known value."""

    def test_energy_is_the_bohr_formula_in_ev(self):
        # -13.6057 eV / n^2 for hydrogen, the number every textbook prints.
        got = resolve_claim({"of": "energy_eV", "system": "h", "n": 1})
        assert got == pytest.approx(-13.6057, abs=1e-3)
        assert resolve_claim({"of": "energy_eV", "system": "h", "n": 2}) == pytest.approx(
            -3.4014, abs=1e-3
        )

    def test_energy_scales_with_reduced_mass(self):
        # Deuterium is bound slightly more tightly than protium. If the resolver
        # ignored mu_ratio these would be equal, and the isotope-shift step of
        # any tour would be checking nothing.
        h = resolve_claim({"of": "energy_eV", "system": "h", "n": 1})
        d = resolve_claim({"of": "energy_eV", "system": "d", "n": 1})
        assert d < h
        assert abs(d - h) == pytest.approx(0.0037, abs=5e-4)

    def test_mean_radius_is_the_closed_form_in_pm(self):
        # <r> = 1.5 a0 for the 1s, and a0 = 52.9177 pm.
        got = resolve_claim({"of": "mean_r_pm", "system": "h", "n": 1, "l": 0})
        assert got == pytest.approx(1.5 * 52.9177, abs=1e-2)

    def test_mean_radius_depends_on_l_not_only_n(self):
        # (3n^2 - l(l+1)) / 2: the 2s is larger than the 2p. A resolver that
        # dropped l would return the same number for both.
        s = resolve_claim({"of": "mean_r_pm", "system": "h", "n": 2, "l": 0})
        p = resolve_claim({"of": "mean_r_pm", "system": "h", "n": 2, "l": 1})
        assert s > p

    def test_wavelength_is_lyman_alpha(self):
        # 121.567 nm, vacuum. The single most recognisable number in the app.
        got = resolve_claim(
            {"of": "wavelength_nm", "system": "h", "n_upper": 2, "n_lower": 1}
        )
        assert got == pytest.approx(121.567, abs=0.01)

    def test_wavelength_is_h_alpha(self):
        got = resolve_claim(
            {"of": "wavelength_nm", "system": "h", "n_upper": 3, "n_lower": 2}
        )
        assert got == pytest.approx(656.47, abs=0.05)

    def test_ionization_energy_of_helium(self):
        # Hartree-Fock by Koopmans. HF has no correlation, so this is above the
        # measured 24.587 eV rather than equal to it; the tolerance below is
        # the model's error, not the solver's.
        got = resolve_claim({"of": "ionization_eV", "system": "he", "model": "hf"})
        assert got == pytest.approx(24.98, abs=0.3)

    def test_every_declared_kind_resolves(self):
        # A kind in CLAIM_KINDS with no branch in the dispatch would raise only
        # when a tour first used it, which is the wrong time to find out.
        assert set(CLAIM_KINDS) == {
            "energy_eV", "mean_r_pm", "wavelength_nm", "ionization_eV",
        }

    def test_unknown_kind_raises_rather_than_returning_zero(self):
        with pytest.raises(ValueError, match="unknown claim kind"):
            resolve_claim({"of": "spin_of_the_universe", "system": "h"})

    def test_missing_input_raises_rather_than_defaulting(self):
        # Defaulting n to 1 would let a claim about the 3d silently check the 1s.
        with pytest.raises(KeyError):
            resolve_claim({"of": "wavelength_nm", "system": "h", "n_upper": 3})


class TestTourContent:
    def test_tours_load(self):
        tours = load_tours()
        assert tours, "no tour JSON found; check the path in load_tours"

    def test_every_claim_holds(self):
        checked = 0
        for tour_id, step_id, claim in iter_claims():
            got = resolve_claim(claim)
            assert math.isfinite(got), f"{tour_id}/{step_id}: {claim['of']} is not finite"
            assert got == pytest.approx(claim["is"], abs=claim["tol"]), (
                f"{tour_id}/{step_id} claims {claim['of']} = {claim['is']} "
                f"+/- {claim['tol']}, engine says {got:.6g}. "
                f"Either the prose is now wrong or the engine changed."
            )
            checked += 1
        assert checked > 0, "no claims checked; the tours declare none"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_tour_claims.py -v`
Expected: FAIL, `ModuleNotFoundError: atomsim.tour_claims`.

- [ ] **Step 3: Write `tour_claims.py`**

```python
"""Resolve the numeric claims a guided-tour step makes, against the real engine.

Tour content is data (``web/src/tours/*.json``) precisely so this module can
read exactly what the browser renders. The alternative, restating each claim in
Python, would break single-source-of-truth on the very thing being checked.

The dispatch is deliberately narrow. A resolver that silently returns the wrong
quantity reports a green tick on prose that lies, which is worse than having no
test at all, so every kind here is pinned to a closed-form value in
``tests/test_tour_claims.py`` before any tour leans on it. Adding a kind means
adding a function, an entry in ``_RESOLVERS``, an entry in ``CLAIM_KINDS`` in
``web/src/tours/types.ts``, and a test against a value known independently.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from atomsim.analytic.hydrogen import energy, mean_radius
from atomsim.atoms import ATOM_KEYS
from atomsim.constants import BOHR_RADIUS_PM, HARTREE_EV
from atomsim.hf_atom import hf_atom, hf_valence_ionization_energy
from atomsim.spectra import transition_lines
from atomsim.systems import get_system

CLAIM_KINDS: tuple[str, ...] = (
    "energy_eV",
    "mean_r_pm",
    "wavelength_nm",
    "ionization_eV",
)

#: Repo root, from ``src/atomsim/tour_claims.py``.
_TOUR_DIR = Path(__file__).resolve().parents[2] / "web" / "src" / "tours"


def load_tours() -> list[dict[str, Any]]:
    """Every tour, read from the same JSON the browser bundles."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(_TOUR_DIR.glob("*.json"))]


def iter_claims() -> Iterator[tuple[str, str, dict[str, Any]]]:
    """(tour id, step id, claim) for every claim in every tour.

    A claim inherits the step's ``state`` for anything it does not name itself,
    restricted to the keys the resolvers read. Restricted rather than merged
    wholesale so that a step changing a display toggle can never quietly change
    what one of its claims asserts.
    """
    inherit = (
        "system", "n", "l", "m", "model",
        "fineStructure", "dirac", "exchange", "pauli",
    )
    for tour in load_tours():
        for step in tour["steps"]:
            state = step.get("state", {})
            for claim in step.get("claims", []):
                merged = {k: state[k] for k in inherit if k in state}
                merged.update(claim)
                yield tour["id"], step["id"], merged


def _system_of(claim: dict[str, Any]):
    return get_system(claim.get("system", "h"))


def _energy_ev(claim: dict[str, Any]) -> float:
    system = _system_of(claim)
    q = energy(claim["n"], Z=system.Z, mu_ratio=system.mu_ratio.value)
    return q.value * HARTREE_EV


def _mean_r_pm(claim: dict[str, Any]) -> float:
    system = _system_of(claim)
    q = mean_radius(claim["n"], claim["l"], Z=system.Z, mu_ratio=system.mu_ratio.value)
    return q.value * BOHR_RADIUS_PM


def _wavelength_nm(claim: dict[str, Any]) -> float:
    system = _system_of(claim)
    n_up = claim["n_upper"]
    n_lo = claim["n_lower"]
    lines = transition_lines(system, n_max=max(n_up, n_lo), fine_structure=False)
    for line in lines.lines:
        if line.n_upper == n_up and line.n_lower == n_lo:
            return line.wavelength.value
    raise ValueError(f"no {n_up} -> {n_lo} line in {system.key}")


def _ionization_ev(claim: dict[str, Any]) -> float:
    key = claim.get("system", "h")
    if key not in ATOM_KEYS:
        raise ValueError(f"ionization_eV needs a many-electron atom, got {key!r}")
    result = hf_atom(key, exchange=claim.get("exchange", True), pauli=claim.get("pauli", True))
    return hf_valence_ionization_energy(result).value * HARTREE_EV


_RESOLVERS = {
    "energy_eV": _energy_ev,
    "mean_r_pm": _mean_r_pm,
    "wavelength_nm": _wavelength_nm,
    "ionization_eV": _ionization_ev,
}


def resolve_claim(claim: dict[str, Any]) -> float:
    """The engine's answer for one claim, in the claim's stated unit.

    Raises rather than defaulting on a missing input: defaulting ``n`` to 1
    would let a claim about the 3d silently check the 1s and pass.
    """
    kind = claim.get("of")
    if kind not in _RESOLVERS:
        raise ValueError(f"unknown claim kind {kind!r}; known: {', '.join(CLAIM_KINDS)}")
    return _RESOLVERS[kind](claim)
```

- [ ] **Step 4: Fix the `hf_atom` entry point if the name differs**

Run: `python -c "from atomsim.hf_atom import hf_atom"`

If that fails, run `grep -n "^def " src/atomsim/hf_atom.py` and use the function that solves an atom from its key and returns an `HFResult`. Adjust the import and `_ionization_ev` accordingly; the signature is what the test pins, not the name.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_tour_claims.py -v`
Expected: PASS. If `test_ionization_energy_of_helium` is outside its tolerance, do not widen the tolerance to fit. Print the value, check it against `tests/test_hf_atom.py`'s reference expectations, and correct the *expected* number in the test to what the engine's own validated reference says.

- [ ] **Step 6: Lint and commit**

```bash
ruff check .
git add src/atomsim/tour_claims.py tests/test_tour_claims.py
git commit -m "Check every number a tour quotes against the engine

Tour content is data so this can read exactly what the browser renders;
restating the claims in Python would break single-source-of-truth on the
very thing being checked.

The dispatch is four kinds and each is pinned to a closed-form value
before any tour leans on it, because a resolver that quietly returns the
wrong quantity would pass every claim in every tour while checking
nothing. Missing inputs raise rather than defaulting: defaulting n to 1
would let a claim about the 3d silently check the 1s and pass."
```

---

## Task 9: The prose lint

**Files:**
- Create: `web/src/tours/prose.ts`
- Create: `web/src/tours/prose.test.ts`

**Interfaces:**
- Produces: `measurementsIn(text: string): string[]`.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import { TOURS } from "./registry";
import { measurementsIn } from "./prose";

describe("measurementsIn", () => {
  it("finds a number carrying a unit", () => {
    expect(measurementsIn("sits at -3.40 eV, always")).toEqual(["-3.40 eV"]);
    expect(measurementsIn("121.567 nm in vacuum")).toEqual(["121.567 nm"]);
    expect(measurementsIn("about 53 pm across")).toEqual(["53 pm"]);
  });

  it("ignores numbers that are not measurements", () => {
    // Orbital labels, counts, and ordinals are not claims about a value, and
    // a lint that flagged them would be turned off within a week.
    expect(measurementsIn("the 2p has four lobes, not five")).toEqual([]);
    expect(measurementsIn("step 3 of 11")).toEqual([]);
    expect(measurementsIn("a 1s2 2s2 2p6 core")).toEqual([]);
  });

  it("does not mistake a unit inside a word", () => {
    // "even" ends in "en", "nm" appears inside no English word, but "K" for
    // kelvin must not match the K in a capitalised word.
    expect(measurementsIn("10 Kelvins of nothing")).toEqual([]);
    expect(measurementsIn("10 K of nothing")).toEqual(["10 K"]);
  });

  it("catches every unit the tours actually use", () => {
    for (const u of ["eV", "nm", "pm", "bohr", "K", "%"]) {
      expect(measurementsIn(`7 ${u}`), u).toHaveLength(1);
    }
  });
});

describe("tour prose", () => {
  it("backs every measurement with a claim", () => {
    // The failure this guards: a step quotes a number, the engine improves,
    // and the prose keeps asserting the old value with nothing to catch it.
    for (const t of TOURS) {
      for (const s of t.steps) {
        const found = s.body.flatMap(measurementsIn);
        if (found.length > 0) {
          expect(
            s.claims?.length ?? 0,
            `${t.id}/${s.id} quotes ${found.join(", ")} with no claim behind it`,
          ).toBeGreaterThan(0);
        }
      }
    }
  });
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd web && npx vitest run src/tours/prose.test.ts`
Expected: FAIL, cannot resolve `./prose`.

- [ ] **Step 3: Write `prose.ts`**

```ts
/**
 * Units a tour quotes. Order matters: longer names first, so "bohr" is not
 * matched as "b" by a shorter alternative added later.
 */
const UNITS = ["bohr", "eV", "nm", "pm", "MHz", "GHz", "K", "%"];

/**
 * Numbers in a body that carry a unit, and are therefore claims about a value.
 *
 * Targeting a unit rather than any numeral is what keeps "the 2p", "four
 * lobes", "step 3 of 11" and "1s2 2s2 2p6" out of the results. A lint that
 * flagged those would be turned off within a week, and a lint that is off
 * catches nothing.
 *
 * The trailing boundary is what stops "10 K" matching inside "10 Kelvins":
 * a unit has to be the whole token, not its first letter.
 */
export function measurementsIn(text: string): string[] {
  const units = UNITS.join("|");
  const re = new RegExp(`-?\\d+(?:\\.\\d+)?\\s(?:${units})(?![A-Za-z])`, "g");
  return text.match(re) ?? [];
}
```

- [ ] **Step 4: Run the tests**

Run: `cd web && npx vitest run src/tours/prose.test.ts`
Expected: PASS, 5 tests. The two existing flagship steps quote no measurements, so the content test passes trivially until Task 10.

- [ ] **Step 5: Commit**

```bash
cd .. && git add web/src/tours/ && git commit -m "Require a claim behind any number a tour quotes

Targets a numeral carrying a unit rather than any numeral at all, so the
2p, four lobes and 1s2 2s2 2p6 do not trip it. A lint with false
positives gets turned off, and a lint that is off catches nothing."
```

---

## Task 10: The flagship's remaining eight steps

**Files:**
- Modify: `web/src/tours/hydrogen-honestly.json`
- Modify: `web/src/components/Controls.tsx` (anchors `m-picker`, `dirac-toggle`, `spectrum-options`, `badges`)

Write the steps one at a time and run both suites after each. A claim that is out by more than its tolerance is the system working: fix the prose, or fix the tolerance if the number was right and the tolerance was too tight, but never widen a tolerance to hide a real disagreement.

- [ ] **Step 1: Append steps 3 to 6**

```json
    {
      "id": "n-sets-size-l-sets-shape",
      "title": "n sets the size, l sets the shape",
      "body": [
        "Same n, different l. The 2s is a ball with a gap inside it and the 2p is two lobes, and the gap is a real node: a surface where the electron is never found.",
        "The mean radius moves too. The 2s sits at 317 pm and the 2p at 265 pm, so the rounder state is also the bigger one."
      ],
      "state": { "n": 2, "l": 1, "m": 0, "view": "cloud" },
      "spotlight": "l-picker",
      "claims": [
        { "of": "mean_r_pm", "n": 2, "l": 0, "is": 317.5, "tol": 1.0 },
        { "of": "mean_r_pm", "n": 2, "l": 1, "is": 264.6, "tol": 1.0 }
      ]
    },
    {
      "id": "accidental-degeneracy",
      "title": "Same energy, different shape",
      "body": [
        "Those two states look nothing alike, and they sit at exactly the same -3.40 eV. Nothing about their shapes says they should.",
        "This is the accidental degeneracy, and accidental is the right word: it belongs to the 1 over r potential and to nothing else. Put a second electron in the way to screen the nucleus and it breaks immediately, which is why no other atom does this."
      ],
      "state": { "n": 2, "l": 1, "m": 0, "view": "levels" },
      "spotlight": "view-list",
      "claims": [
        { "of": "energy_eV", "n": 2, "is": -3.4014, "tol": 0.002 }
      ]
    },
    {
      "id": "the-basis-is-a-choice",
      "title": "m, and a basis is a choice",
      "body": [
        "The chemistry orbitals you were taught, p_x and p_y, are not more real than the complex states with definite m. They are sums of them, and which set you call the answer is a choice of basis rather than a fact about the atom.",
        "Both are first-class here and the badge says which one you are looking at. Switch and watch the same physics wear a different face."
      ],
      "state": { "n": 2, "l": 1, "m": 1, "view": "cloud", "basis": "real" },
      "spotlight": "basis-picker"
    },
    {
      "id": "a-contour-not-a-boundary",
      "title": "A surface is a contour, not a boundary",
      "body": [
        "Textbook orbital pictures draw a surface and leave you to assume the electron is inside it. This one encloses 90% of the electron, and it says so, because the other 10% is genuinely outside and goes on forever.",
        "Drag the fraction and the surface breathes. There is no radius at which the atom stops."
      ],
      "state": { "n": 3, "l": 2, "m": 0, "view": "cloud", "surfaceMode": "surface", "isoFraction": 0.9 },
      "spotlight": "surface-controls"
    },
```

- [ ] **Step 2: Run both suites**

```bash
cd web && npx vitest run src/tours/ && cd .. && pytest tests/test_tour_claims.py -v
```

Expected: PASS. If a `mean_r_pm` claim fails, print the engine's value with
`python -c "from atomsim.tour_claims import resolve_claim; print(resolve_claim({'of':'mean_r_pm','system':'h','n':2,'l':0}))"`
and correct the prose and the claim to that number.

- [ ] **Step 3: Append steps 7 to 10**

```json
    {
      "id": "the-degeneracy-was-the-models-lie",
      "title": "So the degeneracy was the model's lie",
      "body": [
        "Turn on fine structure and the level that was one line becomes several. The 2s and the 2p were never exactly equal; the model that said so had left three things out.",
        "The badge changes from EXACT to APPROXIMATION, and it names what is missing rather than implying nothing is. This is an alpha-squared perturbation, correct to that order and no further."
      ],
      "state": { "n": 2, "l": 1, "m": 0, "view": "levels", "fineStructure": true },
      "spotlight": "fine-structure"
    },
    {
      "id": "dirac-and-the-gap-it-leaves",
      "title": "Dirac, exactly, and the gap it still leaves",
      "body": [
        "The Dirac equation solves hydrogen in closed form, relativity included, so this goes back to EXACT. It gives the fine structure the perturbation was approximating.",
        "It is still not the measured answer. The 2s and 2p at the same j are exactly degenerate here and are not in the laboratory, and that difference is the Lamb shift. This engine does not model it, and says so rather than quietly closing the gap."
      ],
      "state": { "n": 2, "l": 1, "m": 0, "view": "levels", "fineStructure": true, "dirac": true },
      "spotlight": "dirac-toggle"
    },
    {
      "id": "levels-become-lines",
      "title": "Levels become lines",
      "body": [
        "Every arrow between two levels is a photon, and its wavelength is fixed by the gap. The 2 to 1 transition gives 121.57 nm and the 3 to 2 gives 656.47 nm, the red line in every hydrogen discharge tube you have seen.",
        "The dots on the axis are vendored NIST measurements, not a fit. The residual panel shows the engine agreeing with them, which is the only reason to believe anything above."
      ],
      "state": { "n": 3, "l": 2, "m": 0, "view": "spectrum" },
      "spotlight": "view-list",
      "claims": [
        { "of": "wavelength_nm", "n_upper": 2, "n_lower": 1, "is": 121.57, "tol": 0.02 },
        { "of": "wavelength_nm", "n_upper": 3, "n_lower": 2, "is": 656.47, "tol": 0.05 }
      ]
    },
    {
      "id": "what-the-picture-cost",
      "title": "What the picture cost",
      "body": [
        "Every badge on this screen is a claim about how much to trust what is next to it. EXACT means a closed-form solution of the stated model. APPROXIMATION names its assumptions. VISUAL LIBERTY means the choice was presentational and it is disclosed anyway.",
        "The nucleus marker is a liberty. The colour map is a liberty. The point size is a liberty. None of them changes a number, and all of them are labelled, because a picture that hides its choices is a picture you cannot check."
      ],
      "state": { "n": 3, "l": 2, "m": 0, "view": "cloud", "colorMode": "density" },
      "spotlight": "state-card"
    }
```

- [ ] **Step 4: Add the anchors these steps name**

`dirac-toggle` on the Dirac checkbox's `<label>` in `Controls.tsx`. `state-card` on the state card in `InfoPanel.tsx` if Task 6 did not already add it.

- [ ] **Step 5: Run everything**

```bash
cd web && npm test && npm run build
cd .. && pytest tests/test_tour_claims.py -v && ruff check .
```

- [ ] **Step 6: Walk it in the browser**

Open `http://127.0.0.1:8021/?tour=hydrogen-honestly&step=0` and click through all ten. Check that each step's picture matches its prose, the ring lands on the right control, and stepping backward from the fine-structure step leaves fine structure off.

- [ ] **Step 7: Commit**

```bash
git add web/src/ && git commit -m "Write the flagship tour

Ten steps from the only atom we can solve to the ledger of what the
picture cost. Every quoted number carries a claim the Python suite
resolves against the engine, so the day one of them stops being true CI
says so rather than a reader finding out."
```

---

## Task 11: The three side tours

**Files:**
- Create: `web/src/tours/break-the-physics.json`, `many-electrons.json`, `a-real-spectrum.json`
- Modify: `web/src/tours/registry.ts`
- Modify: `web/src/components/Controls.tsx`, `web/src/components/WhatIfView.tsx`, `web/src/components/SpectrumView.tsx` (new anchors)

This task is content plus anchors. No new code, no new claim kinds unless a step wants a quantity the four do not cover, in which case add the kind by the recipe in `tour_claims.py`'s docstring and pin it with its own closed-form test first.

- [ ] **Step 1: Write `break-the-physics.json`**

```json
{
  "id": "break-the-physics",
  "title": "Break the physics",
  "blurb": "Change a constant, change a force law, switch off exclusion. Rigorously.",
  "steps": [
    {
      "id": "turn-up-alpha",
      "title": "Turn up the fine-structure constant",
      "body": [
        "Alpha is about 1 over 137 and nothing in the theory says why. Here you can move it and watch the level structure respond.",
        "The badge says COUNTERFACTUAL, which is a specific promise: the physics was altered deliberately and then solved properly under the altered rules. It is not a cartoon."
      ],
      "state": { "view": "whatif", "n": 2, "l": 1, "m": 0 },
      "spotlight": "const-sliders"
    },
    {
      "id": "change-the-force-law",
      "title": "Change the force law itself",
      "body": [
        "Coulomb is 1 over r. Replace it with a Yukawa, a power law, or an expression you type, and the same numerical solver finds the states of whatever you wrote.",
        "This is why the engine is a radial Schrodinger solver rather than a table of hydrogen formulas. The counterfactual and the real atom go through the same code."
      ],
      "state": { "view": "forcelaw" },
      "spotlight": "force-preset"
    },
    {
      "id": "distinguishable-electrons",
      "title": "Electrons that repel but do not exclude",
      "body": [
        "Switch off exchange and helium's electrons still repel, but they stop being identical particles. The energy moves, and the direction it moves is the whole content of the exchange term.",
        "Nothing here is a fudge. It is the same Hartree-Fock solver with one term dropped, and the result is labelled for what it is."
      ],
      "state": { "system": "he", "model": "hf", "view": "radial", "exchange": false },
      "spotlight": "exchange-toggle"
    },
    {
      "id": "pauli-off",
      "title": "Switch off exclusion and watch the atom collapse",
      "body": [
        "With no occupancy cap every electron falls into the 1s. Argon becomes 1s to the eighteenth, and the periodic table stops existing.",
        "This is the strongest counterfactual in the app and it contains the previous one: antisymmetry is what exclusion is, so you cannot switch off one and keep the other."
      ],
      "state": { "system": "ar", "model": "hf", "view": "radial", "exchange": false, "pauli": false },
      "spotlight": "pauli-toggle"
    }
  ]
}
```

- [ ] **Step 2: Write `many-electrons.json`**

```json
{
  "id": "many-electrons",
  "title": "More than one electron",
  "blurb": "Two honest models of the same atom, and the size of their disagreement.",
  "steps": [
    {
      "id": "screening",
      "title": "The cheap model: pretend the nucleus is weaker",
      "body": [
        "A screened central field replaces the other electrons with a fitted reduction of the nuclear charge. It is fast, it is APPROXIMATION, and its parameters came from a 1974 paper.",
        "It works well enough to be worth having and badly enough to be worth checking, which is what the next step does."
      ],
      "state": { "system": "na", "model": "gsz", "view": "radial" },
      "spotlight": "model-picker"
    },
    {
      "id": "hartree-fock",
      "title": "The honest one: solve for the field",
      "body": [
        "Hartree-Fock gives every subshell its own field and iterates until the field and the orbitals agree. Nothing is fitted.",
        "Sodium's ionization energy comes out near 4.95 eV against a measured 5.14. The gap is correlation, which this model does not have, and that gap is stated rather than tuned away."
      ],
      "state": { "system": "na", "model": "hf", "view": "radial" },
      "spotlight": "model-picker",
      "claims": [
        { "of": "ionization_eV", "system": "na", "model": "hf", "is": 4.95, "tol": 0.15 }
      ]
    },
    {
      "id": "atoms-gsz-cannot-do",
      "title": "Atoms the cheap model cannot do at all",
      "body": [
        "Sulfur and chlorine have no published screening parameters, so the fitted model cannot draw them. Hartree-Fock needs none and solves them from scratch.",
        "The app moves you to Hartree-Fock rather than offering a model that would refuse every request, which is the difference between a limit and a bug."
      ],
      "state": { "system": "cl", "model": "hf", "view": "radial" },
      "spotlight": "system-picker"
    },
    {
      "id": "both-densities",
      "title": "Both, on one axis",
      "body": [
        "The total radial density is the one thing both models claim to describe, so it is the one place they can be compared without arguing about conventions.",
        "The disagreement is drawn rather than described. Where the curves separate is where the fitted model is spending its error."
      ],
      "state": { "system": "ar", "model": "hf", "view": "radial", "compare": true },
      "spotlight": "compare-toggle"
    }
  ]
}
```

- [ ] **Step 3: Write `a-real-spectrum.json`**

```json
{
  "id": "a-real-spectrum",
  "title": "A spectrum you could observe",
  "blurb": "From which lines exist to what a telescope would actually record.",
  "steps": [
    {
      "id": "which-lines-are-strong",
      "title": "Which lines are strong",
      "body": [
        "Selection rules say which transitions happen. The dipole engine says how fast, and the bars are scaled by the spontaneous emission rate.",
        "That is a rate, not a brightness. Nothing about how many atoms are in the upper level has been said yet."
      ],
      "state": { "n": 4, "l": 3, "m": 0, "view": "spectrum", "intensities": true },
      "spotlight": "view-list"
    },
    {
      "id": "how-many-atoms",
      "title": "How many atoms are actually there",
      "body": [
        "Boltzmann populates the levels, Saha decides how much of the gas is still neutral, and now the bars mean emission rather than capability.",
        "Move the temperature and watch lines appear and vanish. Above about twenty thousand kelvin almost nothing is neutral and every line fades, however hot it gets."
      ],
      "state": { "n": 4, "l": 3, "m": 0, "view": "spectrum", "intensities": true, "thermal": true, "temperatureK": 10000 },
      "spotlight": "spectrum-options"
    },
    {
      "id": "what-shape-is-a-line",
      "title": "What shape is a line",
      "body": [
        "A line is not a spike. The upper level's finite lifetime gives Lorentzian wings, thermal motion gives a Gaussian core, and the real profile is their convolution.",
        "The full-range axis cannot show this: a line there is a fraction of a pixel wide. Click one to plot it linearly and see the actual shape."
      ],
      "state": { "n": 4, "l": 3, "m": 0, "view": "spectrum", "intensities": true, "thermal": true, "profile": true },
      "spotlight": "spectrum-options"
    },
    {
      "id": "when-a-line-stops-measuring",
      "title": "When a line stops measuring anything",
      "body": [
        "Put more gas in the way and a weak line grows in proportion. A strong one does not: once its core is black it cannot absorb more, and a hundred times the gas barely widens it.",
        "That is the curve of growth, and its middle branch is where a line stops being a measurement of how much gas there is."
      ],
      "state": { "n": 4, "l": 3, "m": 0, "view": "spectrum", "intensities": true, "thermal": true, "profile": true },
      "spotlight": "spectrum-options"
    },
    {
      "id": "absorption",
      "title": "Put the gas in front of a star",
      "body": [
        "Now the same atoms sit between you and a continuum. Each line absorbs using only the atoms in its own lower level, which is why the Lyman lines go black while the Balmer lines are invisible in the very same gas.",
        "That is the thing an emission spectrum cannot tell you, and it is why absorption spectroscopy is how we know what stars are made of."
      ],
      "state": { "n": 4, "l": 3, "m": 0, "view": "spectrum", "intensities": true, "thermal": true, "absorption": true, "logColumn": 22 },
      "spotlight": "spectrum-options"
    }
  ]
}
```

- [ ] **Step 4: Register them**

```ts
import aRealSpectrum from "./a-real-spectrum.json";
import breakThePhysics from "./break-the-physics.json";
import hydrogenHonestly from "./hydrogen-honestly.json";
import manyElectrons from "./many-electrons.json";
import type { Tour } from "./types";

export const TOURS: Tour[] = [
  hydrogenHonestly as Tour,
  breakThePhysics as Tour,
  manyElectrons as Tour,
  aRealSpectrum as Tour,
];
```

- [ ] **Step 5: Add the anchors these tours name**

`const-sliders` (WhatIfView), `force-preset` (ForceLawView), `exchange-toggle`, `pauli-toggle`, `model-picker`, `system-picker`, `compare-toggle` (Controls.tsx), `spectrum-options` on the checkbox group in `SpectrumView.tsx`.

- [ ] **Step 6: Run everything**

```bash
cd web && npm test && npm run build
cd .. && pytest tests/test_tour_claims.py -v && ruff check .
```

The anchors test names any anchor you missed. The sodium ionization claim is the one number here; if it is out, print the engine's value and correct the prose, do not widen the tolerance.

- [ ] **Step 7: Walk all three in the browser, then commit**

```bash
git add web/src/ && git commit -m "Add the three side tours

Break the physics, more than one electron, and a spectrum you could
observe. Content and anchors only: no new code and no new claim kinds,
which is what the tour format was for."
```

---

## Self-Review

**Spec coverage.** Section 2 (format, `state`, `claims`, `spotlight`) is Tasks 1, 2, 6. Section 2.1's full-reset rule is Task 1 Step 4 with its own test, and Task 3 enforces it against the store. Section 3's file table is Tasks 1 to 7. Section 3's URL addressability is Task 4; the save-and-restore invariant is Task 3 Step 6. Section 3.1's live controls are Task 6 (`pointer-events: none`) and its no-auto-advance rule is met by never building one. Section 4.1's ten steps are Tasks 2 and 10; 4.2's three side tours are Task 11. Section 5's three checks are Tasks 2 and 6 (structural), 8 (physical), 9 (prose lint). Section 6's staging cut falls between Tasks 10 and 11.

One spec item is deliberately reinterpreted: 4.1 step 10 says "the liberties ledger, every disclosed liberty currently on screen, in one place". Building a ledger component is a new surface the spec's own file table does not list, so the step instead lands on a state with several liberties visible and narrates them against the existing `Badge` components. If a real ledger is wanted it should be its own phase.

**Placeholders.** None. Task 8 Step 4 and Task 10 Step 2 are conditional recovery instructions with the exact command to run, not deferred work.

**Type consistency.** `stepState` and `clampStep` are used with those names in Tasks 1, 3 and 6. `tourReset(state, systems)` is defined in Task 3 and used only there. `tourById` is defined in Task 2 and used in Tasks 3, 5, 6, 7. `CLAIM_KINDS` exists twice on purpose, in `types.ts` and `tour_claims.py`, and Task 8's `test_every_declared_kind_resolves` plus Task 2's "only claims quantities the Python resolver implements" are the two halves that keep them equal. Store fields are `tourId`/`stepIndex` throughout; the URL's are `tour`/`step`, bridged in Task 4 Step 5.

**Known risk.** Task 6 Step 5 and Task 11 Step 5 name anchors by description rather than by line number, because the exact markup in `Controls.tsx` was not read while writing this plan. The anchors test is what closes that gap: it fails with the missing name, and the browser check at Task 6 Step 8 is what catches an anchor put somewhere silly.
