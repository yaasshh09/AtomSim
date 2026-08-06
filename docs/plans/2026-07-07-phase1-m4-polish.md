# Phase 1 M4: Polish Implementation Plan

> Scope from spec
> `2026-07-05-phase1-hydrogen-honestly-design.md` §3 M4: performance tuning,
> nucleus scale modes (true-scale vs visible marker with VISUAL LIBERTY),
> remaining disclosures, docs, demo-script hooks for the Phase 2 guided tour.
> Scope choices below derive from the approved spec, not new design ground.

**Goal:** close out Phase 1 with the four M4 items, each landing green
(`pytest` + `npm test` + build) and committed conventionally.

## Global constraints

Same as M3 (PYTHONUTF8 before conda run; TDD for engine/server and web logic;
boundary rule: every physical value crossing a boundary is a
`Quantity`/`Field`/provenance-carrying container; no frontend physics;
commits end with the Fable trailer).

---

### Task 1: Engine: nuclear rms charge radii on system presets

- [x] `System` gains `nuclear_radius: Quantity | None` (unit **bohr**, engine
  canonical; label "nuclear rms charge radius").
  - h, mu-h: scipy CODATA `proton rms charge radius` (0.84075 fm).
  - d: `deuteron rms charge radius` (2.12778 fm).
  - he+: `alpha particle rms charge radius` (1.6785 fm).
  - t: literature 1.7591 ± 0.0363 fm, Angeli & Marinova (2013) ADNDT 99, 69
    (not in scipy CODATA table), provenance cites the source.
  - ps: **None**, the "nucleus" is a positron, a point lepton; honesty note in
    provenance-free absence + description. `hydrogen_like`: None (no data).
  - Fidelity EXACT (reference measurement, not a model approximation), with
    `error_estimate` = quoted uncertainty converted to bohr.
- [x] Tests: values/units/provenance for all six presets; ps/generic → None.

### Task 2: Server: nucleus quantities in SystemModel

- [x] `constants.BOHR_RADIUS_FM`; `_to_fm` conversion next to `_to_pm`.
- [x] `SystemModel` gains `nuclear_radius: QuantityModel | None` (bohr) and
  `nuclear_radius_fm: QuantityModel | None`, /api/systems and /api/state both
  carry it (SystemModel is embedded in both).
- [x] Tests: h has 0.84075 fm (±), ps has null, unit strings honest.

### Task 3: Web: nucleus scale modes in the cloud view

- [x] types.ts mirrors SystemModel additions.
- [x] Store: `nucleusMode: "hidden" | "true-scale" | "marker"` (default
  "marker") + setter; store test.
- [x] `lib/liberties.ts`: `NUCLEUS_MARKER_LIBERTY` provenance (marker radius is
  camera-relative presentation; true position is exact: origin).
- [x] CloudView: nucleus sphere at origin, true-scale uses r_rms in bohr
  (invisibly small at atomic zoom: the honest teaching moment, stated in a
  caption with the EXACT radius); marker uses camera-relative radius with the
  VISUAL LIBERTY badge. Positronium: no sphere; caption "positron: point
  lepton". Controls: nucleus select. InfoPanel: r_rms readout (fm) with badge.
- [x] Tests: nucleus sizing/caption logic in a lib module (`lib/nucleus.ts`).

### Task 4: Web: URL deep links (demo-script hooks for the Phase 2 tour)

- [x] `lib/urlState.ts`: `parseAppUrl(search)` → validated partial state
  (n/l/m clamped via `clampState`, enum whitelists for view/colour/basis/
  nucleus, fine structure flag, system key sanitized); `serializeAppUrl(state)`
  → canonical query string (defaults omitted). Tests both ways + fuzz-ish
  invalid inputs.
- [x] Wiring: apply parsed URL once at store creation; `history.replaceState`
  on every relevant store change (subscribe). The Phase 2 guided tour can
  drive the app by URL alone, that is the demo-script hook surface.
- [x] README: "Deep links" note.

### Task 5: Performance: bundle code-split (Iris Xe budget already met)

- [x] Measured FPS on the dev Iris Xe: 29-62 at 100k points (M3 walkthrough), 
  within the spec's ≤500k comfort budget; no render-path change needed.
- [x] vite `manualChunks`: split `three`/`react`/`katex` vendors so the app
  chunk drops below the 500 kB warning; verify `npm run build` output.

### Task 6: Ship

- [x] Remaining-disclosures audit (RENDER_LIBERTIES covers z-vertical, glow,
  gamma; add nucleus-marker liberty from Task 3; spectrum log-λ axis is
  labeled on-axis).
- [x] Full gates; extended Playwright walkthrough (nucleus modes × systems
  incl. positronium, deep-link load); screenshot review.
- [x] README M4/Phase-1-complete status; check plan boxes; push; CI green;
  memory update.
