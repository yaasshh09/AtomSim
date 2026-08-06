# Phase 1: "Hydrogen, Honestly": Design

**Date:** 2026-07-05 · **Status:** Approved (design interview 2026-07-05) · **Author:** Yash Gupta
**Parent spec:** `2026-07-04-atom-sim-requirements-design.md` (§9, Phase 1) · **Carry-forwards:** Phase 0 final-review findings

---

## 1. Goal and success criteria

Deliver the hydrogen deep-dive: the full analytic + numerical engine with provenance threaded end-to-end, a local Python server, and a browser frontend rendering honest 3D/2D visualizations of hydrogen-like atoms, including the exotic-but-real systems (isotopes, muonic hydrogen, positronium, ions) and perturbative fine structure.

**Phase 1 is done when:**

1. `atomsim serve` (one command, Miniforge Prompt) opens a browser app.
2. The app shows any hydrogen-like (n, l, m) state as: 3D Monte-Carlo point cloud, 2D inferno cross-section, radial R(r) and P(r) plots, energy-level diagram with fine-structure zoom, and computed-vs-NIST spectrum overlay, with a thumbnail gallery of neighboring states for navigation.
3. System presets (H, D, T, μ-H, positronium, He⁺, generic Z) work via exact reduced mass.
4. Real ↔ complex orbital toggle works, with complex phase rendered as hue.
5. **Every displayed physical quantity carries a visible fidelity badge with a click-through inspector** (method, assumptions, error scale, refinement path).
6. Layered math: the app works with math hidden; "show the physics" expands KaTeX-rendered equations.
7. Validation suite green in CI: analytic identities, sampler statistics, fine-structure vs. published values, spectra vs. vendored NIST data with stated tolerances.

## 2. Decisions locked in the 2026-07-05 interview

| Question | Decision |
|---|---|
| Sequencing | **Vertical slice first**, minimal end-to-end path early, then deepen every layer |
| Frontend stack | **React + TypeScript + react-three-fiber** (Three.js under the hood) |
| Scope adds from design inspiration | 2D cross-section view **IN**; thumbnail gallery **IN**; poster mode **DEFERRED to Phase 2** (pairs with What-If Lab) |

## 3. Milestones

Each milestone gets its own implementation plan and lands on `main` with green CI.

- **M1, Walking skeleton (~1.5 wk):** `Field` provenance upgrade; minimal ψ sampler; FastAPI server with the job/progress pattern; Vite+React app showing a rotating point cloud of any (n, l, m) with a real provenance badge. Fixes the three Phase 0 deferred minors (CODATA test tolerance, CI action SHA-pinning, pyproject license form) since those files are touched.
- **M2, Engine depth (~2 wk):** angular/real-basis completeness, fine structure, spectra + NIST comparison, exotic-system preset registry, full validation suite.
- **M3, UI depth (~2.5 wk):** three-panel layout, 2D cross-section view, radial plots, level diagram with fine-structure zoom, spectrum overlay, real/complex + phase-as-hue, honesty inspector, layered math, thumbnail gallery.
- **M4, Polish (~1 wk):** performance tuning, nucleus scale modes (true-scale vs. visible marker with VISUAL LIBERTY), remaining disclosures, docs, demo-script hooks for the Phase 2 guided tour.

## 4. Provenance architecture upgrade (resolves the Phase 0 carry-forward)

**The Phase 0 contradiction:** the Global Constraint said every physical quantity crossing a module boundary is a `Quantity` with `Provenance`, but arrays (`radial_wavefunction() -> np.ndarray`, `RadialSolution.u`) crossed bare, and `mean_radius()` returned a raw `float`.

**Resolution, the amended boundary rule:** every physical value crossing a module boundary is one of:

1. a `Quantity` (scalar + unit + label + provenance),
2. a **`Field`** (new), array-valued physical quantity: `values: np.ndarray`, `grid: np.ndarray`, `unit: str`, `grid_unit: str`, `label: str`, `provenance: Provenance`, or
3. a container dataclass that itself carries a `Provenance` describing how its contents were produced (per-value `Quantity` members may refine it, e.g. per-state energy error estimates).

**Concrete changes:**

- `provenance.py`: add frozen `Field` dataclass (validated: `values` and `grid` same leading shape).
- `analytic/hydrogen.py`: `radial_wavefunction()` returns `Field` (EXACT); `mean_radius()` returns `Quantity` (EXACT).
- `numerics/radial_solver.py`: `RadialSolution` gains `provenance: Provenance` (the NUMERICAL method description currently duplicated across energies); `u`/`r` stay ndarrays *inside* the provenance-carrying container; per-state `energies` keep individual `Quantity` values.
- **Serialization contract:** `Quantity`, `Field`, and `Provenance` each have exactly one canonical JSON form, defined once in the server layer (Pydantic models mirroring the dataclasses 1:1) and consumed by matching TypeScript types. Provenance reaches the browser by construction, a UI element without badge data is a type error, not an oversight.

*Alternative rejected:* container-level provenance only (no `Field`). A bare `np.ndarray` handed to a renderer carries no honesty metadata, precisely the quiet-lying failure mode this project exists to prevent.

## 5. Engine extensions

New subpackage layout: `atomsim/analytic/angular.py`, `atomsim/analytic/fine_structure.py`, `atomsim/sampling.py`, `atomsim/spectra.py`, `atomsim/systems.py` (final naming may be adjusted in plans; boundaries as described here).

### 5.1 Angular structure and full wavefunctions

- Complex spherical harmonics Y_lm (via `scipy.special`), and **real orbitals** as the standard ± combinations (p_x, p_y, d_xy, …) with the chemistry-vs-physics basis choice surfaced as a labeled teaching moment.
- Full ψ_nlm(r, θ, φ) = R_nl(r) · Y_lm(θ, φ); densities |ψ|²; complex phase retained for phase-as-hue rendering.
- Both bases are first-class engine outputs, not frontend transformations.

### 5.2 Monte-Carlo position sampling (in the engine, never the frontend)

Sampling is physics and must carry provenance:

- Radial: inverse-CDF sampling from P(r) = r²|R_nl(r)|² on a dense grid (numerically exact to grid resolution; provenance states grid).
- Angular: sampling from |Y_lm|², for complex Y_lm, φ is uniform and cos θ follows a 1D inverse-CDF; for real orbitals, 2D sampling on (cos θ, φ).
- Output: positions as float32 arrays wrapped in a provenance-carrying result (NUMERICAL; method, N, grid resolution, RNG seed stated).

*Alternative rejected:* frontend/GPU sampling from a density texture, faster for huge N but makes the sampling method invisible to the honesty layer.

### 5.3 Fine structure (perturbative, labeled)

- ΔE(n, l, j) to order α²: spin-orbit + relativistic kinetic energy + Darwin term (the standard combined result as a function of n, j), reduced-mass-aware, `APPROXIMATION` with stated α⁴-scale error and refinement note pointing at the Phase 3 Dirac solution.
- Powers the level-diagram fine-structure zoom and spectral doublets.

### 5.4 Spectra and NIST comparison

- Line lists from level differences with selection rules (Δl = ±1; Δj = 0, ±1 when fine structure is on), wavelengths in nm and energies in eV, per-line provenance.
- **Vendored NIST reference data:** a small curated JSON in-repo (H I lines at minimum; D and He II as the presets require), with citation, retrieval date, and quoted uncertainties. No live NIST queries (offline-hostile, CI-flaky, irreproducible).
- Comparison API: computed vs. reference with explicit tolerance statements per fidelity tier.

### 5.5 System preset registry

`systems.py`: H, D, T, muonic hydrogen, positronium, He⁺, and generic hydrogen-like (user Z), each supplying nuclear charge and exact reduced-mass ratio with provenance (mass values via CODATA/scipy). Mostly plumbing, the physics (μ-scaling) already exists and is tested.

## 6. Server

**FastAPI + uvicorn**, in-package (`atomsim/server/`), launched by a console entry point: `atomsim serve` (serves API + built frontend static files; opens browser).

- **Pydantic models** mirror `Quantity`/`Field`/`Provenance` 1:1, the single canonical JSON contract of §4.
- **Fast path (plain GET, < ~100 ms):** presets list; state info (energies, ⟨r⟩, fine-structure shifts); radial R(r)/P(r) Fields (downsampled for plotting); spectrum lines + NIST comparison.
- **Job pattern (from M1, so it is architecture rather than retrofit):** POST create job → `{job_id}`; WebSocket progress channel; result fetched as two parts, `meta` (JSON: provenance, count, dtype, layout) and `data` (binary `application/octet-stream`, float32). Used for MC samples (100k points ≈ 1.2 MB xyz float32) and 2D plane-density grids (e.g. 512² floats); later reused for volumetrics.
- **Thumbnails:** server-rendered small PNGs (matplotlib Agg, inferno) with cache; the same rendering path Phase 2's poster mode will reuse.
- Local-only binding (127.0.0.1); no auth (single-user local app by design).

## 7. Frontend

**Vite + React + TypeScript (strict) + react-three-fiber**; zustand for app state; KaTeX for math; **custom SVG plots with `d3-scale`** (no charting library, Plotly rejected: heavy, fights the scientific-cinematic aesthetic, and the level diagram/spectrum are custom drawings anyway; hand-rolled plots are the better portfolio artifact). Lives in `web/` at repo root; production build ships as static files served by `atomsim serve`.

### 7.1 Layout (from the user's design references)

Three-panel dark UI, scientific-cinematic:

- **Left, info card:** system name, state label, live readouts (E with fine-structure toggle, ⟨r⟩ in a₀/pm, |L|, node counts, sample count, FPS), each physical quantity with its badge. Color-by-l legend.
- **Center, canvas:** the active view, rotatable/zoomable.
- **Right, controls:** system preset ladder (element/Z → n → l → m), basis toggle (real/complex), view-mode select, sampling controls, "show the physics" expander.
- Aesthetic: dark background, green-on-dark accent family, inferno/magma colormap for density-only views, complex phase as hue (separate, clearly labeled mode), formula-as-typography via KaTeX. Purely aesthetic choices (glow, point size) get VISUAL LIBERTY disclosure.

### 7.2 Views

1. **3D point cloud**, custom point shader; density-colormap mode and phase-as-hue mode (complex basis).
2. **2D cross-section**, engine-computed plane grid through the z-axis, inferno, with an explicit "what is plotted" label (|ψ|² vs. ψ, the honesty fix over the classic poster).
3. **Radial plots**, R(r) and P(r) = r²|R|², with ⟨r⟩ marker.
4. **Energy-level diagram**, degeneracy labels, fine-structure zoom (labeled APPROXIMATION), transition arrows.
5. **Spectrum**, computed lines vs. NIST overlay with residuals.
6. **Thumbnail gallery**, neighboring (n, l, m) states as server-rendered thumbnails; click to switch.

### 7.3 Honesty UI

Badge component renders on every displayed quantity (color-coded by fidelity tier); clicking opens the inspector drawer: method, assumptions list, error estimate, "what would make this more accurate." Data arrives with provenance by construction (§4); the badge component *requires* provenance props (type-enforced).

### 7.4 Layered math

Every view usable with math hidden. "Show the physics" expands per-view KaTeX content: governing equation, quantum-number meaning, derivation sketch. Content authored per view in M3.

## 8. Testing and CI

- **Python (TDD as Phase 0):** Y_lm orthonormality by quadrature; real-basis unitarity; sampler χ²/KS tests of r- and angle-histograms against P(r) and |Y_lm|² (seeded RNG, stated tolerance); fine structure vs. published hydrogen values (e.g. 2p splitting / Lyman-α doublet); spectrum vs. vendored NIST within per-tier tolerance; serialization round-trip tests (every payload carries provenance); FastAPI endpoints via TestClient, including job lifecycle.
- **TypeScript:** vitest for logic, binary payload decoding, scale/axis math, state transitions, badge-prop completeness. No pixel/screenshot tests in Phase 1.
- **CI:** existing Python job (windows-latest) + new Node job (windows-latest: `tsc --noEmit`, vitest, production build). Actions pinned by SHA (Phase 0 deferred minor). CI stays green on every push to `main`.

## 9. Explicitly not in Phase 1

What-If Lab UI, screening/multi-electron models, Pauli/classical-ghost toggles, guided tour (all Phase 2); poster mode (Phase 2, per interview); volumetric rendering, Dirac, hyperfine, Zeeman/Stark, Hartree-Fock, time dependence, free-form V(r) (Phase 3+).

## 10. Risks and mitigations

- **OneDrive vs. `node_modules`:** the repo lives under OneDrive; syncing tens of thousands of `node_modules` files causes churn and file-lock errors. Mitigation: `node_modules` is git-ignored and created as a **directory junction to a local non-synced path** (e.g. `%LOCALAPPDATA%\atomsim\node_modules`), set up by a documented script in M1.
- **New toolchain for Yash (Node.js):** M1 plan includes complete copy-paste install/verify blocks (per working agreement: name the prompt, include `cd`, describe success).
- **Iris Xe GPU budget:** point clouds ≤ ~500k points render comfortably; view-level FPS readout keeps performance honest; volumetrics stay in Phase 3.
- **Solo schedule:** vertical-slice order means a demoable artifact exists from M1 onward. All §7.2 views are firm scope per the interview; if the schedule guardrail ever forces a cut, that is a renegotiation with Yash (likely candidates: spectrum residual detail, gallery breadth), never a silent drop. Protected centerpiece: 3D cloud + cross-section + radial plots.
