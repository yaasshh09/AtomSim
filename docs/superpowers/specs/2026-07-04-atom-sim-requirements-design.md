# Atom Sim — Requirements Specification

**Date:** 2026-07-04 · **Status:** Awaiting approval · **Author:** interview between Yash Gupta and Claude (architect)
**Working name:** `atom_sim` (final name is an open item)

---

## 1. Vision

A physically rigorous, deeply customizable quantum-mechanical atom model and visualization platform, serving three purposes at once:

1. **Portfolio piece** — demo-ready for college applications by ~October–November 2026.
2. **Teaching tool** — primary v1 audience: university/QM students; later phases add high-school, general-public, and self-guided modes.
3. **Learning sandbox** — both for users exploring atomic physics and for the author, whose hand-writing of the core solvers is itself a project goal.

**Prime directive: the model never quietly lies about physics.** Every quantity and visual carries provenance. Approximations are labeled and inspectable. Counterfactual physics is unmissably flagged.

### Fidelity taxonomy (used everywhere)

| Badge | Meaning |
|---|---|
| `EXACT` | Closed-form solution of the stated model (e.g., analytic hydrogen-like states) |
| `NUMERICAL` | Converged numerical solution of the stated model, with quantified error |
| `APPROXIMATION` | Honest simplified model (e.g., screening/effective charge), assumptions stated |
| `COUNTERFACTUAL` | Deliberately altered physics (What-If Lab), computed rigorously under the altered rules |
| `VISUAL LIBERTY` | Purely presentational choice (point sizes, glow, visible nucleus marker), disclosed |

---

## 2. Users and the teaching layer

- **v1 primary audience:** university students taking or approaching QM. End-state audiences (later phases): high-school students, curious general public, the author.
- **Layered math disclosure:** every view works with math hidden; an expandable "show the physics" layer reveals the governing equations, quantum numbers, and derivation sketches. One codebase serves all audiences via this layering.
- **v1 teaching scope:** contextual explainers on every control and displayed quantity, plus **one flagship guided tour — "The Hydrogen Atom, Honestly"** — which doubles as the demo script for portfolio viewers.
- **Not in v1:** full lesson system, assessment/quizzes (revisit only if a classroom use case emerges).

---

## 3. Physics requirements

### 3.1 Tier 1 — exactly solvable (v1 core)

- **Hydrogen-like atoms** (1 electron, arbitrary nuclear charge Z): analytic eigenstates and energies, with **exact reduced-mass treatment** — which natively delivers the exotic-but-real systems: deuterium/tritium (isotope shifts), muonic atoms, positronium, highly charged ions, Rydberg states.
- **Fine structure in v1 as labeled perturbative corrections:** spin-orbit coupling, relativistic kinetic energy, Darwin term (`APPROXIMATION` relative to Dirac; exact within stated perturbative order).
- **Later phase:** exact analytic **Dirac equation** solution for hydrogen-like atoms (rare flagship feature); **hyperfine structure** (nuclear spin, the 21 cm line); **Zeeman and Stark effects** — the Hamiltonian architecture treats external fields as first-class perturbations from day one so these are features, not rewrites.

### 3.2 The engine core: a numerical radial solver

Because the What-If Lab requires arbitrary central potentials V(r), the heart of the engine is a **numerical radial Schrödinger solver** (NumPy/SciPy, hand-written), not a lookup table of hydrogen formulas. One engine powers three product pillars:

1. Real Coulomb physics (validated against the analytic solutions to tight tolerance — the analytic tier doubles as the solver's proof of correctness),
2. Screened effective potentials for v1 multi-electron atoms,
3. Counterfactual potentials (Yukawa, power-law, etc.).

Solver method choice (finite-difference eigensolver vs. Numerov shooting, grid design, convergence criteria) is an architecture-phase decision with documented error analysis.

### 3.3 Tier 2 — multi-electron atoms

- **v1:** central-field **screening/effective-charge models** (`APPROXIMATION`, honestly labeled with expected error scale). **Fully arbitrary electron configurations** — Aufbau ground state by default; user may place electrons in any orbitals (excited, hollow, non-physical), with honesty labels on non-ground configurations.
- **Later phase:** genuine self-consistent **Hartree-Fock** — hand-written radial HF (learning goal) cross-validated against **Psi4**; DFT via Psi4 if pursued. Periodic-table-wide rigor arrives with this phase.

### 3.4 Tier 3 — time-dependent phenomena (later phase)

Transitions, absorption/emission dynamics, wavepacket evolution. **Not in v1**, but state representations are designed to be time-evolvable (complex coefficients over a basis, not baked-in stationary assumptions). QuTiP is the candidate library (pure-Python, Windows-native).

Static **spectra are in v1**: line spectra computed from level differences with selection rules, overlaid against NIST reference data.

---

## 4. The What-If Lab (flagship differentiator)

All counterfactual states run the **real consequences of the altered rules** — no hand-waving — under an unmissable `COUNTERFACTUAL` banner.

- **Fundamental constants:** users vary raw constants (ℏ, e, mₑ, ε₀, c) individually; the honesty layer displays which **dimensionless combinations** (α, mass ratios) actually changed, with live consequence readouts (Bohr radius, Rydberg energy, atom size). The degeneracy — e.g., doubling e while quadrupling ε₀ changes nothing observable — is itself a teaching moment.
- **Force laws:** v1 ships a curated preset library — Yukawa/screened, power-law 1/r^p, harmonic, finite well, Coulomb-plus-core — each with parameter sliders and commentary on known physics. Free-form V(r) expression entry (safely parsed) in a later phase.
- **Quantum rule toggles** (all wanted; phased by physical prerequisite):
  - **v1:** Pauli exclusion OFF (configuration collapse — "why chemistry exists"); **classical-physics ghost** overlay (Bohr orbits, radiative collapse timescale); **spinless electrons** (degeneracy and fine-structure changes) if schedule allows, else early phase 3.
  - **HF phase:** distinguishable electrons (exchange effects only become honestly demonstrable once exchange energy exists in the model).

---

## 5. Visualization

- **3D orbitals/densities, phased:** v1 = Monte-Carlo **point-cloud** sampling of |ψ|² (honest by construction) + **isosurfaces** (textbook lobes). Next phase = volumetric ray-marched density clouds (WebGL2, tuned to integrated GPU).
- **Real ↔ complex orbital toggle** with **complex phase rendered as hue**; the chemistry-vs-physics basis choice is surfaced as a teaching moment.
- **2D companions:** radial wavefunction R(r) and radial probability P(r) = r²|R(r)|² plots; effective-potential curves (with centrifugal term); energy-level diagrams with degeneracy labels and fine-structure zoom; computed spectrum vs. NIST overlay.
- **Nucleus:** honest scale handling — true-scale mode (nucleus invisibly small) vs. visible-marker mode (`VISUAL LIBERTY`).
- **Interactivity:** radial solves run live as sliders move (target well under ~1 s on the dev machine); expensive 3D fields recompute asynchronously with progress indication.
- **Aesthetic bar: scientific-cinematic.** Dark theme, glow, careful color — but every visual element maps to a physical quantity; purely aesthetic choices are disclosed via `VISUAL LIBERTY` badges.

---

## 6. Honesty layer (core architecture, not UI garnish)

Every computed quantity and rendered element carries **provenance metadata** from the engine outward: badge tier, method used, assumptions, expected error scale, and "what would make this more accurate." The UI renders these as small badges with a click-through **inspector panel**. This is a first-class data structure threaded through the whole system.

---

## 7. Platform, stack, and hardware envelope

| Decision | Choice |
|---|---|
| OS | **Native Windows** — no WSL2/Docker anywhere in the toolchain |
| Engine | Python (conda env), **hand-written NumPy/SciPy solvers** as the core |
| Heavy QC / validation | **Psi4** (native Windows conda-forge builds) — validator in v1-era, HF/DFT engine later. *Verify current Windows build status at setup time.* |
| Dynamics (later) | QuTiP (Windows-native) |
| Not used | PySCF (no native Windows), ASE (materials-oriented; irrelevant for isolated atoms) |
| Frontend | Browser UI, WebGL2 3D (framework chosen in architecture phase; TypeScript), served by a local Python server (FastAPI-class, with async job/progress channel) |
| Delivery | Local app (clone + conda env + one launch command). Hosted **demo subset** (precomputed showcase) is a post-v1 goal |
| Hardware envelope | i7-13700H (14C/20T), 32 GB RAM, Intel Iris Xe iGPU — CPU-side numerics (no CUDA), moderate-resolution WebGL rendering |

---

## 8. Validation and testing (a headline feature)

- **Analytic-identity tests:** numerical solver vs. closed-form hydrogen energies/wavefunctions at production grid resolution, with stated tolerances; normalization, orthogonality, node counts, virial theorem, hydrogenic ⟨r⟩ expectation-value formulas.
- **Empirical tests:** computed spectra vs. **NIST Atomic Spectra Database** values (H, D, He⁺, and alkali atoms as the screening models permit), with per-tier tolerance statements.
- **Cross-validation:** multi-electron results vs. Psi4 (in the HF phase).
- **CI:** GitHub Actions (Windows runner), test-driven development throughout, CI badges in README. The validation suite is presented as a portfolio artifact in its own right.

---

## 9. Phasing and timeline

Budget: 10–20 h/week. Hard checkpoint: **demo-ready ~Oct–Nov 2026** (≈16–17 weeks).

- **Phase 0 — Foundation (~2 wks):** public GitHub repo, conda env, CI skeleton, radial-solver spike validated against analytic hydrogen.
- **Phase 1 — "Hydrogen, Honestly" (~6–8 wks):** analytic + numerical engine with provenance; exotic real systems via reduced mass; perturbative fine structure; 3D point-cloud + isosurface rendering with real/complex toggle and phase hue; radial plots; level diagram + spectrum vs. NIST; layered math UI; honesty badges + inspector.
- **Phase 2 — "What-If Lab" + v1 demo (~4–6 wks):** fundamental-constants lab with degeneracy lesson; preset potential library with live morphing; Pauli-off and classical-ghost toggles (spinless if schedule allows); screening multi-electron with arbitrary configurations; flagship guided tour. **→ v1 ships.**
- **Schedule guardrail:** the cut-list if time runs short, in order: spinless toggle → alkali screening breadth → tour length. The centerpiece (hydrogen deep-dive + live force-law morphing) is protected.
- **Phase 3 — post-deadline (ordered flexibly):** volumetric renderer; exact Dirac hydrogen; hyperfine (21 cm); Zeeman/Stark; real Hartree-Fock + distinguishable-electrons toggle + periodic-table breadth; free-form V(r) entry; time-dependent dynamics (QuTiP); lesson system; high-school/public modes; hosted demo.

---

## 10. Repository and workflow

Solo developer; **public GitHub from day one** (commit history as portfolio evidence). MIT license (default — open item). Spec and architecture docs live in `docs/`. Conventional commits; CI on every push.

---

## 11. Open items (non-blocking)

1. **Project name** (working: `atom_sim`).
2. Frontend framework specifics — architect's decision in the architecture phase.
3. Hosted-demo mechanism (static precomputed vs. Pyodide subset) — post-v1.
4. License confirmation (MIT default).
5. Psi4 Windows build verification at Phase 0 setup (fallback if broken: keep custom solvers primary, defer Psi4 cross-checks to HF phase or use WSL only for offline validation data generation — would revisit with user).
