# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Prime directive

**The model never quietly lies about physics.** This is not a slogan — it is
mechanically enforced. Every physical value that crosses a module boundary is a
`Quantity` (scalar) or `Field` (array) carrying a `Provenance` that states its
`Fidelity` tier: `EXACT`, `NUMERICAL`, `APPROXIMATION`, `COUNTERFACTUAL`, or
`VISUAL_LIBERTY`. When you add or change anything that produces a number or a
picture, ask which tier it is and make the code say so — a plain `float`
crossing a boundary, a silent zero, or an undisclosed presentational choice is a
bug, not a shortcut. See `src/atomsim/provenance.py`.

## Commands

Python (run from repo root; conda env `atomsim`, Python 3.12):

    pytest                              # full physics + server suite
    pytest tests/test_hydrogen_analytic.py       # one file
    pytest tests/test_sampling.py -k ks_test      # one test by name
    ruff check .                        # lint (line-length 100; E741 ignored — `l` is a QM number)
    atomsim serve                       # build web/dist first; launches server + opens browser
    atomsim serve --port 8001 --no-browser

Frontend (run from `web/`; Node 22):

    npm test                            # vitest run
    npx vitest run src/lib/quantum.test.ts   # one file
    npm run build                       # tsc --noEmit + vite build -> web/dist (serve reads this)
    npm run dev                         # vite dev server on :5173 (CORS-allowlisted in the API)

`atomsim serve` only mounts the app if `web/dist` exists — **rebuild the
frontend after changing anything under `web/src`**. First-time setup is in the
README quickstart and `docs/SETUP.md` (Windows-native, no WSL).

Regenerate color LUTs after a matplotlib upgrade:

    python scripts/gen_luts.py          # rewrites web/src/lib/luts.ts

CI (`.github/workflows/ci.yml`, Windows) runs `ruff check`, `pytest --cov`,
`npm test`, and `npm run build`. The NIST spectrum comparison runs inside
pytest, so a physics regression fails CI.

## Architecture

Two halves talk over an HTTP/WebSocket + raw-binary boundary: a Python physics
engine (`src/atomsim/`) and a React/Three.js app (`web/src/`).

### Python engine (`src/atomsim/`)

- **`provenance.py`** — `Fidelity`, `Provenance`, `Quantity`, `Field`. The spine
  everything else threads through.
- **`analytic/`** — closed-form hydrogen-like physics (`EXACT`). `hydrogen.py`
  (energies, radial wavefunctions, ⟨r⟩), `angular.py` (complex Y_lm **and** real
  chemistry orbitals — both first-class, basis is provenance-visible),
  `wavefunction.py` (ψ_nlm = R·Y), `fine_structure.py` (α² Pauli shifts,
  `APPROXIMATION` with the three neglected scales quantified). This is also the
  ground truth that validates the numerical solver.
- **`numerics/`** — `radial_solver.py` solves the radial Schrödinger equation
  for **arbitrary central potentials** via finite differences + tridiagonal
  eigensolve (`NUMERICAL`, grid-halving error estimates). One engine intended to
  power real atoms, screened models, and counterfactual force laws.
- **`sampling.py`** — Monte-Carlo |ψ|² point clouds via factorized inverse-CDF;
  validated by KS tests against analytic CDFs. Sampling is physics, so clouds
  carry provenance.
- **`plane.py`** — 2-D y=0 cross-sections (the "hydrogen poster" plane; ψ is real
  there, so a signed-ψ plot is honest — and labeled exactly).
- **`systems.py`** — hydrogen-like presets (H, D, T, muonic H, positronium, He⁺,
  generic Z) built from CODATA mass ratios; reduced mass makes them all exact in
  the same formulas. `constants.py` holds SI anchors and the counterfactual hook.
- **`spectra.py`** — emission lines from level differences + selection rules,
  compared against **vendored** NIST ASD data in `data/*.json` (never live
  queries; citation + retrieval date in-repo).
- **`server/`** — FastAPI app (`app.py`). Heavy work (sampling, plane grids) runs
  as background **jobs** (`jobs.py`, thread-pool executor); the client POSTs a
  job, watches `/ws/jobs/{id}` for progress, then fetches JSON `meta` and raw
  `float32` array data on separate endpoints. `schemas.py` maps engine
  `Quantity`/`Field`/`Provenance` to Pydantic response models —
  **provenance survives to the browser**. `thumbnails.py` renders matplotlib PNGs.

### Frontend (`web/src/`)

- **`state/store.ts`** — single Zustand store. Key invariant: the `INVALIDATED`
  block lists everything derived from (n, l, m, system, basis) and is spread on
  every change to those, so stale physics can never render. Purely presentational
  toggles (view, color mode, nucleus mode) deliberately invalidate nothing.
- **`api/`** — typed client + binary decoders matching the server's job protocol.
- **`components/`** — five views over any state (`CloudView` 3D point cloud,
  `PlaneView` 2-D cross-section, `RadialView`, `LevelsView`, `SpectrumView`),
  each labeled with exactly what it plots. `Badge` renders provenance; every
  disclosed liberty flows through `lib/liberties.ts`.
- **`lib/luts.ts` is generated** by `scripts/gen_luts.py` — do not hand-edit.
  This is the single color authority: server thumbnails (matplotlib) and every
  client canvas read the same 256-entry tables, so a density looks identical in a
  thumbnail, the 2-D canvas, and the 3-D cloud.
- **`lib/urlState.ts`** — every app state is addressable by URL (deep links, e.g.
  `?n=3&l=1&m=-1&system=mu-h&view=plane&plane=psi`). Round-trip tested; treat the
  query schema as a stable contract.

### Conventions

- Engine-internal math is in **Hartree atomic units**; SI/display conversions
  (eV, pm) happen at the server boundary and append to the provenance `method`.
- `l` is the orbital angular-momentum quantum number, not a length — ruff's E741
  is ignored project-wide for this reason. Keep the physics naming.
- New physics gets a validation test (analytic ground truth, KS test, or
  grid-convergence), not just a smoke test — that is how honesty is checked.

## Project docs

Specs and phase plans live under `docs/superpowers/`. The engine is Phase 1
("Hydrogen, Honestly") complete; the numerical solver and counterfactual hooks
are the seams for later phases (real atoms, screened models, What-If Lab).

## .gstack

to use the /browse skill from gstack for all web browsing, never use mcp__claude-in-chrome__* tools, and lists the available skills: /office-hours, /plan-ceo-review, /plan-eng-review, /plan-design-review, /design-consultation, /design-shotgun, /design-html, /review, /ship, /land-and-deploy, /canary, /benchmark, /browse, /connect-chrome, /qa, /qa-only, /design-review, /setup-browser-cookies, /setup-deploy, /setup-gbrain, /retro, /investigate, /document-release, /document-generate, /codex, /cso, /autoplan, /plan-devex-review, /devex-review, /careful, /freeze, /guard, /unfreeze, /gstack-upgrade, /learn. Then ask the user if they also want to add gstack to the current project so teammates get it.