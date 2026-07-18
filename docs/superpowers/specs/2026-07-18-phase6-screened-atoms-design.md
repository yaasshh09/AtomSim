# Phase 6 — Screened Multi-Electron Atoms (Design)

**Status:** approved design, pre-plan.
**Date:** 2026-07-18.
**Depends on:** the numerical radial solver (`numerics/radial_solver.py`), the
provenance spine (`provenance.py`), and the existing system/level/spectrum
machinery.

## 1. Goal

Add the second product pillar the numerical solver was always meant to power:
**real multi-electron atoms** via a central-field screening model. Each atom is
selectable like any other system (H, He⁺, muonic H); its orbital energies, term
diagram, spectrum, and radial functions render in the existing views. Every
number it produces is honestly labeled `APPROXIMATION` with a stated error
scale. Real Hartree-Fock remains a later phase — this phase is the honest,
independent-particle stepping stone.

Scope is deliberately bounded (chosen during brainstorming):

- **Model:** Green-Sellin-Zachor (GSZ) analytic screened potential, with
  Garvey-Jackman-Green (1975) closed-form parameters — no self-consistency.
- **UI:** atom-as-a-selectable-system plus a small configuration panel; the
  existing views are reused. A dedicated periodic-table `AtomView` is deferred.
- **Views:** the energy side first — `LevelsView`, `SpectrumView`, `RadialView`.
  `CloudView`/`PlaneView` show an honest labeled placeholder for screened atoms
  (numerical wavefunction sampling is deferred to a later phase).
- **Coverage:** a generic engine for any (Z, N) in the GJG range; named presets
  He–Ar (Z = 2–18); vendored NIST reference expanded to add He, Li, Na.

## 2. Physics model

### 2.1 The screened potential

An electron in a neutral atom or ion (nuclear charge `Z`, total electron count
`N`) moves, in the central-field approximation, in an effective potential
(Hartree atomic units):

    V_eff(r) = -(1/r) * [ (Z - N + 1) + (N - 1) * Ω(r; d, H) ]

where `Ω` is the GSZ screening function with the limits `Ω(0) = 1`,
`Ω(∞) = 0`:

    Ω(r) = 1 / [ H * (exp(r/d) - 1) + 1 ]

The two limits pin the physics and are the correctness anchors:

- `r → 0`:  `Z_eff → Z` — the electron sees the bare nucleus.
- `r → ∞`:  `Z_eff → Z - N + 1` — it sees the net core charge (for a neutral
  atom, `N = Z`, so `Z_eff → 1`, the correct asymptotic pull on an outer
  electron).

The screening electron count is `N - 1` (an electron does not screen itself).

### 2.2 Parameters

The two GSZ parameters `(d, H)` are **not** hand-tabulated per element. They come
from the **Garvey-Jackman-Green (1975)** analytic independent-particle-model fit,
which gives `d` and `H` as closed-form functions of `(Z, N)`. This covers the
periodic table (well beyond periods 1–3) with a small, auditable set of universal
fit coefficients transcribed from the paper — not a large vendored data blob.

> Reference: A. E. S. Garvey, C. H. Jackman, A. E. S. Green, *Phys. Rev. A* **12**,
> 1144 (1975), "Independent-particle-model potentials for atoms and ions." The
> GSZ form is Green, Sellin, Zachor, *Phys. Rev.* **184**, 1 (1969). Exact
> coefficients are transcribed at implementation time and pinned by the tests in
> §6.

### 2.3 Provenance layering (the prime directive)

Two distinct fidelities compose here and must not be conflated:

- The **potential** V_eff is an `APPROXIMATION` — a model choice with a real error
  scale. Its provenance names the model (GSZ/GJG) and its regime of validity.
- The **radial solve** is `NUMERICAL` — finite differences with a grid-halving
  error estimate (unchanged from the existing solver).
- The **resulting orbital energy** ε_nl is reported as `APPROXIMATION`: the model
  error dominates the numerical error, and the tier is the weaker of the two. Its
  `method` states both ("GSZ/GJG independent-particle screening; radial
  Schrödinger solved numerically; error scale ~X% vs NIST valence energies").

No value silently upgrades its tier. The numerical error estimate still travels
alongside as a quantified sub-scale.

### 2.4 Reduced mass

Multi-electron nuclei are heavy; `mu_ratio = 1` (infinite-nuclear-mass
approximation) is used, and this is disclosed as a negligible sub-approximation
relative to the screening error. No exotic reduced-mass systems in this phase.

## 3. Configuration model

An atom is described by `(Z, N, occupations)`, where `occupations` maps each
subshell `nl` to an electron count.

- **Aufbau/Madelung ground configuration by default** — filled in `(n + l, n)`
  order, capped at `2(2l+1)` per subshell.
- The user may **promote/demote** electrons to reach **excited or hollow**
  configurations. Any non-ground configuration carries a "non-ground
  configuration" honesty note. Non-physical over-filling is prevented by the
  **Pauli cap** (this phase enforces exclusion; the Pauli-*off* configuration
  collapse — "why chemistry exists" — is a separate future phase).
- **Total energy = Σ (occupancy · ε_nl)** over occupied subshells, labeled
  `APPROXIMATION` with the caveat "sum of independent-particle orbital energies;
  not a variational total energy." Because V_eff depends only on `(Z, N)` and not
  on which orbitals are filled, changing the configuration changes the energy
  *sum*, not the field — this is the honest independent-particle picture and is
  stated as such.

Orbital-to-quantum-number mapping: for angular momentum `l`, the solver's radial
states are indexed `k = 0, 1, 2, …`; the principal quantum number is
`n = k + l + 1`. So a subshell `nl` is the `(k = n - l - 1)`-th solution at that
`l`.

## 4. Engine modules (new)

- **`src/atomsim/numerics/screening.py`** — the GSZ V_eff builder and the GJG
  `(d, H)` parameter functions. Produces a `potential(r)` callable plus an
  `APPROXIMATION` `Provenance`. Pure, dependency-light, independently testable.
- **`src/atomsim/atoms.py`** — the element table (Z, symbol, name), the Aufbau
  ground-configuration generator, subshell/occupancy representation, and Pauli
  rules. Defines the `Atom`/`Configuration` types. No solver dependency.
- **`src/atomsim/screened_atom.py`** — the orchestrator. Given `(element or Z, N,
  configuration)`: build V_eff, solve each `l ∈ {0,1,2}` with
  `solve_radial_with_error`, assemble orbital energies ε_nl (mapped by
  `n = k+l+1`), apply the configuration to get occupied levels and the total
  energy, and expose the numerical radial functions `R_nl = u/r`. Returns
  provenance-carrying results (`Quantity`/`Field`), mirroring the shape the
  server already consumes for levels/radial/spectrum.

Each module answers cleanly: what it does, how you call it, what it depends on.

## 5. Server integration

A screened atom is a **selectable system key** (`he`, `li`, `be`, …, `ar`),
resolved alongside the hydrogenic presets. Endpoints branch on the system
**kind**:

- **`/api/levels`, `/api/radial`, `/api/spectrum`** — hydrogenic key → existing
  analytic path; screened key → the screening engine (§4). The configuration is
  passed as a query parameter (compact occupation string, e.g. `1s2 2s2 2p1`),
  defaulting to Aufbau when omitted.
- **Cloud/plane endpoints** — for a screened key, return an honest, labeled
  placeholder payload (a `VISUAL_LIBERTY`/disclosed-absence marker) rather than a
  silently-hydrogenic cloud. The frontend renders the "coming in a later phase"
  message from this signal.
- **Schemas** map the screened `Quantity`/`Field`/`Provenance` to the existing
  response models; the `APPROXIMATION` tier and error scale survive to the
  browser exactly as for every other value.

Back-compat: hydrogenic system keys and all existing endpoints behave exactly as
before; the branch is additive.

## 6. Frontend integration

- **System picker** gains an "Atoms (screened)" group (He–Ar) beside the existing
  hydrogen-like systems.
- **Configuration panel** (shown for screened systems): the current occupation
  string, a "reset to Aufbau" action, and promote/demote controls; a
  non-ground-configuration note appears when the config is not the Madelung
  ground state. The config is part of URL state (deep-linkable, round-trip
  tested), independent of the (n, l, m) orbital selection.
- **Views:** `LevelsView` (term ladder — the accidental Coulomb l-degeneracy is
  visibly gone), `SpectrumView` (transition series vs NIST), and `RadialView`
  (numerical R_nl) render screened output. `CloudView`/`PlaneView` show a
  badge-labeled placeholder for screened atoms. Every screened number carries its
  `APPROXIMATION` badge through `Badge`/`liberties.ts`.
- The store treats atom selection and configuration as physics inputs: they are
  in the `INVALIDATED` set (changing them clears derived physics), consistent with
  the store's stale-physics invariant.

## 7. Coverage & NIST reference data

- **Engine:** generic for any `(Z, N)` within the GJG fit's validity.
- **Named presets:** He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar
  (Z = 2–18) so shell structure and Aufbau filling are visible across three
  periods.
- **Vendored NIST reference:** expand the existing `data/nist_h_i.json`,
  `nist_d_i.json` with **He, Li, Na** (the valence sweet spots where a frozen
  central field is most defensible), each with citation and retrieval date. Other
  named atoms are model-only and honesty-labeled. **No live NIST queries, ever.**

## 8. Validation & testing

Per the project rule that new physics gets a real validation test, not a smoke
test:

1. **Hydrogenic limit (exact):** with `N = 1` the `(N-1)` screening term
   vanishes, so `V_eff = -Z/r` and the solver must reproduce `-Z²/(2n²)` to the
   solver's numerical tolerance. This is the calibration anchor.
2. **Grid convergence:** orbital energies carry a grid-halving error estimate via
   `solve_radial_with_error`; the estimate must shrink under refinement.
3. **NIST valence tolerance:** the Li, Na, and He valence ionization energies must
   land within a **stated tolerance** of the vendored NIST values. This tolerance
   *is* the published error scale that defines the `APPROXIMATION` tier; the test
   asserts it and the provenance reports it. Tolerances are set from the model's
   known accuracy, not loosened to pass.
4. **Aufbau ordering:** the generated ground configuration matches the Madelung
   order for Z = 2–18 (1s, 2s, 2p, 3s, 3p filling — the 4s/3d crossover is beyond
   this phase's preset range but the generator uses the general `(n+l, n)` rule so
   it extends correctly).
5. **Configuration accounting:** total energy equals the occupancy-weighted sum of
   orbital energies; Pauli cap rejects over-filling; excited configs are flagged
   non-ground.

## 9. Rejected alternatives

- **Slater's-rules hydrogenic Z_eff** — too crude; no solver-driven l-splitting;
  weaker NIST agreement. (Kept conceptually as a possible future "transparent
  lesson" toggle, not the engine.)
- **Self-consistent Hartree screening** — heavier, iterative, and overlaps the
  Phase-3 real-Hartree-Fock work; out of scope for the honest independent-particle
  tier.
- **Dedicated periodic-table `AtomView`** — deferred to a later phase; atom-as-
  system keeps this phase focused.
- **Full numerical cloud/plane sampling** — the inverse-CDF-on-numerical-density
  subsystem is deferred; the energy-side views ship first.

## 10. File-structure summary

**Backend (new):** `numerics/screening.py`, `atoms.py`, `screened_atom.py`,
`data/nist_he_i.json`, `data/nist_li_i.json`, `data/nist_na_i.json`.
**Backend (modified):** `systems.py` (screened-atom registry / kind resolution),
`server/app.py` + `server/schemas.py` (kind-routed endpoints, config param,
screened placeholders), `spectra.py` (screened transition source + new
references).
**Frontend (modified):** `api/types.ts`, `api/client.ts` (config param, screened
shapes), `state/store.ts` (atom + configuration in `INVALIDATED`),
`lib/urlState.ts` (config deep link), `components/` (system picker group, config
panel, screened placeholders in Cloud/Plane, Levels/Spectrum/Radial wiring).
**Tests (new/extended):** `tests/test_screening.py`, `tests/test_atoms.py`,
`tests/test_screened_atom.py`, extend `tests/test_server.py`,
`tests/test_spectra.py`; frontend `store.test.ts`, `urlState.test.ts`.
