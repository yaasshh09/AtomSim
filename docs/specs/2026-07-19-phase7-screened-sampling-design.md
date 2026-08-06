# Phase 7: Numerical Wavefunction Sampling for Screened Atoms (Design)

**Status:** approved design, pre-plan.
**Date:** 2026-07-19.
**Depends on:** Phase 6 screened atoms (`screened_atom.py`, `numerics/screening.py`),
the sampling subsystem (`sampling.py`), the plane subsystem (`plane.py`), the
angular basis (`analytic/angular.py`), and the provenance spine
(`provenance.py`).

## 1. Goal

Turn the two honest placeholders Phase 6 deliberately left behind, `CloudView`
(3-D point cloud) and `PlaneView` (2-D cross-section) refusing screened atoms
with a `422`, into real numerical renders. After this phase a screened atom
(He-Ar) is a first-class citizen in *all five* views, not just the energy side
(Levels / Spectrum / Radial).

The design rests on one structural fact: **both sampling and the plane factorize
into a radial part × an angular part, and the angular part of a central-field
atom is identical to hydrogen's.** So this phase swaps only the radial source
(analytic hydrogenic `R_nl` → numerical screened `R_nl`) and relabels the
fidelity tier. No new angular mathematics is introduced.

Scope is deliberately bounded:

- **Both views** (Cloud and Plane) ship together, they share the numerical
  radial source, so the marginal cost of the second is small and both
  placeholders are removed at once.
- **No protocol change.** The job / WebSocket / raw-binary transport and every
  downstream `meta`/binary endpoint are reused verbatim; only the two job
  *producers* change.
- **No new store state or URL schema.** `(n, l, m, system, view)` already
  round-trips; a screened system with `view=cloud`/`view=plane` simply works.

Real Hartree-Fock wavefunctions remain out of scope, the orbitals sampled here
are the same GSZ/GJG independent-particle orbitals Phase 6 already serves.

## 2. Physics and fidelity

### 2.1 What is sampled

For a screened atom of nuclear charge `Z` and electron count `N`, the
single-particle orbital `ψ_nlm = R_nl(r) · Y_lm(θ, φ)` where:

- `R_nl(r)` is the **numerical** radial function from the Phase 6 solver
  (`screened_radial`): finite-difference solution of the radial Schrödinger
  equation in the GSZ/GJG effective potential, `R = u/r`. Units `bohr^-3/2`.
- `Y_lm` is the **hydrogenic** angular factor (complex `Y_lm` or the real
  chemistry orbital `S_lm`) from `analytic/angular.py`, unchanged, because the
  potential is central and the angular equation is `Z`-independent.

The sampling target is `|ψ_nlm|²`; the plane target is signed `ψ` on the `y = 0`
plane (real there, in both bases, because the central-field orbital is real on
that plane).

### 2.2 Fidelity layering (the prime directive)

Both new renders are tagged **`APPROXIMATION`**, the weaker of the composed
tiers, because the GSZ/GJG *model* error dominates the Monte-Carlo / grid error:

- The sampling itself is an exact draw from the given distribution (would be
  `NUMERICAL` in isolation), but the distribution is a modeled orbital, so the
  honest end-to-end tier is `APPROXIMATION`.
- The hydrogen plane is `EXACT` because ψ there is analytic; the screened plane
  is **not**, same geometry, but a modeled radial function, so `APPROXIMATION`.
  A test asserts the tier to guard against a stray `EXACT` copied from the
  hydrogen path.

Provenance carried to the browser:

- **Cloud** `method`: `"factorized inverse-CDF Monte-Carlo of |psi_nlm|^2 over a
  numerical screened R_nl (<basis> basis); GSZ/GJG independent-particle
  screening; radial Schrodinger solved numerically"`.
- **Plane** `method`: `"signed psi on the y=0 plane over a numerical screened
  R_nl (<basis> basis); GSZ/GJG independent-particle screening; radial
  Schrodinger solved numerically"`.
- `error_estimate`: the grid-halving estimate `screened_radial` already produces,
  understood as a floor under the dominating model error (same layering Phase 6
  established for energies).
- `assumptions`: inherited from `screening_provenance(z, n_electrons)`.

### 2.3 Configuration independence

A single orbital's radial shape depends only on `(Z, N)` through the effective
potential, **not** on the excited electron configuration. Therefore the cloud
and plane take `(n, l, m, system)` exactly as hydrogen does today; the `config`
query parameter is irrelevant to these two views and is **not** required or
consumed by them. (Config still governs Levels / Spectrum, as in Phase 6.)

## 3. Engine architecture

Three thin reuse layers; no new angular math.

### 3.1 `sampling.py`: factor out the radial source

Today `_radial_inverse_cdf(n, l, Z, mu_ratio)` hard-codes the analytic
`radial_wavefunction`. Refactor:

- Extract a helper that builds the radial inverse-CDF from a **tabulated**
  `R_nl` given as `(r_grid, R_values)`, i.e. `P(r) = r² R²`, cumulative-trapezoid
  CDF, normalized. `sample_density` (analytic) becomes a caller that first
  evaluates `radial_wavefunction` on its grid, then delegates to this helper.
- Add `sample_screened_density(z, n_electrons, n, l, m, count, *, seed, progress,
  basis)` that sources its `(r_grid, R_values)` from `screened_radial(z,
  n_electrons, n, l, points=…)` and reuses the **untouched** angular helpers
  `_costheta_inverse_cdf(l, m)` and `_phi_inverse_cdf(m)`. Returns a
  `SampleCloud` whose `provenance.fidelity = APPROXIMATION` and whose method
  string names GSZ/GJG (see §2.2).

The `SampleCloud` container gains no required new fields; `Z` stores the nuclear
charge and a nucleus/electron-count note lives in provenance. (If the container
needs `n_electrons` for labeling, it is added as an optional field defaulting to
`None` so the hydrogen path is unaffected.)

### 3.2 `screened_atom.py`: `evaluate_screened_state`

`evaluate_screened_state(z, n_electrons, n, l, m, positions, *, basis)` returns
`ψ` at arbitrary Cartesian `positions` (shape `(N, 3)`):

- radii `r = |positions|`; interpolate `R_nl` (from `screened_radial`) onto `r`;
- multiply by the hydrogenic `Y_lm` (complex) or `S_lm` (real) angular factor
  from `angular.py` evaluated at the positions' angles.

Used by (a) the complex-basis **phase channel** of the cloud job and (b) the
plane grid. Fidelity `APPROXIMATION`; mirrors the existing `evaluate_state`
signature so the server swap is a one-line branch.

### 3.3 `plane.py`: generalize the grid

Generalize `plane_grid` to accept a **radial/state evaluator** instead of calling
`evaluate_state` directly, so a new `screened_plane_grid(z, n_electrons, n, l, m,
*, basis, half_extent, resolution)` reuses the exact grid geometry and
`default_half_extent` logic, swapping in `evaluate_screened_state`. Fidelity
`APPROXIMATION`. The extent heuristic keys off the screened `r_max` scale
(reuse `screened_atom._r_max`) rather than the hydrogenic `n²/Z`, so the orbital
fills the frame sensibly.

## 4. Server wiring (`server/app.py`)

Replace the two `422` refusals with screened branches; the transport is untouched.

- **`create_sample_job`** (currently lines 641-644): when `_is_screened(system)`,
  resolve the atom and, in the worker, call
  `sample_screened_density(element.z, element.z, n, l, m, count, …)` and
  `evaluate_screened_state(…)` for the phase channel. Same `SampleJobResult`
  shape, same binary/meta endpoints downstream.
- **`create_plane_job`** (currently lines 670-673): same pattern, calling
  `screened_plane_grid(…)`.
- `_validate_state(n, l, m)` still applies; `n_electrons = element.z` (neutral
  atom), consistent with the other screened endpoints.

Because the downstream `meta`/binary serializers already emit whatever
`Provenance` the cloud/plane carry, the `APPROXIMATION` tier and GSZ/GJG method
string reach the browser with **no schema change**.

## 5. Frontend (`web/src/`)

- **`CloudView.tsx` / `PlaneView.tsx`**, remove the screened placeholder
  early-return so the normal render path runs. The existing `Badge` reads
  provenance from the job meta and shows `APPROXIMATION` + the GSZ method
  automatically (identical to how screened Levels/Radial already render).
- **API client**, drop any client-side screened guard that mirrored the server
  refusal so cloud/plane jobs POST for screened systems.
- **No new store state, no URL schema change.** Existing `(n, l, m, system, view)`
  round-trip already covers screened cloud/plane deep links.

## 6. Validation

New physics gets real tests, not smoke tests.

1. **Radial self-consistency (KS test)**, `tests/test_sampling.py`: sample the
   r-marginal of `sample_screened_density` for a representative atom/orbital and
   KS-test it against the CDF of `P(r)=r²R²` built independently from
   `screened_radial`. Proves the sampler faithfully reproduces its numerical
   input distribution (the screened analogue of the existing hydrogen KS test,
   which compares against an analytic CDF unavailable here).
2. **Angular marginals**, extend the existing cosθ / φ KS tests to cover a
   screened case; expected distributions are the hydrogenic ones (central field).
3. **Normalization / ⟨r⟩ sanity**, sampled point count and mean radius finite
   and within a physically sane band for the chosen atom.
4. **Plane**, `tests/test_plane.py`: ψ real on `y=0`; radial node count along a
   ray equals `n−l−1`; values finite; provenance asserted `APPROXIMATION`
   (guards against a stray `EXACT`).
5. **Server**, `tests/test_server.py`: the sample and plane job endpoints now
   return `200` for a screened system; `meta` reports `APPROXIMATION`; binary
   payload shape correct. The two old "expect 422" assertions are updated.
6. **Frontend (vitest)**, a screened system in `cloud`/`plane` no longer renders
   the placeholder and mounts the canvas.

## 7. File plan

| File | Change |
|------|--------|
| `src/atomsim/sampling.py` | factor out tabulated-radial CDF; add `sample_screened_density` |
| `src/atomsim/screened_atom.py` | add `evaluate_screened_state` |
| `src/atomsim/plane.py` | generalize `plane_grid`; add `screened_plane_grid` |
| `src/atomsim/server/app.py` | swap the two `422` branches for screened producers |
| `web/src/components/CloudView.tsx` | remove screened placeholder early-return |
| `web/src/components/PlaneView.tsx` | remove screened placeholder early-return |
| `web/src/api/client.ts` | drop client-side screened cloud/plane guard |
| `tests/test_sampling.py` | screened radial KS + angular + normalization |
| `tests/test_plane.py` | screened plane real/node/provenance |
| `tests/test_screened_atom.py` | `evaluate_screened_state` unit tests |
| `tests/test_server.py` | screened sample/plane endpoints → 200 + provenance |
| `web/src/components/*.test.tsx` | screened cloud/plane render (no placeholder) |

## 8. Non-goals

- Hartree-Fock or self-consistent-field wavefunctions (still the GSZ/GJG orbital).
- Any change to the job/WS/binary transport or response schemas.
- A dedicated periodic-table `AtomView` (deferred, as in Phase 6).
- Multi-orbital / summed-density clouds, one selected `(n, l, m)` at a time,
  matching current app behavior.
