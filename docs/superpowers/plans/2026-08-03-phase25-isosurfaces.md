# Phase 25: Isosurfaces Implementation Plan

**Goal:** Ship the enclosed-probability isosurface end to end, from a new
geometry kernel through the job protocol to a surface mode inside the Cloud view.

**Spec:** `docs/superpowers/specs/2026-08-03-phase25-isosurfaces-design.md`

## Deviation from the spec's file plan

The spec put everything in `src/atomsim/isosurface.py`. This plan splits it in
two, for the same reason Phase 21 split `hf_terms` from `hartree_fock`: the
geometry is pure and checkable against closed forms with no physics in sight, and
burying it under a density evaluator is exactly what lets a broken triangle
survive behind a plausible-looking orbital.

- `src/atomsim/numerics/marching_tets.py` — scalar field in, welded oriented
  mesh out. Knows nothing about atoms.
- `src/atomsim/isosurface.py` — density grid, box sizing, level solve,
  provenance, and the `Field`s that cross the boundary.

## Deviation from the spec's algorithm: tetrahedra, not cubes

The spec said marching cubes with the classical 256-case tables. This plan uses
marching tetrahedra instead, decomposing each cell into six tetrahedra sharing
the main diagonal.

The reason is that the 256-row triangulation table is transcribed data. Nothing
inside this repo can check it, and a single wrong row produces a hole or a
flipped facet in one rare configuration that no orbital in the test set may
happen to hit. Every case of a tetrahedron, by contrast, is derivable in four
lines: zero, one, or two corners above the level give nothing, one triangle, or
one quad, and the edges they sit on follow from which corners those are. It is
also free of the ambiguous-face problem that marching cubes has to resolve by
convention.

The cost is roughly twice the triangles for the same grid and slightly worse
triangle aspect ratios. Both are acceptable for a 96³ render budget, and neither
touches the numbers the phase is about.

The Kuhn decomposition is used, identically in every cell, so neighbouring cells
split their shared face along the same diagonal and the mesh stays watertight.

## Global constraints

- Hartree atomic units in the engine; bohr for lengths crossing to the client.
- Everything crossing a module boundary is a `Quantity` or a `Field` with
  `Provenance`. The mesh arrays are `Field`s.
- ruff line-length 100; run from repo root in conda env `atomsim`.
- Validation tests, not smoke tests. Commit per task, no AI attribution.

---

### Task 1: The geometry kernel

**Files:** `src/atomsim/numerics/marching_tets.py`, `tests/test_marching_tets.py`

- [ ] Kuhn decomposition of the cell grid into six tetrahedra, vectorized over
      all cells, chunked over z so a 128³ grid never materializes more than a
      slab of tetrahedra at once.
- [ ] Per-tetrahedron classification, linear interpolation of the crossing point
      on each cut edge, triangle emission with outward orientation (normal
      pointing down the field gradient, out of the enclosed region).
- [ ] Vertex welding by the sorted pair of grid-corner indices, so the mesh is
      indexed and shared vertices are one vertex.
- [ ] `enclosed_volume(vertices, triangles)` by the divergence theorem.

**Verification:** a sphere field `r² - a²` at N = 32, 64, 128: area against
`4 pi a²` and volume against `(4/3) pi a³`, both converging as the grid refines;
watertightness (every edge in exactly two triangles); orientation (enclosed
volume positive); a plane field for exactness on a case with no curvature error.

### Task 2: The density grid and the level solve

**Files:** `src/atomsim/isosurface.py`, `tests/test_isosurface.py`

- [ ] `density_grid(state, half_width, n)` evaluating |ψ|² on the cubic grid for
      analytic states, reusing `analytic/wavefunction.py`.
- [ ] Automatic half-width from the state's ⟨r⟩ and the tail, then the measured
      captured mass and the escaped remainder.
- [ ] `solve_level(grid, target_fraction)` by descending sort and cumulative sum,
      interpolating between bracketing densities; raises when the box cannot
      support the target.
- [ ] `isosurface(...)` assembling grid, level, mesh, and provenance.

**Verification:** hydrogen 1s at f = 0.9 gives the closed-form sphere radius
2.6612 bohr; achieved fraction matches the target; escaped mass shrinks as the
box grows; grid halving gives the error estimate.

### Task 3: The independent checks

**Files:** `tests/test_isosurface.py`

- [ ] Cross-check against the Phase 1 sampler: the fraction of sampled points
      with `rho >= level` matches the target within binomial error.
- [ ] Topology: 1s at 90% is one component, 2p_z is two, counted by union-find
      over the welded mesh.
- [ ] Provenance: `NUMERICAL` for analytic states, inherits `APPROXIMATION` from
      screened and HF ones, and the disclosures the spec names.

### Task 4: Screened and Hartree-Fock states

**Files:** `src/atomsim/isosurface.py`, `tests/test_isosurface.py`

- [ ] Density from `screened_atom.evaluate_state` / `hf_atom.evaluate_hf_state`
      on the same grid path, with the tier taken as the weaker of the two.

### Task 5: Server

**Files:** `src/atomsim/server/app.py`, `schemas.py`, `tests/test_server_iso.py`

- [ ] `isosurface` job, meta JSON, binary vertices / triangles / vertex signs.
- [ ] Validation: fraction in (0, 1), n in the allowed grid sizes, 422 on a
      target the box cannot support.

### Task 6: Frontend

**Files:** `web/src/api/types.ts`, `client.ts`, `state/store.ts`,
`components/IsoSurface.tsx`, `components/CloudView.tsx`, `lib/urlState.ts`,
`lib/liberties.ts`, plus tests

- [ ] `surfaceMode` (`cloud` | `surface` | `both`) and `isoFraction` in the
      store, cleared by the same invalidation that clears the cloud.
- [ ] `IsoSurface` mesh with vertex colors from the shared LUTs, smooth shading
      disclosed as a liberty.
- [ ] URL keys `surf` and `iso`, round-trip tested.

### Task 7: Close out

- [ ] `pytest`, `ruff check .`, `npm test`, `npm run build` all green.
- [ ] "What building it changed" section appended to the spec.
