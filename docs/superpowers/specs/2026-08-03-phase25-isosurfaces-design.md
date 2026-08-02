# Phase 25 — Isosurfaces (the textbook lobes, with a number on them)

**Date:** 2026-08-03 · **Status:** in build · **Tier:** `NUMERICAL` (inheriting
the state's tier when that is weaker)

---

## 1. What this is

Requirements spec §5, the one v1 visualization item that was never built:
*"3D orbitals/densities, phased: v1 = Monte-Carlo point-cloud sampling of |ψ|²
(honest by construction) + isosurfaces (textbook lobes)."* The point cloud
shipped in Phase 1. This is the other half.

The reason it was left until last is that a lobe is the least honest picture in
chemistry. Every textbook draws one, none of them says what it is a picture of,
and the reader is left believing the orbital has a boundary. It does not. A
surface drawn through |ψ|² is a choice of contour, and the only thing that makes
it a statement about an atom rather than about a rendering setting is the
fraction of the electron it contains.

So the control is that fraction. You ask for 90% and the engine solves for the
contour that holds 90%, then says so on the picture, including the part textbooks
omit: the electron is outside this surface one time in ten.

## 2. The level is solved, not chosen

Given a target fraction `f`, the level `c` is the solution of

    integral over {rho >= c} of rho dV = f

On the grid this is exact and cheap. Sort the cell densities descending,
cumulative-sum `rho * dV`, find where the sum crosses `f`, and interpolate
between the bracketing densities. One sort of N³ values, no root-finding loop and
no tolerance to tune.

The raw level is reported next to the fraction. Hiding it would be its own kind
of dishonesty: someone comparing this against a published isosurface needs the
number that publication used.

**Why not a raw-threshold slider.** It is one line of code less and it makes the
number on screen meaningless. The enclosed fraction has to be computed either way
to say anything true about the picture, so putting the arbitrary quantity in
control and the physical one in a caption gets the cost without the benefit.

## 3. Grid, box, and the mass that escapes

The density is evaluated on a cubic `N x N x N` grid, N = 96 by default, 64 and
128 selectable. The half-width `L` is sized from the state's radial extent so
that the box holds essentially all of the probability.

`L` is a guess until it is measured, so it is measured: the grid sum of `rho dV`
is compared against 1 and the shortfall is reported as escaped probability. This
matters more than it looks. If 2% of the electron is outside the box, then the
"90%" contour computed inside the box is really the 90% of 98%, and the surface
is wrong in a way that no amount of grid refinement fixes. The disclosure names
the escaped fraction and the solve refuses targets that the box cannot support.

Diffuse states are where this bites: a hydrogen 4f needs a far bigger box than a
1s, and a screened valence orbital bigger still.

## 4. Marching cubes, hand-written

The standard algorithm: for each cell, classify its eight corners against the
level into one of 256 cases, look up which edges are cut, place a vertex on each
cut edge by linear interpolation of the density, and emit the triangles for that
case. The edge and triangle tables are the classical ones and are data, not
physics.

It is written here rather than pulled in because scikit-image is not in the
environment and the project's rule is hand-written numerics in the core. It is
vectorized over all cells at once with NumPy, which is what keeps a 96³ grid
inside the interaction budget.

Vertices are welded on the shared edge index so neighbouring triangles reference
one vertex, which is what lets the client compute smooth normals and lets the
component count below mean something.

**Sign, not just shape.** Each vertex is tagged with `sign(psi)` in the real
basis or `arg(psi)` in the complex one. The surface itself is drawn through
`|psi|²`, which is sign-blind, so the tag is what produces the two-colored lobes
of a p orbital. That is the textbook picture, and here the colors come from the
same LUTs the cloud and the plane use rather than from a palette chosen for the
occasion.

## 5. Four checks, because a wrong surface still looks like an orbital

This is the failure mode that matters. A bug in the tables, the interpolation, or
the level solve produces a smooth closed shape that reads as an orbital to any
eye, so the tests have to be things a plausible-looking wrong answer would fail.

1. **Closed form, hydrogen 1s.** The level set is a sphere, and

       P(r < a) = 1 - e^(-2a) (1 + 2a + 2a²)

   so the 90% contour is at `a = 2.6612` bohr. The mesh's vertex radii and its
   enclosed volume are checked against that sphere. Nothing in this repo chose
   that number.

2. **Mesh geometry against the grid count.** The enclosed volume computed from
   the triangles by the divergence theorem (`sum of (v0 · n) A / 3`) against the
   summed volume of cells above the level. These are two different objects: the
   second is a staircase of voxels, the first is the interpolated surface. Their
   gap is the mesh discretization error, and it gets reported rather than assumed
   away.

3. **Against the Phase 1 sampler.** Draw Monte-Carlo points from |ψ|² with the
   existing KS-validated sampler and count how many satisfy `rho >= c`. That
   fraction must match the target within binomial error. This is the check that
   ties the surface to machinery already validated by a different method, and it
   is a point-in-set test rather than a ray cast, so it tests the level rather
   than the triangles.

4. **Grid halving.** Recompute at N/2. The change in achieved fraction is the
   error estimate the `Quantity` carries, per the project's convention for
   everything `NUMERICAL`.

Plus topology as a separate assertion: at 90%, 2p_z is two disconnected
components and 1s is one. Counted by union-find over the welded mesh. "The
textbook lobes are two lobes" is a claim, so it is tested.

## 6. Provenance

`NUMERICAL` when ψ is analytic, because the grid and the linear interpolation are
the approximation, and the tier is never `EXACT` no matter how exact ψ is. It
inherits `APPROXIMATION` from a screened or Hartree-Fock state, taking the weaker
of the two tiers the way the rest of the engine does.

Disclosed on the surface:

- the target fraction, the fraction actually achieved on this grid, and the
  grid-halving error on it;
- the probability that escaped the box;
- the raw level in atomic units;
- that the surface is a contour and not a boundary, stated as the complementary
  percentage rather than as a caveat.

Smooth-shaded normals are a `VISUAL_LIBERTY` through `lib/liberties.ts`: they
imply a surface finer than the triangles that carry it. Flat shading would be the
literal mesh, and is available.

## 7. Surfaces

- `src/atomsim/isosurface.py` — grid evaluation, level solve, marching cubes,
  mesh `Field`s with provenance. New module; nothing existing changes shape.
- `src/atomsim/server/` — an `isosurface` job in the existing thread-pool
  protocol, meta as JSON, vertices and triangles and vertex signs as raw binary
  on the array endpoints, exactly like the sampling job.
- `web/src/components/IsoSurface.tsx` — a `BufferGeometry` fed from those
  buffers, vertex-colored through the shared LUTs.
- `CloudView` gains a `surfaceMode` of `cloud` / `surface` / `both`, sharing the
  camera, the nucleus modes, the ghost overlay and the gallery strip.
- URL keys `surf` and `iso`, round-trip tested like the rest.

`both` is the frame the phase exists for: the same 90% stated twice in one
camera, once as the points the sampler drew and once as the skin the solve put
around them.
