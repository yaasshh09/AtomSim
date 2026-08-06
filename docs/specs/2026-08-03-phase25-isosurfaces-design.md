# Phase 25: Isosurfaces (the textbook lobes, with a number on them)

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

- `src/atomsim/isosurface.py`, grid evaluation, level solve, marching cubes,
  mesh `Field`s with provenance. New module; nothing existing changes shape.
- `src/atomsim/server/`, an `isosurface` job in the existing thread-pool
  protocol, meta as JSON, vertices and triangles and vertex signs as raw binary
  on the array endpoints, exactly like the sampling job.
- `web/src/components/IsoSurface.tsx`, a `BufferGeometry` fed from those
  buffers, vertex-colored through the shared LUTs.
- `CloudView` gains a `surfaceMode` of `cloud` / `surface` / `both`, sharing the
  camera, the nucleus modes, the ghost overlay and the gallery strip.
- URL keys `surf` and `iso`, round-trip tested like the rest.

`both` is the frame the phase exists for: the same 90% stated twice in one
camera, once as the points the sampler drew and once as the skin the solve put
around them.

---

## What building it changed

**Tetrahedra instead of the 256-case table.** Section 4 planned marching cubes
with the classical tables transcribed by hand. Nothing in this repo can check
transcribed data, and one wrong row is a hole or a flipped facet in a rare
configuration that no orbital in the test set need ever hit. Every tetrahedron
case is derivable in four lines instead, so the table is built at import from the
statement it comes from and the test checks it back against that same statement.
The cost is about twice the triangles for a given grid, which the render budget
absorbed without noticing.

**The box could not be sized on the grid the surface is drawn on.** Section 3
said grow the box until a coarse sum of `|psi|^2 dV` says it holds 99.9% of the
electron. That sum gets worse as the box grows, because a fixed point count over
a wider box is a coarser mesh over a cusp: for a hydrogen 1s it falls from 0.997
at 8 bohr to 0.0005 at 134. The loop written against it grew the box forever and
then failed to enclose 90% of the electron inside a 134 bohr cube. The tail is
now measured where it lives, as a 1-D radial integral with the sphere average
taken exactly, and the box is sized by its inscribed sphere so the escaped mass
it reports is an upper bound rather than an estimate.

**The error bar the spec asked for is nearly blind.** Section 3 said halve the
grid and quote the movement in the enclosed fraction. It converges far too fast
to be the error bar for a picture: the coarse and fine levels agree to fifteen
digits for a hydrogen 2p at 50%, and for a 1s at 90% the fraction error comes out
exactly 0.0 while the enclosed volume moves half a percent. The fraction and the
shape are different claims, so the volume is extracted on the halved grid too and
carries its own bar. A zero beside a surface that is 0.5% off in size would have
been the quiet lie this project exists to prevent.

**"2p is two lobes" is a claim about the continuum, not about any grid we can
afford.** Section 5 listed the component count as a topology check with the
answer assumed. The lobes really are separate for every contour, but the gap
closes as the contour loosens - the surface meets the z axis at 0.60 bohr for a
30% contour and 0.11 bohr for a 90% one - and separating a 90% p orbital needs
about 0.1 bohr cells across a 30 bohr box, which is 300^3. So the count is
reported as measured, with the cell size beside it and the caveat attached
wherever an angular node could be doing the fusing.

**The request names a fraction and cannot name a level.** Neither the API nor the
client has a parameter for a contour value, and the box is not exposed either.
A level without the grid it was measured on is meaningless, and a client able to
send one would be choosing a picture rather than asking a question.

**Triangle indices needed their own decoder.** They are the one channel in this
API that is not float32, and index bytes are perfectly valid float32: reading
them through the existing decoder returns plausible garbage rather than an error,
which is the kind of failure that reaches a screenshot.

**Hartree-Fock orbitals have a surface in the engine and no way to ask for one
over HTTP.** `hf_isosurface` is built and tested; the endpoint accepts the same
system keys the plane job does, which is hydrogen-like presets and GSZ screened
atoms. Wiring the HF path through needs the request shape the `/api/jobs/hf`
endpoint uses (Z, electron count, configuration) rather than a system key, and
that is a phase of its own rather than a field bolted onto this one.
