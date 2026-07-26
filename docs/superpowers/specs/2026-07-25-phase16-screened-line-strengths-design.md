# Phase 16: Screened-atom line strengths

Status: implemented 2026-07-26. One anchor below was predicted wrong and is
corrected in place; see "What measurement changed" at the end. Closes the last
gap the intensity work
disclosed: screened atoms (He, Li, Na, ...) still return null strengths plus a
note saying the dipole integral is only implemented over closed-form hydrogenic
radial functions. This implements it over the numerical ones.

## What this adds

A dipole matrix element for **any central potential**, computed from the radial
solver's own output, plus the screened-atom wiring that turns it into Einstein A
and oscillator strengths on He/Li/Na spectra.

The solver returns `u = r R(r)` normalized to `integral u^2 dr = 1`, so the
dipole integral collapses to a plain overlap:

    R = integral R_b (r) r R_a (r) r^2 dr = integral u_a (r) u_b (r) r dr

No division by r, no reconstruction of R, no interpolation.

## The grid is the whole problem

`screened_radial` chooses `r_max` from `n`, so asking it for two different
states hands back two different grids and the overlap is meaningless. Both
states must be solved on **one common grid**: `r_max` taken from the larger `n`
of the pair, same `n_points`, then `u_a` and `u_b` are sampled at identical
radii and `numpy.trapezoid` is honest.

This is the single thing most likely to be silently wrong, so a test solves a
pair whose two states would naturally get different boxes and checks the result
against the analytic value.

## Fidelity

`APPROXIMATION`, not `NUMERICAL`. Two error sources stack and the larger wins:

- the GSZ screened potential is a model, good to a few percent on valence
  energies and worse on strengths;
- the finite-difference grid, whose error is estimated by grid-halving.

Both are stated in the provenance. Tagging this `NUMERICAL` would imply the only
error is discretization, which would be a lie about the dominant term.

## The formulas live in one place

`transitions.py` already holds `f = (2/3) dE (l_max/(2l+1)) |R|^2` and
`A = (4/3) alpha^3 dE^3 (l_max/(2l'+1)) |R|^2 / t_au`. Rather than transcribe
them into the screened path, they are extracted as
`f_from_radial_dipole` and `A_from_radial_dipole`, taking a dipole integral and
an energy from whatever source. The analytic and numerical paths then share one
copy, the same discipline that put `R_nl` in `hydrogen._radial_eval` alone.

## Validation anchors (locked in tests)

**The hydrogenic limit is the tight anchor.** Feed the numerical engine a pure
Coulomb potential `-Z/r` and it must reproduce the exact closed-form dipole
integrals from Phase 13. This is the project's established pattern of checking
numerics against analytics, and it validates the grid, the normalization
convention, the overlap integral and the error estimate at once, with no
reliance on remembered literature values.

| check | target |
|-------|--------|
| `<2p|r|1s>` via Coulomb potential | 1.290266 bohr, the exact value, to ~1e-3 |
| `<3d|r|2p>`, `<3p|r|1s>` via Coulomb | analytic values to ~1e-3 |
| A(2p->1s) built from the numerical R | 6.27e8 s^-1 |
| grid-halving error shrinks with n_points | convergence, not a fixed claim |
| common-grid pair (n=1 with n=4) | matches analytic despite differing natural boxes |

**Screened atoms are checked loosely and honestly.** GSZ is a crude model, so
the alkali resonance-line strengths are asserted only to the accuracy the model
can support, with the residual reported rather than a tight bound asserted:

- Na 3s -> 3p is the strongest line of its spectrum (corrected below);
- their `f` lands in the right band for an alkali resonance line (order unity);
- every listed screened line gets a positive, finite A;
- He and Li strengths fall monotonically along a Rydberg series.

No literature `f` value is asserted to a digit I cannot derive here.

## Surfacing

`screened_transition_lines(result, intensities=True)` fills the strengths and
drops `intensity_note` to `None`. `/api/spectrum?system=na&intensities=true`
then serves them, and the Spectrum view needs no change: it already scales by
whatever A arrives.

## What measurement changed

**The resonance line is not the strongest line outright.** The anchor above
assumed it would be. It is not, and the reason is structural rather than a bug:
`A` goes as `dE^3`, and an independent-particle orbital spectrum contains the
core transitions (`np -> 1s`, keV scale), which beat every valence line by
orders of magnitude. Measured: Li 2p->2s ranks 6th of 32 by `A`, Na 3p->3s
ranks 10th. Restricted to lines that end above the closed core, Na 3p->3s does
win, and that is what the test asserts.

**The box was four times larger than it needed to be.** `dipole_box_radius`
was `40 (n+1)^2`, which was free when the grid had a fixed point count and
expensive once `grid_points_for` held `h` fixed instead: a full screened
spectrum took 13.4 s. Measured against box size, the hydrogenic `<6p|r|5s>` and
`<4p|r|1s>` and the screened Na 3s->3p agree to six significant digits down to a
coefficient of 10, and only move at 2.5. Set to 10: same numbers, 3.1 s.

**GSZ does better on strengths than on wavelengths.** Li 2p->2s comes out at
`f = 0.725` and Na 3p->3s at `f = 0.956`, both within a few percent of the
literature, while the same lines' wavelengths are off by 2.5% and 6.3%. The
tests assert the order-unity band, not the percent agreement, because a
one-parameter screening model landing that close on `f` is partly luck and
asserting it would overstate what the model can support.

## Deferred

- The two-regime axis problem from Phase 15, unchanged.
- Fine-structure splitting of screened lines (the GSZ model has no spin-orbit
  term, so there is no j to resolve).
- Population modelling, unchanged.
