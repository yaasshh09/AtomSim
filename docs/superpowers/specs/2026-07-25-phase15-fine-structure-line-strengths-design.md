# Phase 15: Fine-structure line strengths (j-resolved rates)

Status: design approved 2026-07-25. Closes the gap Phase 14 disclosed: with fine
structure on, the Spectrum view withholds intensities and prints a note saying
the 6j branching factor is not implemented. This implements it.

## What is missing today

A gross-structure rate `A(n'l' -> nl)` has to be split across the fine-structure
components `j' = l' +/- 1/2` and `j = l +/- 1/2`. The split is not uniform and
not proportional to degeneracy; it is set by a Wigner 6j symbol. Phase 14
refused to guess, so fine-structure spectra currently show uniform bars plus an
apology.

## Physics

With spin as a spectator, the E1 rate for a fine-structure component is

    A(n'l'j' -> nlj) = (4/3) alpha^3 dE^3 (2j+1) {j 1 j'; l' 1/2 l}^2 l_max |R|^2

where `l_max = max(l, l')`, `R` is the same radial dipole integral Phase 13
computes, and the braces are a Wigner 6j symbol. The `(2j'+1)` from the line
strength cancels the `1/(2j'+1)` from averaging over the upper sublevels, which
is why only the lower `(2j+1)` survives.

**The check that makes this self-validating.** Summing over the lower `j` must
return the gross rate, because the total decay rate of a level cannot depend on
how finely the lower level is resolved:

    sum_j (2j+1) {j 1 j'; l' 1/2 l}^2 == 1 / (2l' + 1)

Verified by hand for 2p -> 1s: the 6j with a zero argument collapses to
`(-1)^(a+b+c) / sqrt((2a+1)(2b+1))`, giving `1/6` for both `j'`, so the sum is
`2 * (1/6) = 1/3 = 1/(2*1+1)`. This identity is asserted in tests across many
`(l, l', j')`, and it is what catches a wrong 6j.

## The 6j implementation

New module `src/atomsim/analytic/wigner.py`, evaluating the Racah formula

    {j1 j2 j3; j4 j5 j6} = D(j1 j2 j3) D(j1 j5 j6) D(j4 j2 j6) D(j4 j5 j3)
                           * sum_t (-1)^t (t+1)! / [ product of six factorials ]

with `D(abc) = sqrt( (a+b-c)! (a-b+c)! (-a+b+c)! / (a+b+c+1)! )`. Half-integer
arguments are carried as exact halves (doubled integers internally) so no
floating-point comparison decides a triangle condition.

**A note on the provenance rule.** A 6j symbol is a dimensionless algebraic
constant, like a Clebsch-Gordan coefficient or pi, not a measured physical
value. It is returned as a plain `float`, and the `Quantity` built from it
carries the provenance. This is the one deliberate exception in the module, and
it is stated in the module docstring so it reads as a decision, not an
oversight. Everything crossing into `transitions.py` as physics is still a
`Quantity`.

Selection rules fall out of the triangle conditions: a forbidden combination
returns exactly 0, so `Delta j = 0, +/-1` (excluding `j = j' = 0`) needs no
separate special-casing.

## Validation anchors (locked in tests)

The 6j engine, independent of any physics:

| check | why it bites |
|-------|--------------|
| `{1/2 1 3/2; 1 1/2 0}^2 = 1/6` | the zero-argument closed form, computed by hand |
| symmetry under column permutation | any index slip breaks it |
| symmetry under swapping upper/lower in two columns | as above |
| triangle violations return exactly 0 | selection rules must be structural |
| the sum rule above, over many `(l, l', j')` | the identity the whole phase rests on |

The physics:

- `sum_j A(n'l'j' -> nlj) == A(n'l' -> nl)` from Phase 13, for many levels.
- The D-line ratio: `A(3p_{3/2} -> 3s) / A(3p_{1/2} -> 3s) = 2`, the classic
  doublet intensity ratio (2:1), which is a real spectroscopic fact rather than
  a self-consistency check.
- `tau(2p_{1/2}) == tau(2p_{3/2}) == tau(2p)` to the gross value: for hydrogen
  the only decay is to 1s, so resolving j must not change the lifetime.
- Every fine-structure component rate is positive and finite.

## Surfacing

`transition_lines(..., fine_structure=True, intensities=True)` stops withholding:
each `(n l j -> n' l' j')` line gets its own A and f. `intensity_note` becomes
`None` for that case. The Spectrum view needs no change beyond that, since it
already scales by whatever A arrives, and the screened-atom note stays.

## Found while building: the fine-structure spectrum spans two regimes

Not fixed here, and not caused here, but it should not stay invisible.

A fine-structure line list contains within-n components (2p_3/2 -> 2s_1/2 and
friends) alongside the ordinary n -> n' lines. They are real E1 transitions, and
the two groups do not overlap at all. At n_max = 6:

| group | lines | Einstein A | wavelength |
|-------|-------|-----------|------------|
| across-n (optical) | 140 | 10^2.5 .. 10^8.8 s^-1 | 94 nm .. 7.5 um |
| within-n (microwave) | 25 | 10^-12.2 .. 10^-6.1 s^-1 | 27 mm .. 11 m |

Consequences for the Spectrum view, both of which predate this phase since those
lines were always in the list:

- the log-lambda axis runs to 11 m, so every optical line is squeezed into a
  sliver at the left;
- the log-A bar scale now spans 21 decades, so the 140 optical lines occupy only
  t = 0.70 .. 1.00 of the available bar height.

Options, none taken yet because each is a product decision rather than a physics
one: split the view by regime, put a disclosed wavelength-range control on it,
or scale bars within the displayed range. Dropping the microwave lines silently
is **not** an option; they are real, and hiding them to tidy an axis is exactly
the kind of quiet lie the prime directive forbids.

## Deferred

- The two-regime axis problem above.
- Screened-atom dipole integrals (still the numerical-radials gap).
- Hyperfine-resolved line strengths (another 6j layer, on F rather than j).
- Population modelling, unchanged from Phase 14.
