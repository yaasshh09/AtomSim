# Phase 24 — Pauli Exclusion OFF (configuration collapse)

**Date:** 2026-07-31 · **Status:** in build · **Tier:** `COUNTERFACTUAL`

---

## 1. What this is

Spec §4, a v1 What-If Lab item that was never built: *"Pauli exclusion OFF
(configuration collapse — why chemistry exists)"*. Remove the occupancy cap and
every electron falls into the 1s. The atom stops having shells, so it stops
having chemistry, and the number on the screen says by how much.

## 2. Pauli off implies exchange off

These are not two switches. The Pauli exclusion principle *is* the antisymmetry
of the fermionic wavefunction; the occupancy cap and the exchange energy are two
consequences of the same thing. Phase 22 removed the second and deliberately
kept the first, and its disclosure says so in as many words ("the Pauli
principle is NOT switched off"). This phase removes the root.

So `pauli=False` forces `exchange=False`, and the combination `pauli=False,
exchange=True` is refused rather than computed. It is not a model: a Slater
determinant with two electrons in the same spin-orbital is identically zero, so
there is no wavefunction there to take an exchange integral over. Returning a
number for it would be inventing physics for a state that does not exist.

Phase 22 is therefore the honest half-step, and this is the whole step. The two
tiers of disclosure have to distinguish them or the second one teaches nothing.

## 3. Why no coefficient changes are needed

The average-of-configuration functional survives `q > 2(2l+1)` untouched, and it
is worth being precise about why, because "the code did not crash" is not a
reason. The surviving terms are

    direct potential:   (q_a - 1) U_0[a,a] + sum_{b != a} q_b U_0[b,b]
    energy functional:  (q_a (q_a - 1) / 2) F_0(aa) + sum_{a<b} q_a q_b F_0(ab)

`(q - 1)` is "how many other electrons an electron sees" and `q(q-1)/2` is "how
many pairs there are". Both are pure combinatorics on a count. Neither knows
about capacity, and neither should: the classical electrostatic repulsion
between ten electrons sharing an orbital is exactly 45 pair interactions, cap or
no cap. Every coefficient that *would* have needed rederiving carries a squared
3j symbol and belongs to exchange, which is gone.

What does NOT survive is term structure. `subshell_terms` counts the microstates
of `l^q` by enumerating distinct spin-orbital assignments, which is Pauli's
combinatorics from top to bottom, and it raises above capacity. It should: a
term symbol for 1s^10 is not a hard question, it is a meaningless one. The
provenance says term structure is undefined here rather than reporting a
configuration average over terms that do not exist.

## 4. Independent ground truth for the collapsed atom

Unusually for a counterfactual, this one has closed-form check. For N electrons
in a single hydrogenic 1s of exponent zeta, with no exchange,

    E(zeta) = N (zeta^2/2 - Z zeta) + [N(N-1)/2] (5 zeta / 8)

since <1s|h|1s> = zeta^2/2 - Z zeta and F_0(1s,1s) = 5 zeta / 8. Minimizing,

    zeta* = Z - (5/16)(N - 1)

The SCF solves for the best 1s radial *function*, not the best exponential, so
it is variational over a strictly larger space: `E_SCF <= E(zeta*)`, and close.

This is checkable against a number nobody in this repo chose. At Z = N = 2 the
formula gives zeta* = 1.6875 and E = -2.8477 hartree, which is the textbook
variational helium result, and the Phase 22 measurement of helium's Hartree
energy is -2.8617 — below it, as it must be, by the amount a free radial
function buys over an exponential. Helium is the case where "Pauli off" changes
nothing at all (1s^2 is already the ground configuration and needs no cap
lifted), which is what makes it a clean calibration of the formula rather than a
test of the collapse.

## 5. Expected physics, i.e. what the tests assert

- **The collapsed atom is far more bound.** Neon's ten electrons in one 1s
  should land near `zeta* = 10 - 45/16 = 7.19`, `E ~ -258` hartree, against the
  real -128.5. Nothing is holding them out of the deep well.
- **The collapsed atom is far smaller, and shrinks with Z.** For a neutral atom
  `N = Z`, so `zeta* = (11/16) Z + 5/16` and the mean radius goes as `~ 1/Z`,
  monotonically. That is the teaching payoff stated as an inequality: with Pauli
  on, atomic size oscillates across a period and that oscillation is the
  periodic table; with Pauli off it decreases forever and there is no chemistry.
- **`E_SCF <= E(zeta*)` for every collapsed atom**, and within a few percent.
- **The ladder has exactly one rung.** All N electrons occupy 1s, so a Levels
  view drawn from this has one orbital energy, not a shell structure.
- **`pauli=False, exchange=True` raises**, and the message says why rather than
  quietly flipping the flag for the caller.

## 6. Provenance

`COUNTERFACTUAL`, and the alteration named at the head of the assumption list
the way Phase 22's is. What this one must additionally disclose:

- that the occupancy cap is gone, which is the thing Phase 22 promised was still
  in place, so a reader who learned the earlier disclosure is not misled;
- that exchange went with it, and that this was forced rather than chosen;
- that term structure is undefined, replacing (not merely omitting) the
  configuration-average line the real atom carries.

## 7. Surfaces

- `atoms.py`: `pauli` flag on `aufbau_configuration`, `validate_config`,
  `is_ground`. With it off, Aufbau is `1s^N`.
- `hf_atom.py`: `pauli` on `solve_hartree_fock`; `pauli_collapse` reports the
  real atom and the collapsed one together, with energies and mean radii.
- Server: `pauli` on `HFRequest`, refusal as 422.
- Frontend: a checkbox that disables and forces the Phase 22 one, since the
  weaker counterfactual is contained in the stronger.

---

## 8. What building it changed

**Section 5 quoted the bound as if it were the answer.** "Neon should land near
-258 hartree" is the exponential's number, and the SCF is required to come out
below it. Neon actually lands at -264.35. The gap is the point rather than an
error, and it widens with Z because a free radial function has more to buy the
more electrons are stacked in one orbital: He 0.49%, Li 1.02%, Be 1.39%, C
1.87%, Ne 2.34%, Ar 2.71% below `E(zeta*)`. "Within a few percent" survives
through argon, but it is being spent steadily, so the test's tolerance is a
relative one and argon sits at roughly half of it.

**What the exclusion principle is worth, measured.** Total energy in hartree and
mean radius in bohr, real atom against collapsed one:

    He   -2.8617 / -2.8617     0.9273 / 0.9273  (no-op, exactly)
    Li   -7.4327 / -8.5469     1.6733 / 0.6735
    Be  -14.5729 / -19.0193    1.5322 / 0.5295
    C   -37.6595 / -60.1798    1.1989 / 0.3713
    Ne -128.5464 / -264.3490   0.7891 / 0.2328
    Ar -526.8152 /-1487.9640   0.8928 / 0.1334

Neon's binding roughly doubles and argon's nearly triples. Helium's equality is
bit-exact in both columns, energy and radius, which is what makes it the
calibration case rather than a suspiciously small difference.

**The periodicity assertion is visible twice in that table, and needed to be.**
The real radii are not monotone at Li over He (the 2s opens) and again at Ar
over Ne (the 3s opens), while the collapsed ones fall from 0.927 to 0.133 with
no feature anywhere. The test picks He, Be, Ne so that one such rise is inside
the sample; a sample that missed every period boundary would have passed while
asserting nothing.

**Phase 22's convergence penalty was shell competition, not the missing
exchange.** Neon, coarse SCF iterations: 15 with exchange and shells, 46 with
the shells and no exchange, 16 with neither. Carbon: 16, 28, 14. Phase 22 saw
the middle number next to the first and read the penalty as the cost of the
toggle; the third says it takes both conditions, because one orbital and no
exchange converges as fast as the real atom does. The reading of why, which the
iteration counts support but do not prove, is that exchange-off leaves nothing
holding distinct occupied orbitals apart while the cap still insists they stay
distinct, and the mixing has to do that work instead.

**The size half of the payoff had no number to report.** `<r>` existed
analytically for hydrogen-like states and nowhere for an HF solve, so
`hf_mean_radius` is new machinery rather than a lookup, checked against the one
closed form available (a one-electron solve gives 3/2 bohr). It deliberately
drops the solve's error estimate instead of inheriting it: that spread is in
hartree, and carrying it onto a length would not be a loose bar, it would be a
number in the wrong unit.

**The refusal belongs to the request schema, not the endpoint.** `pauli=false`
with `exchange=true` is answered 422 by validation rather than 400 by a handler
check. 400 is the server declining a well-posed request, and this one is not
well posed: there is no such model to decline.

**The frontend coupling had to run both ways.** Section 7 said the Pauli
checkbox would disable and force the exchange one, which is only half of it.
Turning exchange back on also restores the cap, because a store that can hold
`pauli=false, exchange=true` will eventually send it, and the only thing waiting
there is the 422. Both directions clear the configuration too, so the solve uses
the ground state of whichever rule is now in force rather than carrying a
configuration across and quietly changing which atom is on one side.

**A hand-written collapsed configuration gets no comparison.** `1s3 2s1` is
legal with the cap off and has no twin with the cap on, so the server sends
nothing rather than differencing it against a configuration that is not the same
atom.
