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
