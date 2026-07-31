# Phase 22 — Distinguishable Electrons (exchange off)

**Date:** 2026-07-31 · **Status:** in build · **Tier:** `COUNTERFACTUAL`

---

## 1. Why now

The requirements spec parks one What-If Lab toggle behind a prerequisite:

> **HF phase:** distinguishable electrons (exchange effects only become honestly
> demonstrable once exchange energy exists in the model).

Phase 21 put exchange in the model. Before it, "turn exchange off" had nothing
to turn off: GSZ is a fitted central field with no exchange term anywhere in it,
so a toggle would have been a label over a no-op. Now the Fock operator has a
real non-local exchange term and the energy functional has real G_k integrals,
so removing them removes something, and the difference is a number we can put on
the screen.

Turning exchange off turns Hartree-Fock into **Hartree**: electrons that repel
each other but are not required to be indistinguishable. It is the historically
real model that came first, and the gap between the two is the exchange energy.

## 2. What the toggle actually does

Exactly three terms leave the calculation, and they are the same three terms in
all three places they appear:

1. the same-shell `k > 0` terms — `same_shell_coefficient` × `U_k[a,a]`,
2. the cross-shell terms — `exchange_coefficient` × `U_k[a,b]`,
3. in the energy functional, the matching `F_k(aa)` for `k > 0` and every
   `G_k(ab)`.

Everything else stays: the one-electron integral, the `k = 0` Hartree potential,
and the `(q_a - 1)` factor that stops an electron from repelling itself. That
factor is *not* exchange and does not leave. It is classical electrostatics —
an electron does not act on itself whether or not the wavefunction is
antisymmetric — and dropping it too would silently fold a self-interaction error
into a number the UI is about to call the exchange energy.

### What the toggle does NOT do

It does not turn off the Pauli exclusion principle. The occupation numbers are
still capped at `2(2l + 1)` per subshell and the configuration is still the
Aufbau one, both of which are Pauli's doing. This model is "the wavefunction is
a product instead of a determinant", not "electrons may pile into 1s". Full
configuration collapse is a separate toggle with a separate answer, and the
disclosure has to say which one the reader is looking at, because a badge that
says COUNTERFACTUAL without saying *which* counterfactual is decoration.

## 3. The self-check that comes free

`solve_hartree_fock` already computes the total energy by two routes that share
no code beyond the one-electron integral, and raises if they disagree by more
than `_ROUTE_AGREEMENT`:

- route 1 assembles the functional term by term (`total_energy_direct`),
- route 2 uses `E = ½ Σ q_a (I(a) + ε_a)`, where ε_a comes out of the operator.

Exchange lives in the functional for route 1 and in the operator for route 2. So
a half-applied toggle — off in the operator, still on in the functional, or the
reverse — makes the two routes disagree and raises `HFConvergenceError` on every
atom. The one bug this phase can plausibly ship is already fatal at runtime, on
existing machinery, with no new test needed. New tests still go in; this is the
belt to their braces.

## 4. Expected physics, i.e. what the tests assert

- **Helium's exchange energy is exactly zero.** Not small — zero, to the
  quadrature. A closed 1s shell holds one spin-up and one spin-down electron,
  exchange only couples same-spin pairs, and `exchange_operator` accordingly
  builds no terms at all for `1s²`. The `k = 0` same-shell exchange is already
  carried by the `(q_a - 1)` factor. So He is the case where the toggle is
  visibly, correctly, a no-op, and it is the sharpest test in the phase: any
  leak of the `(q_a - 1)` factor into the "exchange" bucket shows up here as a
  spurious few hartree.
- **Beryllium's is not.** `1s²2s²` has G_k(1s,2s), so exchange stabilizes it.
- **Exchange is stabilizing: `E_HF < E_Hartree` for every atom that has any.**
  Both are variational upper bounds on the same exact energy, and the Hartree
  wavefunction is a special case of nothing — but the HF functional is the
  Hartree functional minus a positive quantity evaluated at its own optimum, so
  the inequality holds and is worth asserting on every vendored atom.
- **Both models still satisfy the virial theorem, `-V/T = 2`.** Hartree is a
  legitimate variational model in its own right, not a broken HF, and the virial
  ratio is the check that says so. If Hartree's virial drifted while HF's did
  not, the toggle would have broken the solve rather than changed the model.
- **The exchange energy grows with Z.** Roughly with the number of same-spin
  pairs available.

## 5. Provenance

The Hartree result is `COUNTERFACTUAL`, not `APPROXIMATION`. The distinction the
tier list draws is between "an honest simplified model of the real thing" and
"deliberately altered physics, computed rigorously under the altered rules", and
this is the second: real electrons are indistinguishable, and this calculation
says they are not. It is not a cheaper HF, it is a different universe's atom,
and it is solved exactly as carefully as the real one.

Its assumption list keeps everything HF's has that is still true (no
correlation, non-relativistic, configuration average) and leads with what
changed, in the form of the sentence that says which counterfactual this is.

The `EXACT`/`NUMERICAL`/`APPROXIMATION` tiers describe distance from the truth.
`COUNTERFACTUAL` does not — it describes distance from *this* universe — which
is why it cannot be expressed as an error bar, and why the Hartree energy is
reported without one against the real atom. The mesh error bar it does carry is
about the numerics of the altered model, and is labeled as such.

## 6. Surfaces

- Engine: `exchange: bool` threaded through `hf_terms` → `hartree_fock` →
  `hf_atom`, keyword-only, defaulting to `True` so no existing caller changes
  meaning.
- `hf_exchange_energy(z, n_electrons, config)` — solves both models and returns
  `E_HF - E_Hartree` as a `Quantity`, which is the number the UI wants.
- Server: the HF job takes `exchange`, and a new endpoint returns the exchange
  energy so the UI does not have to subtract two numbers it fetched separately
  and hope they came from the same mesh.
- Frontend: a checkbox next to the Hartree-Fock model selector, live only when
  the HF model is selected, wired into the store's `INVALIDATED` set because the
  levels it produces are different physics.

---

## 7. What building it changed

**The store toggle is not in `INVALIDATED` after all.** Section 6 said it would
be. It should not be: `INVALIDATED` clears everything derived from (n, l, m,
system, basis), and an HF solve is derived from none of those. Putting the flag
there would have thrown away seconds of solve on every click of n, for an answer
that cannot change. It clears `hf` explicitly instead, the same shape `setConfig`
already uses for the same reason.

**The caption inherited a theorem that does not cover it.** The Hartree-Fock
caption says the total energy "is variational, unlike the screened model's sum
of orbital energies". Rendered under the counterfactual, that sentence was
false: a product wavefunction is not antisymmetric, so it is not an admissible
trial function for electrons, and the variational theorem says nothing about
where its energy lands relative to the true one. It happens to land above. The
counterfactual branch now says stationary-for-this-model and states why the
bound does not apply. The same trap is already noted in `test_hf_exchange.py`,
which deliberately does not assert the bound; the caption was the place it got
through anyway, and it got through by being inherited rather than written.

**Helium's zero is bit-exact, and that is worth more than a tolerance.** The
prediction was "zero". What the code actually does for a single closed s shell
is run bit-identical arithmetic down both paths - the exchange branches are
empty loops, not small numbers - so the test asserts `== 0.0` rather than
`approx(0)`. A tolerance there would hide the first term that ever starts
contributing.

**Magnitudes, measured.** E_HF - E_Hartree, in hartree: He 0 (exactly), Li
0.0203, Be 0.0641, C 0.390, Ne 2.14, Ar 7.38 - between 0% and 1.7% of the total
energy, always stabilizing, monotonic in Z over this set. Cross-checked against
an independent count for neon: 5 spin-up electrons give C(5,2) = 10 same-spin
pairs per spin, 20 in all, and an average exchange integral of order 0.1 hartree
puts the total near 2. It is smaller than the exchange integrals evaluated on
fixed Hartree-Fock orbitals would suggest, which is the expected direction: the
Hartree solve relaxes its own orbitals and recovers part of the gap.

**Convergence, not accuracy, is what the toggle costs.** Hartree takes more SCF
iterations than Hartree-Fock on the closed shells, not fewer: neon needs 46
coarse iterations against 15, carbon 28 against 16. Both converge well inside
the budget and both hold the virial ratio to 2.000004, so this is a wall-clock
note rather than a limit.
