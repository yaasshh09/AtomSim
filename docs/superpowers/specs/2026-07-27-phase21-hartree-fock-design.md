# Phase 21: Hartree-Fock, self-consistently

Status: design, approved to implement.
Supersedes the model error in `numerics/screening.py`, not the module itself.

## 1. What this is for

`screening.py` has carried a promise in its provenance since Phase 6:

> `refinement="self-consistent Hartree-Fock (a later phase) removes the model error"`

This phase pays that off. The Green-Sellin-Zachor potential is an analytic
independent-particle model whose two parameters `(d, K)` are transcribed
per-atom from Szydlik and Green (1974). It is fast and it is honest about being
approximate, but three things are wrong with it as the engine's ceiling:

1. **It needs a table.** No parameters, no atom. Neutral S and Cl have no entry
   in the source paper, so the engine refuses them (`atoms.NO_GSZ_PARAMETERS`).
   That refusal is the correct behaviour under the prime directive and it is
   also an admission that the model does not generalize.
2. **Its total energy is not a total energy.** `solve_screened_atom` says so
   directly: `"not a variational total energy; ignores e-e double counting"`.
   Summing occupancy-weighted eigenvalues double counts every electron-electron
   interaction. The number exists, it is labelled, and it is not the energy of
   anything.
3. **The error has no sign.** GSZ can land above or below the truth and the code
   cannot say which. Hartree-Fock is variational: `E_HF >= E_exact` for the
   non-relativistic, infinite-nuclear-mass Hamiltonian, always. Trading an
   unsigned model error for a one-signed, bounded one is the real upgrade here,
   and it is a statement the provenance can make and a test can check.

There is a fourth point worth stating because it is easy to misread as
circular. Szydlik and Green fitted `(d, K)` *to* Hartree-Fock results, and the
per-atom deviations vendored as fidelity hints in `screening.py` (0.5 ppm for
He up to 240 ppm for Li) are deviations from HF. So when the new solver agrees
with GSZ, that is not two independent methods confirming each other. It is the
new solver reproducing the thing GSZ was built to imitate, which validates the
implementation and says nothing new about the physics. The cross-model test in
section 7.4 is written with that framing so it cannot be quoted as more than it
is.

### 1.1 What this phase does not deliver

The phase removes the parameter table. It does **not**, by itself, extend the
engine to arbitrary Z, and the spec is written that way on purpose.

The blocker is the mesh, not the physics. `radial_solver.py` uses a uniform
grid, and the 1s length scale contracts as 1/Z, so at fixed step `h` the
finite-difference error on the core eigenvalue grows roughly as Z^2. That is
affordable through Z=18, which is exactly the range GSZ already covers, and it
degrades past there. A logarithmic mesh (`r = r_0 (e^x - 1)`, uniform in x) is
what every production atomic-structure code uses and is the actual prerequisite
for Z beyond about 20. It is deferred to Phase 22 (section 10) because it
touches `radial_solver.py` for every existing caller, and doing it inside this
phase would mean changing the numerics and the physics in the same step with no
fixed point to test against.

So: this phase buys parameter-free physics and a real total energy over the
current Z range. The Z range itself moves next phase.

## 2. The physics

### 2.1 Restricted Hartree-Fock, average of configuration

Orbitals are `P_nl(r) = r R_nl(r)`, real, one radial function per `(n, l)`
subshell, shared by all electrons in that subshell regardless of `m` or spin.
This is restricted HF. Open shells are handled by the average-of-configuration
energy functional: the energy is averaged over all the ways the subshell
occupancies can be distributed over `m_l` and `m_s`.

That choice has a consequence which must be disclosed, not buried. Average of
configuration gives one energy per *configuration*, not per *term*. For carbon
`1s2 2s2 2p2` it returns the centroid of `3P`, `1D` and `1S`, and it cannot
split them. The engine already models fine structure, hyperfine structure and
Zeeman and Stark splittings elsewhere; none of those are term splittings, and
nothing in this phase gains the ability to produce one. `HFResult` therefore
carries an explicit assumption string saying the energy is a configuration
average, and the frontend badge repeats it for any atom whose configuration has
a partially filled subshell.

### 2.2 The radial equations

For each occupied subshell `a = (n_a, l_a)` with occupancy `q_a`:

```
[ -1/2 d2/dr2 + l_a(l_a+1)/(2 r^2) - Z/r + V_direct(r) ] P_a(r)
    - (K P_a)(r)  =  eps_a P_a(r) + sum_{b != a, l_b = l_a} lambda_ab P_b(r)
```

`V_direct` is local and spherically symmetric. `K` is the exchange operator,
non-local: it maps `P_a` to a function built from integrals of `P_a` against
every occupied orbital. That non-locality is the whole numerical difficulty of
this phase, and section 5 is about it.

The off-diagonal Lagrange multipliers `lambda_ab` enforce orthogonality between
same-`l` orbitals (1s against 2s, and so on). Solving each `l` channel as a
symmetric eigenproblem makes them unnecessary: eigenvectors of a symmetric
operator belonging to distinct eigenvalues are orthogonal by construction. This
is the main reason the eigenvalue formulation is preferred over the classical
inhomogeneous-ODE formulation, and it is worth the eigensolver.

### 2.3 Slater integrals by cumulative quadrature

Everything two-electron reduces to the pair potential

```
U_k[a,b](r) = integral_0^inf ( r_<^k / r_>^(k+1) ) P_a(s) P_b(s) ds
            = r^-(k+1) integral_0^r  s^k     P_a P_b ds
            + r^k      integral_r^inf s^-(k+1) P_a P_b ds
```

from which `F^k(ab) = int P_a^2 U_k[b,b] dr` and `G^k(ab) = int P_a P_b U_k[a,b] dr`.

Compute `U_k` with two cumulative trapezoid passes, O(N) each, no ODE solve.
Two properties make this self-checking:

- For a normalized density and `k = 0`, `U_0(r) -> 1/r` as r grows past the
  charge. Assert it at the box edge.
- `U_k` is symmetric in its two orbital arguments. Assert it.

Both are cheap, and both catch the endpoint and `r_<`/`r_>` mistakes that this
kind of quadrature invites.

### 2.4 The angular coefficients, and why they are not written here

The direct and exchange terms carry angular coefficients built from Wigner 3j
symbols, of the general shape

```
c_k(l_a, l_b)  proportional to  (2 l_b + 1) * wigner_3j(l_a, k, l_b, 0, 0, 0)^2
```

with `k` running over `|l_a - l_b| .. l_a + l_b` in steps that keep
`l_a + k + l_b` even, plus occupancy factors that differ between the same-shell
and cross-shell cases.

**This spec deliberately does not state the coefficient formulas.** They are the
highest-risk part of the phase and they are easy to get plausibly wrong. A
worked example of exactly that, done while writing this spec: taking the
same-shell occupancy weight as `q_b / (2(2 l_b + 1))` reproduces helium
correctly (net potential comes out to exactly one unit of the 1s Hartree
potential, self-interaction free, which is right) and then fails hydrogen,
leaving a residual half-unit of an electron's own Hartree potential acting on
itself. One-electron HF must reduce to the bare Coulomb problem exactly, and
that convention does not. The pair-counting factor `(q_a - 1)/q_a` fixes
hydrogen and then risks double counting against the same-shell exchange term.

A wrong factor here does not crash. It produces a converged, smooth, entirely
believable set of orbitals with the wrong energy, which is the precise failure
mode the prime directive exists to prevent. So the coefficients get derived by
differentiating the average-of-configuration energy functional with respect to
`P_a`, with the derivation written into the module docstring, and they get
pinned by the anchors in section 7.1 before anything else in the phase is
trusted.

**Update.** That derivation is now done and lives in
`docs/superpowers/plans/2026-07-27-phase21-hartree-fock.md` under "The derived
equations", where it is checked four independent ways: hydrogen, helium,
agreement with a separate closed-shell derivation on full shells, and
beryllium's textbook `4J - 2K`. This section stands as the record of why it was
not written from memory, which remains the right call.

`analytic/wigner.py` currently exports `triangular` and `wigner_6j` only. This
phase adds `wigner_3j` to that module, following the same conventions already
established there: doubled integer momenta internally so triangle conditions are
integer comparisons, plain `float` return (a 3j symbol is an algebraic constant,
not a modelled physical value, exactly as the module docstring already argues
for 6j), and exact zeros when a selection rule fails.

## 3. Fidelity model

Two distinct errors live in an HF number and the design keeps them apart,
because collapsing them is how a model starts lying quietly.

**Error one: solving the HF equations imperfectly.** Finite grid, finite box,
finite SCF convergence. All three are measurable, and all three are `NUMERICAL`.

**Error two: HF is not the atom.** HF omits electron correlation entirely, a
few tenths of a hartree for a light atom, well under one percent of the total
energy but far larger than any grid error. That is `APPROXIMATION`, and
unusually for this codebase it is a *signed and bounded* approximation:
`E_HF >= E_exact`. The magnitude quoted in the provenance string is not to be
written from memory: it comes from the measured gap between the computed HF
energy and a vendored exact non-relativistic energy, or it is omitted and the
assumption string says only that the deficit is positive and unquantified.

Following the convention `screened_atom.py` already established (label by the
dominant error, carry the smaller one as a quantified sub-scale):

| Quantity | Fidelity | error_estimate | Assumptions carry |
|---|---|---|---|
| `total_energy` | `APPROXIMATION` | grid + SCF residual, hartree | no correlation; variational upper bound; configuration average if open shell |
| `orbitals[i].energy` | `APPROXIMATION` | grid + SCF residual | Koopmans: neglects orbital relaxation and correlation, which partly cancel |
| `kinetic`, `potential` | `NUMERICAL` | grid-halving | these are properties of the computed solution, not claims about the atom |
| `virial_ratio` | `NUMERICAL` | departure of `-V/T` from 2 | a convergence diagnostic, not a physical result |
| `P_nl(r)` fields | `APPROXIMATION` | grid-halving on the eigenvalue | mean-field orbital, no correlation |

`kinetic`, `potential` and `virial_ratio` being `NUMERICAL` while
`total_energy` is `APPROXIMATION` is intentional and is the sharpest expression
of the split: the virial ratio is a statement about whether the equations were
solved, and the total energy is a statement about an atom.

The refinement string on `total_energy` becomes
`"configuration interaction or many-body perturbation theory would recover the
correlation energy"`, replacing GSZ's promise of this phase with the next
honest one.

**Non-convergence must raise, never return.** A `HFResult` with `converged=False`
is a quiet lie in object form. `solve_hartree_fock` raises
`HFConvergenceError` carrying the residual history, and `converged` stays on the
dataclass only so the exception can be inspected.

## 4. Modules

Three new modules, one extension.

**`analytic/wigner.py` (extend).** Add `wigner_3j`. Pure algebra, plain floats,
no provenance, consistent with the existing 6j rationale.

**`numerics/slater.py` (new).** The two-electron radial machinery: `y_k(P_a,
P_b, r, k)` returning `U_k`, and `slater_f`, `slater_g` returning `F^k` and
`G^k`. Plain arrays and floats in, plain out. This module is quadrature, not
physics, and it is where the section 2.3 self-checks live.

**`numerics/hartree_fock.py` (new).** The SCF loop, the Fock operator as a
`scipy.sparse.linalg.LinearOperator`, the preconditioner, the mixing, and the
angular coefficients from section 2.4. Returns `HFResult`.

**`hf_atom.py` (new, top level).** The counterpart of `screened_atom.py`:
element and configuration in, `HFResult` out, provenance assembled, with
`hf_radial`, `hf_valence_ionization_energy` and `evaluate_hf_state` mirroring
the screened-atom API one for one so callers can switch models by swapping a
function, not a call shape.

```python
@dataclass(frozen=True)
class HFOrbital:
    n: int
    l: int
    occupancy: int
    energy: Quantity          # APPROXIMATION, hartree
    P: Field                  # r R_nl(r), on the solver grid

@dataclass(frozen=True)
class HFResult:
    key: str
    z: int
    n_electrons: int
    config: Configuration
    is_ground: bool
    orbitals: tuple[HFOrbital, ...]
    total_energy: Quantity    # APPROXIMATION
    kinetic: Quantity         # NUMERICAL
    potential: Quantity       # NUMERICAL
    virial_ratio: Quantity    # NUMERICAL, target 2
    iterations: int
    residual_history: tuple[float, ...]
    converged: bool
    provenance: Provenance
```

`HFResult` is deliberately shaped like `ScreenedAtomResult` plus the fields that
only mean something once the energy is variational. Everything downstream
(`spectra.py`, `populations.py`, `transfer.py`) consumes orbital energies and
dipole integrals, so a small adapter is all that is needed to let the existing
spectroscopy stack run on HF orbitals. That adapter is in scope; rewiring the
spectroscopy defaults to HF is not (section 10).

## 5. Numerics

### 5.1 The eigenproblem stops being tridiagonal

`eigh_tridiagonal` cannot be used, because the Fock operator is not tridiagonal:
exchange couples every grid point to every other. Dense `eigh` is not an
alternative either. The grid is N of order 1e4 to 4e4 (r_max about 40 bohr,
h about 0.005 bohr for light atoms and 0.001 for argon, set by the 1/Z core
scale). A dense matrix at N = 2e4 is 3.2 GB and O(N^3) to diagonalize. So
matrix-free iteration is mandatory here, not a preference.

The Fock operator is applied as a `LinearOperator`:

```
F psi = H_local psi - K psi
```

`H_local psi` is a tridiagonal matvec, O(N). `K psi` requires `U_k[b, psi]` for
every occupied `b` and every allowed `k`, recomputed each matvec because it
depends on `psi`. For argon that is five occupied subshells times a handful of
`k` values, so roughly 20 to 30 cumulative-integral passes of length N per
matvec, order 1e6 flops. Cheap in numpy. **Iteration count, not matvec cost, is
the thing to control.**

### 5.2 The preconditioner (resolved)

LOBPCG needs one. The largest eigenvalue of the discretized kinetic operator is
`2/h^2` (the `2 inv2m / h^2` diagonal of `radial_solver.py` with
`inv2m = 1/2`, times the factor 2 at the top of the finite-difference
spectrum), which is 8e4 hartree at h = 0.005 and 2e6 at h = 0.001, against level
spacings of order 1 hartree in the valence region. Unpreconditioned LOBPCG on
that stagnates.

The preconditioner is essentially free: `M = (H_local - sigma I)^-1`, applied by
a banded solve in O(N), with `sigma` placed below the lowest eigenvalue sought
in that `l` channel so `H_local - sigma I` is positive definite and the
factorization is stable without pivoting. It is the right preconditioner
because `H_local` carries all the high-frequency content of the spectrum, while
exchange is a smooth integral kernel whose spectrum decays fast, so the
preconditioned operator has a small effective condition number. The
factorization is computed once per SCF iteration per `l` channel and reused
across all LOBPCG iterations.

Two supporting decisions:

- **Warm start.** Iteration zero starts from the GSZ orbitals for atoms that
  have parameters, and from a hydrogenic `Z_eff` guess otherwise. Later
  iterations start from the previous iteration's orbitals, which is LOBPCG's
  best case and the main reason the block method is affordable here.
- **Block size.** `n_states + 2` guard vectors per channel, because LOBPCG loses
  accuracy on the highest requested eigenpair.

**Fallback, named now so it is not invented under pressure.** If LOBPCG proves
fragile, replace it with Rayleigh-Ritz in the local eigenbasis: take the lowest
~40 eigenvectors of `H_local` from the existing `eigh_tridiagonal`, project the
full Fock operator into that subspace, and diagonalize the small dense matrix.
It is robust and reuses machinery already in the repo. It is the fallback and
not the primary because it introduces a second error channel (basis truncation)
that would have to be separately quantified and disclosed, where LOBPCG
converges to the true grid eigenpair and leaves discretization as the only
numerical error. Fewer error channels wins under the prime directive.

### 5.3 SCF loop

```
warm start -> build V_direct and K from current orbitals
           -> solve each l channel (LOBPCG, preconditioned)
           -> assemble new density
           -> mix: rho_new = alpha rho_out + (1 - alpha) rho_in,  alpha = 0.4
           -> converged when max |delta eps| < 1e-8 hartree
              and max |delta rho| < 1e-6
```

Damped linear mixing, not bare iteration: undamped SCF on atoms with a diffuse
valence shell oscillates. `alpha = 0.4` is a starting value to be tuned against
measured iteration counts, not a claim. DIIS would converge faster and is
deferred (section 10); linear mixing has one parameter and no failure modes
worth debugging in the same phase as the angular algebra.

`residual_history` is recorded and returned, so stagnation is visible in the
result rather than silently absorbed into a "converged" flag.

### 5.4 Grid

Uniform, reusing `radial_solver.py`'s conventions and its O(h^2) grid-halving
error estimator unchanged. `r_max` and `n_points` scale with Z and with the
outermost occupied `n`; both are validated by a box-convergence test rather
than asserted, since GSZ's `_r_max` of `40 (n_top + 1)^2 / Z_net` is far larger
than an HF atom needs and would waste most of the grid on vacuum.

See section 1.1 for why this stays uniform in this phase.

## 6. Total energy, computed three ways

The energy is assembled by direct summation, and then checked by two
independent routes that share no code with it. This is the same discipline as
the Phase 18 flux-closure check, which caught two invisible grid bugs.

1. **Direct assembly.** One-electron integrals plus direct plus exchange, from
   the energy functional.
2. **Orbital identity.** `E = 1/2 sum_a q_a ( I_a + eps_a )`, where `I_a` is the
   one-electron integral. The two routes were checked against each other on
   helium while writing this spec, using approximate figures purely to exercise
   the algebra: taking `I_1s = -1.944` and `eps_1s = -0.918`, route 2 gives
   `1/2 * 2 * (-2.862) = -2.862`, and the direct route `2 I + F^0` with
   `F^0 = eps - I = 1.026` gives `-3.888 + 1.026 = -2.862`. They agree, which is
   what makes this a real cross-check and not a restatement of route 1. Those
   two input figures are illustrative and unverified; they are **not** benchmark
   values and must not be copied into a test. See section 7.3.
3. **Virial.** At a converged HF solution in a pure Coulomb external potential,
   `E = -T` and `-V/T = 2` exactly. Any departure is a convergence or grid
   defect, and it is reported as `virial_ratio` rather than swallowed.

Routes 1 and 2 must agree to 1e-8 hartree (they are algebraically identical, so
disagreement means a coding error). Route 3 must agree to 1e-4 relative (it is
sensitive to grid and box, so disagreement means a numerics problem, and its
size is the honest measure of that).

## 7. Validation

### 7.1 Exact anchors, in ladder order

These pin the angular algebra of section 2.4 and are written before the SCF loop
is trusted.

| Case | What it pins | Expected |
|---|---|---|
| H, Z=1, N=1 | no self-interaction at all | `E = -0.5` hartree exactly, direct and exchange contributions both exactly zero |
| He `1s2` | direct term and the same-shell factor | net potential on an electron is exactly one unit of the 1s Hartree potential; `E_HF` matches the vendored value |
| Be `1s2 2s2` | cross-shell direct and `k=0` exchange | vendored value |
| Ne `1s2 2s2 2p6` | `k > 0` exchange (l_a=0, l_b=1) | vendored value |
| Mg, Ar | three shells, larger Z, grid stress | vendored value |

Hydrogen is the sharpest of these and costs nothing: it is the same calibration
anchor `screening.py` already uses (N=1 reduces to bare Coulomb), and it is what
the discarded coefficient convention in section 2.4 failed.

### 7.2 Internal identities

Three-way total energy agreement (section 6). `U_0 -> 1/r` at the box edge and
`U_k` argument symmetry (section 2.3). Orbital orthonormality within each `l`
channel. Grid-halving convergence on every reported energy, and box convergence
on `r_max`.

### 7.3 Absolute benchmark data

Source: **Bunge, Barrientos and Bunge (1993), "Roothaan-Hartree-Fock
ground-state atomic wave functions: Slater-type orbital expansions and
expectation values for Z = 2 to 54", Atomic Data and Nuclear Data Tables 53,
113 to 162.**

Chosen over Froese Fischer's numerical HF tables because it is a single citation
covering Z = 2 to 54 in one table, which is the range the engine wants once the
mesh moves (section 1.1), so the vendored file will not need re-sourcing next
phase.

It is a finite-basis Roothaan HF calculation, so its energies sit slightly
*above* the basis-set-free numerical HF limit that a grid solver converges to.
That gap is handled structurally rather than argued about: **the test tolerance
is relative and grid-dominated** (1e-4 relative on the total energy), which is
one to two orders of magnitude larger than any plausible basis-set-incompleteness
gap for these atoms. Either source therefore satisfies the same test, and the
choice cannot silently corrupt the validation.

Vendored to `src/atomsim/data/hf_reference_energies.json`, in the same metadata
shape as the existing NIST files (`species`, `citation`, `retrieved`, `note`,
`values`), transcribed from the source at implementation time with the retrieval
date recorded. **No values are written into this spec**, deliberately: a
benchmark energy is a load-bearing number that gets compared against, and one
recalled from memory rather than transcribed from the source is exactly the kind
of number this project refuses to ship. Transcription is a task, not a
formality.

### 7.4 Cross-model comparison against GSZ

Compare HF valence ionization energies against the existing GSZ ones and against
the NIST values already vendored for He, Li and Na
(`test_screened_atom.py::test_valence_ionization_matches_nist`). HF must do at
least as well as GSZ on all three.

The test docstring must state what agreement with GSZ does and does not mean,
per section 1: GSZ was fitted to HF, so agreement is a check on this
implementation, not independent confirmation of the physics. The NIST comparison
is the one that carries external weight.

### 7.5 Performance

Record and assert wall time per atom. Phase 16 established that the box size and
the shared-grid strategy dominate screened-atom cost (13.4s down to 1.9s); HF
adds an SCF loop on top of the same eigenproblem, so a regression here is easy
to introduce and easy to miss. Target: argon in under 10 seconds. Iteration
counts go into `residual_history`, so the preconditioner claim in section 5.2 is
falsifiable rather than asserted.

## 8. Server and frontend

Minimal surface this phase, because the physics is the risk.

- `server/schemas.py`: `HFResultModel`, mirroring the existing screened-atom
  model plus `kinetic`, `potential`, `virial_ratio`, `iterations`, `converged`.
  Provenance survives to the browser as always.
- `server/app.py`: HF runs as a background job (`jobs.py`), not a synchronous
  request. An SCF loop is seconds of work, which is what the job protocol exists
  for.
- Frontend: a model selector (GSZ / Hartree-Fock) on the screened-atom views,
  with the `Badge` showing the fidelity split from section 3 and, for open
  shells, the configuration-average disclosure from section 2.1. The virial
  ratio is shown as a convergence readout, labelled as a diagnostic and not as
  physics.
- `lib/urlState.ts`: one new query key `model=gsz|hf`, default `gsz` so existing
  deep links keep resolving to what they resolved to before. Round-trip tested,
  as the query schema is a stable contract.

## 9. File plan

```
src/atomsim/analytic/wigner.py          extend: wigner_3j
src/atomsim/numerics/slater.py          new: y_k, slater_f, slater_g
src/atomsim/numerics/hartree_fock.py    new: Fock operator, preconditioner, SCF
src/atomsim/hf_atom.py                  new: HFResult, solve_hartree_fock, adapters
src/atomsim/data/hf_reference_energies.json   new: vendored, cited, dated
src/atomsim/numerics/screening.py       edit: refinement string points to hf_atom
src/atomsim/server/schemas.py           extend: HFResultModel
src/atomsim/server/app.py               extend: HF job endpoint
tests/test_wigner_3j.py                 new
tests/test_slater.py                    new
tests/test_hartree_fock.py              new: anchors, identities, benchmarks
tests/test_hf_atom.py                   new: end to end, NIST, cross-model
web/src/...                             model selector, badge, urlState key
docs/superpowers/plans/2026-07-27-phase21-hartree-fock.md   implementation plan
```

## 10. Milestones

1. `wigner_3j` plus `slater.py`, with the section 2.3 self-checks. No SCF yet.
2. Closed-shell HF (He, Be, Ne, Mg, Ar) against the anchors and the vendored
   energies. This is the milestone that either validates the angular algebra or
   exposes it.
3. Average-of-configuration open shells (Li, B, C, N, O, F, Na, Al, Si, P), plus
   S and Cl, which GSZ cannot do at all and which are the visible payoff.
4. Server plus frontend surface.

Milestone 2 is the gate. If the anchors do not pin the coefficients, nothing
past it is worth building.

## 11. Deferred

- **Logarithmic mesh, and with it Z > 18.** Phase 22. See section 1.1.
- **DIIS acceleration.** Linear mixing first; DIIS if measured iteration counts
  justify it.
- **Correlation.** CI or MBPT is a separate project. HF's variational bound is
  the honest stopping point here.
- **Term energies.** Requires multiconfiguration HF. Section 2.1.
- **Switching the spectroscopy stack to HF by default.** The adapter ships this
  phase so HF orbitals can drive `spectra.py` and `transfer.py`; changing the
  default is its own phase with its own regression surface.
- **Relativistic corrections for heavier atoms.** `analytic/dirac.py` exists for
  hydrogen; combining it with HF is Dirac-Hartree-Fock and is out of scope.

## 12. Risks

- **Angular coefficients wrong but convergent.** The main risk. Mitigated by the
  section 7.1 ladder, hydrogen first. If wrong: energies are smooth, plausible
  and off by a few percent, and only the anchors catch it.
- **LOBPCG stagnates.** Mitigated by the preconditioner, the warm start, and the
  named fallback in section 5.2. If wrong: iteration counts in
  `residual_history` show it immediately, and non-convergence raises.
- **Uniform grid insufficient at argon.** Mitigated by the relative tolerance and
  by grid-halving reporting the actual error. If wrong: argon misses the
  benchmark tolerance and the log mesh moves from Phase 22 into this one.
- **SCF oscillation on diffuse valence shells (Na, Al).** Mitigated by damped
  mixing. If wrong: reduce `alpha`, or bring DIIS forward.
