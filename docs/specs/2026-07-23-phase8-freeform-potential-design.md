# Phase 8: What-If: Free-Form V(r) (design)

**Date:** 2026-07-23
**Branch:** `phase8-freeform-potential`
**Status:** approved (standing authorization: user pre-approved "plan + build Phase 8", away during build)
**Parent spec:** `2026-07-04-atom-sim-requirements-design.md` (§4 What-If Lab, "Free-form V(r) expression entry (safely parsed) in a later phase") · **Builds on:** Phase 4/5 force-law lab, `numerics/radial_solver.py`, `numerics/force_law.py`

---

## 1. Goal and success criteria

The What-If Lab's force-law prong shipped a **curated preset library** (power law, Yukawa, Coulomb-plus-core, harmonic, finite well) in Phases 4/5. The requirements deferred one piece to "a later phase": letting the user type **their own** radial potential `V(r)` as a math expression, parse it **safely**, and run the real numerical consequences.

This phase delivers that. The user types an expression in `r` (bohr) returning `V` in hartree; the engine compiles it through a **whitelist AST parser** (no `eval`), feeds it to the existing radial solver, and shows the bound-state ladder against the EXACT hydrogen baseline under the `COUNTERFACTUAL` banner.

The honest heart of this phase is not the parser, it is the **trust gate**: an arbitrary `V(r)` can make a finite-difference solver return plausible-looking garbage without failing (continuum contamination, fall-to-center). The engine must detect those regimes and label them, never present them as real bound states.

**Done when:**

1. A user can select a **Custom** force law and type `V(r)`; the ladder, the sampled `V(r)` curve, and the EXACT hydrogen reference render in the existing Force-Law view.
2. The expression is parsed by a **whitelist AST evaluator** that rejects every construct outside a small math whitelist, with a precise error message. No `eval`, no `exec`, no attribute/subscript/call outside the whitelist.
3. Every custom potential is `COUNTERFACTUAL`; every level energy is `NUMERICAL` with a grid-halving **and** box-doubling error estimate.
4. The **trust gate** marks each returned level `trusted` or not: a level is trusted only if it is box-converged (not continuum-contaminated) and grid-converged (not falling to center). Untrusted levels are shown and labeled, never silently dropped.
5. Parse errors return HTTP 422 with the offending construct named; the frontend shows the message inline.
6. The custom expression is addressable by deep link (`?view=forcelaw&preset=custom&expr=...`), round-trip tested.
7. Validation suite green: parser accept/reject tests, DoS-guard tests, physics-recovery tests (`-1/r` recovers hydrogen, `0.5*r**2` recovers the oscillator within grid error), trust-gate tests, server route tests, web parse-error + URL round-trip tests.

## 2. Physics grounding (why this is honest)

`V(r)` here is the **bare radial potential** in Hartree atomic units, exactly as the presets are: the solver adds the centrifugal barrier `l(l+1)/(2μr²)` internally (`radial_solver.py`). `μ` (reduced-mass ratio) and `Z` come from the selected system, so H, D, muonic H, He⁺, positronium, generic Z all scale correctly. `Z` is *not* implicit, a custom expression is literal, so hydrogen is `expr = "-1/r"`, and `He⁺` selected as the system solves that same `-1/r` at `μ_He`.

The teaching contract is unchanged from the presets: the counterfactual ladder is shown against the **EXACT** hydrogen ladder for the selected `l` (`E_n = -(Z²μ)/(2n²)`, `n ≥ l+1`), so the contrast is always honest.

### 2.1 Where a finite-difference solver silently lies

Two failure modes must be caught, because the solver returns numbers in both:

- **Continuum contamination.** A potential that does not decay to a well-defined floor (or decays too slowly) has no genuine discrete spectrum in the region the box covers; the lowest "eigenvalues" are really box modes that move when the box moves. Detection: solve on a box of size `r_max` and again on `2·r_max`; a level whose energy shifts by more than a tolerance is **not box-converged** → untrusted.
- **Fall-to-center.** A potential more singular than `-c/r²` near the origin (e.g. `-1/r**3`) has no ground state; on a grid the "ground energy" dives as the grid refines. Detection: the grid-halving error already returned by `solve_radial_with_error` blows up relative to `|E|` → **not grid-converged** → untrusted.

The continuum **threshold** (the energy above which states are unbound) is estimated as `V(r_max)` from the larger box. A level counts as a bound state only if `E < threshold` **and** it passes both convergence checks.

## 3. Decisions locked in

| Question | Decision |
|---|---|
| Parser | **Whitelist AST evaluator** (`ast.parse(mode="eval")` + node walk). Rejected: `eval` with restricted globals (unsafe, well-known escapes), a third-party math-expr lib (new dependency, still needs auditing). |
| Allowed grammar | Numbers, `r`, constants `pi`/`e`; binary `+ - * / **`; unary `+ -`; calls to a fixed function whitelist (`exp log log10 sqrt sin cos tan sinh cosh tanh abs sign minimum maximum where`); comparisons (`< <= > >= == !=`) **only as arguments to `where`** for piecewise potentials. Nothing else. |
| DoS guards | Expression length ≤ 200 chars; AST node count ≤ 80; literal exponent in `**` bounded to `|k| ≤ 12`. Over-limit → 422. |
| Vectorization | Compiled closure evaluates over a NumPy `r`-array; whitelisted names map to NumPy ufuncs. `where(cond, a, b)` gives safe piecewise. |
| Trust gate | Box-doubling (continuum) + grid-halving-relative (fall-to-center). Untrusted levels returned and labeled, not dropped. |
| `r_max` selection | Deterministic: default box `40·(n_states+1)²/max(Z,1)` bohr (mirrors the Coulomb-family heuristic, generous), used for both the base and the ×2 box; no adaptive expansion loop (keeps it cheap and reproducible). |
| Reference overlay | EXACT hydrogen ladder for the selected `l`, same as the power-law preset. |
| API surface | Reuse `/api/forcelaw` with `preset=custom` + `expr` query param. No new endpoint. |
| Client validation | Lightweight only (non-empty, length cap, balanced parens) for instant feedback; **the server parser is the sole authority** on the whitelist. No duplicated whitelist on the client. |
| Provenance | Potential choice → `COUNTERFACTUAL`; sampled `V(r)` curve → `EXACT` (definitional given the expression, as presets); levels → `NUMERICAL` with both error estimates; untrusted levels annotated in `method`. |

## 4. Engine

### 4.1 `numerics/expression.py` (new): the safe parser

- `compile_potential(expr: str) -> PotentialFn`, parse, walk, and return a NumPy-vectorized closure `f(r: np.ndarray) -> np.ndarray`.
- `ExpressionError(ValueError)`, raised with a precise, user-facing message (which construct, and that it is not allowed). Every rejection path names the disallowed node type.
- Walk allows only: `Expression`, `BinOp{Add,Sub,Mult,Div,Pow}`, `UnaryOp{UAdd,USub}`, `Constant`(numeric), `Name`(`r`,`pi`,`e` only), `Call`(whitelisted func names, positional args only), `Compare`(only inside a `where` call). Reject `keywords`, `starargs`, attributes, subscripts, comprehensions, lambdas, assignments, names outside the whitelist, and any function name outside the whitelist.
- DoS guards enforced during the walk (node count, `**` exponent bound) and before it (length cap).
- Evaluation is guarded: if the closure produces non-finite values across the sampled interior grid, that surfaces as an `ExpressionError` ("V(r) is not finite at r = …") at solve time (§4.2), not at compile time (a valid expression can still be singular).

### 4.2 `numerics/force_law.py`: custom driver + trust gate

- `free_form_levels(expr, l, system="h", n_states=4) -> ForceLawResult`, compiles `expr`, chooses `r_max` (§3), runs `solve_radial_with_error` on `r_max` and `2·r_max`, applies the trust gate, and returns a `ForceLawResult` shaped like the presets.
- **Struct changes (additive, presets stay byte-identical):**
  - `ForceLawLevel` gains `trusted: bool = True`. Presets never set it → always `True`, existing behavior unchanged.
  - `ForceLawResult` gains `expression: str | None = None`. `preset_key="custom"`, `params={}` for custom.
- Threshold = `V(r_max_big)`. A level is `trusted` iff `|E_big - E_small| ≤ box_tol` **and** its grid-halving error `≤ grid_tol · max(|E|, floor)`. Bound = `E < threshold`. Return trusted bound levels first, then any untrusted-but-plausible levels flagged; `bound_count` counts trusted bound levels only, `requested_count` unchanged.
- The `V(r)` curve is sampled via the existing `_sample_curve` (EXACT), with the expression string in its `method`.
- Non-finite interior values from the closure → raise `ValueError` (→ 422 at the server), message names the first bad `r`.

## 5. Server: `server/app.py` + `schemas.py`

- `/api/forcelaw` gains `expr: str | None = None`. When `preset == "custom"`: require `expr`, call `free_form_levels`; on `ExpressionError`/`ValueError` → HTTP 422 with the message. `expr` is inert for the built-in presets.
- `ForceLawLevelModel` gains `trusted: bool`. `ForceLawModel` gains `expression: str | None`. Provenance already rides through `QuantityModel`, so `COUNTERFACTUAL`/`NUMERICAL` survive unchanged.
- Validation: `l ≥ 0`, `1 ≤ n_states ≤ 8` (existing); for custom, empty/blank `expr` → 422.

## 6. Web: `web/src`

- **`lib/forceLaw.ts`**: add `"custom"` to `ForcePreset`; `PRESET_LABELS.custom = "Custom  V(r) = …"`; custom has no numeric `ParamSpec`s. Add `DEFAULT_EXPR = "-1/r"` and a light `validateExprClient(expr): string | null` (empty / length / balanced-parens only).
- **`state/store.ts`**: `forceExpr: string` (default `DEFAULT_EXPR`) + setter, in the force-law slice (independent of main physics, no `INVALIDATED`). `loadForceLaw()` sends `expr` when preset is custom.
- **`api/client.ts` + `api/types.ts`**: `getForceLaw` gains optional `expr`; types mirror `trusted` and `expression`.
- **`components/ForceLawView.tsx`**: custom mode renders the expr text input with inline error (client pre-check + server 422 message), reuses the existing `V(r)` curve and ladder renderers, always shows the `COUNTERFACTUAL` banner, and marks untrusted levels with a visible "not box/grid-converged, not a real bound state" tag.
- **`lib/urlState.ts`**: `expr` written only when `preset === "custom"`; validated on read (length cap, non-empty) and clamped; junk drops. Round-trip tested.

## 7. Honesty / provenance (the heart)

- The potential is a **made-up universe** → `COUNTERFACTUAL`, banner always on for custom.
- The sampled `V(r)` curve is `EXACT`, it is definitionally the expression the user typed, evaluated exactly.
- The levels are `NUMERICAL`, each carrying **two** error estimates (grid-halving from the solver, box-doubling from the trust gate). Untrusted levels are surfaced with the reason, honoring the prime directive: the model never quietly presents a box artifact or a fall-to-center divergence as a bound state.
- No client-side physics: the client never evaluates `V(r)` for the ladder; it only draws server-provided data. (A client-side curve preview, if added, would be `VISUAL_LIBERTY` and labeled, deferred, YAGNI.)

## 8. Out of scope (YAGNI)

- Free variables other than `r` (no user-named parameters/sliders for custom exprs, a slider needs a named param, deferred).
- Complex potentials, time dependence, non-central potentials.
- Adaptive `r_max` search beyond the single ×2 box-doubling check.
- Client-side live curve preview before the server responds.
- Saving/naming custom potentials.

## 9. Testing (validation, not smoke)

- **Parser (`tests/test_expression.py`)**: accepts `-1/r`, `0.5*r**2`, `-exp(-r)/r`, `where(r<3, -2, 0)`, `pi*sin(r)`; rejects `__import__("os")`, `r.__class__`, `a`, `foo(r)`, `r[0]`, `lambda r: r`, `[r for r in x]`, `r := 1`, `**` with exponent 99, a 300-char string, an 81-node tree; each rejection message names the construct.
- **Physics recovery (`tests/test_free_form.py`)**: `-1/r` recovers hydrogen `n=1,2,3` at `l=0` within grid error (cross-check `analytic/hydrogen.energy`); the same `-1/r` custom run must equal the `powerlaw` preset at `p=1` for identical `l`/system (same potential, same solver path, a byte-for-byte-of-physics equivalence check); `0.5*r**2` recovers `oscillator_energy` within grid error. Trust gate: `-1/r**3` flags not-grid-converged; a shallow constant-tail potential flags not-box-converged; `exp(-r)` (purely positive) yields zero trusted bound states.
- **Server (`tests/test_server.py`)**: `preset=custom&expr=-1/r` → 200 with `expression` echoed and `trusted` present; `expr` missing → 422; `expr=r.__class__` → 422 with a whitelist message; untrusted flags survive to the response.
- **Web (`*.test.ts`)**: `validateExprClient` accepts/rejects; URL round-trip for `preset=custom&expr=...`; parse-error display path.
