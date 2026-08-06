# Free-Form V(r) Implementation Plan

**Goal:** Let a What-If Lab user type an arbitrary radial potential `V(r)`, parse it safely, solve it numerically, and show the honest bound-state ladder with a trust gate.

**Architecture:** A whitelist AST evaluator (`numerics/expression.py`) compiles a user string into a NumPy-vectorized closure. A `free_form_levels` driver in `numerics/force_law.py` runs the existing `solve_radial_with_error` on two box sizes, applies a trust gate (box-doubling + grid-halving) and returns a `ForceLawResult`. The existing `/api/forcelaw` route and Force-Law view gain a `custom` mode.

**Tech Stack:** Python 3.12 (stdlib `ast`, NumPy, FastAPI/Pydantic), React/TypeScript/Zustand, Vitest.

## Global Constraints

- Every physical value crossing a module boundary is a `Quantity`/`Field` with `Provenance` stating a `Fidelity` tier. Custom potential → `COUNTERFACTUAL`; sampled `V(r)` curve → `EXACT`; levels → `NUMERICAL` with `error_estimate`.
- Engine math is Hartree atomic units; eV/pm conversions only at the server boundary.
- No `eval`/`exec`. The parser is a whitelist AST walk. No third-party expression library.
- `ruff check .` clean (line-length 100; E741 ignored). `l` is the QM quantum number.
- New physics gets a validation test (analytic ground truth / recovery / convergence), not a smoke test.
- Rebuild `web/dist` (`npm run build` in `web/`) after any `web/src` change.
---

### Task 1: Safe expression parser (`numerics/expression.py`)

**Files:**
- Create: `src/atomsim/numerics/expression.py`
- Test: `tests/test_expression.py`

**Interfaces:**
- Produces: `compile_potential(expr: str) -> Callable[[np.ndarray], np.ndarray]`; `ExpressionError(ValueError)`. Allowed names: `r`, `pi`, `e`. Allowed funcs: `exp log log10 sqrt sin cos tan sinh cosh tanh abs sign minimum maximum where`. DoS caps: length ≤ 200, node count ≤ 80, literal `**` exponent `|k| ≤ 12`.

- [x] **Step 1: Write failing tests**

```python
# tests/test_expression.py
import numpy as np
import pytest

from atomsim.numerics.expression import ExpressionError, compile_potential

def test_coulomb_expression_evaluates():
    f = compile_potential("-1/r")
    r = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(f(r), [-1.0, -0.5, -0.25])

def test_oscillator_and_constants():
    f = compile_potential("0.5*r**2")
    np.testing.assert_allclose(f(np.array([2.0])), [2.0])
    g = compile_potential("pi")
    np.testing.assert_allclose(g(np.array([1.0, 9.0])), [np.pi, np.pi])

def test_whitelisted_functions_and_piecewise():
    f = compile_potential("-exp(-r)/r")
    np.testing.assert_allclose(f(np.array([1.0])), [-np.exp(-1.0)])
    well = compile_potential("where(r < 3, -2, 0)")
    np.testing.assert_allclose(well(np.array([1.0, 5.0])), [-2.0, 0.0])

@pytest.mark.parametrize(
    "expr",
    [
        '__import__("os")',
        "r.__class__",
        "foo(r)",
        "a + 1",
        "r[0]",
        "lambda r: r",
        "[x for x in r]",
        "r and 1",
    ],
)
def test_rejects_non_whitelisted(expr):
    with pytest.raises(ExpressionError):
        compile_potential(expr)

def test_dos_guards():
    with pytest.raises(ExpressionError):
        compile_potential("r**99")
    with pytest.raises(ExpressionError):
        compile_potential("r" * 201)
    with pytest.raises(ExpressionError):
        compile_potential("+".join(["r"] * 60))  # > 80 nodes

def test_error_message_names_construct():
    with pytest.raises(ExpressionError, match="Attribute"):
        compile_potential("r.real")
```

- [x] **Step 2: Run to verify fail**, `pytest tests/test_expression.py -q` → ImportError / fail.

- [x] **Step 3: Implement**

```python
# src/atomsim/numerics/expression.py
"""Safe whitelist-AST compiler: a user V(r) string -> NumPy-vectorized closure.

No eval/exec. Only a small math whitelist is allowed; every other AST construct
is rejected with a message naming it. See docs/specs/
2026-07-23-phase8-freeform-potential-design.md.
"""

import ast
import math
from collections.abc import Callable

import numpy as np

MAX_LEN = 200
MAX_NODES = 80
MAX_POW = 12

PotentialFn = Callable[[np.ndarray], np.ndarray]

class ExpressionError(ValueError):
    """User expression is empty, too large, or uses a non-whitelisted construct."""

_CONSTANTS = {"pi": math.pi, "e": math.e}

_FUNCS: dict[str, Callable] = {
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "abs": np.abs, "sign": np.sign,
    "minimum": np.minimum, "maximum": np.maximum, "where": np.where,
}

_BINOPS = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
    ast.Div: np.divide, ast.Pow: np.power,
}
_CMPOPS = {
    ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
    ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal,
}

def compile_potential(expr: str) -> PotentialFn:
    if not expr or not expr.strip():
        raise ExpressionError("expression is empty")
    if len(expr) > MAX_LEN:
        raise ExpressionError(f"expression too long ({len(expr)} > {MAX_LEN} chars)")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error: {exc.msg}") from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise ExpressionError(f"expression too complex ({len(nodes)} > {MAX_NODES} nodes)")

    _check(tree.body)  # raises ExpressionError on any disallowed construct

    def potential(r: np.ndarray) -> np.ndarray:
        out = _eval(tree.body, r)
        return np.asarray(out, dtype=float) * np.ones_like(r, dtype=float)

    return potential

def _check(node: ast.AST, *, in_where: bool = False) -> None:
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise ExpressionError(f"operator {type(node.op).__name__} is not allowed")
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
            if abs(node.right.value) > MAX_POW:
                raise ExpressionError(f"exponent magnitude > {MAX_POW} is not allowed")
        _check(node.left)
        _check(node.right)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ExpressionError(f"unary {type(node.op).__name__} is not allowed")
        _check(node.operand)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise ExpressionError("only numeric constants are allowed")
    elif isinstance(node, ast.Name):
        if node.id != "r" and node.id not in _CONSTANTS:
            raise ExpressionError(f"unknown name {node.id!r}; only r, pi, e are allowed")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            name = getattr(node.func, "id", type(node.func).__name__)
            raise ExpressionError(f"function {name!r} is not allowed")
        if node.keywords:
            raise ExpressionError("keyword arguments are not allowed")
        for arg in node.args:
            _check(arg, in_where=(node.func.id == "where"))
    elif isinstance(node, ast.Compare):
        if not in_where:
            raise ExpressionError("comparisons are only allowed as arguments to where()")
        if len(node.ops) != 1 or type(node.ops[0]) not in _CMPOPS:
            raise ExpressionError("only a single simple comparison is allowed")
        _check(node.left)
        _check(node.comparators[0])
    else:
        raise ExpressionError(f"{type(node).__name__} is not allowed in a potential")

def _eval(node: ast.AST, r: np.ndarray) -> np.ndarray:
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_eval(node.left, r), _eval(node.right, r))
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, r)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return r if node.id == "r" else _CONSTANTS[node.id]
    if isinstance(node, ast.Call):
        return _FUNCS[node.func.id](*[_eval(a, r) for a in node.args])
    if isinstance(node, ast.Compare):
        return _CMPOPS[type(node.ops[0])](_eval(node.left, r), _eval(node.comparators[0], r))
    raise ExpressionError(f"{type(node).__name__} is not allowed")  # defense in depth
```

- [x] **Step 4: Run**, `pytest tests/test_expression.py -q` → PASS. Then `ruff check src/atomsim/numerics/expression.py tests/test_expression.py`.

- [x] **Step 5: Commit**, `git add -A && git commit -m "Add whitelist-AST safe compiler for custom V(r) expressions"`

---

### Task 2: Free-form driver + trust gate (`numerics/force_law.py`)

**Files:**
- Modify: `src/atomsim/numerics/force_law.py`
- Test: `tests/test_free_form.py`

**Interfaces:**
- Consumes: `compile_potential` (Task 1); existing `solve_radial_with_error`, `_sample_curve`, `_hydrogen_reference`, `ForceLawResult`, `ForceLawLevel`.
- Produces: `free_form_levels(expr: str, l: int, system: str | System = "h", n_states: int = 4) -> ForceLawResult`. `ForceLawLevel` gains `trusted: bool = True`; `ForceLawResult` gains `expression: str | None = None`.

- [x] **Step 1: Write failing tests**

```python
# tests/test_free_form.py
import numpy as np
import pytest

from atomsim.numerics.force_law import force_law_levels, free_form_levels
from atomsim.provenance import Fidelity

def test_coulomb_recovers_hydrogen():
    res = free_form_levels("-1/r", l=0, system="h", n_states=3)
    assert res.expression == "-1/r"
    assert res.preset_key == "custom"
    energies = [lvl.energy.value for lvl in res.counterfactual]
    # hydrogen 1s,2s,3s = -0.5, -0.125, -0.0556 hartree
    np.testing.assert_allclose(energies[:3], [-0.5, -0.125, -1 / 18], rtol=2e-3)
    assert all(lvl.trusted for lvl in res.counterfactual[:3])
    assert res.counterfactual[0].energy.provenance.fidelity is Fidelity.NUMERICAL

def test_custom_matches_powerlaw_preset():
    a = free_form_levels("-1/r", l=0, system="h", n_states=3)
    b = force_law_levels("powerlaw", {"p": 1.0}, l=0, system="h", n_states=3)
    ea = [c.energy.value for c in a.counterfactual]
    eb = [c.energy.value for c in b.counterfactual]
    np.testing.assert_allclose(ea, eb, rtol=1e-9)

def test_oscillator_recovers_levels():
    # 0.5*r^2 => omega=1, l=0 energies 1.5, 3.5, 5.5 hartree
    res = free_form_levels("0.5*r**2", l=0, system="h", n_states=3)
    e = sorted(lvl.energy.value for lvl in res.counterfactual)
    np.testing.assert_allclose(e[:3], [1.5, 3.5, 5.5], rtol=5e-3)

def test_fall_to_center_flagged_untrusted():
    res = free_form_levels("-1/r**3", l=0, system="h", n_states=2)
    assert any(not lvl.trusted for lvl in res.counterfactual) or res.bound_count == 0

def test_purely_positive_has_no_trusted_bound_states():
    res = free_form_levels("exp(-r)", l=0, system="h", n_states=2)
    assert res.bound_count == 0
```

- [x] **Step 2: Run to verify fail**, `pytest tests/test_free_form.py -q` → ImportError.

- [x] **Step 3: Implement**, add to `src/atomsim/numerics/force_law.py`:

Add field to the dataclasses (keep presets identical via defaults):
```python
@dataclass(frozen=True)
class ForceLawLevel:
    radial_index: int
    energy: Quantity  # NUMERICAL, hartree, carries grid-halving error
    trusted: bool = True
```
```python
@dataclass(frozen=True)
class ForceLawResult:
    preset_key: str
    params: Params
    l: int
    z: int
    system_key: str
    counterfactual: tuple[ForceLawLevel, ...]
    bound_count: int
    requested_count: int
    reference: Reference
    potential_curve: Field  # hartree vs bohr, EXACT
    expression: str | None = None
```

Add imports at top: `from atomsim.numerics.expression import compile_potential`.

Add driver + trust gate at the end of the module:
```python
FREE_FORM_BOX_TOL = 5e-3   # hartree; box-doubling shift above this = not converged
FREE_FORM_GRID_FRAC = 1e-2  # grid error above this fraction of |E| = not converged

def _free_form_rmax(z: int, n_states: int) -> float:
    return 40.0 * (n_states + 1) ** 2 / max(z, 1)

def free_form_levels(
    expr: str,
    l: int,
    system: str | System = "h",
    n_states: int = 4,
) -> ForceLawResult:
    if l < 0:
        raise ValueError(f"orbital quantum number l must be >= 0, got {l}")
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")

    potential = compile_potential(expr)  # raises ExpressionError on bad input

    sys = system if isinstance(system, System) else get_system(system)
    z = sys.Z
    mu = sys.mu_ratio.value
    r_max = _free_form_rmax(z, n_states)

    # sanity: the expression must be finite on the interior grid
    r_probe = np.linspace(r_max / CURVE_POINTS, r_max, CURVE_POINTS)
    v_probe = np.asarray(potential(r_probe), dtype=float)
    if not np.all(np.isfinite(v_probe)):
        bad = r_probe[~np.isfinite(v_probe)][0]
        raise ValueError(f"V(r) is not finite at r = {bad:g} bohr")

    small = solve_radial_with_error(potential, l=l, mu_ratio=mu, n_states=n_states, r_max=r_max)
    big = solve_radial_with_error(
        potential, l=l, mu_ratio=mu, n_states=n_states, r_max=2.0 * r_max
    )
    threshold = float(potential(np.array([2.0 * r_max]))[0])

    note = f"; counterfactual free-form V(r) = {expr}"
    levels: list[ForceLawLevel] = []
    for k in range(n_states):
        e = small.energies[k]
        if e.value >= threshold:  # not a bound state of this box
            continue
        box_shift = abs(e.value - big.energies[k].value)
        grid_err = e.provenance.error_estimate or 0.0
        trusted = (
            box_shift <= FREE_FORM_BOX_TOL
            and grid_err <= FREE_FORM_GRID_FRAC * max(abs(e.value), 1e-6)
        )
        reason = "" if trusted else "; UNTRUSTED (not box/grid-converged, not a real bound state)"
        levels.append(
            ForceLawLevel(
                radial_index=len(levels),
                energy=_tag(e, note + reason),
                trusted=trusted,
            )
        )

    bound_count = sum(1 for lvl in levels if lvl.trusted)
    reference = _hydrogen_reference({}, z, mu, l, n_states)
    curve = _sample_curve(potential, r_max, note)

    return ForceLawResult(
        preset_key="custom",
        params={},
        l=l,
        z=z,
        system_key=sys.key,
        counterfactual=tuple(levels),
        bound_count=bound_count,
        requested_count=n_states,
        reference=reference,
        potential_curve=curve,
        expression=expr,
    )
```

- [x] **Step 4: Run**, `pytest tests/test_free_form.py -q` → PASS. Confirm presets still green: `pytest tests/ -k force_law -q`. Then `ruff check src/atomsim/numerics/force_law.py tests/test_free_form.py`.

- [x] **Step 5: Commit**, `git add -A && git commit -m "Solve custom V(r) with a box-doubling and grid-halving trust gate"`

---

### Task 3: Server route + schema

**Files:**
- Modify: `src/atomsim/server/app.py:460-520` (`forcelaw_endpoint`), `src/atomsim/server/schemas.py` (`ForceLawLevelModel`, `ForceLawModel`)
- Test: `tests/test_server.py` (append)

**Interfaces:**
- Consumes: `free_form_levels` (Task 2).
- Produces: `/api/forcelaw?preset=custom&expr=...` → `ForceLawModel` with `expression: str | None` and per-level `trusted: bool`. Parse errors → HTTP 422.

- [x] **Step 1: Write failing tests** (append to `tests/test_server.py`, reuse its existing `client` fixture pattern):

```python
def test_forcelaw_custom_recovers_hydrogen(client):
    r = client.get("/api/forcelaw", params={"preset": "custom", "expr": "-1/r", "l": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["expression"] == "-1/r"
    assert body["preset"] == "custom"
    assert body["counterfactual"][0]["trusted"] is True
    assert body["counterfactual"][0]["energy"]["provenance"]["fidelity"] == "numerical"

def test_forcelaw_custom_missing_expr_is_422(client):
    r = client.get("/api/forcelaw", params={"preset": "custom", "l": 0})
    assert r.status_code == 422

def test_forcelaw_custom_rejects_unsafe_expr(client):
    r = client.get("/api/forcelaw", params={"preset": "custom", "expr": "r.__class__"})
    assert r.status_code == 422
    assert "not allowed" in r.json()["detail"]
```

- [x] **Step 2: Run to verify fail**, `pytest tests/test_server.py -k custom -q` → fail (422 vs 200 / missing field).

- [x] **Step 3: Implement**

In `schemas.py`, add `trusted: bool = True` to `ForceLawLevelModel` and `expression: str | None = None` to `ForceLawModel`.

In `app.py`:
- Import: `from atomsim.numerics.force_law import PRESETS, force_law_levels, free_form_levels` and `from atomsim.numerics.expression import ExpressionError`.
- Add `expr: str | None = None` to `forcelaw_endpoint` signature.
- Branch before the `preset not in PRESETS` check:
```python
        if preset == "custom":
            if not expr or not expr.strip():
                raise HTTPException(status_code=422, detail="custom preset requires 'expr'")
            if l < 0:
                raise HTTPException(status_code=422, detail=f"l must be >= 0, got {l}")
            if not 1 <= n_states <= 8:
                raise HTTPException(status_code=422, detail=f"n_states must be in [1, 8], got {n_states}")
            sys_ = _resolve_system(system)
            try:
                result = free_form_levels(expr, l=l, system=sys_, n_states=n_states)
            except (ExpressionError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return _forcelaw_response(result, sys_)
```
- Refactor the existing response construction (the `return ForceLawModel(...)` block) into a helper `_forcelaw_response(result, sys_) -> ForceLawModel` so both paths share it; set `expression=result.expression` and `trusted=c.trusted` inside it.

- [x] **Step 4: Run**, `pytest tests/test_server.py -q` → PASS (all, including existing preset tests through the refactored helper). `ruff check src/atomsim/server/`.

- [x] **Step 5: Commit**, `git add -A && git commit -m "Serve custom V(r) force laws with trust flags from /api/forcelaw"`

---

### Task 4: Frontend engine mirror, store, client, URL

**Files:**
- Modify: `web/src/lib/forceLaw.ts`, `web/src/state/store.ts`, `web/src/api/client.ts`, `web/src/api/types.ts`, `web/src/lib/urlState.ts`
- Test: `web/src/lib/forceLaw.test.ts` (append), `web/src/lib/urlState.test.ts` (append if present, else add cases to the existing round-trip test file)

**Interfaces:**
- Produces: `ForcePreset` union includes `"custom"`; `DEFAULT_EXPR = "-1/r"`; `validateExprClient(expr: string): string | null`; store `forceExpr`/`setForceExpr`; `getForceLaw` accepts `expr`; URL key `expr`.

- [x] **Step 1: Write failing tests**

```typescript
// web/src/lib/forceLaw.test.ts (append)
import { DEFAULT_EXPR, validateExprClient } from "./forceLaw";

test("validateExprClient accepts a simple expression", () => {
  expect(validateExprClient("-1/r")).toBeNull();
  expect(DEFAULT_EXPR).toBe("-1/r");
});

test("validateExprClient rejects empty, too-long, unbalanced parens", () => {
  expect(validateExprClient("")).not.toBeNull();
  expect(validateExprClient("r".repeat(201))).not.toBeNull();
  expect(validateExprClient("exp(-r")).not.toBeNull();
});
```
```typescript
// URL round-trip (add to the existing urlState test file)
test("custom preset round-trips the expression", () => {
  const s = decodeState(new URLSearchParams("view=forcelaw&preset=custom&expr=-1%2Fr"));
  expect(s.forcePreset).toBe("custom");
  expect(s.forceExpr).toBe("-1/r");
});
```
(Match the actual import names of the existing urlState test, `decodeState`/`parseState`; mirror whatever the file already uses.)

- [x] **Step 2: Run to verify fail**, `cd web && npx vitest run src/lib/forceLaw.test.ts` → fail.

- [x] **Step 3: Implement**

`forceLaw.ts`: add `"custom"` to the `ForcePreset` union; `PRESET_PARAMS.custom = []`; `PRESET_LABELS.custom = "Custom  V(r) = …"`; then:
```typescript
export const DEFAULT_EXPR = "-1/r";

/** Lightweight client pre-check only; the server parser is the authority. */
export function validateExprClient(expr: string): string | null {
  if (!expr.trim()) return "Enter an expression in r";
  if (expr.length > 200) return "Expression is too long (max 200 characters)";
  let depth = 0;
  for (const ch of expr) {
    if (ch === "(") depth++;
    else if (ch === ")" && --depth < 0) return "Unbalanced parentheses";
  }
  if (depth !== 0) return "Unbalanced parentheses";
  return null;
}
```

`store.ts`: in the force-law slice add `forceExpr: string` (init `DEFAULT_EXPR`) and `setForceExpr(expr: string)`; in `loadForceLaw()` pass `expr: get().forcePreset === "custom" ? get().forceExpr : undefined` to `getForceLaw`.

`api/types.ts`: add `trusted: boolean` to the force-law level type and `expression: string | null` to the force-law response type.

`api/client.ts`: `getForceLaw` gains optional `expr?: string`; when present, add `&expr=<encodeURIComponent>` to the query.

`urlState.ts`: after the preset block, read/write `expr` only when preset is custom:
```typescript
  // read:
  if (out.forcePreset === "custom") {
    const rawExpr = q.get("expr");
    if (rawExpr && validateExprClient(rawExpr) === null) out.forceExpr = rawExpr;
  }
  // write (in encodeState):
  if (state.forcePreset === "custom" && state.forceExpr !== DEFAULT_EXPR) {
    q.set("expr", state.forceExpr);
  }
```
Import `DEFAULT_EXPR, validateExprClient` in `urlState.ts`. Add `forceExpr: DEFAULT_EXPR` to `URL_DEFAULTS`.

- [x] **Step 4: Run**, `cd web && npx vitest run src/lib/forceLaw.test.ts src/lib/urlState.test.ts` → PASS.

- [x] **Step 5: Commit**, `git add -A && git commit -m "Wire custom V(r) through the force-law store, client, and deep links"`

---

### Task 5: Force-Law view custom mode + build

**Files:**
- Modify: `web/src/components/ForceLawView.tsx`
- Verify: full build + test

**Interfaces:**
- Consumes: store `forcePreset`/`forceExpr`/`setForceExpr`, `validateExprClient`, force-law response with `expression`/`trusted`.

- [x] **Step 1: Implement the UI**

In `ForceLawView.tsx`:
- Add `custom` to the preset selector options (label from `PRESET_LABELS`).
- When `forcePreset === "custom"`: render a text input bound to `forceExpr` (via `setForceExpr`) instead of the numeric parameter sliders. Show `validateExprClient(forceExpr)` inline in red if non-null; also show the server error (the store's force-law error state) when the request 422s.
- Always render the `COUNTERFACTUAL` badge/banner for custom (the response provenance already says so, reuse the existing `Badge`).
- In the level ladder, when a level's `trusted === false`, render it dimmed with a "not converged, not a real bound state" tag. Keep the `V(r)` curve and hydrogen-reference rendering exactly as-is (they are preset-agnostic).
- Keep a short helper caption listing allowed names/functions: `r, pi, e; + - * / **; exp log sqrt sin cos tan sinh cosh tanh abs sign where(cond, a, b)`.

- [x] **Step 2: Typecheck + build**, `cd web && npm run build` (tsc --noEmit + vite build → web/dist). Expected: clean.

- [x] **Step 3: Full suites**, from repo root `pytest -q` and `ruff check .`; from `web/` `npm test`. Expected: all green.

- [x] **Step 4: Live smoke**, `atomsim serve --port 8021 --no-browser` (background), then `curl "http://127.0.0.1:8021/api/forcelaw?preset=custom&expr=-1/r&l=0"` returns hydrogen levels with `trusted:true`; `curl "...expr=r.__class__"` returns 422. Stop the server.

- [x] **Step 5: Commit**, `git add -A && git commit -m "Render the custom V(r) editor and trust flags in the Force-Law view"`

---

## Self-review notes

- **Spec coverage:** §4.1 parser → Task 1; §4.2 driver+trust gate → Task 2; §5 server/schema → Task 3; §6 web lib/store/URL → Task 4, view → Task 5; §7 provenance enforced in Tasks 2-3; §9 tests distributed across all tasks. Deferred items in §8 are intentionally not tasked.
- **Type consistency:** `free_form_levels`, `compile_potential`, `ExpressionError`, `ForceLawLevel.trusted`, `ForceLawResult.expression`, `DEFAULT_EXPR`, `validateExprClient` used identically across tasks.
- **Trust tolerances** (`FREE_FORM_BOX_TOL`, `FREE_FORM_GRID_FRAC`) may need tuning against the recovery tests in Task 2 Step 4; if hydrogen levels come back flagged untrusted, loosen `FREE_FORM_BOX_TOL` first (the deep 1s is box-insensitive; higher n are more box-sensitive at this r_max).
