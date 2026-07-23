# Exact Dirac Hydrogen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the closed-form exact Dirac-Coulomb energy for hydrogen-like atoms as an EXACT level model, cross-checked against the perturbative α² approximation, surfaced as a Dirac toggle in the Levels view.

**Architecture:** A new pure-analytic `analytic/dirac.py` computes `E(n,j)`; `/api/levels` gains a `dirac` mode that fills the fine-level array from it; the Levels view offers "α² perturbative / Dirac exact" when fine structure is on.

**Tech Stack:** Python 3.12 (math, scipy CODATA α), FastAPI/Pydantic, React/TypeScript/Zustand, Vitest.

## Global Constraints

- Every physical value crossing a boundary is a `Quantity`/`Field` with `Provenance` + `Fidelity`. Dirac energy at real α → `EXACT`; at altered α → `COUNTERFACTUAL`. Assumptions must name the omitted physics (Lamb/QED, hyperfine, finite nuclear size, recoil beyond reduced mass).
- Engine math in Hartree atomic units; eV conversion only at the server boundary.
- `ruff check .` clean (line-length 100; E741 ignored).
- New physics gets a validation test (analytic ground truth / convergence), not a smoke test.
- Rebuild `web/dist` (`npm run build` in `web/`) after any `web/src` change.
- Commit messages carry no AI attribution.
- Physics: `γ = sqrt((j+½)² − (Zα)²)`; `D = n − (j+½) + γ`; `E_bind = (μ/α²)([1 + (Zα/D)²]^(−½) − 1)` hartree. Requires `Zα < j+½` (else supercritical → reject).

---

### Task 1: Engine — `analytic/dirac.py`

**Files:**
- Create: `src/atomsim/analytic/dirac.py`
- Test: `tests/test_dirac.py`

**Interfaces:**
- Consumes: `energy` (Bohr), `ALPHA`, `Quantity`/`Provenance`/`Fidelity`, `fine_structure_shift` (test only).
- Produces: `dirac_energy(n: int, j: float, Z: int = 1, mu_ratio: float = 1.0, alpha: float = ALPHA) -> Quantity`; `dirac_fine_splitting(n: int, l: int, Z: int = 1, mu_ratio: float = 1.0, alpha: float = ALPHA) -> float`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dirac.py
import math

import pytest

from atomsim.analytic.dirac import dirac_energy, dirac_fine_splitting
from atomsim.analytic.fine_structure import fine_structure_shift
from atomsim.analytic.hydrogen import energy
from atomsim.constants import ALPHA
from atomsim.provenance import Fidelity


def test_nonrelativistic_limit_recovers_bohr():
    # alpha -> 0 collapses Dirac onto the Bohr ladder
    for n, j in [(1, 0.5), (2, 0.5), (3, 1.5)]:
        e = dirac_energy(n, j, Z=1, alpha=1e-5).value
        assert e == pytest.approx(energy(n).value, rel=1e-6)


def test_ground_state_matches_published_value():
    # E(1s1/2) = -1/2 - alpha^2/8 exactly for Z=1, mu=1
    expected = -0.5 - ALPHA**2 / 8.0
    assert dirac_energy(1, 0.5, Z=1).value == pytest.approx(expected, abs=1e-12)


def test_agrees_with_perturbative_to_order_alpha4():
    # residual = |E_dirac - (E_bohr + dE_fs)| should scale as alpha^4 (16x under halving)
    n, l, j = 2, 1, 1.5

    def residual(a):
        d = dirac_energy(n, j, Z=1, alpha=a).value
        p = energy(n).value + fine_structure_shift(n, l, j, Z=1, alpha=a).value
        return abs(d - p)

    r1 = residual(ALPHA)
    r2 = residual(ALPHA / 2)
    assert r1 < 1e-8
    assert r2 == pytest.approx(r1 / 16.0, rel=0.15)  # alpha^4 scaling


def test_exact_nj_degeneracy_is_l_independent():
    # 2s1/2 (l=0) and 2p1/2 (l=1) share one Dirac energy -> depends on (n,j) only
    e_from_s = dirac_energy(2, 0.5, Z=1).value
    e_from_p = dirac_energy(2, 0.5, Z=1).value
    assert e_from_s == e_from_p


def test_fine_splitting_matches_perturbative():
    # 2p3/2 - 2p1/2 interval, Dirac vs Pauli, agree to O(alpha^4)
    dirac_gap = dirac_fine_splitting(2, 1, Z=1)
    pert = (
        fine_structure_shift(2, 1, 1.5, Z=1).value
        - fine_structure_shift(2, 1, 0.5, Z=1).value
    )
    assert dirac_gap == pytest.approx(pert, rel=1e-3)
    assert dirac_gap > 0


def test_fidelity_exact_at_real_alpha_counterfactual_when_altered():
    assert dirac_energy(1, 0.5).provenance.fidelity is Fidelity.EXACT
    assert dirac_energy(1, 0.5, alpha=0.2).provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_supercritical_is_rejected():
    with pytest.raises(ValueError):
        dirac_energy(1, 0.5, Z=200)  # Z*alpha > j+1/2


def test_invalid_j_rejected():
    with pytest.raises(ValueError):
        dirac_energy(2, 2.5)  # j must be in {1/2, 3/2} for n=2
```

- [ ] **Step 2: Run to verify fail** — `MKL_THREADING_LAYER=SEQUENTIAL <env>/python.exe -m pytest tests/test_dirac.py -q` → ImportError.

- [ ] **Step 3: Implement**

```python
# src/atomsim/analytic/dirac.py
"""Exact Dirac-Coulomb energy for hydrogen-like atoms (the fine-structure refinement).

Closed form of the one-body Dirac equation in a point Coulomb field. EXACT for that
model, which still omits the Lamb shift/QED, hyperfine structure, finite nuclear size,
and two-body recoil beyond reduced-mass scaling. See docs/superpowers/specs/
2026-07-23-phase9-dirac-hydrogen-design.md.
"""

import math

from atomsim.analytic.hydrogen import energy
from atomsim.constants import ALPHA
from atomsim.provenance import Fidelity, Provenance, Quantity

_DIRAC_ASSUMPTIONS = (
    "exact eigenvalue of the one-body Dirac-Coulomb equation (point nucleus)",
    "no Lamb shift / QED radiative corrections (this splits 2s1/2 from 2p1/2 in reality)",
    "no hyperfine structure, no finite-nuclear-size correction",
    "reduced mass by mu-scaling the rest energy; two-body relativistic recoil neglected",
)


def _validate(n: int, j: float, Z: int, alpha: float) -> None:
    if n < 1:
        raise ValueError(f"principal quantum number n must be >= 1, got {n}")
    if j < 0.5 or abs((j - 0.5) - round(j - 0.5)) > 1e-12:
        raise ValueError(f"j must be a half-integer >= 1/2, got {j}")
    if (j + 0.5) > n:
        raise ValueError(f"j={j} is not allowed for n={n} (need j <= n-1/2)")
    if Z < 1:
        raise ValueError(f"nuclear charge Z must be >= 1, got {Z}")
    if Z * alpha >= j + 0.5:
        raise ValueError(
            f"supercritical: Z*alpha = {Z * alpha:g} >= j+1/2 = {j + 0.5:g}; "
            "the point-Coulomb Dirac solution is not real here"
        )


def dirac_energy(
    n: int, j: float, Z: int = 1, mu_ratio: float = 1.0, alpha: float = ALPHA
) -> Quantity:
    """Exact Dirac-Coulomb binding energy E(n, j) in hartree (rest energy subtracted)."""
    _validate(n, j, Z, alpha)
    gamma = math.sqrt((j + 0.5) ** 2 - (Z * alpha) ** 2)
    d = n - (j + 0.5) + gamma
    e_bind = (mu_ratio / alpha**2) * ((1.0 + (Z * alpha / d) ** 2) ** -0.5 - 1.0)

    altered = not math.isclose(alpha, ALPHA, rel_tol=1e-12)
    bohr = energy(n, Z=Z, mu_ratio=mu_ratio).value
    # Omitted-physics scale (Lamb-dominated), an honesty order-of-magnitude, not a bound.
    omitted = abs(bohr) * (Z * alpha) ** 3
    method = "exact Dirac-Coulomb energy E(n,j) = mu*c^2([1+(Za/D)^2]^(-1/2) - 1)"
    if altered:
        method += f"; altered fine-structure constant alpha = {alpha:g} (real {ALPHA:g})"
    return Quantity(
        value=e_bind,
        unit="hartree",
        label=f"E_Dirac {n},j={j:g} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.COUNTERFACTUAL if altered else Fidelity.EXACT,
            method=method,
            assumptions=_DIRAC_ASSUMPTIONS,
            error_estimate=omitted,
            refinement="QED / Lamb shift (2s-2p splitting), then hyperfine structure",
        ),
    )


def dirac_fine_splitting(
    n: int, l: int, Z: int = 1, mu_ratio: float = 1.0, alpha: float = ALPHA
) -> float:
    """E(n, j=l+1/2) - E(n, j=l-1/2) in hartree; requires l >= 1."""
    if l < 1:
        raise ValueError(f"fine splitting needs l >= 1, got {l}")
    hi = dirac_energy(n, l + 0.5, Z=Z, mu_ratio=mu_ratio, alpha=alpha).value
    lo = dirac_energy(n, l - 0.5, Z=Z, mu_ratio=mu_ratio, alpha=alpha).value
    return hi - lo
```

- [ ] **Step 4: Run** — `pytest tests/test_dirac.py -q` → PASS. `ruff check src/atomsim/analytic/dirac.py tests/test_dirac.py`.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "Add exact Dirac-Coulomb hydrogen energy (EXACT, cross-checked vs perturbative)"`

---

### Task 2: Server — `/api/levels` Dirac mode

**Files:**
- Modify: `src/atomsim/server/app.py` (`LevelsResponse` at :107, `levels_endpoint` at :373)
- Test: `tests/test_server.py` (append)

**Interfaces:**
- Consumes: `dirac_energy` (Task 1); existing `energy`, `FineLevelModel`.
- Produces: `/api/levels?...&dirac=true`; `LevelsResponse.dirac: bool`.

- [ ] **Step 1: Write failing tests** (append to `tests/test_server.py`):

```python
def test_levels_dirac_is_exact_and_degenerate(client):
    r = client.get("/api/levels", params={"system": "h", "n_max": 3, "dirac": "true"})
    assert r.status_code == 200
    body = r.json()
    assert body["dirac"] is True
    fine = body["fine"]
    f0 = fine[0]
    assert f0["energy"]["provenance"]["fidelity"] == "exact"
    # 2s1/2 (l=0,j=0.5) and 2p1/2 (l=1,j=0.5) must share one Dirac energy
    n2 = [f for f in fine if f["n"] == 2 and f["j"] == 0.5]
    assert len(n2) == 2
    assert n2[0]["energy"]["value"] == pytest.approx(n2[1]["energy"]["value"], abs=1e-14)


def test_levels_perturbative_still_default(client):
    r = client.get("/api/levels", params={"system": "h", "n_max": 2, "fine_structure": "true"})
    body = r.json()
    assert body["dirac"] is False
    assert body["fine"][0]["energy"]["provenance"]["fidelity"] == "approximation"


def test_levels_dirac_supercritical_is_422(client):
    # a high-Z generic hydrogenic system pushes Z*alpha past j+1/2 for j=1/2
    r = client.get("/api/levels", params={"system": "z200", "n_max": 1, "dirac": "true"})
    assert r.status_code == 422
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_server.py -k "dirac or perturbative_still" -q` → fail.

- [ ] **Step 3: Implement**

- Import at top of app.py: `from atomsim.analytic.dirac import dirac_energy`.
- `LevelsResponse`: add `dirac: bool = False`.
- In `levels_endpoint`, add `dirac: bool = False` to the signature; replace the fine-array block so it builds when `dirac or fine_structure`:

```python
        fine = None
        if dirac or fine_structure:
            fine = []
            for n in range(1, n_max + 1):
                for l in range(n):
                    for j in ([0.5] if l == 0 else [l - 0.5, l + 0.5]):
                        if dirac:
                            try:
                                le = dirac_energy(
                                    n, j, Z=sys_.Z, mu_ratio=mu, alpha=alpha_used
                                )
                            except ValueError as exc:
                                raise HTTPException(status_code=422, detail=str(exc)) from exc
                            e_bohr = energy(n, Z=sys_.Z, mu_ratio=mu)
                            sh = replace(
                                le,
                                value=le.value - e_bohr.value,
                                label=f"dE_Dirac {n},{l},j={j:g}",
                            )
                        else:
                            le = level_energy(
                                n, l, j, Z=sys_.Z, mu_ratio=mu,
                                m_over_M=sys_.m_over_M, alpha=alpha_used,
                            )
                            sh = fine_structure_shift(
                                n, l, j, Z=sys_.Z, mu_ratio=mu,
                                m_over_M=sys_.m_over_M, alpha=alpha_used,
                            )
                        fine.append(FineLevelModel(
                            n=n, l=l, j=j,
                            energy=QuantityModel.from_quantity(le),
                            energy_ev=QuantityModel.from_quantity(_to_ev(le)),
                            shift=QuantityModel.from_quantity(sh),
                            shift_ev=QuantityModel.from_quantity(_to_ev(sh)),
                        ))
        return LevelsResponse(
            system=SystemModel.from_system(sys_), n_max=n_max,
            fine_structure=fine_structure, alpha=alpha_used, gross=gross, fine=fine,
            dirac=dirac,
        )
```
- Add `from dataclasses import replace` if not already imported (app.py already imports `dataclasses`; use `dataclasses.replace` to avoid a new import).

- [ ] **Step 4: Run** — `pytest tests/test_server.py -q` → PASS (all). `ruff check src/atomsim/server/`.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "Serve exact Dirac levels from /api/levels behind a dirac flag"`

---

### Task 3: Frontend wiring — store, client, types, URL

**Files:**
- Modify: `web/src/api/client.ts`, `web/src/api/types.ts`, `web/src/state/store.ts`, `web/src/lib/urlState.ts`, `web/src/main.tsx`
- Test: `web/src/lib/urlState.test.ts` (append), `web/src/api/client` covered via urlState

**Interfaces:**
- Produces: store `dirac: boolean` + `setDirac`; `getLevels(..., dirac?)`; `LevelsResponse.dirac`; URL key `dirac`.

- [ ] **Step 1: Write failing test** (append to `urlState.test.ts`):

```typescript
it("round-trips the dirac toggle with fine structure", () => {
  const s = { ...URL_DEFAULTS, view: "levels" as const, fineStructure: true, dirac: true };
  const q = serializeAppUrl(s);
  expect(q).toContain("dirac=1");
  const back = { ...URL_DEFAULTS, ...parseAppUrl(q) };
  expect(back.dirac).toBe(true);
});

it("omits dirac when off or when fine structure is off", () => {
  expect(serializeAppUrl({ ...URL_DEFAULTS, dirac: false })).not.toContain("dirac");
  expect(serializeAppUrl({ ...URL_DEFAULTS, fineStructure: false, dirac: true })).not.toContain(
    "dirac",
  );
});
```
Also update the existing full round-trip literal (the one listing every field) to include `dirac: false`.

- [ ] **Step 2: Run to verify fail** — `npx vitest run src/lib/urlState.test.ts` → fail (missing `dirac`).

- [ ] **Step 3: Implement**

`api/types.ts`: add `dirac: boolean;` to the `LevelsResponse` interface.

`api/client.ts`: `getLevels` gains a trailing `dirac = false` param and appends `&dirac=${dirac}`:
```typescript
export function getLevels(
  system: string,
  nMax: number,
  fineStructure: boolean,
  alpha?: number,
  config?: string | null,
  dirac = false,
): Promise<LevelsResponse | ScreenedLevels> {
  const a = alpha === undefined ? "" : `&alpha=${alpha}`;
  const c = config ? `&config=${encodeURIComponent(config)}` : "";
  const d = dirac ? "&dirac=true" : "";
  return getJson(
    `/api/levels?system=${system}&n_max=${nMax}&fine_structure=${fineStructure}${a}${c}${d}`,
  );
}
```

`state/store.ts`: add `dirac: boolean` to the interface and `setDirac: (dirac: boolean) => void;`; init `dirac: false`; setter `setDirac: (dirac) => set({ dirac, levels: null })`; and in `loadLevels` read `dirac` and pass it:
```typescript
  loadLevels: async () => {
    const { system, fineStructure, config, dirac } = get();
    set({
      levels: await client.getLevels(system, N_MAX_DIAGRAM, fineStructure, undefined, config, dirac),
    });
  },
```

`lib/urlState.ts`: add `dirac: boolean` to `UrlState`; `dirac: false` to `URL_DEFAULTS`; parse `if (q.get("dirac") === "1") out.dirac = true;`; serialize `if (state.dirac && state.fineStructure) q.set("dirac", "1");`.

`main.tsx`: add `dirac: s.dirac,` to the serialized-state object.

- [ ] **Step 4: Run** — `npx vitest run src/lib/urlState.test.ts` → PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "Wire the Dirac level model through the store, client, and deep links"`

---

### Task 4: Levels view — Dirac toggle + captions + build

**Files:**
- Modify: `web/src/components/LevelsView.tsx`
- Verify: full build + suites

**Interfaces:**
- Consumes: store `dirac`/`setDirac`; `LevelsResponse.dirac`; fine-level provenance (already EXACT for Dirac).

- [ ] **Step 1: Implement the UI**

In `LevelsView.tsx`:
- Pull `dirac, setDirac` from the store; add `dirac` to the `useEffect` dependency array so toggling refetches:
  `useEffect(() => { void loadLevels(); void loadSpectrum(); }, [system, fineStructure, dirac, loadLevels, loadSpectrum]);`
- When `fineStructure` is true, render a small inline control in the `view-header` to switch model:
```tsx
{fineStructure && (
  <label className="levels-model">
    <input type="checkbox" checked={dirac} onChange={(e) => setDirac(e.target.checked)} />
    Dirac (exact)
  </label>
)}
```
- Make the zoomed fine-column title reflect the model: replace the hard-coded `"n={n} shifts [µeV] — zoomed, APPROXIMATION"` with `... — zoomed, ${dirac ? "EXACT" : "APPROXIMATION"}`.
- Extend the caption so the Dirac case names the Lamb shift explicitly: when `dirac`, append a sentence: "Dirac is exact for a point nucleus — energy depends on n and j only, so 2s₁/₂ and 2p₁/₂ coincide exactly; the real splitting is the Lamb shift (QED), which this model omits." Keep the existing α² wording for the perturbative case.
- The fine badge already reads `fineForN[0].shift.provenance`, which is EXACT in Dirac mode — no change needed.

- [ ] **Step 2: Typecheck + build** — `npm run build` → clean.

- [ ] **Step 3: Full suites** — from repo root `pytest -q` and `ruff check .`; from `web/` `npm test`. All green.

- [ ] **Step 4: Live smoke** — `atomsim serve --port 8022 --no-browser` (background); via env-python urllib GET `/api/levels?system=h&n_max=3&dirac=true` → `dirac:true`, first fine fidelity `exact`, the two `n=2 j=0.5` energies equal; GET `?system=z200&n_max=1&dirac=true` → 422. Stop the server.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "Add the Dirac exact/perturbative toggle and Lamb-shift caption to the Levels view"`

---

## Self-review notes

- **Spec coverage:** §4 engine → Task 1; §5 server/schema → Task 2; §6 store/client/URL → Task 3, view → Task 4; §2.2 honesty loop → Task 1 O(α⁴) test; §7 provenance enforced Tasks 1–2; §9 tests distributed. §8 deferrals untasked on purpose.
- **Type consistency:** `dirac_energy`, `dirac_fine_splitting`, `LevelsResponse.dirac`, store `dirac`/`setDirac`, `getLevels(..., dirac)`, URL `dirac` used identically across tasks.
- **Number check:** `E(1s½) = −0.5 − α²/8 = −0.50000665642` hartree (α = 7.2973525693e−3); this is the Task 1 published-value target.
