# Phase 11: Stark Effect (parabolic manifold) Implementation Plan

**Goal:** Add an external electric-field (Stark) mode to the Levels view that splits each degenerate hydrogen n-shell into its parabolic (n₁, n₂, m) sublevels, showing the linear-in-field fan that is hydrogen's l-degeneracy signature.

**Architecture:** A new closed-form engine module (`analytic/stark.py`) computes, per shell n, every parabolic sublevel's energy = Bohr + linear + quadratic Stark (closed form, no eigensolver). The server attaches per-**gross**-level `sublevels` to `/api/levels` behind an `e_field` param (MV/m). The Levels view adds an always-on F slider and fans each gross level into its parabolic sublevels in a zoomed column. Stark is a gross-structure effect: independent of the fine-structure toggle, no spin, no α.

**Tech Stack:** Python 3.12 (numpy-free, stdlib `math`), FastAPI + Pydantic, React + TypeScript + Zustand, vitest, pytest.

## Global Constraints

- Engine-internal math in Hartree atomic units; SI/display conversion (eV, MV/m) only at the boundary, appended to the provenance `method`.
- Every physical value crossing a module boundary is a `Quantity`/`Field` carrying `Provenance` with a `Fidelity` tier. Stark sublevels are `APPROXIMATION` always (no α dependence, so no counterfactual branch).
- `l` is the orbital quantum number, not a length (ruff E741 ignored project-wide). Here `m` is the magnetic quantum number; `n1`/`n2` the parabolic numbers.
- New physics gets a validation test (analytic ground truth), not a smoke test.
- Line length 100. Run `ruff check .` before each engine/server commit.
- Rebuild the frontend (`npm run build`) after changing anything under `web/src`; `atomsim serve` only mounts `web/dist`.
- Run pytest with `MKL_THREADING_LAYER=SEQUENTIAL` set (env quirk); env python is `C:\Users\yashg\.conda\envs\atomsim\python.exe`.

---

## File Structure

- **Modify** `src/atomsim/constants.py`, add `E0_V_PER_M` anchor.
- **Create** `src/atomsim/analytic/stark.py`, the closed-form Stark engine (`StarkSublevel`, `stark_sublevels`).
- **Create** `tests/test_stark.py`, engine validation.
- **Modify** `src/atomsim/server/app.py`, `StarkSublevelModel`, `GrossLevelModel.sublevels`, `LevelsResponse.e_field`, `e_field` param + attach logic.
- **Modify** `tests/test_server.py`, server route tests (append).
- **Modify** `web/src/api/types.ts`, `StarkSublevel`, `GrossLevel.sublevels`, `LevelsResponse.e_field`.
- **Modify** `web/src/api/client.ts`, `getLevels(..., eField)`.
- **Modify** `web/src/state/store.ts`, `eField`/`setEField`, thread through `loadLevels`.
- **Modify** `web/src/lib/urlState.ts`, `e` param parse/serialize + `URL_DEFAULTS`.
- **Modify** `web/src/main.tsx`, serialize `eField`.
- **Modify** `web/src/components/LevelsView.tsx`, F slider, Stark-manifold zoom column, caption.
- **Modify** `web/src/lib/urlState.test.ts` and `web/src/state/store.test.ts`, web logic tests (append).

---

### Task 1: Engine: `analytic/stark.py` (parabolic Stark core)

**Files:**
- Modify: `src/atomsim/constants.py`
- Create: `src/atomsim/analytic/stark.py`
- Test: `tests/test_stark.py`

**Interfaces:**
- Consumes: `energy(n, Z, mu_ratio) -> Quantity` (`analytic/hydrogen.py`); `Fidelity, Provenance, Quantity` (`provenance.py`); `E0_V_PER_M` (`constants.py`).
- Produces:
  - `E0_V_PER_M: float`, atomic unit of electric field in V/m (in `constants.py`).
  - `@dataclass(frozen=True) StarkSublevel(n1: int, n2: int, m: int, k: int, energy: Quantity)`.
  - `stark_sublevels(n, Z=1, mu_ratio=1.0, field_mv_per_m=0.0) -> list[StarkSublevel]`.

- [ ] **Step 1: Add the `E0_V_PER_M` constant**

In `src/atomsim/constants.py`, after the `B0_TESLA` block (before the `@dataclass` on line 32), add:

```python
# Real-universe display anchor ONLY (same caveat): atomic unit of electric field
# (E_h / (e a0)), in volts per metre: for MV/m<->a.u. conversion at the boundary.
E0_V_PER_M: float = _sc.physical_constants["atomic unit of electric field"][0]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_stark.py`:

```python
"""Validation for the parabolic Stark engine (linear + quadratic in a static field)."""

import math

import pytest

from atomsim.analytic.hydrogen import energy
from atomsim.analytic.stark import StarkSublevel, stark_sublevels
from atomsim.constants import E0_V_PER_M
from atomsim.provenance import Fidelity

# field magnitude helpers
_AU_PER_MVM = 1e6 / E0_V_PER_M  # a.u. of field per (MV/m)

def _field_au(mvm: float) -> float:
    return mvm * _AU_PER_MVM

def test_zero_field_recovers_bohr():
    for n in (1, 2, 3):
        want = energy(n).value
        for s in stark_sublevels(n, field_mv_per_m=0.0):
            assert s.energy.value == pytest.approx(want, rel=1e-12)

def test_sublevel_count_is_n_squared():
    assert len(stark_sublevels(1, field_mv_per_m=5.0)) == 1
    assert len(stark_sublevels(2, field_mv_per_m=5.0)) == 4
    assert len(stark_sublevels(3, field_mv_per_m=5.0)) == 9
    assert len(stark_sublevels(4, field_mv_per_m=5.0)) == 16

def test_parabolic_constraint():
    for n in (1, 2, 3, 4):
        for s in stark_sublevels(n, field_mv_per_m=1.0):
            assert s.n1 + s.n2 + abs(s.m) + 1 == n
            assert s.k == s.n1 - s.n2

def test_low_field_slope_is_linear_stark():
    # dE/dF (a.u.) at small F equals (3/2) n k for Z=mu=1.
    mvm = 1e-3
    f = _field_au(mvm)
    n = 3
    e0 = {(s.n1, s.n2, s.m): s.energy.value for s in stark_sublevels(n, field_mv_per_m=0.0)}
    for s in stark_sublevels(n, field_mv_per_m=mvm):
        slope = (s.energy.value - e0[(s.n1, s.n2, s.m)]) / f
        assert slope == pytest.approx(1.5 * n * s.k, rel=1e-4, abs=1e-12)

def test_linear_fan_is_traceless():
    # Sum of first-order shifts over a shell vanishes (sum of k = 0).
    n = 4
    assert sum(s.k for s in stark_sublevels(n, field_mv_per_m=1.0)) == 0

def test_ground_state_polarizability():
    # n=1: no linear term; quadratic coefficient -9/4 -> polarizability 9/2 a.u.
    n = 1
    mvm = 10.0
    f = _field_au(mvm)
    e0 = energy(1).value
    s = stark_sublevels(n, field_mv_per_m=mvm)[0]
    assert s.k == 0
    coeff = (s.energy.value - e0) / (f * f)
    assert coeff == pytest.approx(-9.0 / 4.0, rel=1e-6)

def test_pm_m_degeneracy():
    subs = {(s.n1, s.n2, s.m): s.energy.value for s in stark_sublevels(3, field_mv_per_m=7.0)}
    for (n1, n2, m), val in subs.items():
        assert subs[(n1, n2, -m)] == pytest.approx(val, rel=1e-12)

def test_z_scaling_halves_linear_shift():
    # Linear shift ~ 1/(Z mu); He+ (Z=2) has half the H linear shift for the same F.
    mvm = 1.0
    f = _field_au(mvm)
    n = 2
    def linshift(Z):
        s = next(s for s in stark_sublevels(n, Z=Z, field_mv_per_m=mvm) if s.k == 1)
        return s.energy.value - energy(n, Z=Z).value
    # extract the linear part via small field (quadratic negligible)
    small = 1e-4
    fs = _field_au(small)
    def lin(Z):
        s = next(s for s in stark_sublevels(n, Z=Z, field_mv_per_m=small) if s.k == 1)
        return (s.energy.value - energy(n, Z=Z).value) / fs
    assert lin(2) == pytest.approx(lin(1) / 2.0, rel=1e-3)

def test_provenance_tier_and_error():
    s = stark_sublevels(3, field_mv_per_m=10.0)[0]
    assert s.energy.provenance.fidelity is Fidelity.APPROXIMATION
    assert s.energy.provenance.error_estimate is not None
    # error grows with field (leading neglected term ~ F^3).
    def err(mvm):
        return next(
            x for x in stark_sublevels(3, field_mv_per_m=mvm) if x.k == 2
        ).energy.provenance.error_estimate
    assert err(20.0) > err(10.0) > 0.0

def test_negative_field_rejected():
    with pytest.raises(ValueError):
        stark_sublevels(2, field_mv_per_m=-1.0)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_stark.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.analytic.stark'`.

- [ ] **Step 4: Implement the module**

Create `src/atomsim/analytic/stark.py`:

```python
"""Stark effect: hydrogen in a static electric field, the parabolic manifold.

A uniform field along z keeps axial symmetry (m good) but breaks rotational
symmetry (l not good); the hydrogen + (-F z) Hamiltonian separates exactly in
parabolic coordinates with non-negative integers n1, n2 and n = n1 + n2 + |m| + 1.
Second-order degenerate perturbation theory is closed form: each shell n splits
into n^2 sublevels labelled by the electric quantum number k = n1 - n2, linear in
the field (the l-degeneracy signature) with a quadratic correction.

APPROXIMATION by construction: second order only, and the Stark manifold is not
truly bound (a static field ionizes; the series is asymptotic and diverges near
F_ion ~ Z^3 mu^2 / (16 n^4) a.u.). No alpha dependence (non-relativistic), so no
COUNTERFACTUAL branch. See docs/specs/2026-07-24-phase11-stark-effect-design.md.
"""

import math
from dataclasses import dataclass

from atomsim.analytic.hydrogen import energy
from atomsim.constants import E0_V_PER_M
from atomsim.provenance import Fidelity, Provenance, Quantity

_S_ASSUMPTIONS = (
    "second-order perturbation theory (linear + quadratic); third and higher orders neglected",
    "static field: the manifold is a resonance, not a true bound state (field ionization neglected)",
    "gross-structure only: fine structure and its low-field crossover neglected",
    "non-relativistic: independent of alpha (altering alpha does not change this shift)",
)

@dataclass(frozen=True)
class StarkSublevel:
    n1: int
    n2: int
    m: int
    k: int            # n1 - n2, the electric quantum number
    energy: Quantity

def stark_sublevels(
    n: int, Z: int = 1, mu_ratio: float = 1.0, field_mv_per_m: float = 0.0,
) -> list[StarkSublevel]:
    """Parabolic Stark sublevels of the shell n in a field F (MV/m)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if Z < 1:
        raise ValueError(f"Z must be >= 1, got {Z}")
    if field_mv_per_m < 0:
        raise ValueError(f"field_mv_per_m must be >= 0, got {field_mv_per_m}")

    f_au = field_mv_per_m * 1e6 / E0_V_PER_M  # atomic units of field
    e_bohr = energy(n, Z=Z, mu_ratio=mu_ratio).value
    zm = Z * mu_ratio
    # classical field-ionization scale (a.u.); guards the F/F_ion error ratio.
    f_ion = (Z ** 3) * (mu_ratio ** 2) / (16.0 * n ** 4)

    method = (
        "parabolic Stark, 2nd-order perturbation theory (linear + quadratic); "
        f"F = {field_mv_per_m:g} MV/m = {f_au:.3e} a.u. "
        f"(F/F_ion = {f_au / f_ion:.2e})"
    )

    out: list[StarkSublevel] = []
    for m in range(-(n - 1), n):
        am = abs(m)
        for n1 in range(0, n - am):
            n2 = n - am - 1 - n1
            k = n1 - n2
            # linear ~ F/(Z*mu); quadratic ~ F^2/(Z^4 * mu^3) (reduced-mass powers
            # differ: 1 vs 3, NOT both Z*mu). Polarizability E2(n=1)=-(9/4)F^2/(Z^4 mu^3).
            lin = 1.5 * n * k * f_au / zm
            quad = (
                -(1.0 / 16.0) * n ** 4
                * (17 * n * n - 3 * k * k - 9 * m * m + 19)
                * f_au * f_au / (Z ** 4 * mu_ratio ** 3)
            )
            value = e_bohr + lin + quad
            err = abs(quad) * (f_au / f_ion) if f_au > 0.0 else 0.0
            out.append(StarkSublevel(
                n1=n1, n2=n2, m=m, k=k,
                energy=Quantity(
                    value=value, unit="hartree",
                    label=f"E_Stark {n},n1={n1},n2={n2},m={m} (F={field_mv_per_m:g}MV/m)",
                    provenance=Provenance(
                        fidelity=Fidelity.APPROXIMATION, method=method,
                        assumptions=_S_ASSUMPTIONS, error_estimate=err,
                        refinement=(
                            "third-order Stark, then the full non-perturbative "
                            "(field-ionization) resonance treatment"
                        ),
                    ),
                ),
            ))
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_stark.py -q`
Expected: PASS (10 tests).

- [ ] **Step 6: Lint**

Run: `"C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check src/atomsim/analytic/stark.py src/atomsim/constants.py tests/test_stark.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/atomsim/analytic/stark.py src/atomsim/constants.py tests/test_stark.py
git commit -m "Add the closed-form parabolic Stark engine (linear + quadratic field)"
```

---

### Task 2: Server: `/api/levels` Stark mode

**Files:**
- Modify: `src/atomsim/server/app.py` (`GrossLevelModel` ~line 92; `LevelsResponse` ~119; endpoint ~387)
- Test: `tests/test_server.py` (append)

**Interfaces:**
- Consumes: `stark_sublevels(...) -> list[StarkSublevel]` (Task 1); `QuantityModel.from_quantity`, `_to_ev` (existing in `app.py`).
- Produces: `/api/levels?...&e_field=<MV/m>` returning `GrossLevelModel.sublevels: list[StarkSublevelModel] | None`; `LevelsResponse.e_field: float`.

- [ ] **Step 1: Write the failing server tests**

Append to `tests/test_server.py`:

```python
def test_levels_stark_splits_gross_levels(client):
    r = client.get("/api/levels?system=h&n_max=3&e_field=50")
    assert r.status_code == 200
    body = r.json()
    assert body["e_field"] == 50.0
    g2 = next(g for g in body["gross"] if g["n"] == 2)
    assert g2["sublevels"] is not None and len(g2["sublevels"]) == 4  # n^2
    s0 = g2["sublevels"][0]
    assert s0["energy"]["provenance"]["fidelity"] == "approximation"
    assert "k" in s0 and "n1" in s0

def test_levels_stark_absent_without_field(client):
    r = client.get("/api/levels?system=h&n_max=2")
    body = r.json()
    assert body["e_field"] == 0.0
    assert all(g.get("sublevels") is None for g in body["gross"])

def test_levels_stark_independent_of_fine_structure(client):
    r = client.get("/api/levels?system=h&n_max=2&fine_structure=false&e_field=30")
    assert r.status_code == 200
    g2 = next(g for g in r.json()["gross"] if g["n"] == 2)
    assert g2["sublevels"] is not None and len(g2["sublevels"]) == 4

def test_levels_stark_negative_field_rejected(client):
    r = client.get("/api/levels?system=h&n_max=2&e_field=-1")
    assert r.status_code == 422

def test_levels_stark_ignored_for_screened(client):
    r = client.get("/api/levels?system=he&e_field=50")
    assert r.status_code == 200
    assert "orbitals" in r.json()  # ScreenedLevelsModel, no sublevels
```

(If the test-file `client` fixture has a different name, match the existing tests in `tests/test_server.py`.)

- [ ] **Step 2: Run to verify failure**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_server.py -q -k stark`
Expected: FAIL (`KeyError: 'e_field'` / missing `sublevels`).

- [ ] **Step 3: Add the schema model + fields**

In `src/atomsim/server/app.py`, add `StarkSublevelModel` immediately before `class GrossLevelModel` (line 92) and extend `GrossLevelModel` and `LevelsResponse`:

```python
class StarkSublevelModel(BaseModel):
    n1: int
    n2: int
    m: int
    k: int
    energy: QuantityModel
    energy_ev: QuantityModel

class GrossLevelModel(BaseModel):
    n: int
    degeneracy: int
    energy: QuantityModel
    energy_ev: QuantityModel
    sublevels: list[StarkSublevelModel] | None = None
```

And in `LevelsResponse` add after `b_field`:

```python
    e_field: float = 0.0
```

- [ ] **Step 4: Add the import**

At the top of `app.py`, alongside the `from atomsim.analytic.zeeman import zeeman_sublevels` line, add:

```python
from atomsim.analytic.stark import stark_sublevels
```

- [ ] **Step 5: Wire the endpoint**

Add `e_field: float = 0.0` to the `levels_endpoint` signature (after `b_field: float = 0.0`).

After the `b_field` validation (line 420-421), add:

```python
        if e_field < 0.0:
            raise HTTPException(status_code=422, detail="e_field must be >= 0")
```

Replace the gross build loop (lines 425-432) so each gross level carries its Stark sublevels:

```python
        gross = []
        for n in range(1, n_max + 1):
            e = energy(n, Z=sys_.Z, mu_ratio=mu)
            gsubs = None
            if e_field > 0.0:
                sss = stark_sublevels(
                    n, Z=sys_.Z, mu_ratio=mu, field_mv_per_m=e_field,
                )
                gsubs = [
                    StarkSublevelModel(
                        n1=s.n1, n2=s.n2, m=s.m, k=s.k,
                        energy=QuantityModel.from_quantity(s.energy),
                        energy_ev=QuantityModel.from_quantity(_to_ev(s.energy)),
                    )
                    for s in sss
                ]
            gross.append(GrossLevelModel(
                n=n, degeneracy=2 * n * n,
                energy=QuantityModel.from_quantity(e),
                energy_ev=QuantityModel.from_quantity(_to_ev(e)),
                sublevels=gsubs,
            ))
```

Finally add `e_field=e_field,` to the `LevelsResponse(...)` return (after `b_field=b_field,`).

- [ ] **Step 6: Run the server tests**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_server.py -q -k stark`
Expected: PASS (5 tests).

- [ ] **Step 7: Full engine+server suite + lint**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest -q` then `"C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check .`
Expected: all pass, no lint errors.

- [ ] **Step 8: Commit**

```bash
git add src/atomsim/server/app.py tests/test_server.py
git commit -m "Serve Stark sublevels from /api/levels behind an e_field flag"
```

---

### Task 3: Web: store, client, types, URL round-trip

**Files:**
- Modify: `web/src/api/types.ts` (GrossLevel; LevelsResponse)
- Modify: `web/src/api/client.ts` (getLevels)
- Modify: `web/src/state/store.ts` (fields; setters; loadLevels)
- Modify: `web/src/lib/urlState.ts` (UrlState; URL_DEFAULTS; parse; serialize)
- Modify: `web/src/main.tsx`
- Test: `web/src/lib/urlState.test.ts`, `web/src/state/store.test.ts` (append)

**Interfaces:**
- Consumes: server `e_field` + gross `sublevels` (Task 2).
- Produces: store `eField: number` / `setEField`; `getLevels(..., eField)`; URL `e` param.

- [ ] **Step 1: Write the failing web tests**

Append to `web/src/lib/urlState.test.ts`:

```ts
it("round-trips the e_field", () => {
  const url = serializeAppUrl({ ...URL_DEFAULTS, eField: 40 });
  expect(url).toContain("e=40");
  expect(parseAppUrl(url).eField).toBe(40);
});

it("omits e when field is zero", () => {
  const url = serializeAppUrl({ ...URL_DEFAULTS, eField: 0 });
  expect(url).not.toContain("e=");
});
```

Append to `web/src/state/store.test.ts` (match the existing store-test style/imports):

```ts
it("setEField clears cached levels", () => {
  useStore.setState({ levels: { fake: true } as never, eField: 0 });
  useStore.getState().setEField(25);
  expect(useStore.getState().eField).toBe(25);
  expect(useStore.getState().levels).toBeNull();
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd web && npx vitest run src/lib/urlState.test.ts src/state/store.test.ts`
Expected: FAIL (`eField` not on type / setter missing).

- [ ] **Step 3: Extend the API types**

In `web/src/api/types.ts`, add before `GrossLevel` (or near the other sublevel type):

```ts
export interface StarkSublevel {
  n1: number;
  n2: number;
  m: number;
  k: number;
  energy: Quantity;
  energy_ev: Quantity;
}
```

Add `sublevels?: StarkSublevel[] | null;` to `GrossLevel`, and `e_field: number;` to `LevelsResponse`.

- [ ] **Step 4: Thread `eField` through the client**

In `web/src/api/client.ts`, change `getLevels` to take `eField` last and add it to the query:

```ts
export function getLevels(
  system: string,
  nMax: number,
  fineStructure: boolean,
  alpha?: number,
  config?: string | null,
  dirac = false,
  bField = 0,
  eField = 0,
): Promise<LevelsResponse | ScreenedLevels> {
  const a = alpha === undefined ? "" : `&alpha=${alpha}`;
  const c = config ? `&config=${encodeURIComponent(config)}` : "";
  const d = dirac ? "&dirac=true" : "";
  const b = bField > 0 ? `&b_field=${bField}` : "";
  const e = eField > 0 ? `&e_field=${eField}` : "";
  return getJson(
    `/api/levels?system=${system}&n_max=${nMax}&fine_structure=${fineStructure}${a}${c}${d}${b}${e}`,
  );
}
```

- [ ] **Step 5: Add store state + setter + loadLevels wiring**

In `web/src/state/store.ts`: add `eField: number;` to the state interface near `bField: number;`, `setEField: (eField: number) => void;` near `setBField`, the default `eField: 0,` near `bField: 0,`, the setter near `setBField`:

```ts
  setEField: (eField) => set({ eField, levels: null }),
```

and in `loadLevels` pull `eField` and pass it:

```ts
  loadLevels: async () => {
    const { system, fineStructure, config, dirac, bField, eField } = get();
    set({
      levels: await client.getLevels(
        system, N_MAX_DIAGRAM, fineStructure, undefined, config, dirac, bField, eField,
      ),
    });
  },
```

- [ ] **Step 6: Add URL parse/serialize**

In `web/src/lib/urlState.ts`: add `eField: number;` to `UrlState` (near `bField`), `eField: 0,` to `URL_DEFAULTS`. In the parse function after the `b` parse add (the short param is `ef`, not `e`: `e` is already the charge-multiplier param in `CONST_PARAMS` and reusing it corrupts the What-If deep links):

```ts
  const ef = Number(q.get("ef"));
  if (Number.isFinite(ef) && ef > 0) out.eField = ef;
```

In serialize after the `b` line add (no fine-structure gate, Stark is independent):

```ts
  if (state.eField > 0) q.set("ef", String(state.eField));
```

- [ ] **Step 7: Serialize from main.tsx**

In `web/src/main.tsx`, in the object passed to `serializeAppUrl` (where `bField: s.bField,` is), add `eField: s.eField,`.

- [ ] **Step 8: Run the web tests**

Run: `cd web && npx vitest run src/lib/urlState.test.ts src/state/store.test.ts`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add web/src/api/types.ts web/src/api/client.ts web/src/state/store.ts web/src/lib/urlState.ts web/src/main.tsx web/src/lib/urlState.test.ts web/src/state/store.test.ts
git commit -m "Wire the Stark e_field through the client, store, and deep links"
```

---

### Task 4: Levels view: F slider, Stark-manifold fan, caption + build

**Files:**
- Modify: `web/src/components/LevelsView.tsx`
- Verify: full build + suites

**Interfaces:**
- Consumes: store `eField`/`setEField`; `GrossLevel.sublevels`.

- [ ] **Step 1: Refetch on field change + destructure**

Pull `eField, setEField` from the store destructure (line 79) and add `eField` to the `useEffect` dependency array (line 85):

```tsx
  }, [system, fineStructure, dirac, bField, eField, loadLevels, loadSpectrum]);
```

- [ ] **Step 2: Add the F slider to the header (always on)**

After the existing Zeeman `B` slider block and the `!fineStructure` hint, add an always-visible F slider:

```tsx
        <label className="levels-field">
          F{" "}
          <input
            type="range" min={0} max={100} step={0.5} value={eField}
            onChange={(e) => setEField(Number(e.target.value))}
          />
          {eField > 0 ? ` ${eField.toFixed(1)} MV/m` : " 0 MV/m"}
        </label>
```

- [ ] **Step 3: Add the Stark-manifold zoom column for the selected n**

Add a zoomed column, rendered when `eField > 0`, that fans the selected n's gross level into its parabolic sublevels. It reuses the right-column geometry (`zx1..sx2`) of the Zeeman zoom and takes precedence there. Gate the existing fine/Zeeman zoom to render only when `eField === 0` so the two never overlap.

Implementation sketch (place before or after the existing fine/Zeeman zoom IIFE, and add `&& eField === 0` to the existing fine/Zeeman zoom's outer condition):

```tsx
        {eField > 0 && (() => {
          const gsel = levels.gross.find((g) => g.n === n);
          const subs = gsel?.sublevels ?? [];
          if (subs.length === 0) return null;
          const bohrN = gsel!.energy_ev.value;
          const shifts = subs.map((s) => s.energy_ev.value - bohrN);
          const lo = Math.min(...shifts);
          const hi = Math.max(...shifts);
          const pad = (hi - lo || 1e-9) * 0.15;
          const yz = scaleLinear([lo - pad, hi + pad], [H - 60, 48]);
          const zx1 = 470;
          const zx2 = 620;
          const kMax = Math.max(...subs.map((s) => Math.abs(s.k)));
          return (
            <g>
              <text x={(zx1 + zx2) / 2} y={26} textAnchor="middle" className="tick">
                n={n} Stark manifold [meV], APPROXIMATION
              </text>
              {/* Bohr reference at zero shift */}
              <line x1={zx1} x2={zx2} y1={yz(0)} y2={yz(0)} className="zero" opacity={0.5} />
              {subs.map((s) => {
                const yS = yz(s.energy_ev.value - bohrN);
                const extreme = Math.abs(s.k) === kMax && s.m === 0;
                return (
                  <g key={`${s.n1}-${s.n2}-${s.m}`}>
                    <line x1={zx1} x2={zx2} y1={yS} y2={yS} className="rung" opacity={0.8} />
                    {extreme && (
                      <text x={zx2 + 6} y={yS} dy="0.32em" className="tick">
                        k={s.k}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          );
        })()}
```

(Adjust `W`/viewBox use only if labels clip; keep within the existing `W=680` canvas. Match the existing SVG class names, `rung`, `zero`, `tick`, already in the file/CSS.)

- [ ] **Step 4: Update the caption**

Extend the caption so that when `eField > 0` it names the linear-fan degeneracy signature and the caveat:

```tsx
        {eField > 0 && (
          <>
            {" "}An electric field splits each n-shell into n² parabolic (n₁,n₂,m)
            sublevels fanned by the electric quantum number k = n₁−n₂. The splitting is
            linear in F, hydrogen's accidental l-degeneracy lets a first-order shift
            appear, unlike non-degenerate atoms (quadratic only). Second-order model on
            the gross shells; the series is asymptotic and breaks down near field
            ionization, see the badge.
          </>
        )}
```

- [ ] **Step 5: Typecheck + build**

Run: `cd web && npm run build`
Expected: clean (tsc --noEmit + vite build).

- [ ] **Step 6: Full suites**

Run from repo root: `MKL_THREADING_LAYER=SEQUENTIAL "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest -q` and `"C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check .`; from `web/`: `npm test`.
Expected: all green.

- [ ] **Step 7: Live smoke**

Start `atomsim serve --port 8024 --no-browser` (background). GET `/api/levels?system=h&n_max=3&e_field=50` → `e_field:50.0`, gross n=2 has 4 `sublevels`, first sublevel fidelity `approximation`, `k`/`n1` present. GET with `e_field=-1` → 422. GET `?system=h&n_max=3&fine_structure=true&b_field=2&e_field=50` → both `b_field` and `e_field` present (fine levels carry Zeeman sublevels, gross levels carry Stark sublevels). Stop the server.

- [ ] **Step 8: Commit**

```bash
git add web/src/components/LevelsView.tsx
git commit -m "Add the Stark E-field slider and parabolic fan to the Levels view"
```

---

## Self-review notes

- **Spec coverage:** §2 physics → Task 1 (parabolic enumeration, linear+quadratic closed form, Z/μ scaling); §2.3 fidelity/error → Task 1 provenance; §5 engine → Task 1; §6 server → Task 2; §7 web store/client/URL → Task 3, view → Task 4; §8 tests distributed across all tasks. §3 deferrals untasked on purpose.
- **Type consistency:** `stark_sublevels`, `StarkSublevel(n1, n2, m, k, energy)`, `E0_V_PER_M`, `StarkSublevelModel`, `GrossLevelModel.sublevels`, `LevelsResponse.e_field`, store `eField`/`setEField`, `getLevels(..., eField)`, URL `e`, TS `StarkSublevel`/`GrossLevel.sublevels`/`LevelsResponse.e_field` used identically across tasks.
- **Number checks:** F(a.u.)/(MV/m) = 1e6/5.14220674763e11 = 1.944e-6. At 100 MV/m, F = 1.944e-4 a.u.; n=2 k=1 linear shift = 3F = 5.83e-4 hartree = 15.9 meV, visible. n=1 quadratic coefficient −9/4 → polarizability 9/2 a.u. Sublevel count per shell = n² (1,4,9,16). F_ion(n=4) ≈ 125 MV/m, so the n=4 error estimate rises sharply inside the 0-100 MV/m range.
- **Contrast with Zeeman (Phase 10):** attaches to gross (not fine) levels; independent of the fine-structure toggle; no α/COUNTERFACTUAL branch; Stark zoom column and Zeeman zoom column are mutually exclusive (Stark wins when e_field > 0).
