# Phase 10: Zeeman Field (Breit-Rabi crossover) Implementation Plan

**Goal:** Add an external magnetic-field (Zeeman) mode to the Levels view that shows the full Breit-Rabi crossover from the anomalous Zeeman effect to Paschen-Back for hydrogen-like atoms.

**Architecture:** A new closed-form engine module (`analytic/zeeman.py`) diagonalizes, per (n, l, mⱼ), the 2×2 {fine-structure + linear-Zeeman} block with the analytic Breit-Rabi roots (no numerical eigensolver). The diagonal comes from the selected level model (α² fine structure or exact Dirac). The server attaches per-level `sublevels` to `/api/levels` behind a `b_field` param; the Levels view adds a B slider and fans each j-level into its mⱼ sublevels.

**Tech Stack:** Python 3.12 (numpy-free, stdlib `math`), FastAPI + Pydantic, React + TypeScript + Zustand, vitest, pytest.

## Global Constraints

- Engine-internal math in Hartree atomic units; SI/display conversion (eV, µeV, Tesla) only at the server boundary, appended to the provenance `method`.
- Every physical value crossing a module boundary is a `Quantity`/`Field` carrying `Provenance` with a `Fidelity` tier. The Zeeman levels are `APPROXIMATION` at real α, `COUNTERFACTUAL` when α is altered.
- `l` is the orbital quantum number, not a length (ruff E741 ignored project-wide).
- New physics gets a validation test (analytic ground truth), not a smoke test.
- Line length 100. Run `ruff check .` before each engine/server commit.
- Rebuild the frontend (`npm run build`) after changing anything under `web/src`; `atomsim serve` only mounts `web/dist`.
- Run pytest with `MKL_THREADING_LAYER=SEQUENTIAL` set (env quirk); env python is `C:\Users\yashg\.conda\envs\atomsim\python.exe`.

---

## File Structure

- **Create** `src/atomsim/analytic/zeeman.py`, the Breit-Rabi engine (`ZeemanSublevel`, `zeeman_sublevels`, `lande_g`, `MU_B_PER_TESLA`).
- **Modify** `src/atomsim/constants.py`, add `B0_TESLA` anchor.
- **Create** `tests/test_zeeman.py`, engine validation.
- **Modify** `src/atomsim/server/app.py`, `ZeemanSublevelModel`, `FineLevelModel.sublevels`, `LevelsResponse.b_field`, `b_field` param + attach logic.
- **Modify** `tests/test_server.py`, server route tests (append).
- **Modify** `web/src/api/types.ts`, `ZeemanSublevel`, `FineLevel.sublevels`, `LevelsResponse.b_field`.
- **Modify** `web/src/api/client.ts`, `getLevels(bField)`.
- **Modify** `web/src/state/store.ts`, `bField`/`setBField`, thread through `loadLevels`.
- **Modify** `web/src/lib/urlState.ts`, `b` param parse/serialize + `URL_DEFAULTS`.
- **Modify** `web/src/main.tsx`, serialize `bField`.
- **Modify** `web/src/components/LevelsView.tsx`, B slider, sublevel fan render, caption.
- **Modify** `web/src/lib/urlState.test.ts` and `web/src/state/store.test.ts`, web logic tests (append).

---

### Task 1: Engine: `analytic/zeeman.py` (Breit-Rabi core)

**Files:**
- Modify: `src/atomsim/constants.py`
- Create: `src/atomsim/analytic/zeeman.py`
- Test: `tests/test_zeeman.py`

**Interfaces:**
- Consumes: `level_energy(n, l, j, Z, mu_ratio, m_over_M, alpha) -> Quantity` (`analytic/fine_structure.py`); `dirac_energy(n, j, Z, mu_ratio, alpha) -> Quantity` (`analytic/dirac.py`); `energy(n, Z, mu_ratio) -> Quantity` (`analytic/hydrogen.py`); `validate_quantum_numbers(n, l)` (`analytic/hydrogen.py`); `Fidelity, Provenance, Quantity` (`provenance.py`); `ALPHA` (`constants.py`).
- Produces:
  - `MU_B_PER_TESLA: float`, µ_B in hartree per Tesla.
  - `lande_g(l: int, j: float) -> float`.
  - `@dataclass(frozen=True) ZeemanSublevel(m_j: float, branch: str, j_label: float, high_field_label: str, energy: Quantity)`.
  - `zeeman_sublevels(n, l, Z=1, mu_ratio=1.0, m_over_M=0.0, alpha=ALPHA, b_tesla=0.0, dirac=False) -> list[ZeemanSublevel]`.

- [ ] **Step 1: Add the `B0_TESLA` constant**

In `src/atomsim/constants.py`, after the `BOHR_RADIUS_FM` block (before the `@dataclass` on line 28), add:

```python
# Real-universe display anchor ONLY (same caveat): atomic unit of magnetic field
# (hbar / (e a0^2)), in tesla: for Tesla<->a.u. conversion at the server boundary.
B0_TESLA: float = _sc.physical_constants["atomic unit of mag. flux density"][0]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_zeeman.py`:

```python
"""Validation for the Zeeman/Breit-Rabi engine (fine structure + linear Zeeman)."""

import math

import pytest

from atomsim.analytic.dirac import dirac_energy
from atomsim.analytic.fine_structure import level_energy
from atomsim.analytic.zeeman import (
    MU_B_PER_TESLA,
    ZeemanSublevel,
    lande_g,
    zeeman_sublevels,
)
from atomsim.constants import ALPHA
from atomsim.provenance import Fidelity

def _by_mj(subs, m_j, branch):
    return next(s for s in subs if s.m_j == m_j and s.branch == branch)

def test_zero_field_recovers_fine_structure():
    # At B=0 every sublevel equals the underlying (n,l,j) perturbative level.
    for l, j, branch in [(0, 0.5, "single"), (1, 1.5, "upper"), (1, 0.5, "lower")]:
        subs = zeeman_sublevels(2, l, b_tesla=0.0)
        want = level_energy(2, l, j).value
        for s in (s for s in subs if s.j_label == j and s.branch == branch):
            assert s.energy.value == pytest.approx(want, rel=1e-12)

def test_sublevel_count():
    # l>=1 gives 4l+2 sublevels (both j present); l=0 gives 2.
    assert len(zeeman_sublevels(2, 0, b_tesla=1.0)) == 2
    assert len(zeeman_sublevels(2, 1, b_tesla=1.0)) == 6   # 4*1+2
    assert len(zeeman_sublevels(3, 2, b_tesla=1.0)) == 10  # 4*2+2

def test_lande_g_values():
    assert lande_g(1, 1.5) == pytest.approx(4.0 / 3.0)
    assert lande_g(1, 0.5) == pytest.approx(2.0 / 3.0)
    assert lande_g(0, 0.5) == pytest.approx(2.0)

def test_low_field_slope_is_lande():
    # dE/dB at small B equals g_J * mu_B * m_j for each sublevel.
    b = 1e-4
    e0 = {(s.j_label, s.m_j, s.branch): s.energy.value for s in zeeman_sublevels(2, 1, b_tesla=0.0)}
    for s in zeeman_sublevels(2, 1, b_tesla=b):
        slope = (s.energy.value - e0[(s.j_label, s.m_j, s.branch)]) / b
        want = lande_g(1, s.j_label) * MU_B_PER_TESLA * s.m_j
        assert slope == pytest.approx(want, rel=1e-3, abs=1e-16)

def test_high_field_slope_is_paschen_back():
    # At large B the slopes approach integer (m_l + 2 m_s) * mu_B.
    b1, b2 = 1.0e4, 2.0e4  # far above the ~1e-6 hartree fine-structure scale
    e1 = {(s.m_j, s.branch): s.energy.value for s in zeeman_sublevels(2, 1, b_tesla=b1)}
    for s in zeeman_sublevels(2, 1, b_tesla=b2):
        slope = (s.energy.value - e1[(s.m_j, s.branch)]) / (b2 - b1)
        integer = round(slope / MU_B_PER_TESLA)
        assert slope / MU_B_PER_TESLA == pytest.approx(integer, abs=1e-3)
        assert integer in (-2, -1, 0, 1, 2)

def test_trace_invariance():
    # Each 2x2 block: sum of eigenvalues == trace for all B.
    n, l = 2, 1
    e_up = level_energy(n, l, l + 0.5).value
    e_dn = level_energy(n, l, l - 0.5).value
    for b in (0.5, 5.0, 50.0):
        subs = zeeman_sublevels(n, l, b_tesla=b)
        for m_j in (0.5, -0.5):  # interior mj -> 2x2 blocks
            up = _by_mj(subs, m_j, "upper").energy.value
            lo = _by_mj(subs, m_j, "lower").energy.value
            trace = e_up + e_dn + 2.0 * MU_B_PER_TESLA * b * m_j
            assert up + lo == pytest.approx(trace, rel=1e-12)

def test_stretched_states_linear():
    # |m_j| = l+1/2 (and all l=0) are exactly linear in B: second difference ~ 0.
    def stretched(b):
        return _by_mj(zeeman_sublevels(2, 1, b_tesla=b), 1.5, "single").energy.value
    a, mid, c = stretched(0.0), stretched(1.0), stretched(2.0)
    assert (a - 2 * mid + c) == pytest.approx(0.0, abs=1e-18)

def test_dirac_diagonal_zero_field():
    subs = zeeman_sublevels(2, 1, b_tesla=0.0, dirac=True)
    for s in subs:
        assert s.energy.value == pytest.approx(dirac_energy(2, s.j_label).value, rel=1e-12)

def test_provenance_tiers_and_error():
    real = zeeman_sublevels(2, 1, b_tesla=2.0)[0].energy.provenance
    assert real.fidelity is Fidelity.APPROXIMATION
    assert real.error_estimate is not None and real.error_estimate > 0.0
    # error grows with B (diamagnetic ~ B^2).
    e_small = zeeman_sublevels(2, 1, b_tesla=1.0)[0].energy.provenance.error_estimate
    e_big = zeeman_sublevels(2, 1, b_tesla=10.0)[0].energy.provenance.error_estimate
    assert e_big > e_small
    altered = zeeman_sublevels(2, 1, b_tesla=2.0, alpha=ALPHA * 1.5)[0].energy.provenance
    assert altered.fidelity is Fidelity.COUNTERFACTUAL

def test_negative_field_rejected():
    with pytest.raises(ValueError):
        zeeman_sublevels(2, 1, b_tesla=-1.0)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_zeeman.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.analytic.zeeman'`.

- [ ] **Step 4: Implement the module**

Create `src/atomsim/analytic/zeeman.py`:

```python
"""Zeeman effect: fine structure + linear Zeeman, the Breit-Rabi crossover.

For a shell n, each (l, m_j) with both j = l +/- 1/2 present forms a 2x2 block
(fine-structure diagonal + linear-Zeeman coupling via <S_z>); stretched states
(|m_j| = l+1/2) and all of l=0 are 1x1 blocks, exactly linear in B. The 2x2
eigenvalues are the closed-form Breit-Rabi roots, so no numerical eigensolver is
needed and the result is exact-of-the-model with zero numerical error.

APPROXIMATION by construction: the linear-Zeeman model omits the diamagnetic B^2
term and uses g_s = 2. COUNTERFACTUAL when alpha is altered. See docs/
specs/2026-07-23-phase10-zeeman-field-design.md.
"""

import math
from dataclasses import dataclass

from atomsim.analytic.dirac import dirac_energy
from atomsim.analytic.fine_structure import level_energy
from atomsim.analytic.hydrogen import validate_quantum_numbers
from atomsim.constants import ALPHA, B0_TESLA
from atomsim.provenance import Fidelity, Provenance, Quantity

_MU_B_AU = 0.5  # Bohr magneton in atomic units (e = hbar = m_e = 1)
MU_B_PER_TESLA = _MU_B_AU / B0_TESLA  # hartree per tesla, prefactor on (J_z + S_z)
_G2 = 2.0 * 0.00116  # anomalous-moment scale on the spin Zeeman part

_Z_ASSUMPTIONS = (
    "linear (paramagnetic) Zeeman only; diamagnetic B^2 term neglected",
    "electron g_s = 2 exactly (anomalous moment ~0.1% of the spin part neglected)",
    "coupling to other n manifolds neglected",
    "diagonal from the selected level model (alpha^2 fine structure or exact Dirac)",
)

@dataclass(frozen=True)
class ZeemanSublevel:
    m_j: float
    branch: str            # "upper" | "lower" | "single"
    j_label: float         # low-field good quantum number (the j at B=0)
    high_field_label: str  # (m_l, m_s) the state approaches at large B
    energy: Quantity

def lande_g(l: int, j: float) -> float:
    """Lande g-factor for a one-electron (s = 1/2) state."""
    s = 0.5
    return 1.0 + (j * (j + 1.0) + s * (s + 1.0) - l * (l + 1.0)) / (2.0 * j * (j + 1.0))

def _high_field_label(m_j: float, m_s: float) -> str:
    return f"m_l={m_j - m_s:g}, m_s={m_s:+g}"

def _mean_sq_radius(n: int, l: int, Z: int) -> float:
    """<r^2> in bohr^2 for a hydrogenic (n,l) state (diamagnetic scale)."""
    return (n * n / (2.0 * Z * Z)) * (5.0 * n * n + 1.0 - 3.0 * l * (l + 1.0))

def zeeman_sublevels(
    n: int, l: int, Z: int = 1, mu_ratio: float = 1.0, m_over_M: float = 0.0,
    alpha: float = ALPHA, b_tesla: float = 0.0, dirac: bool = False,
) -> list[ZeemanSublevel]:
    """Breit-Rabi sublevels of the (n, l) shell in a field B (tesla)."""
    validate_quantum_numbers(n, l)
    if Z < 1:
        raise ValueError(f"Z must be >= 1, got {Z}")
    if b_tesla < 0:
        raise ValueError(f"b_tesla must be >= 0, got {b_tesla}")

    def diag(j: float) -> Quantity:
        if dirac:
            return dirac_energy(n, j, Z=Z, mu_ratio=mu_ratio, alpha=alpha)
        return level_energy(
            n, l, j, Z=Z, mu_ratio=mu_ratio, m_over_M=m_over_M, alpha=alpha
        )

    muB_b = MU_B_PER_TESLA * b_tesla  # hartree
    altered = not math.isclose(alpha, ALPHA, rel_tol=1e-12)
    fidelity = Fidelity.COUNTERFACTUAL if altered else Fidelity.APPROXIMATION
    diamag = 0.125 * (b_tesla / B0_TESLA) ** 2 * _mean_sq_radius(n, l, Z)
    denom = 2 * l + 1
    method = (
        "Breit-Rabi (fine structure + linear Zeeman, g_s=2) eigenvalue; "
        f"mu_B*B = {muB_b:.3e} hartree at B = {b_tesla:g} T"
        + ("; exact Dirac diagonal split by a perturbative linear-Zeeman model" if dirac else "")
        + (f"; altered alpha = {alpha:g} (real {ALPHA:g})" if altered else "")
    )

    def make(value: float, m_j: float, branch: str, j_label: float, m_s: float,
             underlying_err: float | None) -> ZeemanSublevel:
        err = (underlying_err or 0.0) + diamag + _G2 * abs(muB_b * m_j)
        return ZeemanSublevel(
            m_j=m_j, branch=branch, j_label=j_label,
            high_field_label=_high_field_label(m_j, m_s),
            energy=Quantity(
                value=value, unit="hartree",
                label=f"E_Zeeman {n},{l},m_j={m_j:g},{branch} (B={b_tesla:g}T)",
                provenance=Provenance(
                    fidelity=fidelity, method=method, assumptions=_Z_ASSUMPTIONS,
                    error_estimate=err,
                    refinement="diamagnetic (B^2) term, then Paschen-Back beyond the two-effect model",
                ),
            ),
        )

    j_up = l + 0.5
    e_up = diag(j_up)
    m_values = [(-(l + 0.5) + k) for k in range(2 * l + 2)]  # -(l+1/2) .. +(l+1/2)
    out: list[ZeemanSublevel] = []
    for m_j in m_values:
        stretched = abs(m_j) > l  # |m_j| == l+1/2
        if l == 0 or stretched:
            # 1x1 block, only j = l+1/2, exactly linear in B.
            zeeman_diag = muB_b * m_j * (2 * l + 2) / denom
            m_s = math.copysign(0.5, m_j)
            out.append(make(
                e_up.value + zeeman_diag, m_j, "single", j_up, m_s,
                e_up.provenance.error_estimate,
            ))
            continue
        # 2x2 block for j = l+1/2 (upper) and j = l-1/2 (lower).
        e_dn = diag(l - 0.5)
        h00 = e_up.value + muB_b * m_j * (2 * l + 2) / denom
        h11 = e_dn.value + muB_b * m_j * (2 * l) / denom
        h01 = muB_b * math.sqrt((l + 0.5) ** 2 - m_j * m_j) / denom
        mean = 0.5 * (h00 + h11)
        disc = math.hypot(0.5 * (h00 - h11), h01)
        out.append(make(  # upper -> state A: (m_l=m_j-1/2, m_s=+1/2)
            mean + disc, m_j, "upper", j_up, 0.5, e_up.provenance.error_estimate,
        ))
        out.append(make(  # lower -> state B: (m_l=m_j+1/2, m_s=-1/2)
            mean - disc, m_j, "lower", l - 0.5, -0.5, e_dn.provenance.error_estimate,
        ))
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_zeeman.py -q`
Expected: PASS (10 tests).

- [ ] **Step 6: Lint**

Run: `"C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check src/atomsim/analytic/zeeman.py src/atomsim/constants.py tests/test_zeeman.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/atomsim/analytic/zeeman.py src/atomsim/constants.py tests/test_zeeman.py
git commit -m "Add the closed-form Breit-Rabi Zeeman engine (fine structure + linear field)"
```

---

### Task 2: Server: `/api/levels` Zeeman mode

**Files:**
- Modify: `src/atomsim/server/app.py` (schemas near lines 98-115; endpoint 375-457)
- Test: `tests/test_server.py` (append)

**Interfaces:**
- Consumes: `zeeman_sublevels(...) -> list[ZeemanSublevel]`, `MU_B_PER_TESLA` (Task 1); `QuantityModel.from_quantity`, `_to_ev` (existing in `app.py`).
- Produces: `/api/levels?...&b_field=<T>` returning `FineLevelModel.sublevels: list[ZeemanSublevelModel] | None`; `LevelsResponse.b_field: float`.

- [ ] **Step 1: Write the failing server tests**

Append to `tests/test_server.py`:

```python
def test_levels_zeeman_splits_fine_levels(client):
    r = client.get("/api/levels?system=h&n_max=3&fine_structure=true&b_field=2")
    assert r.status_code == 200
    body = r.json()
    assert body["b_field"] == 2.0
    # A 2p level (l=1) fans into its m_j sublevels.
    p = next(f for f in body["fine"] if f["n"] == 2 and f["l"] == 1 and f["j"] == 1.5)
    assert p["sublevels"] is not None and len(p["sublevels"]) == 4  # m_j = +-3/2, +-1/2
    s0 = p["sublevels"][0]
    assert s0["energy"]["provenance"]["fidelity"] == "approximation"
    assert "m_l" in s0["high_field_label"]

def test_levels_zeeman_absent_without_field(client):
    r = client.get("/api/levels?system=h&n_max=2&fine_structure=true")
    body = r.json()
    assert body["b_field"] == 0.0
    assert all(f.get("sublevels") is None for f in body["fine"])

def test_levels_zeeman_negative_field_rejected(client):
    r = client.get("/api/levels?system=h&n_max=2&fine_structure=true&b_field=-1")
    assert r.status_code == 422

def test_levels_zeeman_ignored_for_screened(client):
    r = client.get("/api/levels?system=he&fine_structure=true&b_field=5")
    assert r.status_code == 200
    assert "orbitals" in r.json()  # ScreenedLevelsModel, no sublevels
```

(If the test-file `client` fixture has a different name, match the existing tests in `tests/test_server.py`.)

- [ ] **Step 2: Run to verify failure**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_server.py -q -k zeeman`
Expected: FAIL (`KeyError: 'b_field'` / missing `sublevels`).

- [ ] **Step 3: Add the schema models**

In `src/atomsim/server/app.py`, add `ZeemanSublevelModel` immediately before `class FineLevelModel` (line 98) and extend `FineLevelModel` and `LevelsResponse`:

```python
class ZeemanSublevelModel(BaseModel):
    m_j: float
    branch: str
    j_label: float
    high_field_label: str
    energy: QuantityModel
    energy_ev: QuantityModel

class FineLevelModel(BaseModel):
    n: int
    l: int
    j: float
    energy: QuantityModel
    energy_ev: QuantityModel
    shift: QuantityModel
    shift_ev: QuantityModel
    sublevels: list[ZeemanSublevelModel] | None = None
```

And in `LevelsResponse` add after `dirac`:

```python
    b_field: float = 0.0
```

- [ ] **Step 4: Add the import**

At the top of `app.py`, alongside line 18-19 imports, add:

```python
from atomsim.analytic.zeeman import zeeman_sublevels
```

- [ ] **Step 5: Wire the endpoint**

Change the endpoint signature (line 376-380) to add `b_field`:

```python
    def levels_endpoint(system: str = "h", n_max: int = 6,
                        fine_structure: bool = False,
                        alpha: float | None = None,
                        config: str | None = None,
                        dirac: bool = False,
                        b_field: float = 0.0):
```

After the `alpha` validation (line 405-406), add:

```python
        if b_field < 0.0:
            raise HTTPException(status_code=422, detail="b_field must be >= 0")
```

Inside the `fine` build loop, after each `fine.append(FineLevelModel(...))` (line 446-452), the append must include the sublevels. Replace the `fine.append(...)` block with:

```python
                        subs = None
                        if b_field > 0.0:
                            zss = zeeman_sublevels(
                                n, l, Z=sys_.Z, mu_ratio=mu, m_over_M=sys_.m_over_M,
                                alpha=alpha_used, b_tesla=b_field, dirac=dirac,
                            )
                            subs = [
                                ZeemanSublevelModel(
                                    m_j=z.m_j, branch=z.branch, j_label=z.j_label,
                                    high_field_label=z.high_field_label,
                                    energy=QuantityModel.from_quantity(z.energy),
                                    energy_ev=QuantityModel.from_quantity(_to_ev(z.energy)),
                                )
                                for z in zss
                            ]
                        fine.append(FineLevelModel(
                            n=n, l=l, j=j,
                            energy=QuantityModel.from_quantity(le),
                            energy_ev=QuantityModel.from_quantity(_to_ev(le)),
                            shift=QuantityModel.from_quantity(sh),
                            shift_ev=QuantityModel.from_quantity(_to_ev(sh)),
                            sublevels=subs,
                        ))
```

Finally add `b_field=b_field,` to the `LevelsResponse(...)` return (after `dirac=dirac,`).

- [ ] **Step 6: Run the server tests**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_server.py -q -k zeeman`
Expected: PASS (4 tests).

- [ ] **Step 7: Full engine+server suite + lint**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest -q` then `"C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check .`
Expected: all pass, no lint errors.

- [ ] **Step 8: Commit**

```bash
git add src/atomsim/server/app.py tests/test_server.py
git commit -m "Serve Zeeman sublevels from /api/levels behind a b_field flag"
```

---

### Task 3: Web: store, client, types, URL round-trip

**Files:**
- Modify: `web/src/api/types.ts` (FineLevel line 84-92; LevelsResponse 94-102)
- Modify: `web/src/api/client.ts` (getLevels 67-81)
- Modify: `web/src/state/store.ts` (fields ~50/150; setters ~200; loadLevels ~352)
- Modify: `web/src/lib/urlState.ts` (UrlState ~26; URL_DEFAULTS ~42; parse ~148; serialize ~212)
- Modify: `web/src/main.tsx`
- Test: `web/src/lib/urlState.test.ts`, `web/src/state/store.test.ts` (append)

**Interfaces:**
- Consumes: server `b_field` + `sublevels` (Task 2).
- Produces: store `bField: number` / `setBField`; `getLevels(..., bField)`; URL `b` param.

- [ ] **Step 1: Write the failing web tests**

Append to `web/src/lib/urlState.test.ts`:

```ts
it("round-trips the b_field with fine structure", () => {
  const url = serializeAppUrl({ ...URL_DEFAULTS, fineStructure: true, bField: 2.5 });
  expect(url).toContain("b=2.5");
  expect(parseAppUrl(url).bField).toBe(2.5);
});

it("omits b when field is zero", () => {
  const url = serializeAppUrl({ ...URL_DEFAULTS, fineStructure: true, bField: 0 });
  expect(url).not.toContain("b=");
});

it("omits b when fine structure is off", () => {
  const url = serializeAppUrl({ ...URL_DEFAULTS, fineStructure: false, bField: 3 });
  expect(url).not.toContain("b=");
});
```

Append to `web/src/state/store.test.ts` (match the existing store-test style/imports):

```ts
it("setBField clears cached levels", () => {
  useStore.setState({ levels: { fake: true } as never, bField: 0 });
  useStore.getState().setBField(4);
  expect(useStore.getState().bField).toBe(4);
  expect(useStore.getState().levels).toBeNull();
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd web && npx vitest run src/lib/urlState.test.ts src/state/store.test.ts`
Expected: FAIL (`bField` not on type / setter missing).

- [ ] **Step 3: Extend the API types**

In `web/src/api/types.ts`, add before `FineLevel`:

```ts
export interface ZeemanSublevel {
  m_j: number;
  branch: string;
  j_label: number;
  high_field_label: string;
  energy: Quantity;
  energy_ev: Quantity;
}
```

Add `sublevels?: ZeemanSublevel[] | null;` to `FineLevel`, and `b_field: number;` to `LevelsResponse`.

- [ ] **Step 4: Thread `bField` through the client**

In `web/src/api/client.ts`, change `getLevels` to take `bField` and add it to the query:

```ts
export function getLevels(
  system: string,
  nMax: number,
  fineStructure: boolean,
  alpha?: number,
  config?: string | null,
  dirac = false,
  bField = 0,
): Promise<LevelsResponse | ScreenedLevels> {
  const a = alpha === undefined ? "" : `&alpha=${alpha}`;
  const c = config ? `&config=${encodeURIComponent(config)}` : "";
  const d = dirac ? "&dirac=true" : "";
  const b = bField > 0 ? `&b_field=${bField}` : "";
  return getJson(
    `/api/levels?system=${system}&n_max=${nMax}&fine_structure=${fineStructure}${a}${c}${d}${b}`,
  );
}
```

- [ ] **Step 5: Add store state + setter + loadLevels wiring**

In `web/src/state/store.ts`: add `bField: number;` to the state interface near `dirac: boolean;` (line 50), `setBField: (bField: number) => void;` near `setDirac` (line 109), the default `bField: 0,` near `dirac: false,` (line 150), the setter near line 200:

```ts
  setBField: (bField) => set({ bField, levels: null }),
```

and in `loadLevels` (line 352-356) pull `bField` and pass it:

```ts
  loadLevels: async () => {
    const { system, fineStructure, config, dirac, bField } = get();
    set({
      levels: await client.getLevels(
        system, N_MAX_DIAGRAM, fineStructure, undefined, config, dirac, bField,
      ),
    });
  },
```

- [ ] **Step 6: Add URL parse/serialize**

In `web/src/lib/urlState.ts`: add `bField: number;` to `UrlState` (near line 27), `bField: 0,` to `URL_DEFAULTS` (near line 51). In the parse function after the `dirac` parse (line 148) add:

```ts
  const b = Number(q.get("b"));
  if (Number.isFinite(b) && b > 0) out.bField = b;
```

In serialize after the `dirac` line (212) add:

```ts
  if (state.bField > 0 && state.fineStructure) q.set("b", String(state.bField));
```

- [ ] **Step 7: Serialize from main.tsx**

In `web/src/main.tsx`, in the object passed to `serializeAppUrl` (where `dirac: s.dirac,` is), add `bField: s.bField,`.

- [ ] **Step 8: Run the web tests**

Run: `cd web && npx vitest run src/lib/urlState.test.ts src/state/store.test.ts`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add web/src/api/types.ts web/src/api/client.ts web/src/state/store.ts web/src/lib/urlState.ts web/src/main.tsx web/src/lib/urlState.test.ts web/src/state/store.test.ts
git commit -m "Wire the Zeeman b_field through the client, store, and deep links"
```

---

### Task 4: Levels view: B slider, sublevel fan, caption + build

**Files:**
- Modify: `web/src/components/LevelsView.tsx`
- Verify: full build + suites

**Interfaces:**
- Consumes: store `bField`/`setBField`; `FineLevel.sublevels`; `MU_B_PER_TESLA` value for the µeV readout (recompute in TS: `0.5 / 2.35051757e5` hartree/T, × 27.211386245 × 1e6 µeV/hartree).

- [ ] **Step 1: Refetch on field change**

Add `bField` to the `useEffect` dependency array (line 81-84) so changing B refetches:

```tsx
  }, [system, fineStructure, dirac, bField, loadLevels, loadSpectrum]);
```

and pull `bField, setBField` from the store destructure (line 79).

- [ ] **Step 2: Add the B slider to the header**

When `fineStructure` is on, render, next to the existing `levels-model` control (line 109-112), a slider with a live µ_B·B µeV readout. When fine structure is off, render it disabled with a hint. Add:

```tsx
        {fineStructure && (
          <label className="levels-field">
            B{" "}
            <input
              type="range" min={0} max={20} step={0.1} value={bField}
              onChange={(e) => setBField(Number(e.target.value))}
            />
            {bField > 0
              ? ` ${bField.toFixed(1)} T (µ_B·B = ${(bField * 0.5 / 2.35051757e5 * 27.211386245e6).toFixed(1)} µeV)`
              : " 0 T"}
          </label>
        )}
        {!fineStructure && (
          <span className="levels-field-hint">turn on fine structure to add a field</span>
        )}
```

- [ ] **Step 3: Fan the sublevels in the zoomed fine column**

In the zoomed fine-structure block (line 150-162), when `bField > 0` and a level has `sublevels`, draw each sublevel as a short horizontal tick offset from the parent line by its energy, with a faint connector, and label the extreme mⱼ. Follow the existing SVG/positioning idiom in that block (reuse the same y-scaling the fine shifts use, applied to `sublevel.energy_ev.value` relative to the parent `energy_ev.value`). Keep the parent line visible as the B=0 reference.

- [ ] **Step 4: Update the caption**

Extend the caption (line 184-188) so that when `bField > 0` it names the crossover:

```tsx
        {bField > 0 && (
          <>
            {" "}A magnetic field splits each j-level into 2j+1 m_j sublevels (anomalous
            Zeeman, spacing g_J·µ_B·B); as B rises they reorganize toward the Paschen-Back
            pattern where (m_l, m_s) become the good labels. Linear model, the diamagnetic
            B² term is omitted.
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

Start `atomsim serve --port 8023 --no-browser` (background). GET `/api/levels?system=h&n_max=3&fine_structure=true&b_field=2` → `b_field:2.0`, a 2p level has 4 `sublevels`, first sublevel fidelity `approximation`. GET with `b_field=-1` → 422. GET `?system=h&n_max=3&fine_structure=true&dirac=true&b_field=2` → sublevels present, method mentions Dirac. Stop the server.

- [ ] **Step 8: Commit**

```bash
git add web/src/components/LevelsView.tsx
git commit -m "Add the Zeeman B-field slider and Breit-Rabi fan to the Levels view"
```

---

## Self-review notes

- **Spec coverage:** §2 physics → Task 1 (block math, closed-form roots, both labels); §2.2 fidelity → Task 1 provenance + error; §2.3 Dirac diagonal → Task 1 `dirac` param + test; §4 engine → Task 1; §5 server → Task 2; §6 web store/client/URL → Task 3, view → Task 4; §9 tests distributed across all tasks. §8 deferrals untasked on purpose.
- **Type consistency:** `zeeman_sublevels`, `ZeemanSublevel(m_j, branch, j_label, high_field_label, energy)`, `MU_B_PER_TESLA`, `lande_g`, `ZeemanSublevelModel`, `FineLevelModel.sublevels`, `LevelsResponse.b_field`, store `bField`/`setBField`, `getLevels(..., bField)`, URL `b`, TS `ZeemanSublevel`/`FineLevel.sublevels`/`LevelsResponse.b_field` used identically across tasks.
- **Number checks:** `MU_B_PER_TESLA = 0.5/2.35051757e5 = 2.127e-6` hartree/T = 57.9 µeV/T. n=2 fine-structure gap ≈ 45 µeV → crossover near B ≈ 0.8 T, inside the 0-20 T slider range. `lande_g(1,1.5)=4/3`, `lande_g(1,0.5)=2/3`. Shell (n,l≥1) sublevel count = 4l+2; l=0 → 2.
