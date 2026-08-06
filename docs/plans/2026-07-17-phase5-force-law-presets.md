# Phase 5: Force-Law Preset Library Implementation Plan

**Goal:** Generalize the phase-4 single power-law force law into a five-preset curated library (power-law, Yukawa/screened, harmonic, finite well, Coulomb-plus-core), each solved by the existing numerical radial solver, each drawn against its own honest reference, with a potential-well diagram plus an energy-ladder toggle.

**Architecture:** Refactor the hardcoded `force_law.py` driver into a **preset registry**, each preset supplies a `V(r)` builder, a parameter schema, a binding rule, an honest reference builder, and a curve range. One generalized `GET /api/forcelaw` serves all five (back-compatible with the phase-4 `?p=` links). The frontend gains a preset picker, dynamic per-preset parameter controls, and a potential-well diagram that draws the returned `V(r)` curve with bound levels and reference overlay.

**Tech Stack:** Python 3.12 (NumPy, SciPy, FastAPI, Pydantic, pytest), React + TypeScript + Zustand + d3-scale (vitest).

## Global Constraints

- Engine-internal math is in **Hartree atomic units**; eV/pm display conversion happens only at the server boundary and appends to the provenance `method`.
- **The prime directive:** every physical value crossing a module boundary is a `Quantity`/`Field` carrying a `Provenance` with a `Fidelity` tier. No bare floats, no silent zeros, no undisclosed liberties.
- `l` is the orbital quantum number, not a length (ruff E741 ignored project-wide). Keep the physics naming.
- Line length 100. `ruff check .` must stay clean.
- New physics gets a **validation test** (analytic ground truth / independent method), not a smoke test.
- `web/dist` is only rebuilt by `npm run build`; the server mounts it only if it exists.
- Python tests in this environment require the MKL sequential-BLAS workaround: prefix pytest with `MKL_THREADING_LAYER=SEQUENTIAL` (PowerShell: `$env:MKL_THREADING_LAYER="SEQUENTIAL"`). Env python: `C:\Users\yashg\.conda\envs\atomsim\python.exe`.
- Bound-state rule is **per binding kind**: potentials that vanish at infinity ("decay": power-law, Yukawa, finite well, Coulomb-core) are bound iff `E < 0`; the harmonic oscillator is "confining", all states bound, `E > 0`. Never apply an `E < 0` filter to a confining preset.
- **Back-compat contract:** `GET /api/forcelaw` with no `preset` behaves exactly as phase 4 (`powerlaw`, reading `p`); existing `?view=forcelaw&p=…&fl=…` deep links keep working.

---

## File Structure

**Backend**
- Create `src/atomsim/analytic/oscillator.py`, EXACT 3-D isotropic QHO levels `E = ω(2k + l + 3/2)`.
- Modify `src/atomsim/numerics/force_law.py`, preset registry, binding-aware bound filter, per-preset reference, potential curve.
- Modify `src/atomsim/server/schemas.py`, generalized force-law response models.
- Modify `src/atomsim/server/app.py`, generalized `/api/forcelaw`.
- Create `tests/test_oscillator.py`; extend `tests/test_force_law.py`, `tests/test_server.py`.

**Frontend**
- Modify `web/src/api/types.ts`, new force-law response shape.
- Modify `web/src/api/client.ts`, `getForceLaw(preset, params, l, system)`.
- Create `web/src/lib/forceLaw.ts`, preset parameter specs (single source of ranges/defaults/labels) + classically-allowed-span helper; test `web/src/lib/forceLaw.test.ts`.
- Modify `web/src/state/store.ts`, force slice (preset, params, `forceViz`).
- Modify `web/src/lib/urlState.ts`, preset + params serialization.
- Modify `web/src/components/ForceLawView.tsx`, preset picker, dynamic params, well diagram + ladder toggle.
- Extend `web/src/state/store.test.ts`, `web/src/lib/urlState.test.ts`.

---

## Task 1: 3-D isotropic harmonic-oscillator exact levels

**Files:**
- Create: `src/atomsim/analytic/oscillator.py`
- Test: `tests/test_oscillator.py`

**Interfaces:**
- Consumes: `atomsim.provenance.{Fidelity, Provenance, Quantity}`.
- Produces:
  - `oscillator_energy(k: int, l: int, omega: float) -> Quantity`, EXACT, hartree, value `omega * (2*k + l + 1.5)`.
  - `oscillator_levels(omega: float, l: int, n_states: int) -> tuple[Quantity ...]`, `k = 0 .. n_states-1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oscillator.py
import math

import pytest

from atomsim.analytic.oscillator import oscillator_energy, oscillator_levels
from atomsim.provenance import Fidelity

def test_ground_state_is_three_halves_omega():
    q = oscillator_energy(k=0, l=0, omega=0.5)
    assert q.unit == "hartree"
    assert q.provenance.fidelity is Fidelity.EXACT
    assert math.isclose(q.value, 0.5 * 1.5, rel_tol=1e-12)

@pytest.mark.parametrize("k,l,omega,expected", [
    (0, 0, 0.5, 0.75),
    (1, 0, 0.5, 1.75),
    (0, 1, 0.5, 1.25),
    (2, 3, 0.3, 0.3 * (4 + 3 + 1.5)),
])
def test_level_formula(k, l, omega, expected):
    assert math.isclose(oscillator_energy(k, l, omega).value, expected, rel_tol=1e-12)

def test_levels_are_ascending_and_counted():
    levels = oscillator_levels(omega=0.4, l=1, n_states=4)
    assert len(levels) == 4
    values = [q.value for q in levels]
    assert values == sorted(values)
    assert math.isclose(values[0], 0.4 * (0 + 1 + 1.5), rel_tol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_oscillator.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.analytic.oscillator'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/atomsim/analytic/oscillator.py
"""3-D isotropic harmonic oscillator: exact bound-state energies (EXACT).

For V(r) = 1/2 mu omega^2 r^2 the radial spectrum is closed form,
E = omega (2 k + l + 3/2) in Hartree atomic units (hbar = 1), where k is the
radial node count. Independent of the Coulomb formulas, this is a second exact
ground truth for the numerical radial solver, see tests/test_force_law.py.
"""

from atomsim.provenance import Fidelity, Provenance, Quantity

_PROV = Provenance(
    fidelity=Fidelity.EXACT,
    method="3-D isotropic harmonic oscillator closed form E = omega(2k + l + 3/2)",
    assumptions=("Hartree atomic units, hbar = 1",),
)

def oscillator_energy(k: int, l: int, omega: float) -> Quantity:
    if k < 0:
        raise ValueError(f"radial index k must be >= 0, got {k}")
    if l < 0:
        raise ValueError(f"orbital quantum number l must be >= 0, got {l}")
    if not omega > 0:
        raise ValueError(f"omega must be positive, got {omega}")
    return Quantity(
        value=omega * (2 * k + l + 1.5),
        unit="hartree",
        label=f"E_osc[k={k}, l={l}]",
        provenance=_PROV,
    )

def oscillator_levels(omega: float, l: int, n_states: int) -> tuple[Quantity, ...]:
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")
    return tuple(oscillator_energy(k, l, omega) for k in range(n_states))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_oscillator.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/atomsim/analytic/oscillator.py tests/test_oscillator.py
git commit -m "Add exact 3-D isotropic harmonic-oscillator levels"
```

---

## Task 2: Preset registry scaffold + power-law parity

Refactor `force_law.py` from one hardcoded shape into a registry, implement the `powerlaw` preset reproducing phase-4 numbers exactly, and migrate the existing tests to the new signature (same asserted values).

**Files:**
- Modify: `src/atomsim/numerics/force_law.py` (full rewrite of the driver; keep `P_MIN`/`P_MAX`)
- Modify: `tests/test_force_law.py` (migrate call sites to new signature)

**Interfaces:**
- Consumes: `solve_radial_with_error` (`numerics/radial_solver.py`), `hydrogen_energy` (`analytic/hydrogen.energy`), `atomsim.systems.{System, get_system}`, `atomsim.provenance.{Field, Fidelity, Provenance, Quantity}`.
- Produces:
  - `ParamSpec(name, min, max, default, unit)` (frozen dataclass).
  - `ReferenceItem(label: str, energy: Quantity)`, `Reference(kind: str, items: tuple[ReferenceItem, ...])`.
  - `ForceLawLevel(radial_index: int, energy: Quantity)`.
  - `ForceLawResult(preset_key, params: dict[str,float], l, z, system_key, counterfactual: tuple[ForceLawLevel,...], bound_count: int, requested_count: int, reference: Reference, potential_curve: Field)`.
  - `ForcePreset(key, params: tuple[ParamSpec,...], uses_Z: bool, binding: str, build_potential, reference, r_max)`.
  - `PRESETS: dict[str, ForcePreset]`.
  - `force_law_levels(preset: str, params: dict[str, float], l: int, system: str | System = "h", n_states: int = 4) -> ForceLawResult`.
  - `P_MIN = 0.5`, `P_MAX = 1.5` (unchanged, re-exported).
  - `CURVE_POINTS = 256`.

- [ ] **Step 1: Write the failing test** (append/replace in `tests/test_force_law.py`, power-law block migrated to the new API)

```python
# tests/test_force_law.py  (power-law parity block)
import math

import numpy as np
import pytest

from atomsim.analytic.hydrogen import energy as hydrogen_energy
from atomsim.numerics.force_law import PRESETS, force_law_levels
from atomsim.provenance import Fidelity

def test_powerlaw_p1_matches_exact_hydrogen():
    res = force_law_levels("powerlaw", {"p": 1.0}, l=0, system="h", n_states=3)
    assert res.preset_key == "powerlaw"
    assert res.bound_count == 3 and res.requested_count == 3
    for k, level in enumerate(res.counterfactual):
        n = k + 1  # l = 0
        exact = hydrogen_energy(n, Z=1, mu_ratio=1.0).value
        assert level.energy.provenance.fidelity is Fidelity.NUMERICAL
        assert math.isclose(level.energy.value, exact, rel_tol=2e-4)

def test_powerlaw_reference_is_exact_hydrogen_ladder():
    res = force_law_levels("powerlaw", {"p": 1.2}, l=1, system="h", n_states=3)
    assert res.reference.kind == "levels"
    assert [item.label for item in res.reference.items] == ["n=2", "n=3", "n=4"]
    assert all(i.energy.provenance.fidelity is Fidelity.EXACT for i in res.reference.items)

def test_powerlaw_degeneracy_breaks_off_p1():
    s = force_law_levels("powerlaw", {"p": 1.2}, l=0, system="h", n_states=2)
    p = force_law_levels("powerlaw", {"p": 1.2}, l=1, system="h", n_states=1)
    e_2s = s.counterfactual[1].energy.value  # (l=0, k=1) -> n=2
    e_2p = p.counterfactual[0].energy.value  # (l=1, k=0) -> n=2
    assert abs(e_2s - e_2p) > 1e-4
    assert e_2s < e_2p  # p > 1 (harder): s below p (alkali ordering)

def test_powerlaw_out_of_range_p_raises():
    with pytest.raises(ValueError, match="p"):
        force_law_levels("powerlaw", {"p": 1.9}, l=0)

def test_potential_curve_is_field_in_hartree():
    res = force_law_levels("powerlaw", {"p": 1.0}, l=0, system="h", n_states=2)
    curve = res.potential_curve
    assert curve.values.shape == curve.grid.shape
    assert curve.unit == "hartree" and curve.grid_unit == "bohr"
    assert curve.provenance.fidelity is Fidelity.EXACT
    assert np.all(curve.grid > 0)

def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown preset"):
        force_law_levels("nope", {}, l=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_force_law.py -v`
Expected: FAIL, `force_law_levels()` old signature / missing `PRESETS`, `preset_key`, `bound_count`.

- [ ] **Step 3: Write the implementation** (full rewrite of `src/atomsim/numerics/force_law.py`)

```python
"""What-If force laws: energy levels under a counterfactual central potential.

A preset registry drives the numerical radial solver with different V(r) shapes,
pairs each with an honest per-preset reference (EXACT hydrogen, EXACT harmonic-
oscillator levels, or structural markers), and returns a sampled V(r) curve so
the view can draw the potential itself. See docs/specs/
2026-07-17-phase5-force-law-presets-design.md.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from atomsim.analytic.hydrogen import energy as hydrogen_energy
from atomsim.analytic.oscillator import oscillator_energy
from atomsim.numerics.radial_solver import solve_radial_with_error
from atomsim.provenance import Field, Fidelity, Provenance, Quantity
from atomsim.systems import System, get_system

P_MIN = 0.5
P_MAX = 1.5
CURVE_POINTS = 256

Params = dict[str, float]
PotentialFn = Callable[[np.ndarray], np.ndarray]

@dataclass(frozen=True)
class ParamSpec:
    name: str
    min: float
    max: float
    default: float
    unit: str

@dataclass(frozen=True)
class ReferenceItem:
    label: str
    energy: Quantity  # EXACT, hartree

@dataclass(frozen=True)
class Reference:
    kind: str  # "levels" | "markers"
    items: tuple[ReferenceItem, ...]

@dataclass(frozen=True)
class ForceLawLevel:
    radial_index: int
    energy: Quantity  # NUMERICAL, hartree, carries grid-halving error

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

@dataclass(frozen=True)
class ForcePreset:
    key: str
    params: tuple[ParamSpec, ...]
    uses_Z: bool
    binding: str  # "decay" (bound iff E<0) | "confining" (all bound)
    build_potential: Callable[[Params, int, float], PotentialFn]
    reference: Callable[[Params, int, float, int, int], Reference]
    r_max: Callable[[Params, int, int], float]

# ---- hydrogen-ladder reference shared by the Coulomb-family presets ----------

def _hydrogen_reference(params: Params, z: int, mu: float, l: int, n_states: int) -> Reference:
    items = tuple(
        ReferenceItem(
            label=f"n={l + 1 + k}",
            energy=hydrogen_energy(l + 1 + k, Z=z, mu_ratio=mu),
        )
        for k in range(n_states)
    )
    return Reference(kind="levels", items=items)

# ---- power-law ---------------------------------------------------------------

def _powerlaw_potential(params: Params, z: int, mu: float) -> PotentialFn:
    p = params["p"]
    return lambda r: -z / r**p

def _powerlaw_rmax(params: Params, z: int, n_states: int) -> float:
    return 20.0 * (n_states + 1) ** 2 / z

POWERLAW = ForcePreset(
    key="powerlaw",
    params=(ParamSpec("p", P_MIN, P_MAX, 1.0, ""),),
    uses_Z=True,
    binding="decay",
    build_potential=_powerlaw_potential,
    reference=_hydrogen_reference,
    r_max=_powerlaw_rmax,
)

PRESETS: dict[str, ForcePreset] = {POWERLAW.key: POWERLAW}

# ---- driver ------------------------------------------------------------------

def _validate(preset: ForcePreset, params: Params, l: int, n_states: int) -> None:
    if l < 0:
        raise ValueError(f"orbital quantum number l must be >= 0, got {l}")
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")
    for spec in preset.params:
        if spec.name not in params:
            raise ValueError(f"preset {preset.key!r} requires parameter {spec.name!r}")
        v = params[spec.name]
        if not spec.min <= v <= spec.max:
            raise ValueError(
                f"{spec.name} must be in [{spec.min}, {spec.max}], got {v}"
            )

def _bound(preset: ForcePreset, energy: Quantity) -> bool:
    return preset.binding == "confining" or energy.value < 0.0

def _tag(energy: Quantity, note: str) -> Quantity:
    return replace(
        energy,
        provenance=replace(energy.provenance, method=energy.provenance.method + note),
    )

def _sample_curve(potential: PotentialFn, r_max: float, note: str) -> Field:
    r = np.linspace(r_max / CURVE_POINTS, r_max, CURVE_POINTS)
    v = np.asarray(potential(r), dtype=float)
    return Field(
        values=v,
        grid=r,
        unit="hartree",
        grid_unit="bohr",
        label="V(r)",
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method=f"analytic potential sampled on {CURVE_POINTS}-point grid{note}",
        ),
    )

def force_law_levels(
    preset: str,
    params: Params,
    l: int,
    system: str | System = "h",
    n_states: int = 4,
) -> ForceLawResult:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
    spec = PRESETS[preset]
    _validate(spec, params, l, n_states)

    sys = system if isinstance(system, System) else get_system(system)
    z = sys.Z
    mu = sys.mu_ratio.value

    potential = spec.build_potential(params, z, mu)
    r_max = spec.r_max(params, z, n_states)
    sol = solve_radial_with_error(potential, l=l, mu_ratio=mu, n_states=n_states, r_max=r_max)

    note = f"; counterfactual preset {preset} params={params}"
    bound: list[ForceLawLevel] = []
    for k in range(n_states):
        e = sol.energies[k]
        if _bound(spec, e):
            bound.append(ForceLawLevel(radial_index=len(bound), energy=_tag(e, note)))

    reference = spec.reference(params, z, mu, l, n_states)
    curve = _sample_curve(potential, r_max, note)

    return ForceLawResult(
        preset_key=preset,
        params=dict(params),
        l=l,
        z=z,
        system_key=sys.key,
        counterfactual=tuple(bound),
        bound_count=len(bound),
        requested_count=n_states,
        reference=reference,
        potential_curve=curve,
    )
```

Note: `solve_radial_with_error` already accepts `r_max`; passing a preset-specific `r_max` keeps every preset box-converged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_force_law.py -v`
Expected: PASS (all power-law parity tests). If any phase-4 assertion referenced the old signature, it is now migrated.

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/atomsim/numerics/force_law.py tests/test_force_law.py
git add src/atomsim/numerics/force_law.py tests/test_force_law.py
git commit -m "Refactor force law into a preset registry (power-law parity)"
```

---

## Task 3: Yukawa and Coulomb-plus-core presets

Both are Coulomb-family (hydrogen reference). Yukawa has a finite bound spectrum (exercises the `E<0` filter); Coulomb-core keeps the Coulomb tail (always `n_states` bound).

**Files:**
- Modify: `src/atomsim/numerics/force_law.py` (add two presets to `PRESETS`)
- Modify: `tests/test_force_law.py` (add validation tests)

**Interfaces:**
- Consumes: everything from Task 2.
- Produces: `PRESETS["yukawa"]` (param `lambda`), `PRESETS["coulombcore"]` (param `core`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_force_law.py  (append)
def test_yukawa_large_lambda_approaches_hydrogen():
    res = force_law_levels("yukawa", {"lambda": 20.0}, l=0, system="h", n_states=1)
    exact = hydrogen_energy(1, Z=1, mu_ratio=1.0).value
    assert math.isclose(res.counterfactual[0].energy.value, exact, rel_tol=5e-3)

def test_yukawa_screening_removes_bound_states():
    loose = force_law_levels("yukawa", {"lambda": 20.0}, l=0, system="h", n_states=4)
    tight = force_law_levels("yukawa", {"lambda": 1.0}, l=0, system="h", n_states=4)
    assert tight.bound_count < loose.bound_count
    assert all(level.energy.value < 0 for level in tight.counterfactual)

def test_yukawa_reference_is_full_hydrogen_ladder_even_under_shortfall():
    res = force_law_levels("yukawa", {"lambda": 1.0}, l=0, system="h", n_states=4)
    assert len(res.reference.items) == 4  # full ideal ladder
    assert res.bound_count <= 4

def test_coulombcore_c0_recovers_hydrogen():
    res = force_law_levels("coulombcore", {"core": 0.0}, l=0, system="h", n_states=2)
    for k, level in enumerate(res.counterfactual):
        exact = hydrogen_energy(k + 1, Z=1, mu_ratio=1.0).value
        assert math.isclose(level.energy.value, exact, rel_tol=2e-4)

def test_coulombcore_repulsive_core_raises_penetrating_s_above_p():
    s = force_law_levels("coulombcore", {"core": 0.5}, l=0, system="h", n_states=2)
    p = force_law_levels("coulombcore", {"core": 0.5}, l=1, system="h", n_states=1)
    e_2s = s.counterfactual[1].energy.value  # (l=0, k=1) -> n=2
    e_2p = p.counterfactual[0].energy.value  # (l=1, k=0) -> n=2
    assert abs(e_2s - e_2p) > 1e-4
    assert e_2s > e_2p  # +c/r^2 repulsion hits the penetrating low-l state harder
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_force_law.py -k "yukawa or coulombcore" -v`
Expected: FAIL, `unknown preset 'yukawa'` / `'coulombcore'`.

- [ ] **Step 3: Write the implementation** (insert before the `PRESETS` dict in `force_law.py`, then extend the dict)

```python
# ---- Yukawa / screened -------------------------------------------------------

def _yukawa_potential(params: Params, z: int, mu: float) -> PotentialFn:
    lam = params["lambda"]
    return lambda r: -(z / r) * np.exp(-r / lam)

def _yukawa_rmax(params: Params, z: int, n_states: int) -> float:
    # a few screening lengths, but at least the Coulomb box for the ideal ladder
    return max(8.0 * params["lambda"], 20.0 * (n_states + 1) ** 2 / z)

YUKAWA = ForcePreset(
    key="yukawa",
    params=(ParamSpec("lambda", 0.5, 20.0, 3.0, "bohr"),),
    uses_Z=True,
    binding="decay",
    build_potential=_yukawa_potential,
    reference=_hydrogen_reference,
    r_max=_yukawa_rmax,
)

# ---- Coulomb plus 1/r^2 core -------------------------------------------------

def _coulombcore_potential(params: Params, z: int, mu: float) -> PotentialFn:
    c = params["core"]
    return lambda r: -z / r + c / r**2

def _coulombcore_rmax(params: Params, z: int, n_states: int) -> float:
    return 20.0 * (n_states + 1) ** 2 / z

COULOMBCORE = ForcePreset(
    key="coulombcore",
    params=(ParamSpec("core", 0.0, 1.0, 0.2, ""),),
    uses_Z=True,
    binding="decay",
    build_potential=_coulombcore_potential,
    reference=_hydrogen_reference,
    r_max=_coulombcore_rmax,
)
```

Then extend the registry:

```python
PRESETS: dict[str, ForcePreset] = {
    POWERLAW.key: POWERLAW,
    YUKAWA.key: YUKAWA,
    COULOMBCORE.key: COULOMBCORE,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_force_law.py -v`
Expected: PASS. (If `test_yukawa_large_lambda_approaches_hydrogen` is slightly outside `5e-3`, the box is too small, confirm `_yukawa_rmax` returns the Coulomb box for large λ; do not loosen the tolerance without cause.)

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/atomsim/numerics/force_law.py tests/test_force_law.py
git add src/atomsim/numerics/force_law.py tests/test_force_law.py
git commit -m "Add Yukawa and Coulomb-plus-core force-law presets"
```

---

## Task 4: Harmonic and finite-well presets

Harmonic uses the EXACT QHO reference (Task 1) and is **confining** (all states bound). Finite well uses **markers** (floor `-V₀`, threshold `0`) and can have zero bound states.

**Files:**
- Modify: `src/atomsim/numerics/force_law.py` (add two presets)
- Modify: `tests/test_force_law.py` (validation tests, incl. an independent transcendental root for the well)

**Interfaces:**
- Consumes: `oscillator_energy` (Task 1); everything from Task 2.
- Produces: `PRESETS["harmonic"]` (param `omega`, binding `confining`, reference kind `levels` from QHO), `PRESETS["finitewell"]` (params `v0`, `a`, binding `decay`, reference kind `markers`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_force_law.py  (append)
from scipy.optimize import brentq

def test_harmonic_matches_exact_qho():
    for omega in (0.3, 0.6):
        for l in (0, 1):
            res = force_law_levels("harmonic", {"omega": omega}, l=l, system="h", n_states=3)
            assert res.bound_count == 3  # confining: all bound
            for k, level in enumerate(res.counterfactual):
                exact = oscillator_energy(k, l, omega).value
                assert math.isclose(level.energy.value, exact, rel_tol=2e-3)

def test_harmonic_reference_is_qho_levels():
    res = force_law_levels("harmonic", {"omega": 0.5}, l=0, system="h", n_states=2)
    assert res.reference.kind == "levels"
    assert all(i.energy.provenance.fidelity is Fidelity.EXACT for i in res.reference.items)
    assert math.isclose(res.reference.items[0].energy.value, 0.5 * 1.5, rel_tol=1e-9)

def _well_ground_state(v0: float, a: float, mu: float) -> float:
    # s-wave spherical well: k1 cot(k1 a) = -k2, E in (-v0, 0), independent of the FD solver
    def f(E):
        k1 = math.sqrt(2 * mu * (E + v0))
        k2 = math.sqrt(-2 * mu * E)
        return k1 / math.tan(k1 * a) + k2
    lo, hi = -v0 + 1e-9, -1e-9
    return brentq(f, lo, hi)

def test_finitewell_ground_state_matches_transcendental():
    v0, a = 2.0, 3.0
    res = force_law_levels("finitewell", {"v0": v0, "a": a}, l=0, system="h", n_states=3)
    assert res.bound_count >= 1
    ref = _well_ground_state(v0, a, mu=1.0)
    assert math.isclose(res.counterfactual[0].energy.value, ref, rel_tol=5e-3)
    for level in res.counterfactual:
        assert -v0 < level.energy.value < 0

def test_finitewell_markers_reference():
    res = force_law_levels("finitewell", {"v0": 2.0, "a": 3.0}, l=0, system="h", n_states=2)
    assert res.reference.kind == "markers"
    labels = {i.label for i in res.reference.items}
    assert labels == {"well floor", "continuum threshold"}

def test_finitewell_too_shallow_has_no_bound_states():
    # sqrt(2*mu*v0)*a < pi/2  =>  no s-wave bound state
    res = force_law_levels("finitewell", {"v0": 0.1, "a": 0.5}, l=0, system="h", n_states=3)
    assert res.bound_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_force_law.py -k "harmonic or finitewell" -v`
Expected: FAIL, `unknown preset 'harmonic'` / `'finitewell'`.

- [ ] **Step 3: Write the implementation** (add before the `PRESETS` dict, then extend it)

```python
# ---- harmonic oscillator -----------------------------------------------------

def _harmonic_potential(params: Params, z: int, mu: float) -> PotentialFn:
    omega = params["omega"]
    k = mu * omega**2
    return lambda r: 0.5 * k * r**2

def _harmonic_reference(params: Params, z: int, mu: float, l: int, n_states: int) -> Reference:
    omega = params["omega"]
    items = tuple(
        ReferenceItem(label=f"k={k}", energy=oscillator_energy(k, l, omega))
        for k in range(n_states)
    )
    return Reference(kind="levels", items=items)

def _harmonic_rmax(params: Params, z: int, n_states: int) -> float:
    # classical turning point of the highest level, with headroom:
    # E ~ omega(2(n_states-1)+3/2); r_turn = sqrt(2E/(mu omega^2)) -> use mu=1 upper bound
    omega = params["omega"]
    e_top = omega * (2 * (n_states - 1) + 1.5)
    return 4.0 * math.sqrt(2.0 * e_top / omega**2)

HARMONIC = ForcePreset(
    key="harmonic",
    params=(ParamSpec("omega", 0.05, 1.0, 0.3, ""),),
    uses_Z=False,
    binding="confining",
    build_potential=_harmonic_potential,
    reference=_harmonic_reference,
    r_max=_harmonic_rmax,
)

# ---- finite spherical well ---------------------------------------------------

def _finitewell_potential(params: Params, z: int, mu: float) -> PotentialFn:
    v0 = params["v0"]
    a = params["a"]
    return lambda r: np.where(r < a, -v0, 0.0)

def _finitewell_reference(params: Params, z: int, mu: float, l: int, n_states: int) -> Reference:
    v0 = params["v0"]
    marker = Provenance(
        fidelity=Fidelity.EXACT,
        method="finite-well structural marker (definitional given V0, a)",
    )
    items = (
        ReferenceItem(
            label="well floor",
            energy=Quantity(value=-v0, unit="hartree", label="-V0", provenance=marker),
        ),
        ReferenceItem(
            label="continuum threshold",
            energy=Quantity(value=0.0, unit="hartree", label="E=0", provenance=marker),
        ),
    )
    return Reference(kind="markers", items=items)

def _finitewell_rmax(params: Params, z: int, n_states: int) -> float:
    return max(6.0 * params["a"], 40.0)

FINITEWELL = ForcePreset(
    key="finitewell",
    params=(
        ParamSpec("v0", 0.1, 5.0, 2.0, "hartree"),
        ParamSpec("a", 0.5, 10.0, 3.0, "bohr"),
    ),
    uses_Z=False,
    binding="decay",
    build_potential=_finitewell_potential,
    reference=_finitewell_reference,
    r_max=_finitewell_rmax,
)
```

Add `import math` at the top of `force_law.py` (needed by `_harmonic_rmax`). Extend the registry:

```python
PRESETS: dict[str, ForcePreset] = {
    POWERLAW.key: POWERLAW,
    YUKAWA.key: YUKAWA,
    COULOMBCORE.key: COULOMBCORE,
    HARMONIC.key: HARMONIC,
    FINITEWELL.key: FINITEWELL,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_force_law.py tests/test_oscillator.py -v`
Expected: PASS. (If harmonic misses `2e-3`, increase the box headroom in `_harmonic_rmax`; the FD grid is uniform, so very large `r_max` at fixed `n_points` can *hurt*, keep the turning-point scaling.)

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/atomsim/numerics/force_law.py tests/test_force_law.py
git add src/atomsim/numerics/force_law.py tests/test_force_law.py
git commit -m "Add harmonic and finite-well force-law presets"
```

---

## Task 5: Generalized schemas + `/api/forcelaw` endpoint

**Files:**
- Modify: `src/atomsim/server/schemas.py` (replace `ForceLawLevelModel`/`ReferenceLevelModel`/`ForceLawModel`)
- Modify: `src/atomsim/server/app.py` (rewrite `forcelaw_endpoint`; imports)
- Modify: `tests/test_server.py` (per-preset endpoint tests)

**Interfaces:**
- Consumes: `force_law_levels`, `PRESETS`, `ForceLawResult`, `Reference` (Task 2-4); `FieldModel`, `QuantityModel`, `_to_ev`.
- Produces: JSON contract, `ForceLawModel { preset, params: dict[str,float], l, z, system, counterfactual: [ForceLawLevelModel], bound_count, requested_count, reference: ReferenceModel, potential_curve: PotentialCurveModel }`.
  - `ForceLawLevelModel { radial_index, energy, energy_ev }`.
  - `ReferenceItemModel { label, energy, energy_ev }`.
  - `ReferenceModel { kind: "levels"|"markers", items: [ReferenceItemModel] }`.
  - `PotentialCurveModel { r: [float] (bohr), v_ev: [float] (eV), provenance }`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server.py  (append; assumes `client` TestClient fixture exists)
def test_forcelaw_backcompat_defaults_to_powerlaw(client):
    r = client.get("/api/forcelaw?p=1.0&l=0&system=h&n_states=3")
    assert r.status_code == 200
    body = r.json()
    assert body["preset"] == "powerlaw"
    assert body["reference"]["kind"] == "levels"
    assert len(body["counterfactual"]) == 3
    assert body["potential_curve"]["v_ev"]

def test_forcelaw_yukawa(client):
    r = client.get("/api/forcelaw?preset=yukawa&lambda=3.0&l=0&system=h&n_states=4")
    assert r.status_code == 200
    body = r.json()
    assert body["preset"] == "yukawa"
    assert body["params"]["lambda"] == 3.0
    assert body["bound_count"] <= body["requested_count"]

def test_forcelaw_harmonic_reference_is_levels(client):
    r = client.get("/api/forcelaw?preset=harmonic&omega=0.5&l=0&n_states=2")
    assert r.status_code == 200
    body = r.json()
    assert body["reference"]["kind"] == "levels"
    assert body["reference"]["items"][0]["energy"]["provenance"]["fidelity"] == "exact"

def test_forcelaw_finitewell_markers(client):
    r = client.get("/api/forcelaw?preset=finitewell&v0=2.0&a=3.0&l=0&n_states=3")
    assert r.status_code == 200
    assert r.json()["reference"]["kind"] == "markers"

def test_forcelaw_rejects_unknown_preset(client):
    assert client.get("/api/forcelaw?preset=bogus&l=0").status_code == 422

def test_forcelaw_rejects_out_of_range_param(client):
    assert client.get("/api/forcelaw?preset=yukawa&lambda=999&l=0").status_code == 422

def test_forcelaw_rejects_negative_l(client):
    assert client.get("/api/forcelaw?preset=harmonic&omega=0.5&l=-1").status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_server.py -k forcelaw -v`
Expected: FAIL, old endpoint has no `preset`, returns `p`, no `potential_curve`.

- [ ] **Step 3a: Replace the force-law models in `schemas.py`** (swap the three classes at lines ~200-218)

```python
class ForceLawLevelModel(BaseModel):
    radial_index: int
    energy: QuantityModel
    energy_ev: QuantityModel

class ReferenceItemModel(BaseModel):
    label: str
    energy: QuantityModel
    energy_ev: QuantityModel

class ReferenceModel(BaseModel):
    kind: Literal["levels", "markers"]
    items: list[ReferenceItemModel]

class PotentialCurveModel(BaseModel):
    r: list[float]        # bohr
    v_ev: list[float]     # eV
    provenance: ProvenanceModel

class ForceLawModel(BaseModel):
    preset: str
    params: dict[str, float]
    l: int
    z: int
    system: SystemModel
    counterfactual: list[ForceLawLevelModel]
    bound_count: int
    requested_count: int
    reference: ReferenceModel
    potential_curve: PotentialCurveModel
```

- [ ] **Step 3b: Rewrite `forcelaw_endpoint` in `app.py`** (replace lines ~380-417)

```python
    @app.get("/api/forcelaw", response_model=ForceLawModel)
    def forcelaw_endpoint(
        preset: str = "powerlaw",
        l: int = 0,
        system: str = "h",
        n_states: int = 4,
        p: float = 1.0,
        lambda_: float = PydanticField(default=3.0, alias="lambda"),
        omega: float = 0.3,
        v0: float = 2.0,
        a: float = 3.0,
        core: float = 0.2,
    ) -> ForceLawModel:
        if preset not in PRESETS:
            raise HTTPException(
                status_code=422,
                detail=f"unknown preset {preset!r}; known: {sorted(PRESETS)}",
            )
        if l < 0:
            raise HTTPException(status_code=422, detail=f"l must be >= 0, got {l}")
        if not 1 <= n_states <= 8:
            raise HTTPException(
                status_code=422, detail=f"n_states must be in [1, 8], got {n_states}"
            )
        supplied = {
            "p": p, "lambda": lambda_, "omega": omega, "v0": v0, "a": a, "core": core,
        }
        params = {spec.name: supplied[spec.name] for spec in PRESETS[preset].params}
        sys_ = _resolve_system(system)
        try:
            result = force_law_levels(preset, params, l=l, system=sys_, n_states=n_states)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        curve = result.potential_curve
        return ForceLawModel(
            preset=result.preset_key,
            params=result.params,
            l=result.l,
            z=result.z,
            system=SystemModel.from_system(sys_),
            counterfactual=[
                ForceLawLevelModel(
                    radial_index=c.radial_index,
                    energy=QuantityModel.from_quantity(c.energy),
                    energy_ev=QuantityModel.from_quantity(_to_ev(c.energy)),
                )
                for c in result.counterfactual
            ],
            bound_count=result.bound_count,
            requested_count=result.requested_count,
            reference=ReferenceModel(
                kind=result.reference.kind,
                items=[
                    ReferenceItemModel(
                        label=item.label,
                        energy=QuantityModel.from_quantity(item.energy),
                        energy_ev=QuantityModel.from_quantity(_to_ev(item.energy)),
                    )
                    for item in result.reference.items
                ],
            ),
            potential_curve=PotentialCurveModel(
                r=curve.grid.tolist(),
                v_ev=(curve.values * HARTREE_EV).tolist(),
                provenance=ProvenanceModel.from_provenance(
                    dataclasses.replace(
                        curve.provenance,
                        method=curve.provenance.method
                        + "; converted to eV via CODATA Hartree-eV factor",
                    )
                ),
            ),
        )
```

Update the imports in `app.py`:
- Replace `from atomsim.numerics.force_law import P_MAX, P_MIN, force_law_levels` with `from atomsim.numerics.force_law import PRESETS, force_law_levels`.
- In the `schemas` import block, replace `ForceLawLevelModel, ForceLawModel, ReferenceLevelModel` with `ForceLawLevelModel, ForceLawModel, PotentialCurveModel, ReferenceItemModel, ReferenceModel`.
- `HARTREE_EV` is already imported; `PydanticField` and `dataclasses` are already imported.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_server.py -k forcelaw -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Full backend suite + lint + commit**

```bash
$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest -q
ruff check .
git add src/atomsim/server/schemas.py src/atomsim/server/app.py tests/test_server.py
git commit -m "Generalize the /api/forcelaw endpoint to all five presets"
```

Expected: full suite green; ruff clean.

---

## Task 6: Frontend types + API client

**Files:**
- Modify: `web/src/api/types.ts` (replace `ForceLawLevel`, `ReferenceLevel`, `ForceLawResult`)
- Modify: `web/src/api/client.ts` (rewrite `getForceLaw`)

**Interfaces:**
- Produces (TS mirror of Task 5 JSON):

```ts
export interface ForceLawLevel { radial_index: number; energy: Quantity; energy_ev: Quantity; }
export interface ReferenceItem { label: string; energy: Quantity; energy_ev: Quantity; }
export interface Reference { kind: "levels" | "markers"; items: ReferenceItem[]; }
export interface PotentialCurve { r: number[]; v_ev: number[]; provenance: Provenance; }
export interface ForceLawResult {
  preset: string;
  params: Record<string, number>;
  l: number;
  z: number;
  system: SystemInfo;
  counterfactual: ForceLawLevel[];
  bound_count: number;
  requested_count: number;
  reference: Reference;
  potential_curve: PotentialCurve;
}
```

- `getForceLaw(system: string, preset: string, params: Record<string, number>, l: number, nStates?: number): Promise<ForceLawResult>`.

- [ ] **Step 1: Replace the force-law block in `types.ts`** (lines ~167-186) with the interfaces above.

- [ ] **Step 2: Rewrite `getForceLaw` in `client.ts`** (replace lines ~83-92)

```ts
export function getForceLaw(
  system: string,
  preset: string,
  params: Record<string, number>,
  l: number,
  nStates = 4,
): Promise<ForceLawResult> {
  const q = new URLSearchParams({
    system,
    preset,
    l: String(l),
    n_states: String(nStates),
  });
  for (const [k, v] of Object.entries(params)) q.set(k, String(v));
  return getJson(`/api/forcelaw?${q.toString()}`);
}
```

- [ ] **Step 3: Typecheck**

Run: `cd web; npx tsc --noEmit`
Expected: errors only in `store.ts` / `ForceLawView.tsx` (fixed in Tasks 7 & 9), `types.ts` and `client.ts` themselves compile.

- [ ] **Step 4: Commit**

```bash
git add web/src/api/types.ts web/src/api/client.ts
git commit -m "Update force-law frontend types and client for presets"
```

---

## Task 7: Preset param specs + store slice

**Files:**
- Create: `web/src/lib/forceLaw.ts` (single source of preset param specs + span helper)
- Create: `web/src/lib/forceLaw.test.ts`
- Modify: `web/src/state/store.ts` (force slice)
- Modify: `web/src/state/store.test.ts`

**Interfaces:**
- Produces in `lib/forceLaw.ts`:

```ts
export type ForcePreset = "powerlaw" | "yukawa" | "harmonic" | "finitewell" | "coulombcore";
export interface ParamSpec { name: string; min: number; max: number; default: number; unit: string; }
export const PRESET_PARAMS: Record<ForcePreset, ParamSpec[]>;
export const PRESET_LABELS: Record<ForcePreset, string>;
export function defaultParams(preset: ForcePreset): Record<string, number>;
export function clampParam(spec: ParamSpec, value: number): number;
/** classically-allowed r-span [rIn, rOut] where E > V(r) along the curve, or null. */
export function allowedSpan(r: number[], vEv: number[], energyEv: number): [number, number] | null;
```

- Produces in `store.ts` (force slice additions): `forcePreset: ForcePreset`, `forceParams: Record<string, number>`, `forceViz: "well" | "ladder"`, `setForcePreset`, `setForceParam(name, value)`, `setForceViz`. `forceP`/`forceL` remain (`forceP` kept as the power-law param mirror is dropped, `p` now lives in `forceParams`; keep `forceL`).

- [ ] **Step 1: Write the failing test**

```ts
// web/src/lib/forceLaw.test.ts
import { describe, expect, it } from "vitest";
import { allowedSpan, clampParam, defaultParams, PRESET_PARAMS } from "./forceLaw";

describe("forceLaw preset specs", () => {
  it("every preset has at least one param with a default in range", () => {
    for (const specs of Object.values(PRESET_PARAMS)) {
      expect(specs.length).toBeGreaterThan(0);
      for (const s of specs) expect(s.default).toBeGreaterThanOrEqual(s.min);
    }
  });

  it("defaultParams returns the spec defaults", () => {
    expect(defaultParams("yukawa")).toEqual({ lambda: 3 });
    expect(defaultParams("finitewell")).toEqual({ v0: 2, a: 3 });
  });

  it("clampParam bounds to the spec range", () => {
    const spec = PRESET_PARAMS.yukawa[0];
    expect(clampParam(spec, 999)).toBe(spec.max);
    expect(clampParam(spec, -1)).toBe(spec.min);
  });

  it("allowedSpan finds the E>V window", () => {
    const r = [1, 2, 3, 4, 5];
    const v = [-10, -8, -6, -4, -2];
    expect(allowedSpan(r, v, -5)).toEqual([1, 3]); // V<-5 at r=1,2,3
    expect(allowedSpan(r, v, -20)).toBeNull(); // E below the whole well
  });
});
```

```ts
// web/src/state/store.test.ts  (append inside the existing suite)
import { defaultParams } from "../lib/forceLaw";

it("setForcePreset swaps params to that preset's defaults and clears data", () => {
  const s = useAppStore.getState();
  s.setForcePreset("yukawa");
  const st = useAppStore.getState();
  expect(st.forcePreset).toBe("yukawa");
  expect(st.forceParams).toEqual(defaultParams("yukawa"));
  expect(st.forceLaw).toBeNull();
  expect(st.forceStatus).toBe("idle");
});

it("setForceParam clamps and clears force-law data", () => {
  useAppStore.getState().setForcePreset("yukawa");
  useAppStore.getState().setForceParam("lambda", 999);
  expect(useAppStore.getState().forceParams.lambda).toBe(20); // spec max
  expect(useAppStore.getState().forceLaw).toBeNull();
});

it("setForceViz is presentational: it does not clear force-law data", () => {
  useAppStore.setState({ forceLaw: { preset: "powerlaw" } as never, forceStatus: "ready" });
  useAppStore.getState().setForceViz("ladder");
  expect(useAppStore.getState().forceViz).toBe("ladder");
  expect(useAppStore.getState().forceLaw).not.toBeNull(); // untouched
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web; npx vitest run src/lib/forceLaw.test.ts src/state/store.test.ts`
Expected: FAIL, `forceLaw.ts` missing; store lacks `setForcePreset`/`setForceParam`/`setForceViz`.

- [ ] **Step 3a: Create `web/src/lib/forceLaw.ts`**

```ts
// Single source of truth for force-law preset parameters, mirrors the Python
// ParamSpec ranges in src/atomsim/numerics/force_law.py.
export type ForcePreset =
  | "powerlaw"
  | "yukawa"
  | "harmonic"
  | "finitewell"
  | "coulombcore";

export interface ParamSpec {
  name: string;
  min: number;
  max: number;
  default: number;
  unit: string;
  step: number;
}

export const PRESET_PARAMS: Record<ForcePreset, ParamSpec[]> = {
  powerlaw: [{ name: "p", min: 0.5, max: 1.5, default: 1.0, unit: "", step: 0.05 }],
  yukawa: [{ name: "lambda", min: 0.5, max: 20, default: 3, unit: "a₀", step: 0.5 }],
  harmonic: [{ name: "omega", min: 0.05, max: 1.0, default: 0.3, unit: "", step: 0.05 }],
  finitewell: [
    { name: "v0", min: 0.1, max: 5, default: 2, unit: "Ha", step: 0.1 },
    { name: "a", min: 0.5, max: 10, default: 3, unit: "a₀", step: 0.5 },
  ],
  coulombcore: [{ name: "core", min: 0, max: 1, default: 0.2, unit: "", step: 0.05 }],
};

export const PRESET_LABELS: Record<ForcePreset, string> = {
  powerlaw: "Power law  −Z/rᵖ",
  yukawa: "Yukawa / screened  −(Z/r)e^(−r/λ)",
  harmonic: "Harmonic  ½kr²",
  finitewell: "Finite well  −V₀ (r<a)",
  coulombcore: "Coulomb + core  −Z/r + c/r²",
};

export function defaultParams(preset: ForcePreset): Record<string, number> {
  const out: Record<string, number> = {};
  for (const spec of PRESET_PARAMS[preset]) out[spec.name] = spec.default;
  return out;
}

export function clampParam(spec: ParamSpec, value: number): number {
  return Math.min(Math.max(value, spec.min), spec.max);
}

export function allowedSpan(
  r: number[],
  vEv: number[],
  energyEv: number,
): [number, number] | null {
  const inside: number[] = [];
  for (let i = 0; i < r.length; i++) if (energyEv > vEv[i]) inside.push(r[i]);
  if (inside.length === 0) return null;
  return [inside[0], inside[inside.length - 1]];
}
```

- [ ] **Step 3b: Modify the store force slice in `web/src/state/store.ts`**

Add to `ViewMode`/state imports: `import { defaultParams, type ForcePreset } from "../lib/forceLaw";`

Replace the force-law state declarations (near lines 72-78) with:

```ts
  forcePreset: ForcePreset;
  forceParams: Record<string, number>;
  forceL: number;
  forceViz: "well" | "ladder";
  forceLaw: ForceLawResult | null;
  forceStatus: SampleStatus;
  setForcePreset: (preset: ForcePreset) => void;
  setForceParam: (name: string, value: number) => void;
  setForceL: (l: number) => void;
  setForceViz: (viz: "well" | "ladder") => void;
  loadForceLaw: () => Promise<void>;
```

Replace the initial values (near lines 142-145) with:

```ts
  forcePreset: "powerlaw",
  forceParams: defaultParams("powerlaw"),
  forceL: 0,
  forceViz: "well",
  forceLaw: null,
  forceStatus: "idle",
```

Replace the force-law actions (near lines 189-208). Import the clamp helper too:
`import { PRESET_PARAMS, clampParam, defaultParams, type ForcePreset } from "../lib/forceLaw";`

```ts
  // force-law slice: its own axis (preset, params, l), independent of the main
  // (n,l,m,system) physics, so never in INVALIDATED. Changing the preset, a
  // param, or l clears only the force-law data. forceViz is presentational and
  // clears nothing (store invariant). System changes clear it too (Z/mu change).
  setForcePreset: (preset) =>
    set({
      forcePreset: preset,
      forceParams: defaultParams(preset),
      forceLaw: null,
      forceStatus: "idle",
    }),
  setForceParam: (name, value) => {
    const { forcePreset, forceParams } = get();
    const spec = PRESET_PARAMS[forcePreset].find((s) => s.name === name);
    if (spec === undefined) return;
    set({
      forceParams: { ...forceParams, [name]: clampParam(spec, value) },
      forceLaw: null,
      forceStatus: "idle",
    });
  },
  setForceL: (l) =>
    set({ forceL: Math.max(0, Math.round(l)), forceLaw: null, forceStatus: "idle" }),
  setForceViz: (viz) => set({ forceViz: viz }),
  loadForceLaw: async () => {
    const { forcePreset, forceParams, forceL, system } = get();
    set({ forceStatus: "sampling", error: null });
    try {
      const forceLaw = await client.getForceLaw(system, forcePreset, forceParams, forceL);
      set({ forceLaw, forceStatus: "ready" });
    } catch (err) {
      set({ forceStatus: "error", error: err instanceof Error ? err.message : String(err) });
    }
  },
```

Also update the `setSystem` reset block (near lines 152-158), it already sets `forceLaw: null, forceStatus: "idle"`; leave `forcePreset`/`forceParams` intact (a system change keeps the chosen preset, just re-solves). No change needed there beyond confirming those two lines remain.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd web; npx vitest run src/lib/forceLaw.test.ts src/state/store.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/forceLaw.ts web/src/lib/forceLaw.test.ts web/src/state/store.ts web/src/state/store.test.ts
git commit -m "Add force-law preset specs and generalize the store slice"
```

---

## Task 8: URL state for preset + params

**Files:**
- Modify: `web/src/lib/urlState.ts`
- Modify: `web/src/lib/urlState.test.ts`
- Modify: `web/src/main.tsx` (wire the new fields into serialize/apply, mirror the existing `forceP`/`forceL` wiring)

**Interfaces:**
- Consumes: `PRESET_PARAMS`, `defaultParams`, `clampParam`, `ForcePreset` (Task 7).
- Produces: `UrlState` gains `forcePreset: ForcePreset` and `forceParams: Record<string, number>` (replacing `forceP`). Query keys: `preset` (omitted when `powerlaw`), and each active param by its name (`p`, `lambda`, `omega`, `v0`, `a`, `core`); `fl` unchanged. `forceViz` stays out of the URL.

- [ ] **Step 1: Write the failing test**

```ts
// web/src/lib/urlState.test.ts  (append)
import { defaultParams } from "./forceLaw";

it("round-trips a yukawa force-law deep link", () => {
  const state = { ...decodeUrl(""), view: "forcelaw" as const,
                  forcePreset: "yukawa" as const, forceParams: { lambda: 5 }, forceL: 1 };
  const q = encodeUrl(state);
  expect(q).toContain("preset=yukawa");
  expect(q).toContain("lambda=5");
  const back = decodeUrl(q);
  expect(back.forcePreset).toBe("yukawa");
  expect(back.forceParams.lambda).toBe(5);
  expect(back.forceL).toBe(1);
});

it("omits preset for the default power-law and reads p", () => {
  const state = { ...decodeUrl(""), view: "forcelaw" as const,
                  forcePreset: "powerlaw" as const, forceParams: { p: 1.2 }, forceL: 0 };
  const q = encodeUrl(state);
  expect(q).not.toContain("preset=");
  expect(q).toContain("p=1.2");
  expect(decodeUrl(q).forceParams.p).toBe(1.2);
});

it("clamps an out-of-range param from the URL", () => {
  const back = decodeUrl("view=forcelaw&preset=yukawa&lambda=999");
  expect(back.forceParams.lambda).toBe(20); // spec max
});

it("falls back to preset defaults when a param is missing", () => {
  const back = decodeUrl("view=forcelaw&preset=finitewell&v0=1.5");
  expect(back.forceParams.v0).toBe(1.5);
  expect(back.forceParams.a).toBe(defaultParams("finitewell").a); // default
});
```

(Use the actual encode/decode function names already in `urlState.test.ts`, match them.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web; npx vitest run src/lib/urlState.test.ts`
Expected: FAIL, `forcePreset`/`forceParams` not handled.

- [ ] **Step 3: Implement in `urlState.ts`**

In the `UrlState` interface, replace `forceP: number;` with:

```ts
  forcePreset: ForcePreset;
  forceParams: Record<string, number>;
```

Add the import: `import { PRESET_PARAMS, clampParam, defaultParams, type ForcePreset } from "./forceLaw";`

Add `forcelaw` presets to a known list and update `URL_DEFAULTS` (replace `forceP: 1.0`):

```ts
const PRESETS: ForcePreset[] = ["powerlaw", "yukawa", "harmonic", "finitewell", "coulombcore"];
// in URL_DEFAULTS:
  forcePreset: "powerlaw",
  forceParams: defaultParams("powerlaw"),
```

In the decode function (near the current `forceP` handling ~line 134), replace the `forceP` block with:

```ts
  const preset = pickEnum(q.get("preset"), PRESETS) ?? "powerlaw";
  out.forcePreset = preset;
  const params = defaultParams(preset);
  for (const spec of PRESET_PARAMS[preset]) {
    const raw = q.get(spec.name);
    if (raw !== null) {
      const v = Number(raw);
      if (Number.isFinite(v)) params[spec.name] = clampParam(spec, v);
    }
  }
  out.forceParams = params;
```

In the encode function (replace the `forceP` line ~162):

```ts
  if (state.forcePreset !== "powerlaw") q.set("preset", state.forcePreset);
  for (const spec of PRESET_PARAMS[state.forcePreset]) {
    const v = state.forceParams[spec.name];
    if (v !== undefined && Math.abs(v - spec.default) > 1e-9) q.set(spec.name, String(v));
  }
```

- [ ] **Step 3b: Wire `main.tsx`**

Wherever `main.tsx` reads `forceP` from the store into `UrlState` and writes it back (mirror of the phase-4 wiring), replace `forceP` with `forcePreset` and `forceParams`. On apply, call `useAppStore.setState({ forcePreset, forceParams, forceL })` (these are plain state fields; the load effect in the view re-fetches). Match the existing pattern used for `forceL`.

- [ ] **Step 4: Run tests + typecheck**

Run: `cd web; npx vitest run src/lib/urlState.test.ts; npx tsc --noEmit`
Expected: PASS; tsc errors only remain in `ForceLawView.tsx` (Task 9).

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/urlState.ts web/src/lib/urlState.test.ts web/src/main.tsx
git commit -m "Deep-link the force-law preset and its parameters"
```

---

## Task 9: ForceLawView: well diagram + ladder toggle

**Files:**
- Modify: `web/src/components/ForceLawView.tsx` (substantial rewrite)
- Modify: `web/src/index.css` (styles for the new controls/diagram, follow existing `.forcelaw*` classes)

**Interfaces:**
- Consumes: store (`forcePreset`, `forceParams`, `forceL`, `forceViz`, `forceLaw`, `forceStatus`, setters, `loadForceLaw`), `PRESET_PARAMS`, `PRESET_LABELS`, `allowedSpan` (Task 7), `Badge`.
- Produces: no new exports (the view is mounted by the existing router entry in `App.tsx`).

- [ ] **Step 1: Write the implementation** (this is a view; validate by build + render test in Step 2-3). Replace `ForceLawView.tsx`:

```tsx
import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import { allowedSpan, PRESET_LABELS, PRESET_PARAMS, type ForcePreset } from "../lib/forceLaw";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 680;
const H = 460;
const PAD = { top: 32, right: 24, bottom: 44, left: 64 };
const L_CHOICES = [0, 1, 2, 3];
const PRESETS: ForcePreset[] = ["powerlaw", "yukawa", "harmonic", "finitewell", "coulombcore"];

export function ForceLawView() {
  const {
    forcePreset, forceParams, forceL, forceViz, forceLaw, forceStatus, error,
    setForcePreset, setForceParam, setForceL, setForceViz, loadForceLaw,
  } = useAppStore();

  useEffect(() => {
    if (forceLaw === null && forceStatus === "idle") void loadForceLaw();
  }, [forceLaw, forceStatus, loadForceLaw]);

  const cfProv = forceLaw?.counterfactual[0]?.energy.provenance ?? null;
  const refProv = forceLaw?.reference.items[0]?.energy.provenance ?? null;

  const levelsEv = forceLaw ? forceLaw.counterfactual.map((c) => c.energy_ev.value) : [];
  const refEv = forceLaw ? forceLaw.reference.items.map((i) => i.energy_ev.value) : [];
  const curveEv = forceLaw ? forceLaw.potential_curve.v_ev : [];
  const curveR = forceLaw ? forceLaw.potential_curve.r : [];

  const allEv = [...levelsEv, ...refEv, ...curveEv];
  const emin = allEv.length ? Math.min(...allEv) : -14;
  const emax = allEv.length ? Math.max(...allEv, 0.1) : 0;
  const y = scaleLinear([emin, emax], [H - PAD.bottom, PAD.top]);
  const rmax = curveR.length ? curveR[curveR.length - 1] : 1;
  const x = scaleLinear([0, rmax], [PAD.left, W - PAD.right]);

  const shortfall =
    forceLaw !== null && forceLaw.bound_count < forceLaw.requested_count;

  return (
    <div className="forcelaw">
      <div className="whatif-controls">
        <label>
          Potential
          <select
            value={forcePreset}
            onChange={(e) => setForcePreset(e.target.value as ForcePreset)}
          >
            {PRESETS.map((p) => (
              <option key={p} value={p}>{PRESET_LABELS[p]}</option>
            ))}
          </select>
        </label>
        {PRESET_PARAMS[forcePreset].map((spec) => (
          <label key={spec.name}>
            {spec.name} = {(forceParams[spec.name] ?? spec.default).toFixed(2)}
            {spec.unit ? ` ${spec.unit}` : ""}
            <input
              type="range"
              min={spec.min}
              max={spec.max}
              step={spec.step}
              value={forceParams[spec.name] ?? spec.default}
              onChange={(e) => setForceParam(spec.name, Number(e.target.value))}
            />
          </label>
        ))}
        <label>
          Orbital l
          <select value={forceL} onChange={(e) => setForceL(Number(e.target.value))}>
            {L_CHOICES.map((l) => (
              <option key={l} value={l}>{l} ({"spdf"[l]})</option>
            ))}
          </select>
        </label>
        <label>
          View
          <select
            value={forceViz}
            onChange={(e) => setForceViz(e.target.value as "well" | "ladder")}
          >
            <option value="well">Potential well</option>
            <option value="ladder">Energy ladder</option>
          </select>
        </label>
      </div>

      {forceStatus === "error" && <p className="error">{error}</p>}
      {forceStatus === "sampling" && <p className="hint-block">solving force law…</p>}
      {shortfall && (
        <p className="hint-block">
          Only {forceLaw!.bound_count} bound state
          {forceLaw!.bound_count === 1 ? "" : "s"} at these parameters
          {forceLaw!.bound_count === 0 ? ", the potential is too shallow to bind." : "."}
        </p>
      )}

      {forceLaw !== null && (
        <>
          <div className="forcelaw-legend">
            {cfProv && (
              <span>counterfactual {forcePreset} <Badge provenance={cfProv} /></span>
            )}
            {refProv && (
              <span>
                reference ({forceLaw.reference.kind}) <Badge provenance={refProv} />
              </span>
            )}
          </div>

          {forceViz === "well" ? (
            <svg viewBox={`0 0 ${W} ${H}`} className="forcelaw-svg" role="img"
                 aria-label="potential energy curve with bound levels and reference">
              <path
                className="forcelaw-curve"
                d={curveR
                  .map((r, i) => `${i === 0 ? "M" : "L"} ${x(r)} ${y(curveEv[i])}`)
                  .join(" ")}
                fill="none"
              />
              {forceLaw.reference.items.map((item, i) => (
                <line key={`ref-${i}`} className="forcelaw-ref"
                      x1={PAD.left} x2={W - PAD.right}
                      y1={y(item.energy_ev.value)} y2={y(item.energy_ev.value)} />
              ))}
              {forceLaw.counterfactual.map((c) => {
                const span = allowedSpan(curveR, curveEv, c.energy_ev.value);
                const x1 = span ? x(span[0]) : PAD.left;
                const x2 = span ? x(span[1]) : W - PAD.right;
                return (
                  <g key={`cf-${c.radial_index}`}>
                    <line className="forcelaw-cf" x1={x1} x2={x2}
                          y1={y(c.energy_ev.value)} y2={y(c.energy_ev.value)} />
                    <text x={x2 + 4} y={y(c.energy_ev.value) - 4} className="forcelaw-label">
                      {c.energy_ev.value.toFixed(2)} eV
                    </text>
                  </g>
                );
              })}
              <text x={PAD.left} y={PAD.top - 12} className="forcelaw-col">
                V(r) and bound levels, {PRESET_LABELS[forcePreset]}
              </text>
            </svg>
          ) : (
            <svg viewBox={`0 0 ${W} ${H}`} className="forcelaw-svg" role="img"
                 aria-label="energy levels versus reference">
              {forceLaw.reference.items.map((item, i) => (
                <g key={`ref-${i}`}>
                  <line className="forcelaw-ref" x1={PAD.left} x2={W / 2 - 8}
                        y1={y(item.energy_ev.value)} y2={y(item.energy_ev.value)} />
                  <text x={PAD.left} y={y(item.energy_ev.value) - 4} className="forcelaw-label">
                    {item.label}
                  </text>
                </g>
              ))}
              {forceLaw.counterfactual.map((c) => (
                <g key={`cf-${c.radial_index}`}>
                  <line className="forcelaw-cf" x1={W / 2 + 8} x2={W - PAD.right}
                        y1={y(c.energy_ev.value)} y2={y(c.energy_ev.value)} />
                  <text x={W - PAD.right} y={y(c.energy_ev.value) - 4} textAnchor="end"
                        className="forcelaw-label">
                    {c.energy_ev.value.toFixed(2)} eV
                  </text>
                </g>
              ))}
              <text x={W / 4} y={PAD.top - 12} textAnchor="middle" className="forcelaw-col">
                reference
              </text>
              <text x={(3 * W) / 4} y={PAD.top - 12} textAnchor="middle" className="forcelaw-col">
                {forcePreset}
              </text>
            </svg>
          )}

          <p className="hint-block">
            The numerical levels (NUMERICAL) are drawn against this preset's honest
            reference (EXACT). Screened and finite potentials bind only finitely many
            states; the missing upper reference rungs are the states they cannot hold.
          </p>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Add styles** to `web/src/index.css`, add a `.forcelaw-curve` rule beside the existing `.forcelaw-ref` / `.forcelaw-cf` (a stroked path, e.g. `stroke: var(--fg-muted); stroke-width: 1.5;`). Match the existing force-law style block.

- [ ] **Step 3: Typecheck + build**

Run: `cd web; npx tsc --noEmit; npm run build`
Expected: PASS; `web/dist` rebuilt.

- [ ] **Step 4: Run the full frontend suite**

Run: `cd web; npm test`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add web/src/components/ForceLawView.tsx web/src/index.css
git commit -m "Add the potential-well diagram and preset picker to ForceLawView"
```

---

## Task 10: Full verification + live smoke test

**Files:** none (verification only).

- [ ] **Step 1: Backend suite + lint**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest -q; ruff check .`
Expected: all green; ruff clean.

- [ ] **Step 2: Frontend suite + build**

Run: `cd web; npm test; npm run build`
Expected: all green; build clean.

- [ ] **Step 3: Live smoke test**, start the server and exercise each preset.

Run (background): `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m atomsim serve --port 8011 --no-browser`

Then, for each preset, confirm HTTP 200 and a sane payload:

```bash
curl -s "http://127.0.0.1:8011/api/forcelaw?p=1.0&l=0&system=h&n_states=3"        # back-compat
curl -s "http://127.0.0.1:8011/api/forcelaw?preset=yukawa&lambda=2&l=0&n_states=4"
curl -s "http://127.0.0.1:8011/api/forcelaw?preset=harmonic&omega=0.5&l=1&n_states=3"
curl -s "http://127.0.0.1:8011/api/forcelaw?preset=finitewell&v0=2&a=3&l=0&n_states=3"
curl -s "http://127.0.0.1:8011/api/forcelaw?preset=coulombcore&core=0.5&l=0&n_states=3"
curl -s -o /dev/null -w "%{http_code}\n" "http://127.0.0.1:8011/api/forcelaw?preset=bogus&l=0"   # expect 422
```

Expected: powerlaw p=1 lands on hydrogen; yukawa `bound_count` ≤ requested; harmonic levels ≈ `ω(2k+l+1.5)·27.211` eV; finite well levels in `(−V₀, 0)` eV; coulombcore splits; bogus → 422. Stop the server when done.

- [ ] **Step 4: Final commit (if any doc/tidy changes)**

```bash
git add -A
git commit -m "Phase 5: verify force-law preset library end-to-end" --allow-empty
```

---

## Self-Review

**Spec coverage:**
- §2.1 five presets → Tasks 2 (powerlaw), 3 (yukawa, coulombcore), 4 (harmonic, finitewell). ✅
- §2.2 bound-state filtering (binding-aware) → Task 2 `_bound`, tested Tasks 3-4. ✅
- §2.3 per-preset reference (hydrogen/QHO/markers, full ladder under shortfall) → Tasks 2-4 + tests. ✅
- §2.4 potential curve (EXACT Field) → Task 2 `_sample_curve`, eV at boundary Task 5. ✅
- §3 registry driver → Task 2. ✅
- §4 generalized endpoint + back-compat + 422 → Task 5. ✅
- §5.1 store + URL (forceViz presentational, out of URL) → Tasks 7, 8. ✅
- §5.2 view: preset picker, dynamic params, well diagram + ladder toggle, shortfall disclosure, badges → Task 9. ✅
- §6 testing (incl. harmonic vs QHO, finite-well transcendental) → Tasks 1, 3, 4, 5, 7, 8. ✅
- §1 new EXACT QHO ground truth → Task 1. ✅

**Placeholder scan:** none, every step carries real code or an exact command.

**Type consistency:** `force_law_levels(preset, params, l, system, n_states)` used identically in Tasks 2-5; `getForceLaw(system, preset, params, l, nStates)` used in Tasks 6-7; `ForceLawResult`/`Reference`/`PotentialCurve` field names match across Python (Task 5), TS (Task 6), and the view (Task 9); `ForcePreset` union identical in `forceLaw.ts`, `store.ts`, `urlState.ts`.

One deviation from the spec's loose wording, deliberately: §2.2 says the `E<0` filter is "applied uniformly"; the plan instead makes binding **per-preset** (`"decay"` vs `"confining"`) because a blanket `E<0` filter would wrongly discard every (positive-energy) harmonic level. This is the more honest reading and is called out in Global Constraints.
