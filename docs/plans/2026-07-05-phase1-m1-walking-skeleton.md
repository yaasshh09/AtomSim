# Phase 1 M1: Walking Skeleton Implementation Plan

**Goal:** `atomsim serve` (one command) opens a browser app showing a rotating 3D Monte-Carlo point cloud of any hydrogen (n, l, m) state, with real provenance badges, the vertical slice that every later milestone deepens.

**Architecture:** The `Field` type completes the provenance boundary rule (spec §4); a factorized inverse-CDF sampler turns |ψ_nlm|² into positions *inside the engine* (sampling is physics); a FastAPI server exposes fast JSON endpoints plus an async job/progress pattern (WebSocket) with binary float32 transfer for point clouds; a Vite + React + TypeScript + react-three-fiber frontend renders the cloud and the badges. Spec: `docs/specs/2026-07-05-phase1-hydrogen-honestly-design.md`.

**Tech Stack:** Python 3.12 (conda env `atomsim`), NumPy ≥2, SciPy ≥1.13, FastAPI + uvicorn + httpx (tests), pytest, ruff · Node 22 (conda-forge, inside the same env), Vite, React 19, TypeScript strict, three/@react-three/fiber/@react-three/drei, zustand, vitest.

## Global Constraints

- Native Windows only; no WSL2/Docker anywhere in the toolchain.
- Engine-internal units: Hartree atomic units; energies exposed in `hartree` (plus explicit eV conversions at the display boundary), lengths in `bohr`.
- **Boundary rule (spec §4):** every physical value crossing a module boundary is a `Quantity`, a `Field`, or a container dataclass carrying its own `Provenance`. Fidelity tiers exactly: EXACT, NUMERICAL, APPROXIMATION, COUNTERFACTUAL, VISUAL_LIBERTY.
- The model never quietly lies: numerical results carry quantified or estimable error; assumptions listed explicitly; purely presentational choices are marked VISUAL LIBERTY (code comment + UI disclosure).
- TDD for all physics/server code; CI green on `windows-latest` for every push; conventional commits; MIT license.
- Conda is at `C:\ProgramData\miniforge3` (NOT on PATH). Every Python/npm command runs through it. From repo root `C:\Users\yashg\OneDrive\Desktop\atom_sim`:
  - Tests: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q`
  - Lint: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .`
  - npm (after Task 3 env update): `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm <args>` (run from the `web\` directory for web commands).
- `web/node_modules` must be a directory junction to `%LOCALAPPDATA%\atomsim\web_node_modules` (OneDrive must never sync it), created by `scripts/setup_web_node_modules.ps1` (Task 9).
- Server binds 127.0.0.1 only. Canonical JSON for `Quantity`/`Field`/`Provenance` is defined once in `src/atomsim/server/schemas.py` and mirrored exactly by `web/src/api/types.ts`.
- TypeScript `strict: true`; no `any` in committed code.
- Ruff config: line length 100, `E741` ignored (l is a quantum number); keep imports sorted (rule `I`).

## File Structure

```
atom_sim/
├── pyproject.toml                     # MODIFY: license form, deps, console script
├── environment.yml                    # MODIFY: fastapi, uvicorn, httpx, nodejs
├── .gitignore                         # MODIFY: web/node_modules, web/dist
├── .github/workflows/ci.yml           # MODIFY: SHA pins + web job
├── scripts/setup_web_node_modules.ps1 # CREATE: OneDrive-safe node_modules junction
├── src/atomsim/
│   ├── provenance.py                  # MODIFY: add Field
│   ├── analytic/hydrogen.py           # MODIFY: radial_wavefunction→Field, mean_radius→Quantity
│   ├── numerics/radial_solver.py      # MODIFY: RadialSolution gains provenance
│   ├── sampling.py                    # CREATE: SampleCloud, sample_density
│   ├── cli.py                         # CREATE: `atomsim serve`
│   └── server/
│       ├── __init__.py                # CREATE
│       ├── schemas.py                 # CREATE: ProvenanceModel, QuantityModel, FieldModel
│       ├── jobs.py                    # CREATE: Job, JobStatus, JobStore
│       └── app.py                     # CREATE: create_app(), endpoints, WS, static mount
├── tests/
│   ├── test_provenance.py             # MODIFY: Field tests
│   ├── test_hydrogen_analytic.py      # MODIFY: .values/.value migration
│   ├── test_radial_solver.py          # MODIFY: .values migration + solution provenance test
│   ├── test_constants.py              # MODIFY: CODATA tolerance (deferred minor)
│   ├── test_sampling.py               # CREATE
│   ├── test_schemas.py                # CREATE
│   ├── test_jobs.py                   # CREATE
│   ├── test_server.py                 # CREATE
│   └── test_cli.py                    # CREATE
└── web/                               # CREATE (Task 9 onward)
    ├── package.json, tsconfig.json, vite.config.ts, index.html
    └── src/
        ├── main.tsx, App.tsx, index.css
        ├── api/types.ts, api/client.ts, api/decode.test.ts
        ├── lib/quantum.ts, lib/quantum.test.ts
        ├── state/store.ts
        └── components/{Badge,InfoPanel,Controls,PointCloud}.tsx
```

---

### Task 1: `Field`: array-valued quantities with provenance

**Files:**
- Modify: `src/atomsim/provenance.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Consumes: existing `Provenance`, `Fidelity`.
- Produces: `Field(values: np.ndarray, grid: np.ndarray, unit: str, grid_unit: str, label: str, provenance: Provenance)`, frozen dataclass; validates `grid` is 1-D and `values.shape[-1] == grid.shape[0]`; coerces both to `np.ndarray`. Tasks 2 and 5 rely on these exact attribute names.

- [x] **Step 1: Write the failing tests**, in `tests/test_provenance.py`: add `import numpy as np` on its own line directly below the existing `import dataclasses`; change the existing import line to `from atomsim.provenance import Fidelity, Field, Provenance, Quantity`; then append:

```python
def _prov():
    return Provenance(fidelity=Fidelity.EXACT, method="closed-form test fixture")

def test_field_carries_values_grid_and_provenance():
    r = np.linspace(0.1, 10.0, 50)
    f = Field(
        values=np.exp(-r),
        grid=r,
        unit="bohr^-3/2",
        grid_unit="bohr",
        label="R_1,0",
        provenance=_prov(),
    )
    assert f.values.shape == (50,)
    assert f.grid_unit == "bohr"
    assert f.provenance.fidelity is Fidelity.EXACT

def test_field_accepts_stacked_values_with_matching_last_axis():
    r = np.linspace(0.0, 1.0, 20)
    f = Field(
        values=np.zeros((3, 20)), grid=r, unit="", grid_unit="bohr",
        label="u_k", provenance=_prov(),
    )
    assert f.values.shape == (3, 20)

def test_field_rejects_mismatched_grid_length():
    with pytest.raises(ValueError, match="grid"):
        Field(
            values=np.zeros(4), grid=np.zeros(5), unit="", grid_unit="bohr",
            label="bad", provenance=_prov(),
        )

def test_field_rejects_non_1d_grid():
    with pytest.raises(ValueError, match="1-D"):
        Field(
            values=np.zeros(4), grid=np.zeros((2, 2)), unit="", grid_unit="bohr",
            label="bad", provenance=_prov(),
        )
```

(Adjust the existing top-of-file import to `from atomsim.provenance import Fidelity, Field, Provenance, Quantity`, keep whatever names the file already imports plus `Field`; keep import sorting.)

- [x] **Step 2: Run tests to verify they fail**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_provenance.py -q`
Expected: FAIL, `ImportError: cannot import name 'Field'`.

- [x] **Step 3: Implement `Field`**, in `src/atomsim/provenance.py`, add `import numpy as np` below the existing imports, and append:

```python
@dataclass(frozen=True)
class Field:
    """Array-valued physical quantity: samples of a function on a 1-D grid.

    Completes the boundary rule: every physical value crossing a module
    boundary is a Quantity (scalar), a Field (array), or a container that
    carries its own Provenance.
    """

    values: np.ndarray
    grid: np.ndarray
    unit: str
    grid_unit: str
    label: str
    provenance: Provenance

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        grid = np.asarray(self.grid)
        if grid.ndim != 1:
            raise ValueError(f"grid must be 1-D, got shape {grid.shape}")
        if values.shape[-1] != grid.shape[0]:
            raise ValueError(
                f"values last axis ({values.shape[-1]}) must match "
                f"grid length ({grid.shape[0]})"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "grid", grid)
```

Also update the module docstring's first line to mention the rule covers scalars (`Quantity`) and arrays (`Field`).

- [x] **Step 4: Run tests to verify they pass**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_provenance.py -q`
Expected: PASS (all, including pre-existing tests).

- [x] **Step 5: Lint and commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
git add src/atomsim/provenance.py tests/test_provenance.py
git commit -m "feat: Field type - array-valued quantities with provenance"
```

---

### Task 2: Migrate analytic + solver boundaries to the amended rule

**Files:**
- Modify: `src/atomsim/analytic/hydrogen.py` (`radial_wavefunction`, `mean_radius`)
- Modify: `src/atomsim/numerics/radial_solver.py` (`RadialSolution`, `solve_radial`)
- Test: `tests/test_hydrogen_analytic.py`, `tests/test_radial_solver.py`

**Interfaces:**
- Consumes: `Field` from Task 1.
- Produces (exact signatures later tasks rely on):
  - `radial_wavefunction(n, l, r, Z=1, mu_ratio=1.0) -> Field`, `.values` = R_nl on `.grid` = r, `unit="bohr^-3/2"`, `grid_unit="bohr"`, EXACT provenance.
  - `mean_radius(n, l, Z=1, mu_ratio=1.0) -> Quantity`, `.value` in `unit="bohr"`, EXACT provenance.
  - `RadialSolution` gains field `provenance: Provenance` (NUMERICAL, same method text as its energies).

- [x] **Step 1: Update the tests to the new boundary types**

In `tests/test_hydrogen_analytic.py`:
- Add `Field` to the provenance import: `from atomsim.provenance import Fidelity, Field`.
- `test_radial_wavefunctions_are_normalized`: `norm = np.trapezoid(radial_wavefunction(n, l, r).values ** 2 * r**2, r)`
- `test_node_counts_are_n_minus_l_minus_1`: `R = radial_wavefunction(n, l, r).values`
- `test_mean_radius_matches_exact_formula_and_integral`: `R = radial_wavefunction(n, l, r).values`; `exact = mean_radius(n, l).value`; final line `assert mean_radius(1, 0).value == pytest.approx(1.5)`
- `test_orthogonality_same_l`: multiply `.values` of both calls.
- `test_scaling_with_z_and_reduced_mass`: `mean_radius(1, 0, Z=2).value` and `mean_radius(1, 0, mu_ratio=2.0).value`.
- Append two new tests:

```python
def test_radial_wavefunction_returns_exact_field():
    r = np.linspace(1e-6, 20.0, 500)
    f = radial_wavefunction(2, 1, r)
    assert isinstance(f, Field)
    assert f.unit == "bohr^-3/2"
    assert f.grid_unit == "bohr"
    assert f.provenance.fidelity is Fidelity.EXACT
    assert np.array_equal(f.grid, r)

def test_mean_radius_returns_exact_quantity():
    q = mean_radius(2, 1)
    assert q.unit == "bohr"
    assert q.provenance.fidelity is Fidelity.EXACT
    assert "3" in q.provenance.method  # states the closed-form formula
```

In `tests/test_radial_solver.py`:
- `test_numerical_1s_wavefunction_overlaps_analytic`: `u_exact = sol.r * radial_wavefunction(1, 0, sol.r).values`
- Append:

```python
def test_solution_carries_its_own_provenance():
    sol = solve_radial(lambda r: 0.5 * r**2, l=0, r_max=12.0, n_points=1200, n_states=1)
    assert sol.provenance.fidelity is Fidelity.NUMERICAL
    assert "finite-difference" in sol.provenance.method
```

- [x] **Step 2: Run tests to verify they fail**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_hydrogen_analytic.py tests/test_radial_solver.py -q`
Expected: FAIL, `AttributeError: 'numpy.ndarray' object has no attribute 'values'` (and no `provenance` on `RadialSolution`).

- [x] **Step 3: Implement the migration**

`src/atomsim/analytic/hydrogen.py`, change the import to `from atomsim.provenance import Fidelity, Field, Provenance, Quantity`; replace the two functions:

```python
def radial_wavefunction(
    n: int, l: int, r: np.ndarray, Z: int = 1, mu_ratio: float = 1.0
) -> Field:
    """Normalized radial wavefunction R_nl(r) in atomic units (r in bohr).

    R_nl = N * exp(-rho/2) * rho^l * L_{n-l-1}^{2l+1}(rho),  rho = 2 Z mu' r / n.
    """
    validate_quantum_numbers(n, l)
    _validate_physical(Z, mu_ratio)
    kappa = Z * mu_ratio
    grid = np.asarray(r, dtype=float)
    rho = 2.0 * kappa * grid / n
    norm = math.sqrt(
        (2.0 * kappa / n) ** 3
        * math.factorial(n - l - 1)
        / (2.0 * n * math.factorial(n + l))
    )
    values = norm * np.exp(-rho / 2.0) * rho**l * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    return Field(
        values=values,
        grid=grid,
        unit="bohr^-3/2",
        grid_unit="bohr",
        label=f"R_{n},{l} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method="closed-form R_nl (normalized generalized-Laguerre form)",
            assumptions=_EXACT_ASSUMPTIONS
            + ("float64 Laguerre evaluation reliable for n <= 20",),
        ),
    )

def mean_radius(n: int, l: int, Z: int = 1, mu_ratio: float = 1.0) -> Quantity:
    """Exact <r> = (3 n^2 - l(l+1)) / (2 Z mu'), in bohr."""
    validate_quantum_numbers(n, l)
    _validate_physical(Z, mu_ratio)
    value = (3.0 * n**2 - l * (l + 1)) / (2.0 * Z * mu_ratio)
    return Quantity(
        value=value,
        unit="bohr",
        label=f"<r>_{n},{l} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method="closed-form <r> = (3 n^2 - l(l+1)) / (2 Z mu')",
            assumptions=_EXACT_ASSUMPTIONS,
        ),
    )
```

`src/atomsim/numerics/radial_solver.py`, add `provenance: Provenance` as the last field of `RadialSolution`, and in `solve_radial` return `RadialSolution(r=r, u=u, energies=energies, l=l, mu_ratio=mu_ratio, provenance=provenance)` (the `provenance` object already exists in the function). `solve_radial_with_error` needs no change (`dataclasses.replace(fine, energies=energies)` preserves the new field).

- [x] **Step 4: Run the full suite**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q`
Expected: PASS, all tests (39 pre-existing, updated in place, plus 3 new).

- [x] **Step 5: Lint and commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
git add src/atomsim/analytic/hydrogen.py src/atomsim/numerics/radial_solver.py tests/test_hydrogen_analytic.py tests/test_radial_solver.py
git commit -m "feat: complete provenance boundary - Field wavefunctions, Quantity mean radius, solution provenance"
```

---

### Task 3: Dependencies, deferred minors, environment update

**Files:**
- Modify: `pyproject.toml`, `environment.yml`, `tests/test_constants.py`

**Interfaces:**
- Consumes: nothing.
- Produces: env `atomsim` gains `fastapi`, `uvicorn`, `httpx`, `nodejs` 22 (so `npm` exists inside the env); pyproject uses PEP 639 license; CODATA test survives scipy bumps.

- [x] **Step 1: pyproject.toml changes**

Replace `requires = ["setuptools>=68"]` with `requires = ["setuptools>=77"]` (PEP 639 support). Replace `license = { text = "MIT" }` with `license = "MIT"`. Replace the dependency lines:

```toml
dependencies = ["numpy>=2.0", "scipy>=1.13", "fastapi>=0.115", "uvicorn>=0.30"]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov", "ruff", "httpx>=0.27"]
```

- [x] **Step 2: environment.yml**, replace the dependencies block:

```yaml
name: atomsim
channels:
  - conda-forge
dependencies:
  - python=3.12
  - numpy>=2.0
  - scipy>=1.13
  - fastapi>=0.115
  - uvicorn>=0.30
  - httpx>=0.27
  - nodejs=22
  - pytest>=8
  - pytest-cov
  - ruff
  - pip
  - pip:
      - -e .[dev]
```

- [x] **Step 3: Deferred minor, CODATA tolerance** in `tests/test_constants.py`:

```python
def test_hartree_ev_matches_codata():
    # CODATA vintage rides with scipy (2022 value shown); 1e-6 abs tolerance
    # absorbs future CODATA revisions without silently accepting real bugs.
    assert abs(HARTREE_EV - 27.211386) < 1e-6
```

- [x] **Step 4: Update the conda env and verify**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" env update -n atomsim -f environment.yml
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -c "import fastapi, uvicorn, httpx; print('py deps ok')"
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm --version
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
```

Expected: `py deps ok`; an npm version ≥ 10; full test suite PASS.

- [x] **Step 5: Commit**

```powershell
git add pyproject.toml environment.yml tests/test_constants.py
git commit -m "chore: fastapi/uvicorn/httpx/nodejs deps, PEP 639 license, CODATA tolerance"
```

---

### Task 4: Monte-Carlo sampler: `atomsim.sampling`

**Files:**
- Create: `src/atomsim/sampling.py`
- Test: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `radial_wavefunction` (Task 2), `validate_quantum_numbers`, `Provenance`/`Fidelity`.
- Produces (Task 7 relies on these exact names):
  - `SampleCloud` frozen dataclass: `positions: np.ndarray` (shape `(count, 3)`, float32, bohr), `n: int`, `l: int`, `m: int`, `Z: int`, `mu_ratio: float`, `provenance: Provenance`.
  - `sample_density(n, l, m, count, Z=1, mu_ratio=1.0, seed=0, progress=None, n_chunks=10) -> SampleCloud`, `progress` is an optional `Callable[[float], None]` called after each chunk with the completed fraction (last call = 1.0).

- [x] **Step 1: Write the failing tests**, `tests/test_sampling.py`:

```python
import numpy as np
import pytest
from scipy.stats import kstest

from atomsim.provenance import Fidelity
from atomsim.sampling import SampleCloud, sample_density

COUNT = 100_000

def _radii(cloud: SampleCloud) -> np.ndarray:
    return np.linalg.norm(cloud.positions.astype(float), axis=1)

def test_positions_shape_dtype_and_metadata():
    cloud = sample_density(2, 1, 0, count=5_000, seed=1)
    assert cloud.positions.shape == (5_000, 3)
    assert cloud.positions.dtype == np.float32
    assert (cloud.n, cloud.l, cloud.m) == (2, 1, 0)
    assert np.isfinite(cloud.positions).all()

def test_provenance_is_numerical_and_states_seed_and_count():
    cloud = sample_density(1, 0, 0, count=2_000, seed=7)
    assert cloud.provenance.fidelity is Fidelity.NUMERICAL
    joined = " ".join(cloud.provenance.assumptions)
    assert "seed=7" in joined
    assert "2000" in joined.replace(",", "").replace("_", "")

def test_1s_radial_distribution_ks_against_analytic_cdf():
    # 1s: F(r) = 1 - exp(-2r) (1 + 2r + 2r^2)
    cloud = sample_density(1, 0, 0, count=COUNT, seed=42)
    r = _radii(cloud)
    ks = kstest(r, lambda x: 1.0 - np.exp(-2.0 * x) * (1.0 + 2.0 * x + 2.0 * x**2))
    assert ks.statistic < 0.01, ks

def test_1s_mean_radius():
    r = _radii(sample_density(1, 0, 0, count=COUNT, seed=42))
    assert r.mean() == pytest.approx(1.5, abs=0.02)  # <r>_1s = 1.5 bohr

def test_2p_mean_radius():
    r = _radii(sample_density(2, 1, 0, count=COUNT, seed=3))
    assert r.mean() == pytest.approx(5.0, abs=0.05)  # <r>_2,1 = 5 bohr

def test_1s_angular_isotropy():
    cloud = sample_density(1, 0, 0, count=COUNT, seed=11)
    r = _radii(cloud)
    cos_theta = cloud.positions[:, 2].astype(float) / r
    assert cos_theta.mean() == pytest.approx(0.0, abs=0.01)
    assert (cos_theta**2).mean() == pytest.approx(1.0 / 3.0, abs=0.01)

def test_2p_m0_angular_distribution():
    # |Y_10|^2 ~ cos^2(theta): pdf over x=cos(theta) is (3/2) x^2 -> E[x^2] = 3/5
    cloud = sample_density(2, 1, 0, count=COUNT, seed=5)
    r = _radii(cloud)
    cos_theta = cloud.positions[:, 2].astype(float) / r
    assert (cos_theta**2).mean() == pytest.approx(0.6, abs=0.01)

def test_seed_reproducibility():
    a = sample_density(3, 2, 1, count=1_000, seed=99)
    b = sample_density(3, 2, 1, count=1_000, seed=99)
    assert np.array_equal(a.positions, b.positions)

def test_progress_callback_monotonic_and_complete():
    calls: list[float] = []
    sample_density(1, 0, 0, count=10_000, seed=0, progress=calls.append, n_chunks=10)
    assert len(calls) == 10
    assert calls[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(calls, calls[1:]))

def test_rejects_invalid_quantum_numbers():
    with pytest.raises(ValueError):
        sample_density(1, 1, 0, count=100)   # l == n
    with pytest.raises(ValueError):
        sample_density(2, 1, 2, count=100)   # |m| > l
    with pytest.raises(ValueError):
        sample_density(1, 0, 0, count=0)     # count must be positive
```

- [x] **Step 2: Run tests to verify they fail**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_sampling.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.sampling'`.

- [x] **Step 3: Implement**, `src/atomsim/sampling.py`:

```python
"""Monte-Carlo sampling of |psi_nlm|^2, sampling IS physics and carries provenance.

Factorized inverse-CDF sampling in the complex spherical-harmonic basis:
r from P(r) = r^2 R_nl^2, cos(theta) from the normalized |Theta_lm|^2, and
phi uniform (|Y_lm|^2 is phi-independent for complex Y_lm). Real-orbital
sampling (phi-dependent) arrives with the M2 angular module.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import lpmv

from atomsim.analytic.hydrogen import radial_wavefunction, validate_quantum_numbers
from atomsim.provenance import Fidelity, Provenance

_R_GRID_POINTS = 8192
_X_GRID_POINTS = 4096

@dataclass(frozen=True)
class SampleCloud:
    """Positions sampled from |psi_nlm|^2, in bohr. Container carries provenance."""

    positions: np.ndarray  # (count, 3) float32
    n: int
    l: int
    m: int
    Z: int
    mu_ratio: float
    provenance: Provenance

def _radial_inverse_cdf(n: int, l: int, Z: int, mu_ratio: float):
    """Grid r and CDF of P(r) = r^2 R_nl^2 for inverse-CDF sampling."""
    r_max = 20.0 * n * n / (Z * mu_ratio)  # P(r_max)/P_peak < 1e-15 for all l < n
    r = np.linspace(0.0, r_max, _R_GRID_POINTS)
    R = radial_wavefunction(n, l, r, Z=Z, mu_ratio=mu_ratio).values
    p = r * r * R * R
    cdf = cumulative_trapezoid(p, r, initial=0.0)
    cdf /= cdf[-1]
    return r, cdf, r_max

def _costheta_inverse_cdf(l: int, m: int):
    """Grid x = cos(theta) and CDF of |Theta_lm|^2 (normalization cancels)."""
    x = np.linspace(-1.0, 1.0, _X_GRID_POINTS)
    p = lpmv(abs(m), l, x) ** 2
    cdf = cumulative_trapezoid(p, x, initial=0.0)
    cdf /= cdf[-1]
    return x, cdf

def sample_density(
    n: int,
    l: int,
    m: int,
    count: int,
    Z: int = 1,
    mu_ratio: float = 1.0,
    seed: int = 0,
    progress: Callable[[float], None] | None = None,
    n_chunks: int = 10,
) -> SampleCloud:
    """Draw `count` positions from |psi_nlm|^2 (complex Y_lm basis)."""
    validate_quantum_numbers(n, l)
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")

    rng = np.random.default_rng(seed)
    r_grid, r_cdf, r_max = _radial_inverse_cdf(n, l, Z, mu_ratio)
    x_grid, x_cdf = _costheta_inverse_cdf(l, m)

    sizes = np.full(n_chunks, count // n_chunks)
    sizes[: count % n_chunks] += 1
    chunks: list[np.ndarray] = []
    done = 0
    for size in sizes:
        if size == 0:
            if progress is not None:
                progress(done / count if count else 1.0)
            continue
        r = np.interp(rng.random(size), r_cdf, r_grid)
        cos_t = np.interp(rng.random(size), x_cdf, x_grid)
        sin_t = np.sqrt(np.clip(1.0 - cos_t**2, 0.0, 1.0))
        phi = rng.uniform(0.0, 2.0 * np.pi, size)
        xyz = np.stack(
            [r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t], axis=1
        )
        chunks.append(xyz.astype(np.float32))
        done += int(size)
        if progress is not None:
            progress(done / count)

    positions = np.concatenate(chunks)
    provenance = Provenance(
        fidelity=Fidelity.NUMERICAL,
        method=(
            "factorized inverse-CDF Monte-Carlo of |psi_nlm|^2: "
            f"r from P(r)=r^2 R^2 (grid N={_R_GRID_POINTS}, r_max={r_max:g} bohr), "
            f"cos(theta) from |Theta_lm|^2 (grid N={_X_GRID_POINTS}), phi uniform"
        ),
        assumptions=(
            "complex spherical-harmonic basis (|Y_lm|^2 is phi-independent)",
            f"RNG PCG64 seed={seed}, count={count}",
            "positions in bohr",
        ),
        refinement="increase CDF grid resolution or sample count",
    )
    return SampleCloud(
        positions=positions, n=n, l=l, m=m, Z=Z, mu_ratio=mu_ratio, provenance=provenance
    )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_sampling.py -q`
Expected: PASS (11 tests; the 100k-sample tests take ~1 s total).

- [x] **Step 5: Lint, full suite, commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
git add src/atomsim/sampling.py tests/test_sampling.py
git commit -m "feat: Monte-Carlo |psi|^2 sampler with inverse-CDF method and provenance"
```

---

### Task 5: Canonical JSON: `atomsim.server.schemas`

**Files:**
- Create: `src/atomsim/server/__init__.py` (empty docstring module), `src/atomsim/server/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: `Provenance`, `Quantity`, `Field` dataclasses.
- Produces (Tasks 7 and 10 rely on these exact shapes):
  - `ProvenanceModel(fidelity: Literal["exact","numerical","approximation","counterfactual","visual_liberty"], method: str, assumptions: list[str], error_estimate: float | None, refinement: str | None)` with classmethod `from_provenance(p: Provenance)`.
  - `QuantityModel(value: float, unit: str, label: str, provenance: ProvenanceModel)` with `from_quantity(q)`.
  - `FieldModel(values: list[float], grid: list[float], unit: str, grid_unit: str, label: str, provenance: ProvenanceModel)` with `from_field(f)` (1-D fields only; raises `ValueError` for stacked values, M3 extends).

- [x] **Step 1: Write the failing tests**, `tests/test_schemas.py`:

```python
import json

import numpy as np
import pytest

from atomsim.provenance import Fidelity, Field, Provenance, Quantity
from atomsim.server.schemas import FieldModel, ProvenanceModel, QuantityModel

def _prov():
    return Provenance(
        fidelity=Fidelity.NUMERICAL,
        method="test method",
        assumptions=("a1", "a2"),
        error_estimate=1e-6,
        refinement="refine",
    )

def test_provenance_round_trip():
    model = ProvenanceModel.from_provenance(_prov())
    data = json.loads(model.model_dump_json())
    assert data["fidelity"] == "numerical"
    assert data["assumptions"] == ["a1", "a2"]
    assert data["error_estimate"] == pytest.approx(1e-6)
    assert data["refinement"] == "refine"

def test_quantity_serializes_with_provenance():
    q = Quantity(value=-0.5, unit="hartree", label="E_1", provenance=_prov())
    data = json.loads(QuantityModel.from_quantity(q).model_dump_json())
    assert data["value"] == -0.5
    assert data["unit"] == "hartree"
    assert data["provenance"]["method"] == "test method"

def test_field_serializes_values_and_grid():
    r = np.linspace(0.0, 1.0, 5)
    f = Field(values=2.0 * r, grid=r, unit="u", grid_unit="bohr", label="f", provenance=_prov())
    data = json.loads(FieldModel.from_field(f).model_dump_json())
    assert data["grid"] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert data["values"] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])

def test_field_model_rejects_stacked_values():
    r = np.linspace(0.0, 1.0, 4)
    f = Field(values=np.zeros((2, 4)), grid=r, unit="", grid_unit="bohr",
              label="u", provenance=_prov())
    with pytest.raises(ValueError, match="1-D"):
        FieldModel.from_field(f)

def test_every_fidelity_value_is_representable():
    for fid in Fidelity:
        p = Provenance(fidelity=fid, method="m")
        assert ProvenanceModel.from_provenance(p).fidelity == fid.value
```

- [x] **Step 2: Run tests to verify they fail**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_schemas.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.server'`.

- [x] **Step 3: Implement**

`src/atomsim/server/__init__.py`:

```python
"""Local FastAPI server: the engine's honesty-preserving boundary to the browser."""
```

`src/atomsim/server/schemas.py`:

```python
"""THE canonical JSON forms of Provenance, Quantity, and Field.

Defined exactly once here; web/src/api/types.ts mirrors these shapes.
Provenance reaches the browser by construction, never as an afterthought.
"""

from typing import Literal

from pydantic import BaseModel

from atomsim.provenance import Field, Provenance, Quantity

FidelityName = Literal[
    "exact", "numerical", "approximation", "counterfactual", "visual_liberty"
]

class ProvenanceModel(BaseModel):
    fidelity: FidelityName
    method: str
    assumptions: list[str]
    error_estimate: float | None
    refinement: str | None

    @classmethod
    def from_provenance(cls, p: Provenance) -> "ProvenanceModel":
        return cls(
            fidelity=p.fidelity.value,
            method=p.method,
            assumptions=list(p.assumptions),
            error_estimate=p.error_estimate,
            refinement=p.refinement,
        )

class QuantityModel(BaseModel):
    value: float
    unit: str
    label: str
    provenance: ProvenanceModel

    @classmethod
    def from_quantity(cls, q: Quantity) -> "QuantityModel":
        return cls(
            value=q.value,
            unit=q.unit,
            label=q.label,
            provenance=ProvenanceModel.from_provenance(q.provenance),
        )

class FieldModel(BaseModel):
    values: list[float]
    grid: list[float]
    unit: str
    grid_unit: str
    label: str
    provenance: ProvenanceModel

    @classmethod
    def from_field(cls, f: Field) -> "FieldModel":
        if f.values.ndim != 1:
            raise ValueError(f"only 1-D fields serialize in M1, got shape {f.values.shape}")
        return cls(
            values=f.values.tolist(),
            grid=f.grid.tolist(),
            unit=f.unit,
            grid_unit=f.grid_unit,
            label=f.label,
            provenance=ProvenanceModel.from_provenance(f.provenance),
        )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_schemas.py -q`
Expected: PASS (6 tests).

- [x] **Step 5: Lint and commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
git add src/atomsim/server/__init__.py src/atomsim/server/schemas.py tests/test_schemas.py
git commit -m "feat: canonical JSON schemas for Provenance, Quantity, Field"
```

---

### Task 6: Job store: `atomsim.server.jobs`

**Files:**
- Create: `src/atomsim/server/jobs.py`
- Test: `tests/test_jobs.py`

**Interfaces:**
- Consumes: stdlib only.
- Produces (Task 7 relies on these exact names):
  - `JobStatus` enum: `PENDING="pending"`, `RUNNING="running"`, `DONE="done"`, `ERROR="error"`.
  - `Job` dataclass: `id: str`, `status: JobStatus`, `progress: float`, `result: Any`, `error: str | None`.
  - `JobStore` with `create() -> Job`, `get(job_id: str) -> Job | None`, `run(job_id: str, fn: Callable[[Callable[[float], None]], Any]) -> None` (synchronous; the caller decides the thread; `fn` receives a progress callback and returns the result).

- [x] **Step 1: Write the failing tests**, `tests/test_jobs.py`:

```python
import pytest

from atomsim.server.jobs import JobStatus, JobStore

def test_create_and_get():
    store = JobStore()
    job = store.create()
    assert job.status is JobStatus.PENDING
    assert store.get(job.id) is job
    assert store.get("nope") is None

def test_run_success_sets_done_result_and_full_progress():
    store = JobStore()
    job = store.create()
    seen: list[float] = []

    def work(progress):
        progress(0.5)
        seen.append(store.get(job.id).progress)
        return "payload"

    store.run(job.id, work)
    assert seen == [0.5]  # progress visible to observers mid-run
    assert job.status is JobStatus.DONE
    assert job.result == "payload"
    assert job.progress == pytest.approx(1.0)

def test_run_failure_sets_error_status_and_message():
    store = JobStore()
    job = store.create()

    def bad(progress):
        raise ValueError("boom")

    store.run(job.id, bad)
    assert job.status is JobStatus.ERROR
    assert "ValueError" in job.error and "boom" in job.error
    assert job.result is None

def test_progress_is_clamped_to_unit_interval():
    store = JobStore()
    job = store.create()

    def work(progress):
        progress(7.0)
        assert job.progress == 1.0
        progress(-3.0)
        assert job.progress == 0.0
        return None

    store.run(job.id, work)

def test_run_unknown_job_raises():
    store = JobStore()
    with pytest.raises(KeyError):
        store.run("nope", lambda progress: None)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_jobs.py -q`
Expected: FAIL, `ModuleNotFoundError`.

- [x] **Step 3: Implement**, `src/atomsim/server/jobs.py`:

```python
"""Minimal in-memory async-job pattern: create -> run (in any thread) -> poll/stream.

Deliberately simple for a single-user local app; the same pattern later
carries plane-density grids and volumetrics.
"""

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"

@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    result: Any = None
    error: str | None = None

class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def run(self, job_id: str, fn: Callable[[Callable[[float], None]], Any]) -> None:
        """Execute fn in the calling thread, streaming progress into the job."""
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"unknown job id: {job_id}")
        job.status = JobStatus.RUNNING

        def report(fraction: float) -> None:
            job.progress = min(max(fraction, 0.0), 1.0)

        try:
            job.result = fn(report)
        except Exception as exc:  # honest failure: surface type + message
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = JobStatus.ERROR
        else:
            job.progress = 1.0
            job.status = JobStatus.DONE
```

- [x] **Step 4: Run tests to verify they pass**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_jobs.py -q`
Expected: PASS (5 tests).

- [x] **Step 5: Lint and commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
git add src/atomsim/server/jobs.py tests/test_jobs.py
git commit -m "feat: in-memory job store with progress reporting"
```

---

### Task 7: FastAPI app: endpoints, WebSocket progress, binary transfer

**Files:**
- Create: `src/atomsim/server/app.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `energy`, `mean_radius`, `validate_quantum_numbers` (Task 2), `HARTREE_EV`, `sample_density`/`SampleCloud` (Task 4), schemas (Task 5), `JobStore`/`JobStatus` (Task 6).
- Produces (Task 10's TS client mirrors these exactly):
  - `create_app() -> FastAPI` with `app.state.jobs: JobStore` exposed for tests.
  - `GET /api/health` → `{"status": "ok", "version": "<atomsim.__version__>"}`
  - `GET /api/state/{n}/{l}/{m}` → `{n, l, m, energy: QuantityModel, energy_ev: QuantityModel, mean_radius: QuantityModel}`; 422 on invalid quantum numbers.
  - `POST /api/jobs/sample` body `{n, l, m, count?, seed?}` (count default 100000, 1000-1000000) → `JobModel {id, status, progress, error}`.
  - `GET /api/jobs/{job_id}` → `JobModel`; 404 unknown.
  - `GET /api/jobs/{job_id}/meta` → `{count, dtype: "float32", layout: "xyz-interleaved", unit: "bohr", n, l, m, provenance}`; 404 unknown, 409 not done.
  - `GET /api/jobs/{job_id}/data` → `application/octet-stream`, `count*12` bytes; 404/409 as meta.
  - `WS /ws/jobs/{job_id}` → JSON messages `{status, progress, error}` every 0.1 s until terminal, then closes.
  - Static mount of `web/dist` at `/` when that directory exists (repo root resolved as `parents[3]` of `app.py`, valid for the editable-install-from-clone delivery model).

- [x] **Step 1: Write the failing tests**, `tests/test_server.py`:

```python
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import atomsim
from atomsim.server.app import create_app
from atomsim.server.jobs import JobStatus

@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c

def _wait_done(client, job_id, deadline_s=30.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline_s:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish")

def test_health_reports_version(client):
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "version": atomsim.__version__}

def test_state_carries_exact_provenance(client):
    r = client.get("/api/state/2/1/0")
    assert r.status_code == 200
    body = r.json()
    assert body["energy"]["value"] == pytest.approx(-0.125)
    assert body["energy"]["unit"] == "hartree"
    assert body["energy"]["provenance"]["fidelity"] == "exact"
    assert body["energy_ev"]["unit"] == "eV"
    assert body["energy_ev"]["value"] == pytest.approx(-3.40, abs=0.01)
    assert body["mean_radius"]["value"] == pytest.approx(5.0)

def test_state_rejects_invalid_quantum_numbers(client):
    assert client.get("/api/state/1/1/0").status_code == 422   # l == n
    assert client.get("/api/state/2/1/2").status_code == 422   # |m| > l
    assert client.get("/api/state/0/0/0").status_code == 422   # n < 1

def test_sample_job_end_to_end(client):
    r = client.post(
        "/api/jobs/sample", json={"n": 1, "l": 0, "m": 0, "count": 5000, "seed": 7}
    )
    assert r.status_code == 200
    job_id = r.json()["id"]

    final = _wait_done(client, job_id)
    assert final["status"] == "done"
    assert final["progress"] == pytest.approx(1.0)

    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["count"] == 5000
    assert meta["dtype"] == "float32"
    assert meta["layout"] == "xyz-interleaved"
    assert meta["unit"] == "bohr"
    assert meta["provenance"]["fidelity"] == "numerical"

    raw = client.get(f"/api/jobs/{job_id}/data")
    assert raw.headers["content-type"].startswith("application/octet-stream")
    positions = np.frombuffer(raw.content, dtype=np.float32).reshape(-1, 3)
    assert positions.shape == (5000, 3)
    assert np.isfinite(positions).all()

def test_sample_rejects_invalid_body(client):
    assert (
        client.post("/api/jobs/sample", json={"n": 1, "l": 1, "m": 0}).status_code == 422
    )
    assert (
        client.post(
            "/api/jobs/sample", json={"n": 1, "l": 0, "m": 0, "count": 10}
        ).status_code
        == 422
    )  # below pydantic ge=1000

def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404
    assert client.get("/api/jobs/deadbeef/meta").status_code == 404
    assert client.get("/api/jobs/deadbeef/data").status_code == 404

def test_meta_before_done_is_409(client):
    job = client.app.state.jobs.create()  # created but never run
    assert client.get(f"/api/jobs/{job.id}/meta").status_code == 409
    assert client.get(f"/api/jobs/{job.id}/data").status_code == 409

def test_websocket_streams_progress_to_done(client):
    r = client.post(
        "/api/jobs/sample", json={"n": 2, "l": 1, "m": 0, "count": 20000, "seed": 1}
    )
    job_id = r.json()["id"]
    messages = []
    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        for _ in range(600):
            msg = ws.receive_json()
            messages.append(msg)
            if msg["status"] in ("done", "error"):
                break
    assert messages[-1]["status"] == "done"
    assert messages[-1]["progress"] == pytest.approx(1.0)

def test_job_error_surfaces_via_status(client):
    jobs = client.app.state.jobs
    job = jobs.create()

    def bad(progress):
        raise RuntimeError("sampler exploded")

    jobs.run(job.id, bad)
    body = client.get(f"/api/jobs/{job.id}").json()
    assert body["status"] == "error"
    assert "sampler exploded" in body["error"]
    assert body["status"] == JobStatus.ERROR.value
```

- [x] **Step 2: Run tests to verify they fail**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_server.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.server.app'`.

- [x] **Step 3: Implement**, `src/atomsim/server/app.py`:

```python
"""The atomsim local server: honest JSON + binary boundaries for the browser app."""

import asyncio
import dataclasses
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field as PydanticField

import atomsim
from atomsim.analytic.hydrogen import energy, mean_radius, validate_quantum_numbers
from atomsim.constants import HARTREE_EV
from atomsim.provenance import Quantity
from atomsim.sampling import SampleCloud, sample_density
from atomsim.server.jobs import Job, JobStatus, JobStore
from atomsim.server.schemas import ProvenanceModel, QuantityModel

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

class StateResponse(BaseModel):
    n: int
    l: int
    m: int
    energy: QuantityModel
    energy_ev: QuantityModel
    mean_radius: QuantityModel

class SampleRequest(BaseModel):
    n: int
    l: int
    m: int
    count: int = PydanticField(default=100_000, ge=1_000, le=1_000_000)
    seed: int = 0

class JobModel(BaseModel):
    id: str
    status: str
    progress: float
    error: str | None

class SampleMetaModel(BaseModel):
    count: int
    dtype: str
    layout: str
    unit: str
    n: int
    l: int
    m: int
    provenance: ProvenanceModel

def _validate_state(n: int, l: int, m: int) -> None:
    try:
        validate_quantum_numbers(n, l)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if abs(m) > l:
        raise HTTPException(status_code=422, detail=f"|m| must be <= l, got m={m}, l={l}")

def _to_ev(q: Quantity) -> Quantity:
    return Quantity(
        value=q.value * HARTREE_EV,
        unit="eV",
        label=q.label + " [eV]",
        provenance=dataclasses.replace(
            q.provenance,
            method=q.provenance.method + "; converted to eV via CODATA Hartree-eV factor",
        ),
    )

def _job_model(job: Job) -> JobModel:
    return JobModel(id=job.id, status=job.status.value, progress=job.progress, error=job.error)

def _finished_cloud(jobs: JobStore, job_id: str) -> SampleCloud:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    if job.status is not JobStatus.DONE:
        raise HTTPException(status_code=409, detail=f"job is {job.status.value}, not done")
    return job.result

def create_app() -> FastAPI:
    app = FastAPI(title="atomsim", version=atomsim.__version__)
    jobs = JobStore()
    app.state.jobs = jobs
    app.add_middleware(
        CORSMiddleware, allow_origins=_DEV_ORIGINS, allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": atomsim.__version__}

    @app.get("/api/state/{n}/{l}/{m}", response_model=StateResponse)
    def state(n: int, l: int, m: int) -> StateResponse:
        _validate_state(n, l, m)
        e = energy(n)
        return StateResponse(
            n=n,
            l=l,
            m=m,
            energy=QuantityModel.from_quantity(e),
            energy_ev=QuantityModel.from_quantity(_to_ev(e)),
            mean_radius=QuantityModel.from_quantity(mean_radius(n, l)),
        )

    @app.post("/api/jobs/sample", response_model=JobModel)
    async def create_sample_job(req: SampleRequest) -> JobModel:
        _validate_state(req.n, req.l, req.m)
        job = jobs.create()

        def work(progress):
            return sample_density(
                req.n, req.l, req.m, req.count, seed=req.seed, progress=progress
            )

        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, jobs.run, job.id, work)
        return _job_model(job)

    @app.get("/api/jobs/{job_id}", response_model=JobModel)
    def job_status(job_id: str) -> JobModel:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
        return _job_model(job)

    @app.get("/api/jobs/{job_id}/meta", response_model=SampleMetaModel)
    def sample_meta(job_id: str) -> SampleMetaModel:
        cloud = _finished_cloud(jobs, job_id)
        return SampleMetaModel(
            count=cloud.positions.shape[0],
            dtype="float32",
            layout="xyz-interleaved",
            unit="bohr",
            n=cloud.n,
            l=cloud.l,
            m=cloud.m,
            provenance=ProvenanceModel.from_provenance(cloud.provenance),
        )

    @app.get("/api/jobs/{job_id}/data")
    def sample_data(job_id: str) -> Response:
        cloud = _finished_cloud(jobs, job_id)
        return Response(
            content=cloud.positions.tobytes(), media_type="application/octet-stream"
        )

    @app.websocket("/ws/jobs/{job_id}")
    async def job_progress(ws: WebSocket, job_id: str) -> None:
        await ws.accept()
        while True:
            job = jobs.get(job_id)
            if job is None:
                await ws.send_json({"status": "error", "progress": 0.0, "error": "unknown job"})
                break
            await ws.send_json(
                {"status": job.status.value, "progress": job.progress, "error": job.error}
            )
            if job.status in (JobStatus.DONE, JobStatus.ERROR):
                break
            await asyncio.sleep(0.1)
        await ws.close()

    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")

    return app
```

- [x] **Step 4: Run tests to verify they pass**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_server.py -q`
Expected: PASS (9 tests). If `test_websocket_streams_progress_to_done` flakes because the job finishes before the first WS frame: it still receives one terminal `done` frame, the assertion holds; no retry logic needed.

- [x] **Step 5: Lint, full suite, commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
git add src/atomsim/server/app.py tests/test_server.py
git commit -m "feat: FastAPI server - state endpoint, sample jobs, WS progress, binary transfer"
```

---

### Task 8: `atomsim serve` CLI

**Files:**
- Create: `src/atomsim/cli.py`
- Modify: `pyproject.toml` (console script)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `create_app` (Task 7).
- Produces: console command `atomsim serve [--port 8000] [--no-browser]`; `atomsim.cli.main(argv)` entry; `build_parser()`.

- [x] **Step 1: Write the failing tests**, `tests/test_cli.py`:

```python
import atomsim.cli as cli

def test_parser_defaults():
    args = cli.build_parser().parse_args(["serve"])
    assert args.command == "serve"
    assert args.port == 8000
    assert args.no_browser is False

def test_serve_invokes_uvicorn_on_loopback(monkeypatch):
    captured = {}

    def fake_run(app, host, port):
        captured["host"] = host
        captured["port"] = port

    opened = []
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    monkeypatch.setattr(cli, "_open_browser_soon", lambda url: opened.append(url))

    cli.main(["serve", "--port", "8123"])
    assert captured == {"host": "127.0.0.1", "port": 8123}
    assert opened == ["http://127.0.0.1:8123"]

def test_no_browser_flag(monkeypatch):
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, host, port: None)
    opened = []
    monkeypatch.setattr(cli, "_open_browser_soon", lambda url: opened.append(url))
    cli.main(["serve", "--no-browser"])
    assert opened == []
```

- [x] **Step 2: Run tests to verify they fail**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_cli.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.cli'`.

- [x] **Step 3: Implement**, `src/atomsim/cli.py`:

```python
"""Command-line entry point: `atomsim serve` launches the local app."""

import argparse
import threading
import webbrowser

import uvicorn

from atomsim.server.app import create_app

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atomsim")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="launch the local server and open the app")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-browser", action="store_true")
    return parser

def _open_browser_soon(url: str) -> None:
    threading.Timer(1.5, webbrowser.open, args=(url,)).start()

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        url = f"http://127.0.0.1:{args.port}"
        if not args.no_browser:
            _open_browser_soon(url)
        uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
```

In `pyproject.toml`, add after the `[project.optional-dependencies]` table:

```toml
[project.scripts]
atomsim = "atomsim.cli:main"
```

- [x] **Step 4: Reinstall (registers the script), run tests**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pip install -e ".[dev]"
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_cli.py -q
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim atomsim serve --help
```

Expected: 3 tests PASS; help text shows `--port` and `--no-browser`.

- [x] **Step 5: Lint, full suite, commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
git add src/atomsim/cli.py tests/test_cli.py pyproject.toml
git commit -m "feat: atomsim serve command"
```

---

### Task 9: Web scaffold: Vite + React + TS, OneDrive-safe

**Files:**
- Create: `scripts/setup_web_node_modules.ps1`, `web/package.json`, `web/tsconfig.json`, `web/vite.config.ts`, `web/index.html`, `web/src/main.tsx`, `web/src/App.tsx` (placeholder), `web/src/index.css` (minimal)
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nodejs/npm from the conda env (Task 3).
- Produces: `npm run dev` serves the app on :5173 with `/api` + `/ws` proxied to :8000; `npm run build` emits `web/dist/`; `npm test` runs vitest. Tasks 10-12 build on this scaffold.

- [x] **Step 1: OneDrive-safety first**, `scripts/setup_web_node_modules.ps1`:

```powershell
# node_modules must NOT live under OneDrive (sync churn + file-lock errors).
# Creates web/node_modules as a directory junction to a local, non-synced dir.
$target = Join-Path $env:LOCALAPPDATA "atomsim\web_node_modules"
$link = Join-Path (Split-Path $PSScriptRoot -Parent) "web\node_modules"
New-Item -ItemType Directory -Force $target | Out-Null
if (-not (Test-Path $link)) {
    New-Item -ItemType Junction -Path $link -Target $target | Out-Null
    Write-Host "junction created: $link -> $target"
} else {
    Write-Host "web/node_modules already exists; nothing to do"
}
```

Append to `.gitignore`:

```
web/node_modules/
web/dist/
```

- [x] **Step 2: Write the scaffold files** (hand-written, not `create vite`, deterministic and reviewable):

`web/package.json`:

```json
{
  "name": "atomsim-web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "test": "vitest run",
    "preview": "vite preview"
  },
  "dependencies": {
    "@react-three/drei": "^10.7.0",
    "@react-three/fiber": "^9.3.0",
    "react": "^19.1.0",
    "react-dom": "^19.1.0",
    "three": "^0.180.0",
    "zustand": "^5.0.8"
  },
  "devDependencies": {
    "@types/react": "^19.1.0",
    "@types/react-dom": "^19.1.0",
    "@types/three": "^0.180.0",
    "@vitejs/plugin-react": "^5.0.0",
    "typescript": "~5.9.0",
    "vite": "^7.1.0",
    "vitest": "^3.2.0"
  }
}
```

(If `npm install` reports unresolvable peer/version conflicts because newer majors shipped, bump the offending range to the major npm suggests, record the change in the commit message.)

`web/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "skipLibCheck": true,
    "types": ["vite/client"],
    "noEmit": true
  },
  "include": ["src"]
}
```

`web/vite.config.ts`:

```ts
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
```

`web/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>atomsim, Hydrogen, Honestly</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`web/src/main.tsx`:

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

`web/src/App.tsx` (placeholder, replaced in Task 11):

```tsx
export default function App() {
  return <h1>atomsim, walking skeleton</h1>;
}
```

`web/src/index.css` (placeholder, replaced in Task 11):

```css
body {
  margin: 0;
  background: #0a0e12;
  color: #e6edf3;
  font-family: "Segoe UI", system-ui, sans-serif;
}
```

- [x] **Step 3: Junction, install, build**

```powershell
cd C:\Users\yashg\OneDrive\Desktop\atom_sim
powershell -ExecutionPolicy Bypass -File scripts\setup_web_node_modules.ps1
cd web
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm install
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm run build
```

Expected: junction message; `npm install` completes (writes `web/package-lock.json`); build prints `✓ built in …` and creates `web/dist/index.html`. Verify the junction: `Get-Item web\node_modules | Select-Object LinkType` → `Junction`.

- [x] **Step 4: Commit** (from repo root; package-lock.json IS committed, node_modules/dist are ignored)

```powershell
cd C:\Users\yashg\OneDrive\Desktop\atom_sim
git add .gitignore scripts/setup_web_node_modules.ps1 web/package.json web/package-lock.json web/tsconfig.json web/vite.config.ts web/index.html web/src/main.tsx web/src/App.tsx web/src/index.css
git commit -m "feat: web scaffold - Vite + React + TS with OneDrive-safe node_modules"
```

---

### Task 10: TypeScript API layer + quantum-number logic (vitest)

**Files:**
- Create: `web/src/api/types.ts`, `web/src/api/client.ts`, `web/src/lib/quantum.ts`
- Test: `web/src/lib/quantum.test.ts`, `web/src/api/decode.test.ts`

**Interfaces:**
- Consumes: server JSON shapes (Task 7), mirrored, never invented.
- Produces (Task 11 relies on): `types.ts` interfaces; `client.getState(n,l,m)`, `client.createSampleJob(n,l,m,count,seed?)`, `client.watchJob(jobId, onProgress)`, `client.getSampleMeta(jobId)`, `client.getSampleData(jobId)`, `client.decodePositions(buffer)`; `quantum.isValidState(n,l,m)`, `quantum.stateLabel(n,l,m)`, `quantum.clampState(n,l,m)`.

- [x] **Step 1: Write the failing tests**

`web/src/lib/quantum.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { clampState, isValidState, stateLabel } from "./quantum";

describe("isValidState", () => {
  it("accepts physical states", () => {
    expect(isValidState(1, 0, 0)).toBe(true);
    expect(isValidState(2, 1, -1)).toBe(true);
    expect(isValidState(6, 5, 5)).toBe(true);
  });
  it("rejects unphysical states", () => {
    expect(isValidState(1, 1, 0)).toBe(false); // l == n
    expect(isValidState(3, 1, 2)).toBe(false); // |m| > l
    expect(isValidState(0, 0, 0)).toBe(false); // n < 1
    expect(isValidState(2, -1, 0)).toBe(false);
    expect(isValidState(1.5, 0, 0)).toBe(false); // non-integer
  });
});

describe("clampState", () => {
  it("clamps l and m when n shrinks", () => {
    expect(clampState(1, 2, -2)).toEqual({ n: 1, l: 0, m: 0 });
  });
  it("clamps m into [-l, l]", () => {
    expect(clampState(3, 1, 5)).toEqual({ n: 3, l: 1, m: 1 });
  });
  it("keeps valid states unchanged", () => {
    expect(clampState(4, 2, -2)).toEqual({ n: 4, l: 2, m: -2 });
  });
});

describe("stateLabel", () => {
  it("uses spectroscopic letters", () => {
    expect(stateLabel(1, 0, 0)).toBe("1s (m = 0)");
    expect(stateLabel(2, 1, 0)).toBe("2p (m = 0)");
    expect(stateLabel(3, 2, -1)).toBe("3d (m = -1)");
  });
});
```

`web/src/api/decode.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { decodePositions } from "./client";

describe("decodePositions", () => {
  it("decodes interleaved xyz float32", () => {
    const src = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = decodePositions(src.buffer);
    expect(out).toHaveLength(6);
    expect(out[4]).toBe(5);
  });
  it("rejects buffers that are not whole xyz triplets", () => {
    expect(() => decodePositions(new ArrayBuffer(10))).toThrow(/multiple of 12/);
  });
});
```

- [x] **Step 2: Run tests to verify they fail**

```powershell
cd C:\Users\yashg\OneDrive\Desktop\atom_sim\web
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test
```

Expected: FAIL, cannot resolve `./quantum` / `./client`.

- [x] **Step 3: Implement**

`web/src/api/types.ts`:

```ts
// Mirrors src/atomsim/server/schemas.py exactly, the single canonical JSON contract.

export type Fidelity =
  | "exact"
  | "numerical"
  | "approximation"
  | "counterfactual"
  | "visual_liberty";

export interface Provenance {
  fidelity: Fidelity;
  method: string;
  assumptions: string[];
  error_estimate: number | null;
  refinement: string | null;
}

export interface Quantity {
  value: number;
  unit: string;
  label: string;
  provenance: Provenance;
}

export interface StateResponse {
  n: number;
  l: number;
  m: number;
  energy: Quantity;
  energy_ev: Quantity;
  mean_radius: Quantity;
}

export type JobStatus = "pending" | "running" | "done" | "error";

export interface JobInfo {
  id: string;
  status: JobStatus;
  progress: number;
  error: string | null;
}

export interface SampleMeta {
  count: number;
  dtype: string;
  layout: string;
  unit: string;
  n: number;
  l: number;
  m: number;
  provenance: Provenance;
}
```

`web/src/api/client.ts`:

```ts
import type { JobInfo, SampleMeta, StateResponse } from "./types";

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export function getState(n: number, l: number, m: number): Promise<StateResponse> {
  return getJson(`/api/state/${n}/${l}/${m}`);
}

export async function createSampleJob(
  n: number,
  l: number,
  m: number,
  count: number,
  seed = 0,
): Promise<JobInfo> {
  const res = await fetch("/api/jobs/sample", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ n, l, m, count, seed }),
  });
  if (!res.ok) throw new Error(`sample job: HTTP ${res.status}`);
  return res.json() as Promise<JobInfo>;
}

export function watchJob(jobId: string, onProgress: (p: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/jobs/${jobId}`);
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data as string) as {
        status: string;
        progress: number;
        error: string | null;
      };
      onProgress(msg.progress);
      if (msg.status === "done") {
        ws.close();
        resolve();
      } else if (msg.status === "error") {
        ws.close();
        reject(new Error(msg.error ?? "job failed"));
      }
    };
    ws.onerror = () => reject(new Error("websocket error"));
  });
}

export function getSampleMeta(jobId: string): Promise<SampleMeta> {
  return getJson(`/api/jobs/${jobId}/meta`);
}

export async function getSampleData(jobId: string): Promise<Float32Array> {
  const res = await fetch(`/api/jobs/${jobId}/data`);
  if (!res.ok) throw new Error(`sample data: HTTP ${res.status}`);
  return decodePositions(await res.arrayBuffer());
}

export function decodePositions(buffer: ArrayBuffer): Float32Array {
  if (buffer.byteLength % 12 !== 0) {
    throw new Error(
      `positions byte length ${buffer.byteLength} is not a multiple of 12 (xyz float32)`,
    );
  }
  return new Float32Array(buffer);
}
```

`web/src/lib/quantum.ts`:

```ts
const L_LETTERS = "spdfghik";

export function isValidState(n: number, l: number, m: number): boolean {
  return (
    Number.isInteger(n) &&
    Number.isInteger(l) &&
    Number.isInteger(m) &&
    n >= 1 &&
    l >= 0 &&
    l < n &&
    Math.abs(m) <= l
  );
}

export function stateLabel(n: number, l: number, m: number): string {
  const letter = L_LETTERS[l] ?? `(l=${l})`;
  return `${n}${letter} (m = ${m})`;
}

export function clampState(n: number, l: number, m: number): { n: number; l: number; m: number } {
  const cn = Math.max(1, Math.round(n));
  const cl = Math.min(Math.max(0, Math.round(l)), cn - 1);
  const cm = Math.min(Math.max(Math.round(m), -cl), cl);
  return { n: cn, l: cl, m: cm };
}
```

- [x] **Step 4: Run tests to verify they pass**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test
```

Expected: PASS (8 tests, 2 files).

- [x] **Step 5: Type-check and commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm run build
cd C:\Users\yashg\OneDrive\Desktop\atom_sim
git add web/src/api/types.ts web/src/api/client.ts web/src/api/decode.test.ts web/src/lib/quantum.ts web/src/lib/quantum.test.ts
git commit -m "feat: TS API client mirroring canonical JSON, quantum-number logic"
```

---

### Task 11: React UI: store, badges, controls, point cloud

**Files:**
- Create: `web/src/state/store.ts`, `web/src/components/Badge.tsx`, `web/src/components/InfoPanel.tsx`, `web/src/components/Controls.tsx`, `web/src/components/PointCloud.tsx`
- Modify: `web/src/App.tsx`, `web/src/index.css` (full replacement of both placeholders)

**Interfaces:**
- Consumes: everything from Task 10.
- Produces: the three-panel app. `useAppStore` shape: `{ n, l, m, count, stateInfo, positions, meta, status, progress, error, setQuantumNumbers, setCount, loadStateInfo, sample }`.

- [x] **Step 1: Implement the store**, `web/src/state/store.ts`:

```ts
import { create } from "zustand";
import * as client from "../api/client";
import type { SampleMeta, StateResponse } from "../api/types";
import { clampState } from "../lib/quantum";

export type SampleStatus = "idle" | "sampling" | "ready" | "error";

interface AppState {
  n: number;
  l: number;
  m: number;
  count: number;
  stateInfo: StateResponse | null;
  positions: Float32Array | null;
  meta: SampleMeta | null;
  status: SampleStatus;
  progress: number;
  error: string | null;
  setQuantumNumbers: (n: number, l: number, m: number) => void;
  setCount: (count: number) => void;
  loadStateInfo: () => Promise<void>;
  sample: () => Promise<void>;
}

export const useAppStore = create<AppState>((set, get) => ({
  n: 1,
  l: 0,
  m: 0,
  count: 100_000,
  stateInfo: null,
  positions: null,
  meta: null,
  status: "idle",
  progress: 0,
  error: null,
  setQuantumNumbers: (n, l, m) => set(clampState(n, l, m)),
  setCount: (count) => set({ count }),
  loadStateInfo: async () => {
    const { n, l, m } = get();
    set({ stateInfo: await client.getState(n, l, m) });
  },
  sample: async () => {
    const { n, l, m, count } = get();
    set({ status: "sampling", progress: 0, error: null });
    try {
      const job = await client.createSampleJob(n, l, m, count);
      await client.watchJob(job.id, (progress) => set({ progress }));
      const [meta, positions] = await Promise.all([
        client.getSampleMeta(job.id),
        client.getSampleData(job.id),
      ]);
      set({ meta, positions, status: "ready", progress: 1 });
    } catch (err) {
      set({ status: "error", error: err instanceof Error ? err.message : String(err) });
    }
  },
}));
```

- [x] **Step 2: Badge with mini-inspector**, `web/src/components/Badge.tsx`:

```tsx
import { useState } from "react";
import type { Provenance } from "../api/types";

const COLORS: Record<string, string> = {
  exact: "#4ade80",
  numerical: "#60a5fa",
  approximation: "#fbbf24",
  counterfactual: "#f472b6",
  visual_liberty: "#a78bfa",
};

export function Badge({ provenance }: { provenance: Provenance }) {
  const [open, setOpen] = useState(false);
  const color = COLORS[provenance.fidelity] ?? "#ffffff";
  return (
    <span className="badge-wrap">
      <button
        type="button"
        className="badge"
        style={{ borderColor: color, color }}
        onClick={() => setOpen((v) => !v)}
      >
        {provenance.fidelity.replace("_", " ").toUpperCase()}
      </button>
      {open && (
        <div className="badge-inspector">
          <p>
            <strong>Method:</strong> {provenance.method}
          </p>
          {provenance.assumptions.length > 0 && (
            <ul>
              {provenance.assumptions.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          )}
          {provenance.error_estimate !== null && (
            <p>
              <strong>Error scale:</strong> {provenance.error_estimate}
            </p>
          )}
          {provenance.refinement && (
            <p>
              <strong>To improve:</strong> {provenance.refinement}
            </p>
          )}
        </div>
      )}
    </span>
  );
}
```

- [x] **Step 3: Info panel**, `web/src/components/InfoPanel.tsx`:

```tsx
import { useEffect } from "react";
import { stateLabel } from "../lib/quantum";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

export function InfoPanel() {
  const { n, l, m, stateInfo, meta, loadStateInfo } = useAppStore();
  useEffect(() => {
    void loadStateInfo();
  }, [n, l, m, loadStateInfo]);
  return (
    <aside className="panel">
      <h1 className="brand">atomsim</h1>
      <h2>Hydrogen (Z = 1)</h2>
      <p className="state-label">{stateLabel(n, l, m)}</p>
      {stateInfo && (
        <dl className="readouts">
          <dt>
            Energy <Badge provenance={stateInfo.energy.provenance} />
          </dt>
          <dd>
            {stateInfo.energy.value.toFixed(6)} hartree
            <br />
            {stateInfo.energy_ev.value.toFixed(4)} eV
          </dd>
          <dt>
            {"⟨r⟩"} <Badge provenance={stateInfo.mean_radius.provenance} />
          </dt>
          <dd>{stateInfo.mean_radius.value.toFixed(3)} a{"₀"}</dd>
          {meta && (
            <>
              <dt>
                Sampled points <Badge provenance={meta.provenance} />
              </dt>
              <dd>{meta.count.toLocaleString()}</dd>
            </>
          )}
        </dl>
      )}
    </aside>
  );
}
```

- [x] **Step 4: Controls**, `web/src/components/Controls.tsx`:

```tsx
import { useAppStore } from "../state/store";

const N_CHOICES = [1, 2, 3, 4, 5, 6];
const COUNT_CHOICES = [10_000, 50_000, 100_000, 250_000];

export function Controls() {
  const { n, l, m, count, status, progress, error, setQuantumNumbers, setCount, sample } =
    useAppStore();
  const lChoices = Array.from({ length: n }, (_, i) => i);
  const mChoices = Array.from({ length: 2 * l + 1 }, (_, i) => i - l);
  return (
    <aside className="panel">
      <h2>State</h2>
      <label>
        n
        <select value={n} onChange={(e) => setQuantumNumbers(Number(e.target.value), l, m)}>
          {N_CHOICES.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </label>
      <label>
        l
        <select value={l} onChange={(e) => setQuantumNumbers(n, Number(e.target.value), m)}>
          {lChoices.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </label>
      <label>
        m
        <select value={m} onChange={(e) => setQuantumNumbers(n, l, Number(e.target.value))}>
          {mChoices.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </label>
      <h2>Sampling</h2>
      <label>
        points
        <select value={count} onChange={(e) => setCount(Number(e.target.value))}>
          {COUNT_CHOICES.map((v) => (
            <option key={v} value={v}>
              {v.toLocaleString()}
            </option>
          ))}
        </select>
      </label>
      <button
        type="button"
        className="primary"
        disabled={status === "sampling"}
        onClick={() => void sample()}
      >
        {status === "sampling" ? `Sampling ${(progress * 100).toFixed(0)}%` : "Sample"}
      </button>
      {status === "error" && <p className="error">{error}</p>}
    </aside>
  );
}
```

- [x] **Step 5: Point cloud**, `web/src/components/PointCloud.tsx`:

```tsx
import { useMemo } from "react";
import * as THREE from "three";

interface Props {
  positions: Float32Array;
  pointSize: number;
}

export function PointCloud({ positions, pointSize }: Props) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return g;
  }, [positions]);
  return (
    <points geometry={geometry}>
      {/* VISUAL LIBERTY: point size, color, glow are presentational choices,
          not physical quantities. Disclosed in UI copy (M3 inspector). */}
      <pointsMaterial
        size={pointSize}
        sizeAttenuation
        color="#7cffb2"
        transparent
        opacity={0.55}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}
```

- [x] **Step 6: App + styles**, replace `web/src/App.tsx`:

```tsx
import { OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { useEffect } from "react";
import { Controls } from "./components/Controls";
import { InfoPanel } from "./components/InfoPanel";
import { PointCloud } from "./components/PointCloud";
import { useAppStore } from "./state/store";

function CameraRig({ distance }: { distance: number }) {
  const camera = useThree((s) => s.camera);
  useEffect(() => {
    camera.position.set(distance * 0.7, distance * 0.45, distance);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, distance]);
  return null;
}

export default function App() {
  const { n, positions } = useAppStore();
  return (
    <div className="app-grid">
      <InfoPanel />
      <main className="canvas-wrap">
        <Canvas camera={{ fov: 50, near: 0.1, far: 5000 }}>
          <color attach="background" args={["#0a0e12"]} />
          <CameraRig distance={5 * n * n + 3} />
          {positions && <PointCloud positions={positions} pointSize={0.05 * n * n} />}
          <OrbitControls />
        </Canvas>
        {!positions && <p className="hint">Choose a state and press Sample</p>}
      </main>
      <Controls />
    </div>
  );
}
```

Replace `web/src/index.css`:

```css
:root {
  --bg: #0a0e12;
  --panel: #10161d;
  --edge: #1e2a36;
  --text: #e6edf3;
  --muted: #8b98a5;
  --accent: #7cffb2;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: "Segoe UI", system-ui, sans-serif;
}

#root,
.app-grid {
  height: 100vh;
}

.app-grid {
  display: grid;
  grid-template-columns: 300px 1fr 300px;
}

.panel {
  background: var(--panel);
  border-inline: 1px solid var(--edge);
  padding: 1.25rem;
  overflow-y: auto;
}

.brand {
  color: var(--accent);
  font-size: 1.4rem;
  letter-spacing: 0.08em;
  margin: 0 0 0.25rem;
}

.panel h2 {
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--muted);
  margin: 1.25rem 0 0.5rem;
}

.state-label {
  font-size: 1.6rem;
  margin: 0.25rem 0 1rem;
}

.readouts dt {
  color: var(--muted);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 0.9rem;
}

.readouts dd {
  margin: 0.25rem 0 0;
  font-size: 1.05rem;
  font-variant-numeric: tabular-nums;
}

.canvas-wrap {
  position: relative;
}

.hint {
  position: absolute;
  inset: auto 0 2rem 0;
  text-align: center;
  color: var(--muted);
  pointer-events: none;
}

label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.75rem;
  margin: 0.5rem 0;
  color: var(--muted);
}

select {
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--edge);
  border-radius: 4px;
  padding: 0.3rem 0.5rem;
  min-width: 6.5rem;
}

button.primary {
  width: 100%;
  margin-top: 1rem;
  padding: 0.6rem;
  background: transparent;
  color: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 6px;
  font-size: 1rem;
  cursor: pointer;
}

button.primary:disabled {
  opacity: 0.5;
  cursor: wait;
}

.error {
  color: #f87171;
  font-size: 0.85rem;
}

.badge-wrap {
  position: relative;
  display: inline-block;
}

.badge {
  background: transparent;
  border: 1px solid;
  border-radius: 999px;
  font-size: 0.6rem;
  letter-spacing: 0.08em;
  padding: 0.1rem 0.5rem;
  cursor: pointer;
  vertical-align: middle;
}

.badge-inspector {
  position: absolute;
  z-index: 10;
  top: 1.6rem;
  left: 0;
  width: 260px;
  background: var(--panel);
  border: 1px solid var(--edge);
  border-radius: 8px;
  padding: 0.75rem;
  font-size: 0.8rem;
  color: var(--text);
  box-shadow: 0 8px 24px rgb(0 0 0 / 0.5);
}

.badge-inspector ul {
  margin: 0.25rem 0;
  padding-left: 1.1rem;
  color: var(--muted);
}
```

- [x] **Step 7: Verify, tests, types, live app**

```powershell
cd C:\Users\yashg\OneDrive\Desktop\atom_sim\web
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm run build
```

Expected: vitest PASS; build succeeds. Then live verification, terminal 1:

```powershell
cd C:\Users\yashg\OneDrive\Desktop\atom_sim
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim atomsim serve --no-browser
```

Terminal 2:

```powershell
cd C:\Users\yashg\OneDrive\Desktop\atom_sim\web
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm run dev
```

Open http://localhost:5173 and verify the checklist:
1. Info panel shows Energy −0.500000 hartree / −13.6057 eV with a green EXACT badge; clicking the badge opens method + assumptions.
2. Press Sample → button shows percent, then a green 1s cloud appears; drag rotates, wheel zooms.
3. Set n=2, l=1, m=0 → Sample → two-lobed dumbbell along z; ⟨r⟩ reads 5.000 a₀.
4. Set n=3, l=2, m=0 → Sample → torus + two lobes.
5. Sampled-points readout shows a blue NUMERICAL badge whose method names inverse-CDF and the seed.

- [x] **Step 8: Commit**

```powershell
cd C:\Users\yashg\OneDrive\Desktop\atom_sim
git add web/src
git commit -m "feat: walking-skeleton UI - three-panel layout, badges, 3D point cloud"
```

---

### Task 12: One-command launch: static serving end-to-end

**Files:**
- Modify: none expected (`app.py` already mounts `web/dist` when present), this task VERIFIES the integration and fixes anything found.

- [x] **Step 1: Fresh build and serve**

```powershell
cd C:\Users\yashg\OneDrive\Desktop\atom_sim\web
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm run build
cd C:\Users\yashg\OneDrive\Desktop\atom_sim
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim atomsim serve --no-browser
```

- [x] **Step 2: Verify from a second terminal**

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
(Invoke-WebRequest http://127.0.0.1:8000/ -UseBasicParsing).StatusCode
```

Expected: `status ok` + version; `200` (index.html served). Then open http://127.0.0.1:8000 in a browser and re-run the Task 11 checklist items 1-2 (this time with NO vite dev server running, the built app must work standalone).

- [x] **Step 3: Full-stack test sweep**

```powershell
cd C:\Users\yashg\OneDrive\Desktop\atom_sim
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
cd web
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test
```

Expected: all green. Commit only if fixes were needed (message: `fix: <what was wrong> in static serving path`).

---

### Task 13: CI: SHA-pinned actions + web job

**Files:**
- Modify: `.github/workflows/ci.yml` (full replacement)

- [x] **Step 1: Replace the workflow** (SHAs verified 2026-07-05 via `git ls-remote`):

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  python:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4
      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5
        with:
          python-version: "3.12"
      - name: Install
        run: python -m pip install -e ".[dev]"
      - name: Lint
        run: ruff check .
      - name: Test
        run: pytest -v --cov=atomsim --cov-report=term

  web:
    runs-on: windows-latest
    defaults:
      run:
        working-directory: web
    steps:
      - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4
      - uses: actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020 # v4
        with:
          node-version: "22"
          cache: npm
          cache-dependency-path: web/package-lock.json
      - name: Install
        run: npm ci
      - name: Test
        run: npm test
      - name: Build (includes tsc --noEmit)
        run: npm run build
```

- [x] **Step 2: Commit and push**

```powershell
git add .github/workflows/ci.yml
git commit -m "ci: SHA-pinned actions, add web job (npm ci + vitest + build)"
git push
```

- [x] **Step 3: Verify CI green**

```powershell
Start-Sleep -Seconds 90
(Invoke-RestMethod "https://api.github.com/repos/yaasshh09/atomsim/actions/runs?per_page=1").workflow_runs[0] | Select-Object status, conclusion
```

Expected: `status completed`, `conclusion success` (re-poll if still `in_progress`). If red: open the run at https://github.com/yaasshh09/atomsim/actions, fix, recommit, do not proceed with a red main.

---

### Task 14: Docs: README quickstart + M1 wrap-up

**Files:**
- Modify: `README.md`

- [x] **Step 1: Add/replace the Quickstart section**, after the badges/intro, insert (adapting to the README's existing structure; keep existing content that still holds):

```markdown
## Quickstart (Windows)

From the **Miniforge Prompt** (Start menu):

​```
cd C:\Users\yashg\OneDrive\Desktop\atom_sim
conda env update -n atomsim -f environment.yml
powershell -ExecutionPolicy Bypass -File scripts\setup_web_node_modules.ps1
cd web
npm ci
npm run build
cd ..
conda activate atomsim
atomsim serve
​```

Your browser opens the app: pick a state (n, l, m), press **Sample**, and explore a
Monte-Carlo point cloud of |ψ|², every displayed quantity carries a provenance badge
(click it to see method, assumptions, and error scale).
```

(Remove the zero-width characters around the fences, they mark the nested block for this plan only.)

- [x] **Step 2: Final verification sweep**

```powershell
cd C:\Users\yashg\OneDrive\Desktop\atom_sim
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
cd web
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test
```

Expected: everything green.

- [x] **Step 3: Commit and push**

```powershell
cd C:\Users\yashg\OneDrive\Desktop\atom_sim
git add README.md
git commit -m "docs: quickstart for the walking-skeleton app"
git push
```

M1 is complete when: CI is green on `main`, and `atomsim serve` alone reproduces the Task 11 checklist.
