# Phase 0: Foundation & Validated Solver Core Implementation Plan

**Goal:** A public GitHub repo with CI, a conda environment, the provenance system, analytic hydrogen-like physics, and a numerical radial Schrödinger solver validated against exact solutions, the engine core everything else builds on.

**Architecture:** Python package `atomsim` (src layout). Engine computes internally in Hartree atomic units (ħ = mₑ = e = 1/(4πε₀) = 1); SI/eV conversions happen only at display boundaries. Every physical quantity leaving the engine is a `Quantity` carrying `Provenance` (fidelity badge + method + assumptions + error estimate). The numerical core is a 3-point finite-difference eigensolver over u(r) = r·R(r) on a uniform grid (`scipy.linalg.eigh_tridiagonal`), which handles **arbitrary central potentials**, the same engine will later power real Coulomb physics, screened multi-electron models, and counterfactual force laws.

**Tech Stack:** Python 3.12 (conda/Miniforge, conda-forge), NumPy ≥2.0, SciPy ≥1.13, pytest, ruff, GitHub Actions (windows-latest). Frontend comes in a later plan.

## Global Constraints

- Native Windows only; no WSL2/Docker anywhere in the toolchain.
- Hand-written NumPy/SciPy solvers are the core; PySCF and ASE are excluded; Psi4 is validator-only (availability probed in Task 3, never a build dependency).
- Engine-internal units: Hartree atomic units; energies exposed in `hartree`, lengths in `bohr`; conversion constants live in `atomsim.constants`.
- Every physical quantity crossing a module boundary is a `Quantity` with `Provenance`; fidelity tiers are exactly: EXACT, NUMERICAL, APPROXIMATION, COUNTERFACTUAL, VISUAL_LIBERTY.
- The model never quietly lies: numerical results must carry quantified or estimable error; assumptions are listed explicitly.
- TDD for all physics code; CI must be green on `windows-latest` for every push; conventional commit messages; MIT license; public GitHub from day one.
- Conda is at `C:\ProgramData\miniforge3` (all-users Miniforge install; NOT on PATH in most shells). Invoke as `& "C:\ProgramData\miniforge3\condabin\conda.bat" ...` in PowerShell. Run tests via `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q`.

## File Structure (locked)

```
atom_sim/
├── pyproject.toml               # package metadata, deps, pytest + ruff config
├── environment.yml              # conda env "atomsim" (conda-forge)
├── .gitignore
├── LICENSE                      # MIT
├── README.md
├── .github/workflows/ci.yml    # windows-latest: ruff + pytest
├── src/atomsim/
│   ├── __init__.py              # __version__
│   ├── provenance.py            # Fidelity, Provenance, Quantity
│   ├── constants.py             # FundamentalConstants (CODATA via scipy), HARTREE_EV
│   ├── analytic/
│   │   ├── __init__.py
│   │   └── hydrogen.py          # exact hydrogen-like: energy, radial_wavefunction, mean_radius
│   └── numerics/
│       ├── __init__.py
│       ├── analysis.py          # count_sign_changes (node counting)
│       └── radial_solver.py     # solve_radial, solve_radial_with_error, RadialSolution
├── tests/
│   ├── test_import.py
│   ├── test_provenance.py
│   ├── test_constants.py
│   ├── test_hydrogen_analytic.py
│   ├── test_analysis.py
│   └── test_radial_solver.py
├── scripts/convergence_study.py # generates docs/phase0-convergence.md
└── docs/
    ├── psi4-windows-status.md   # Task 3 finding
    └── phase0-convergence.md    # generated convergence table
```

---

### Task 1: Project scaffold, conda environment, first green test

**Files:**
- Create: `pyproject.toml`, `environment.yml`, `.gitignore`, `LICENSE`, `README.md`
- Create: `src/atomsim/__init__.py`, `src/atomsim/analytic/__init__.py`, `src/atomsim/numerics/__init__.py`
- Test: `tests/test_import.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: importable package `atomsim` with `atomsim.__version__ == "0.1.0"`; conda env `atomsim` with the package installed editable; `pytest` runs green from repo root.

- [ ] **Step 1: Write the failing test**

`tests/test_import.py`:
```python
import atomsim

def test_package_imports_and_has_version():
    assert atomsim.__version__ == "0.1.0"
```

- [ ] **Step 2: Create the package and project files**

`src/atomsim/__init__.py`:
```python
"""atomsim, a quantum atom model that never quietly lies about physics."""

__version__ = "0.1.0"
```

`src/atomsim/analytic/__init__.py` and `src/atomsim/numerics/__init__.py`: empty files.

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "atomsim"
version = "0.1.0"
description = "A physically rigorous, deeply customizable quantum atom model. Never quietly lies about physics."
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = ["numpy>=2.0", "scipy>=1.13"]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov", "ruff"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
src = ["src", "tests", "scripts"]

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

`environment.yml`:
```yaml
name: atomsim
channels:
  - conda-forge
dependencies:
  - python=3.12
  - numpy>=2.0
  - scipy>=1.13
  - pytest>=8
  - pytest-cov
  - ruff
  - pip
  - pip:
      - -e .[dev]
```

`.gitignore`:
```
__pycache__/
*.pyc
*.egg-info/
build/
dist/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
node_modules/
.vscode/
```

`LICENSE`: standard MIT text, copyright line `Copyright (c) 2026 Yash Gupta`.

`README.md` (stub, completed in Task 10):
```markdown
# atomsim

A physically rigorous, deeply customizable quantum-mechanical atom model and
visualization platform. **Prime directive: the model never quietly lies about
physics.**

Phase 0 (foundation + validated solver core) in progress. See
`docs/specs/2026-07-04-atom-sim-requirements-design.md` for the full
specification and `docs/SETUP.md` for prerequisites.
```

- [ ] **Step 3: Create the conda environment (installs the package editable)**

Run (PowerShell):
```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" env create -f environment.yml
```
Expected: env `atomsim` created without error (takes a few minutes). If an env named `atomsim` already exists, run `& "C:\ProgramData\miniforge3\condabin\conda.bat" env remove -n atomsim --yes` first.

- [ ] **Step 4: Run the test to verify it passes**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q`
Expected: `1 passed`

- [ ] **Step 5: Run lint**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```powershell
git add -A
git commit -m "feat: project scaffold, conda env, importable atomsim package"
```

---

### Task 2: CI workflow and public GitHub publication

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: green pytest suite from Task 1.
- Produces: public GitHub repo with passing CI on every push; the repo slug (OWNER/atom_sim) recorded for the README badge in Task 10.

- [ ] **Step 1: Write the workflow**

`.github/workflows/ci.yml`:
```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install
        run: python -m pip install -e ".[dev]"
      - name: Lint
        run: ruff check .
      - name: Test
        run: pytest -v --cov=atomsim --cov-report=term
```
(CI uses pip, not conda, faster, and Phase 0 has no conda-only deps. Psi4, when it arrives in a later phase, gets its own optional conda job.)

- [ ] **Step 2: Commit the workflow**

```powershell
git add .github/workflows/ci.yml
git commit -m "ci: windows-latest lint + test workflow"
```

- [ ] **Step 3: Ensure GitHub CLI is available and authenticated**

Run: `gh auth status` (if `gh` missing: `winget install --id GitHub.cli -e`, then open a fresh shell).
If not authenticated: **PAUSE and ask the user** to run `! gh auth login --web` in the session (interactive; only the user can complete it).

- [ ] **Step 4: Create the public repo and push**

```powershell
gh repo create atom_sim --public --source . --push --description "A quantum atom model that never quietly lies about physics"
```
Expected: repo created; `git push` succeeds. Record the owner: `gh api user -q .login` → used as OWNER in Task 10's badge URL.

- [ ] **Step 5: Verify CI goes green**

Run: `gh run watch --exit-status` (or `gh run list --limit 1` until status is `completed success`).
Expected: the CI run passes. If it fails, fix before proceeding, CI green is a global constraint.

---

### Task 3: Psi4 Windows availability probe (spec open item 5)

**Files:**
- Create: `docs/psi4-windows-status.md`

**Interfaces:**
- Consumes: conda from Task 1.
- Produces: a documented go/no-go decision on Psi4-on-Windows that the Phase 1/HF planning will read.

- [ ] **Step 1: Probe conda-forge for native Windows Psi4 builds**

Run:
```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" search -c conda-forge psi4 --platform win-64
```
Expected: EITHER a list of versions (record newest) OR `PackagesNotFoundError`.

- [ ] **Step 2: Write the status doc, fill in the observed branch**

`docs/psi4-windows-status.md`:
```markdown
# Psi4 on native Windows: status probe

**Date:** <run date> · **Command:** `conda search -c conda-forge psi4 --platform win-64`

## Result

<PASTE the actual output summary here, versions found, or PackagesNotFoundError>

## Decision

- **If builds exist:** Psi4 <newest version> is available natively. Validation
  cross-checks proceed as specced in the Hartree-Fock phase. No action now, 
  Psi4 is deliberately NOT part of the Phase 0/1 environment.
- **If no builds:** Spec open item 5 fallback applies, custom solvers remain
  primary, Psi4 cross-checks deferred; revisit options (newer channel, WSL-only
  offline validation data) with the user before the HF phase.
```
(Keep whichever Decision branch matches reality, delete the other, and state the finding as fact.)

- [ ] **Step 3: Commit**

```powershell
git add docs/psi4-windows-status.md
git commit -m "docs: record Psi4 native-Windows availability probe result"
```

---

### Task 4: Provenance core: Fidelity, Provenance, Quantity

**Files:**
- Create: `src/atomsim/provenance.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Consumes: nothing.
- Produces (used by every later task):
  - `Fidelity`, Enum with members `EXACT, NUMERICAL, APPROXIMATION, COUNTERFACTUAL, VISUAL_LIBERTY` (values are the lowercase strings `"exact"`, `"numerical"`, `"approximation"`, `"counterfactual"`, `"visual_liberty"`).
  - `Provenance(fidelity: Fidelity, method: str, assumptions: tuple[str ...] = (), error_estimate: float | None = None, refinement: str | None = None)`, frozen dataclass. `error_estimate` is in the same unit as the quantity it describes.
  - `Quantity(value: float, unit: str, label: str, provenance: Provenance)`, frozen dataclass.

- [ ] **Step 1: Write the failing tests**

`tests/test_provenance.py`:
```python
import dataclasses

import pytest

from atomsim.provenance import Fidelity, Provenance, Quantity

def test_fidelity_has_exactly_the_five_spec_tiers():
    assert {f.value for f in Fidelity} == {
        "exact",
        "numerical",
        "approximation",
        "counterfactual",
        "visual_liberty",
    }

def test_quantity_carries_full_provenance():
    p = Provenance(
        fidelity=Fidelity.EXACT,
        method="closed-form Bohr formula",
        assumptions=("non-relativistic", "point nucleus"),
    )
    q = Quantity(value=-0.5, unit="hartree", label="E_1", provenance=p)
    assert q.value == -0.5
    assert q.unit == "hartree"
    assert q.provenance.fidelity is Fidelity.EXACT
    assert "point nucleus" in q.provenance.assumptions
    assert q.provenance.error_estimate is None

def test_provenance_and_quantity_are_immutable():
    p = Provenance(fidelity=Fidelity.NUMERICAL, method="fd")
    q = Quantity(value=1.0, unit="bohr", label="r", provenance=p)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.method = "changed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.value = 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_provenance.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.provenance'`

- [ ] **Step 3: Implement**

`src/atomsim/provenance.py`:
```python
"""Provenance: every physical quantity says how it was computed and how much to trust it.

This is the mechanical enforcement of the project's prime directive:
the model never quietly lies about physics.
"""

from dataclasses import dataclass, field
from enum import Enum

class Fidelity(Enum):
    EXACT = "exact"                    # closed-form solution of the stated model
    NUMERICAL = "numerical"            # converged numerical solution, quantified error
    APPROXIMATION = "approximation"    # honest simplified model, assumptions stated
    COUNTERFACTUAL = "counterfactual"  # deliberately altered physics, computed rigorously
    VISUAL_LIBERTY = "visual_liberty"  # purely presentational choice, disclosed

@dataclass(frozen=True)
class Provenance:
    fidelity: Fidelity
    method: str
    assumptions: tuple[str, ...] = field(default=())
    error_estimate: float | None = None  # same unit as the quantity it describes
    refinement: str | None = None        # what would make this more accurate

@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    label: str
    provenance: Provenance
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_provenance.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```powershell
git add src/atomsim/provenance.py tests/test_provenance.py
git commit -m "feat: provenance core - Fidelity tiers, Provenance, Quantity"
```

---

### Task 5: Fundamental constants (CODATA via scipy)

**Files:**
- Create: `src/atomsim/constants.py`
- Test: `tests/test_constants.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `HARTREE_EV: float`, Hartree energy in eV (≈27.211386…), for display conversion.
  - `FundamentalConstants`, frozen dataclass with SI fields `hbar, e, m_e, eps0, c`; classmethod `codata() -> FundamentalConstants`; properties `alpha` (dimensionless), `bohr_radius` (m), `hartree_energy` (J). This dataclass is the future What-If Lab hook: counterfactual universes are instances with altered fields.

- [ ] **Step 1: Write the failing tests**

`tests/test_constants.py`:
```python
from scipy.constants import physical_constants

from atomsim.constants import HARTREE_EV, FundamentalConstants

def test_hartree_ev_matches_codata():
    assert abs(HARTREE_EV - 27.2113862460) < 1e-8

def test_derived_alpha_matches_published_value():
    c = FundamentalConstants.codata()
    published = physical_constants["fine-structure constant"][0]
    assert abs(c.alpha - published) / published < 1e-9

def test_derived_bohr_radius_matches_published_value():
    c = FundamentalConstants.codata()
    published = physical_constants["Bohr radius"][0]
    assert abs(c.bohr_radius - published) / published < 1e-9

def test_derived_hartree_matches_published_value():
    c = FundamentalConstants.codata()
    published = physical_constants["Hartree energy"][0]
    assert abs(c.hartree_energy - published) / published < 1e-9

def test_counterfactual_universe_rescales():
    # doubling e quadruples alpha (e^2) and shrinks the atom (a0 ~ 1/e^2)
    real = FundamentalConstants.codata()
    weird = FundamentalConstants(
        hbar=real.hbar, e=2 * real.e, m_e=real.m_e, eps0=real.eps0, c=real.c
    )
    assert abs(weird.alpha / real.alpha - 4.0) < 1e-12
    assert abs(weird.bohr_radius / real.bohr_radius - 0.25) < 1e-12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_constants.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.constants'`

- [ ] **Step 3: Implement**

`src/atomsim/constants.py`:
```python
"""Fundamental constants (CODATA, via scipy) and the counterfactual-universe hook.

Engine-internal computations use Hartree atomic units; this module supplies the
SI anchors and display conversions. A FundamentalConstants instance with altered
fields IS a counterfactual universe (What-If Lab, later phase).
"""

import math
from dataclasses import dataclass

from scipy import constants as _sc

HARTREE_EV: float = _sc.physical_constants["Hartree energy in eV"][0]

@dataclass(frozen=True)
class FundamentalConstants:
    hbar: float  # J s
    e: float     # C
    m_e: float   # kg
    eps0: float  # F/m
    c: float     # m/s

    @classmethod
    def codata(cls) -> "FundamentalConstants":
        return cls(
            hbar=_sc.hbar,
            e=_sc.elementary_charge,
            m_e=_sc.electron_mass,
            eps0=_sc.epsilon_0,
            c=_sc.speed_of_light,
        )

    @property
    def alpha(self) -> float:
        """Fine-structure constant e^2 / (4 pi eps0 hbar c), dimensionless."""
        return self.e**2 / (4 * math.pi * self.eps0 * self.hbar * self.c)

    @property
    def bohr_radius(self) -> float:
        """a0 = 4 pi eps0 hbar^2 / (m_e e^2), in metres."""
        return 4 * math.pi * self.eps0 * self.hbar**2 / (self.m_e * self.e**2)

    @property
    def hartree_energy(self) -> float:
        """E_h = hbar^2 / (m_e a0^2), in joules."""
        return self.hbar**2 / (self.m_e * self.bohr_radius**2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_constants.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```powershell
git add src/atomsim/constants.py tests/test_constants.py
git commit -m "feat: CODATA constants with counterfactual-universe dataclass hook"
```

---

### Task 6: Analytic hydrogen-like energies (with reduced mass → exotic systems)

**Files:**
- Create: `src/atomsim/analytic/hydrogen.py`
- Test: `tests/test_hydrogen_analytic.py`

**Interfaces:**
- Consumes: `Quantity, Provenance, Fidelity` (Task 4); `HARTREE_EV` (Task 5).
- Produces:
  - `energy(n: int, Z: int = 1, mu_ratio: float = 1.0) -> Quantity`, exact non-relativistic energy in hartree: E = −mu_ratio·Z²/(2n²). `mu_ratio` = reduced mass / electron mass (1.0 = infinite-nucleus hydrogen; 0.5 = positronium; ≈185.84 = muonic hydrogen).
  - `validate_quantum_numbers(n: int, l: int) -> None`, raises `ValueError` unless n ≥ 1 and 0 ≤ l < n. (Task 7 reuses it.)

- [ ] **Step 1: Write the failing tests**

`tests/test_hydrogen_analytic.py`:
```python
import pytest
from scipy.constants import electron_mass, physical_constants, proton_mass

from atomsim.analytic.hydrogen import energy
from atomsim.constants import HARTREE_EV
from atomsim.provenance import Fidelity

def test_hydrogen_ground_state_is_minus_half_hartree():
    q = energy(1)
    assert q.value == pytest.approx(-0.5, abs=1e-15)
    assert q.unit == "hartree"
    assert q.provenance.fidelity is Fidelity.EXACT

def test_energy_scales_as_z_squared_over_n_squared():
    assert energy(2, Z=3).value == pytest.approx(-0.5 * 9 / 4, rel=1e-14)

def test_helium_plus_ground_state():
    assert energy(1, Z=2).value == pytest.approx(-2.0, rel=1e-14)

def test_positronium_via_reduced_mass():
    # mu = m_e/2 exactly -> binding 6.803 eV
    q = energy(1, mu_ratio=0.5)
    assert q.value == pytest.approx(-0.25, rel=1e-14)
    assert q.value * HARTREE_EV == pytest.approx(-6.803, abs=0.001)

def test_muonic_hydrogen_ground_state_in_ev():
    m_mu = physical_constants["muon mass"][0]
    mu_ratio = (m_mu * proton_mass / (m_mu + proton_mass)) / electron_mass
    e_ev = energy(1, mu_ratio=mu_ratio).value * HARTREE_EV
    assert e_ev == pytest.approx(-2528.5, abs=2.0)  # known ~ -2.53 keV

def test_invalid_quantum_numbers_raise():
    with pytest.raises(ValueError):
        energy(0)
    with pytest.raises(ValueError):
        energy(-3)

def test_provenance_names_its_assumptions():
    p = energy(1).provenance
    joined = " ".join(p.assumptions).lower()
    assert "non-relativistic" in joined
    assert "point nucleus" in joined
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_hydrogen_analytic.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.analytic.hydrogen'`

- [ ] **Step 3: Implement**

`src/atomsim/analytic/hydrogen.py`:
```python
"""Exact analytic solutions for hydrogen-like (one-electron) atoms.

Hartree atomic units (hbar = m_e = e = 1/(4 pi eps0) = 1). The reduced-mass
ratio mu_ratio = mu/m_e makes isotopes, muonic atoms, and positronium exact
within the same formulas. This module is also the ground truth that validates
the numerical radial solver.
"""

from atomsim.provenance import Fidelity, Provenance, Quantity

_EXACT_ASSUMPTIONS = (
    "non-relativistic Schrodinger equation",
    "point nucleus (no finite-size or QED effects)",
    "nuclear motion included only via reduced-mass ratio",
    "no external fields",
)

def validate_quantum_numbers(n: int, l: int = 0) -> None:
    if n < 1:
        raise ValueError(f"principal quantum number n must be >= 1, got {n}")
    if not 0 <= l < n:
        raise ValueError(f"orbital quantum number l must satisfy 0 <= l < n, got l={l}, n={n}")

def energy(n: int, Z: int = 1, mu_ratio: float = 1.0) -> Quantity:
    """Exact bound-state energy E_n = -mu_ratio * Z^2 / (2 n^2), in hartree."""
    validate_quantum_numbers(n)
    value = -mu_ratio * Z**2 / (2.0 * n**2)
    return Quantity(
        value=value,
        unit="hartree",
        label=f"E_{n} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method="closed-form Bohr formula E_n = -mu' Z^2 / (2 n^2)",
            assumptions=_EXACT_ASSUMPTIONS,
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_hydrogen_analytic.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```powershell
git add src/atomsim/analytic/hydrogen.py tests/test_hydrogen_analytic.py
git commit -m "feat: exact hydrogen-like energies with reduced-mass exotic systems"
```

---

### Task 7: Analytic radial wavefunctions R_nl(r)

**Files:**
- Modify: `src/atomsim/analytic/hydrogen.py` (append two functions)
- Test: `tests/test_hydrogen_analytic.py` (append tests)

**Interfaces:**
- Consumes: `validate_quantum_numbers` (Task 6).
- Produces:
  - `radial_wavefunction(n: int, l: int, r: np.ndarray, Z: int = 1, mu_ratio: float = 1.0) -> np.ndarray`, normalized R_nl(r) on the given grid (r in bohr), satisfying ∫R²r²dr = 1. Documented reliable range: n ≤ 20 (float64 Laguerre evaluation).
  - `mean_radius(n: int, l: int, Z: int = 1, mu_ratio: float = 1.0) -> float`, exact ⟨r⟩ = (3n² − l(l+1)) / (2·Z·mu_ratio), in bohr.

- [ ] **Step 1: Write the failing tests (append to `tests/test_hydrogen_analytic.py`)**

```python
import numpy as np

from atomsim.analytic.hydrogen import mean_radius, radial_wavefunction

def _grid():
    return np.linspace(1e-8, 150.0, 200_001)

def test_radial_wavefunctions_are_normalized():
    r = _grid()
    for n, l in [(1, 0), (2, 0), (2, 1), (3, 1), (5, 3)]:
        norm = np.trapezoid(radial_wavefunction(n, l, r) ** 2 * r**2, r)
        assert norm == pytest.approx(1.0, abs=1e-6), (n, l)

def test_node_counts_are_n_minus_l_minus_1():
    r = _grid()
    for n, l in [(1, 0), (2, 0), (3, 0), (3, 2), (4, 1)]:
        R = radial_wavefunction(n, l, r)
        mask = np.abs(R) > 1e-6 * np.abs(R).max()
        signs = np.sign(R[mask])
        nodes = int(np.sum(signs[1:] != signs[:-1]))
        assert nodes == n - l - 1, (n, l)

def test_mean_radius_matches_exact_formula_and_integral():
    r = _grid()
    for n, l in [(1, 0), (2, 1), (3, 0)]:
        R = radial_wavefunction(n, l, r)
        integral = np.trapezoid(R**2 * r**3, r)
        exact = mean_radius(n, l)
        assert integral == pytest.approx(exact, rel=1e-5), (n, l)
    assert mean_radius(1, 0) == pytest.approx(1.5)  # 1s: <r> = 1.5 bohr

def test_orthogonality_same_l():
    r = _grid()
    overlap = np.trapezoid(
        radial_wavefunction(1, 0, r) * radial_wavefunction(2, 0, r) * r**2, r
    )
    assert abs(overlap) < 1e-6

def test_scaling_with_z_and_reduced_mass():
    # length scale ~ 1/(Z mu'): heavier reduced mass shrinks the atom
    assert mean_radius(1, 0, Z=2) == pytest.approx(0.75)
    assert mean_radius(1, 0, mu_ratio=2.0) == pytest.approx(0.75)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_hydrogen_analytic.py -q`
Expected: FAIL, `ImportError: cannot import name 'mean_radius'`

- [ ] **Step 3: Implement (append to `src/atomsim/analytic/hydrogen.py`)**

```python
import math

import numpy as np
from scipy.special import eval_genlaguerre

def radial_wavefunction(
    n: int, l: int, r: np.ndarray, Z: int = 1, mu_ratio: float = 1.0
) -> np.ndarray:
    """Normalized radial wavefunction R_nl(r) in atomic units (r in bohr).

    R_nl = N * exp(-rho/2) * rho^l * L_{n-l-1}^{2l+1}(rho),  rho = 2 Z mu' r / n.
    Reliable for n <= 20 (float64 generalized-Laguerre evaluation).
    """
    validate_quantum_numbers(n, l)
    kappa = Z * mu_ratio
    rho = 2.0 * kappa * np.asarray(r, dtype=float) / n
    norm = math.sqrt(
        (2.0 * kappa / n) ** 3
        * math.factorial(n - l - 1)
        / (2.0 * n * math.factorial(n + l))
    )
    return norm * np.exp(-rho / 2.0) * rho**l * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)

def mean_radius(n: int, l: int, Z: int = 1, mu_ratio: float = 1.0) -> float:
    """Exact <r> = (3 n^2 - l(l+1)) / (2 Z mu'), in bohr."""
    validate_quantum_numbers(n, l)
    return (3.0 * n**2 - l * (l + 1)) / (2.0 * Z * mu_ratio)
```
(Move the `import math` line to the top of the file with the other imports; `numpy` and `scipy.special` imports go there too.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_hydrogen_analytic.py -q`
Expected: `12 passed`

- [ ] **Step 5: Commit**

```powershell
git add src/atomsim/analytic/hydrogen.py tests/test_hydrogen_analytic.py
git commit -m "feat: analytic radial wavefunctions with normalization and node tests"
```

---

### Task 8: Numerical radial solver core (validated on the harmonic oscillator)

**Files:**
- Create: `src/atomsim/numerics/analysis.py`, `src/atomsim/numerics/radial_solver.py`
- Test: `tests/test_analysis.py`, `tests/test_radial_solver.py`

**Interfaces:**
- Consumes: `Quantity, Provenance, Fidelity` (Task 4).
- Produces:
  - `count_sign_changes(y: np.ndarray, rel_floor: float = 1e-6) -> int` (in `analysis.py`), node counter ignoring numerical noise below `rel_floor * max|y|`.
  - `RadialSolution`, frozen dataclass: `r: np.ndarray` (grid, bohr), `u: np.ndarray` (shape `(n_states, len(r))`, normalized ∫u²dr = 1, sign fixed positive at small r), `energies: tuple[Quantity ...]` (hartree, NUMERICAL provenance), `l: int`, `mu_ratio: float`.
  - `solve_radial(potential: Callable[[np.ndarray], np.ndarray], l: int = 0, mu_ratio: float = 1.0, r_max: float = 120.0, n_points: int = 24000, n_states: int = 3) -> RadialSolution`, solves −(1/2μ')u″ + [V(r) + l(l+1)/(2μ'r²)]u = Eu with u(0) = u(r_max) = 0. `potential` takes r in bohr, returns V in hartree.

- [ ] **Step 1: Write the failing tests**

`tests/test_analysis.py`:
```python
import numpy as np

from atomsim.numerics.analysis import count_sign_changes

def test_counts_true_sign_changes():
    x = np.linspace(0, 2 * np.pi, 1000)
    assert count_sign_changes(np.sin(x + 0.1)) == 2

def test_ignores_noise_below_floor():
    y = np.ones(100)
    y[50:] = -1.0
    y[10] = -1e-12  # noise spike must not count
    assert count_sign_changes(y) == 1
```

`tests/test_radial_solver.py`:
```python
import numpy as np
import pytest

from atomsim.numerics.radial_solver import solve_radial
from atomsim.provenance import Fidelity

def test_3d_harmonic_oscillator_l0_energies():
    # V = r^2/2, mu'=1: exact E = 2k + l + 3/2 -> 1.5, 3.5, 5.5
    sol = solve_radial(lambda r: 0.5 * r**2, l=0, r_max=12.0, n_points=2400, n_states=3)
    got = [q.value for q in sol.energies]
    assert got == pytest.approx([1.5, 3.5, 5.5], abs=1e-4)

def test_3d_harmonic_oscillator_l1_energies():
    sol = solve_radial(lambda r: 0.5 * r**2, l=1, r_max=12.0, n_points=2400, n_states=2)
    got = [q.value for q in sol.energies]
    assert got == pytest.approx([2.5, 4.5], abs=1e-4)

def test_solutions_are_normalized_and_sign_fixed():
    sol = solve_radial(lambda r: 0.5 * r**2, l=0, r_max=12.0, n_points=2400, n_states=3)
    for k in range(3):
        norm = np.trapezoid(sol.u[k] ** 2, sol.r)
        assert norm == pytest.approx(1.0, abs=1e-8)
        first = np.argmax(np.abs(sol.u[k]) > 0.01 * np.abs(sol.u[k]).max())
        assert sol.u[k][first] > 0

def test_state_k_has_k_nodes():
    from atomsim.numerics.analysis import count_sign_changes

    sol = solve_radial(lambda r: 0.5 * r**2, l=0, r_max=12.0, n_points=2400, n_states=4)
    for k in range(4):
        assert count_sign_changes(sol.u[k]) == k

def test_energies_carry_numerical_provenance():
    sol = solve_radial(lambda r: 0.5 * r**2, l=0, r_max=12.0, n_points=1200, n_states=1)
    p = sol.energies[0].provenance
    assert p.fidelity is Fidelity.NUMERICAL
    assert "finite-difference" in p.method
    assert any("grid" in a for a in p.assumptions)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_analysis.py tests/test_radial_solver.py -q`
Expected: FAIL, `ModuleNotFoundError` for both modules.

- [ ] **Step 3: Implement**

`src/atomsim/numerics/analysis.py`:
```python
"""Small physics-meaningful analysis utilities."""

import numpy as np

def count_sign_changes(y: np.ndarray, rel_floor: float = 1e-6) -> int:
    """Count sign changes of y, ignoring values below rel_floor * max|y| (noise)."""
    y = np.asarray(y)
    mask = np.abs(y) > rel_floor * np.abs(y).max()
    signs = np.sign(y[mask])
    return int(np.sum(signs[1:] != signs[:-1]))
```

`src/atomsim/numerics/radial_solver.py`:
```python
"""Numerical radial Schrodinger solver for ARBITRARY central potentials.

Solves  -(1/2mu') u'' + [V(r) + l(l+1)/(2mu' r^2)] u = E u,  u = r R(r),
with u(0) = u(r_max) = 0, via 3-point finite differences on a uniform grid and
scipy.linalg.eigh_tridiagonal. Hartree atomic units throughout.

This one engine powers real Coulomb physics, screened multi-electron models,
and counterfactual force laws alike.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

from atomsim.provenance import Fidelity, Provenance, Quantity

@dataclass(frozen=True)
class RadialSolution:
    r: np.ndarray
    u: np.ndarray  # shape (n_states, len(r)); normalized, sign-fixed
    energies: tuple[Quantity, ...]
    l: int
    mu_ratio: float

def solve_radial(
    potential: Callable[[np.ndarray], np.ndarray],
    l: int = 0,
    mu_ratio: float = 1.0,
    r_max: float = 120.0,
    n_points: int = 24000,
    n_states: int = 3,
) -> RadialSolution:
    h = r_max / (n_points + 1)
    r = h * np.arange(1, n_points + 1)
    inv2m = 1.0 / (2.0 * mu_ratio)

    v_eff = np.asarray(potential(r), dtype=float) + l * (l + 1) * inv2m / r**2
    diag = 2.0 * inv2m / h**2 + v_eff
    offdiag = np.full(n_points - 1, -inv2m / h**2)

    eigvals, eigvecs = eigh_tridiagonal(
        diag, offdiag, select="i", select_range=(0, n_states - 1)
    )
    u = eigvecs.T.copy()

    norms = np.sqrt(np.trapezoid(u**2, r, axis=1))
    u /= norms[:, None]
    for k in range(u.shape[0]):
        first = np.argmax(np.abs(u[k]) > 0.01 * np.abs(u[k]).max())
        if u[k][first] < 0:
            u[k] = -u[k]

    provenance = Provenance(
        fidelity=Fidelity.NUMERICAL,
        method="3-point finite-difference radial Hamiltonian (u = r R), scipy eigh_tridiagonal",
        assumptions=(
            f"uniform grid: h={h:.3e} bohr, r_max={r_max:g} bohr, N={n_points}",
            "Dirichlet boundaries u(0) = u(r_max) = 0",
            "only low-lying bound states are box-converged",
        ),
        refinement="increase n_points / r_max, or use solve_radial_with_error for a quantified error",
    )
    energies = tuple(
        Quantity(
            value=float(e),
            unit="hartree",
            label=f"E[{k}] (l={l})",
            provenance=provenance,
        )
        for k, e in enumerate(eigvals)
    )
    return RadialSolution(r=r, u=u, energies=energies, l=l, mu_ratio=mu_ratio)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_analysis.py tests/test_radial_solver.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```powershell
git add src/atomsim/numerics/ tests/test_analysis.py tests/test_radial_solver.py
git commit -m "feat: finite-difference radial solver validated on 3D harmonic oscillator"
```

---

### Task 9: Coulomb validation against analytic hydrogen + honest error estimates

**Files:**
- Modify: `src/atomsim/numerics/radial_solver.py` (append `solve_radial_with_error`)
- Test: `tests/test_radial_solver.py` (append tests)

**Interfaces:**
- Consumes: `solve_radial`, `RadialSolution` (Task 8); `energy`, `radial_wavefunction` (Tasks 6-7).
- Produces:
  - `solve_radial_with_error(potential, l=0, mu_ratio=1.0, r_max=120.0, n_points=24000, n_states=3) -> RadialSolution`, solves at `n_points` and `2·n_points`; returns the fine solution whose energy `Quantity`s carry `provenance.error_estimate = |E_fine − E_coarse|` (hartree). This is the honesty mechanism: numerical energies ship with a quantified error.

- [ ] **Step 1: Write the failing tests (append to `tests/test_radial_solver.py`)**

```python
from atomsim.analytic.hydrogen import energy as exact_energy
from atomsim.analytic.hydrogen import radial_wavefunction
from atomsim.numerics.radial_solver import solve_radial_with_error

def _coulomb(z):
    return lambda r: -z / r

def test_hydrogen_energies_match_analytic():
    sol = solve_radial(_coulomb(1), l=0, n_states=3)
    for k, q in enumerate(sol.energies):
        exact = exact_energy(k + 1).value
        assert abs(q.value - exact) / abs(exact) < 1e-3, k

def test_helium_plus_and_positronium_match_analytic():
    sol = solve_radial(_coulomb(2), l=0, r_max=60.0, n_states=2)
    assert sol.energies[0].value == pytest.approx(exact_energy(1, Z=2).value, rel=1e-3)

    sol = solve_radial(_coulomb(1), mu_ratio=0.5, l=0, r_max=200.0, n_states=1)
    assert sol.energies[0].value == pytest.approx(
        exact_energy(1, mu_ratio=0.5).value, rel=1e-3
    )

def test_l1_states_match_analytic():
    sol = solve_radial(_coulomb(1), l=1, n_states=2)
    # lowest l=1 state is n=2, then n=3
    assert sol.energies[0].value == pytest.approx(exact_energy(2).value, rel=1e-3)
    assert sol.energies[1].value == pytest.approx(exact_energy(3).value, rel=1e-3)

def test_numerical_1s_wavefunction_overlaps_analytic():
    sol = solve_radial(_coulomb(1), l=0, n_states=1)
    u_exact = sol.r * radial_wavefunction(1, 0, sol.r)
    overlap = np.trapezoid(sol.u[0] * u_exact, sol.r)
    assert overlap > 0.99999

def test_error_estimate_bounds_true_error():
    sol = solve_radial_with_error(_coulomb(1), l=0, n_points=6000, n_states=2)
    for k, q in enumerate(sol.energies):
        est = q.provenance.error_estimate
        assert est is not None and est > 0
        true_err = abs(q.value - exact_energy(k + 1).value)
        assert true_err <= 2.0 * est + 1e-12, (k, true_err, est)
        assert est < 1e-3 * abs(q.value)  # and the estimate itself is small
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_radial_solver.py -q`
Expected: FAIL, `ImportError: cannot import name 'solve_radial_with_error'`

- [ ] **Step 3: Implement (append to `src/atomsim/numerics/radial_solver.py`)**

```python
import dataclasses

def solve_radial_with_error(
    potential: Callable[[np.ndarray], np.ndarray],
    l: int = 0,
    mu_ratio: float = 1.0,
    r_max: float = 120.0,
    n_points: int = 24000,
    n_states: int = 3,
) -> RadialSolution:
    """Solve at n_points and 2*n_points; attach |E_fine - E_coarse| as error estimate.

    Grid-halving is a conservative estimate for this O(h^2)-convergent scheme:
    the reported fine-grid error is smaller than the difference itself.
    """
    coarse = solve_radial(potential, l, mu_ratio, r_max, n_points, n_states)
    fine = solve_radial(potential, l, mu_ratio, r_max, 2 * n_points, n_states)

    energies = tuple(
        dataclasses.replace(
            q,
            provenance=dataclasses.replace(
                q.provenance,
                error_estimate=abs(q.value - coarse.energies[k].value),
                refinement="increase n_points further; estimate from grid-halving",
            ),
        )
        for k, q in enumerate(fine.energies)
    )
    return dataclasses.replace(fine, energies=energies)
```
(Add `import dataclasses` at the top of the file with the other imports.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_radial_solver.py -q`
Expected: `10 passed` (this file; ~24000-point tridiagonal solves take seconds each, the full file should finish well under 2 minutes)

- [ ] **Step 5: Run the ENTIRE suite and lint**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q` then `... ruff check .`
Expected: all tests pass, lint clean.

- [ ] **Step 6: Commit**

```powershell
git add src/atomsim/numerics/radial_solver.py tests/test_radial_solver.py
git commit -m "feat: Coulomb validation vs analytic hydrogen with grid-halving error estimates"
```

---

### Task 10: Convergence study, README completion, Phase 0 wrap

**Files:**
- Create: `scripts/convergence_study.py`
- Create: `docs/phase0-convergence.md` (generated by the script)
- Modify: `README.md`, `tests/test_radial_solver.py` (append convergence-order test)

**Interfaces:**
- Consumes: `solve_radial` (Task 8), `energy` (Task 6).
- Produces: committed convergence evidence (error vs h table + observed order) and a README that presents the project honestly, the Phase 0 exit artifact.

- [ ] **Step 1: Write the failing convergence-order test (append to `tests/test_radial_solver.py`)**

```python
def _observed_order(potential, exact, l, r_max, n_list, **kw):
    errs = [
        abs(solve_radial(potential, l=l, r_max=r_max, n_points=n, n_states=1, **kw)
            .energies[0].value - exact)
        for n in n_list
    ]
    return [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)], errs

def test_convergence_order_harmonic_is_second_order():
    orders, _ = _observed_order(
        lambda r: 0.5 * r**2, exact=1.5, l=0, r_max=12.0, n_list=[600, 1200, 2400]
    )
    for p in orders:
        assert 1.7 < p < 2.3, orders

def test_convergence_order_coulomb_documented():
    # The r=0 Coulomb cusp can reduce the observed order below 2: that finding
    # is recorded in docs/phase0-convergence.md, and the floor asserted here.
    orders, errs = _observed_order(
        lambda r: -1.0 / r, exact=-0.5, l=0, r_max=60.0, n_list=[3000, 6000, 12000]
    )
    assert errs[-1] < 1e-4  # absolute accuracy still good
    for p in orders:
        assert p > 1.3, orders
```

- [ ] **Step 2: Run to verify the new tests' status**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_radial_solver.py -q`
Expected: PASS immediately (they test existing solver behavior, that is fine; they exist to pin the convergence claims the docs make). If either fails, the solver or the tolerances need investigation before proceeding, do not loosen bounds without understanding why.

- [ ] **Step 3: Write the convergence study script**

`scripts/convergence_study.py`:
```python
"""Generate docs/phase0-convergence.md: solver error vs grid spacing, observed order."""

from pathlib import Path

import numpy as np

from atomsim.analytic.hydrogen import energy
from atomsim.numerics.radial_solver import solve_radial

CASES = [
    ("3D harmonic oscillator (l=0, E=1.5)", lambda r: 0.5 * r**2, 1.5, 0, 12.0,
     [600, 1200, 2400, 4800]),
    ("Hydrogen 1s (E=-0.5)", lambda r: -1.0 / r, energy(1).value, 0, 60.0,
     [3000, 6000, 12000, 24000]),
    ("Hydrogen 2p (E=-0.125)", lambda r: -1.0 / r, energy(2).value, 1, 100.0,
     [3000, 6000, 12000, 24000]),
]

lines = [
    "# Phase 0 convergence study",
    "",
    "3-point finite-difference radial solver, uniform grid. Generated by",
    "`scripts/convergence_study.py`, regenerate after any solver change.",
    "",
]
for name, pot, exact, l, r_max, n_list in CASES:
    lines += [f"## {name}", "", "| N (grid points) | h (bohr) | |E_num - E_exact| (hartree) | observed order |",
              "|---|---|---|---|"]
    prev = None
    for n in n_list:
        e = solve_radial(pot, l=l, r_max=r_max, n_points=n, n_states=l + 1).energies[-1].value
        err = abs(e - exact)
        order = f"{np.log2(prev / err):.2f}" if prev else ", "
        lines.append(f"| {n} | {r_max / (n + 1):.2e} | {err:.3e} | {order} |")
        prev = err
    lines.append("")
lines += [
    "**Reading the results:** the smooth harmonic potential shows clean O(h^2)",
    "convergence. The Coulomb cusp at r=0 can reduce the observed order for s",
    "states, exactly the kind of honest numerical caveat this project exists to",
    "surface. Every solver energy can carry a grid-halving error estimate via",
    "`solve_radial_with_error`.",
    "",
]
out = Path(__file__).resolve().parents[1] / "docs" / "phase0-convergence.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {out}")
```

Note for the Hydrogen 2p row: `n_states=l+1` solves for 2 states when l=1 and `energies[-1]` takes the second, wrong: for l=1 the LOWEST state is already n=2 (E=-0.125). Use `n_states=1` and `energies[0]`, and compare against `energy(2).value`. Implement it that way:
replace the line computing `e` with:
```python
        e = solve_radial(pot, l=l, r_max=r_max, n_points=n, n_states=1).energies[0].value
```
and compare `exact = energy(2).value` for the 2p case as listed in CASES.

- [ ] **Step 4: Run the script and inspect the output**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python scripts/convergence_study.py`
Expected: `wrote ...docs\phase0-convergence.md`; the table shows errors decreasing with N and observed orders ≈2 for harmonic, ≥1.3 for Coulomb s states.

- [ ] **Step 5: Complete the README**

Replace `README.md` with:
```markdown
# atomsim

A physically rigorous, deeply customizable quantum-mechanical atom model and
visualization platform, portfolio project, teaching tool, and self-directed
physics sandbox.

![CI](https://github.com/OWNER/atom_sim/actions/workflows/ci.yml/badge.svg)

**Prime directive: the model never quietly lies about physics.** Every quantity
carries provenance:

| Badge | Meaning |
|---|---|
| `EXACT` | Closed-form solution of the stated model |
| `NUMERICAL` | Converged numerical solution with quantified error |
| `APPROXIMATION` | Honest simplified model, assumptions stated |
| `COUNTERFACTUAL` | Deliberately altered physics, computed rigorously |
| `VISUAL LIBERTY` | Purely presentational choice, disclosed |

## Status: Phase 0 complete

- Provenance system: every physical quantity is a `Quantity` with `Provenance`
- Exact hydrogen-like physics: energies + radial wavefunctions, reduced-mass
  exact (deuterium, muonic hydrogen, positronium, He+, ...)
- Numerical radial solver for **arbitrary central potentials** (the engine that
  will power real atoms, screened models, and counterfactual force laws alike),
  validated against closed-form hydrogen and harmonic-oscillator solutions, 
  see [docs/phase0-convergence.md](docs/phase0-convergence.md)
- Numerical energies ship with grid-halving error estimates

## Quickstart (Windows, native: no WSL)

Prerequisites: [docs/SETUP.md](docs/SETUP.md). Then:

    conda env create -f environment.yml
    conda activate atomsim
    pytest

## Roadmap

Full specification: [docs/specs/2026-07-04-atom-sim-requirements-design.md](docs/specs/2026-07-04-atom-sim-requirements-design.md).
Next: Phase 1, "Hydrogen, Honestly": 3D orbital visualization (point cloud +
isosurfaces, complex phase as hue), fine structure, spectra vs NIST, and the
honesty-badge UI, served from a local Python engine into a WebGL browser front
end.

## License

MIT
```
Replace `OWNER` with the GitHub username recorded in Task 2.

- [ ] **Step 6: Full suite, lint, commit, push**

Run: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q` and `... ruff check .`
Expected: all pass.

```powershell
git add scripts/convergence_study.py docs/phase0-convergence.md README.md tests/test_radial_solver.py
git commit -m "feat: convergence study with documented observed orders; complete Phase 0 README"
git push
```

- [ ] **Step 7: Verify CI is green on GitHub**

Run: `gh run watch --exit-status`
Expected: success. Phase 0 exit criteria met: public repo, green CI, validated solver core with honest error reporting.

---

## Self-Review Notes

- **Spec coverage (Phase 0 scope):** repo/CI/env ✓ (Tasks 1-2), Psi4 open item ✓ (Task 3), provenance-as-architecture ✓ (Task 4, threaded through 6-9), exotic real systems via reduced mass ✓ (Tasks 6-7 tests), radial-solver spike validated against analytic hydrogen ✓ (Tasks 8-9), quantified-error honesty ✓ (Task 9), convergence evidence ✓ (Task 10). Phase 1 items (rendering, fine structure, NIST spectra, server, frontend, tour) are deliberately in the NEXT plan.
- **Known physics caveat, by design:** the Coulomb cusp may reduce observed convergence order for s states on a uniform grid; Task 10 documents rather than hides it, and tolerances (rel. error < 1e-3 at spike resolution, order > 1.3) are set accordingly. Improving the grid (log-spaced/Numerov) is a Phase 1+ refinement decision informed by these numbers.
- **Type consistency check:** `Quantity(value, unit, label, provenance)` used identically in Tasks 4, 6, 8, 9; `validate_quantum_numbers(n, l=0)` defined Task 6, reused Task 7; `solve_radial(...)` signature identical in Tasks 8, 9, 10; `count_sign_changes` defined Task 8 and used in Tasks 8/10 tests only.
```
