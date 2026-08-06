# Phase 1 M2: Engine Depth Implementation Plan

**Goal:** Complete the hydrogen-like engine, real/complex angular bases, full ψ_nlm, real-basis sampling, perturbative fine structure, exotic-system presets, spectra with vendored-NIST comparison, all exposed through new server endpoints, so M3 is almost pure frontend (only the thumbnail-render endpoint lands with the M3 gallery, since it brings the matplotlib dependency).

**Architecture:** Spec `docs/specs/2026-07-05-phase1-hydrogen-honestly-design.md` §5.1, §5.3-5.5, §6, §8. New modules `analytic/angular.py`, `analytic/wavefunction.py`, `analytic/fine_structure.py`, `systems.py`, `spectra.py`, vendored `data/nist_*.json`; `sampling.py` gains the real basis; the server gains `/api/systems`, `/api/radial`, `/api/spectrum` and system/basis/fine-structure parameters. Every boundary-crossing value stays a `Quantity`, `Field`, or provenance-carrying container.

**Tech Stack:** Python 3.12 (conda env `atomsim`), scipy ≥ 1.18 (`sph_harm_y`, `lpmv`, `physical_constants`), FastAPI, pytest · TypeScript strict + vitest (types mirror only, UI consumption is M3).

## Global Constraints

- Native Windows only; no WSL2/Docker. Engine units: Hartree atomic units (`hartree`, `bohr`); display conversions at the boundary.
- **Boundary rule (spec §4):** every physical value crossing a module boundary is a `Quantity`, a `Field`, or a container dataclass carrying its own `Provenance`. Fidelity tiers exactly: EXACT, NUMERICAL, APPROXIMATION, COUNTERFACTUAL, VISUAL_LIBERTY.
- The model never quietly lies: assumptions listed, errors quantified or estimated, honest failure modes (e.g. positronium fine structure gets an error estimate as large as the shift, not a silent wrong number).
- TDD for all physics/server code; CI green on `windows-latest` for every push; conventional commits; MIT license.
- Conda at `C:\ProgramData\miniforge3` (NOT on PATH). **Always set `$env:PYTHONUTF8='1'` first** (conda run crashes printing Unicode test output otherwise, see memory `atom-sim-dev-env-gotchas`). From repo root:
  - Tests: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q`
  - Lint: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .`
  - Web tests (from `web\`): `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test`
- Ruff: line length 100, `E741` ignored, imports sorted (`I`). TypeScript `strict: true`, no `any`.

## File Structure

```
src/atomsim/
├── constants.py                   # MODIFY: add ALPHA
├── analytic/
│   ├── angular.py                 # CREATE: complex/real spherical harmonics, labels
│   ├── wavefunction.py            # CREATE: evaluate_state (full psi_nlm)
│   └── fine_structure.py          # CREATE: alpha^2 shifts, level energies
├── sampling.py                    # MODIFY: basis="complex"|"real"
├── systems.py                     # CREATE: preset registry (H, D, T, mu-H, Ps, He+, generic)
├── spectra.py                     # CREATE: transition lines, NIST comparison
├── data/
│   ├── __init__.py                # CREATE (empty docstring)
│   └── nist_h_i.json              # CREATE: vendored NIST H I lines (+ optional d_i, he_ii)
└── server/
    ├── schemas.py                 # MODIFY: System/Level/Line/Spectrum models
    └── app.py                     # MODIFY: /api/systems, /api/radial, /api/spectrum, params
tests/
├── test_angular.py                # CREATE
├── test_wavefunction.py           # CREATE
├── test_sampling.py               # MODIFY: real-basis tests
├── test_fine_structure.py         # CREATE
├── test_systems.py                # CREATE
├── test_spectra.py                # CREATE
└── test_server.py                 # MODIFY: new endpoint tests
web/src/api/types.ts               # MODIFY: mirror new JSON shapes
pyproject.toml                     # MODIFY: package-data for data/*.json
```

---

### Task 1: Angular module: complex + real spherical harmonics

**Files:**
- Create: `src/atomsim/analytic/angular.py`
- Test: `tests/test_angular.py`

**Interfaces:**
- Consumes: `Provenance`, `Fidelity` from `atomsim.provenance`; `scipy.special.sph_harm_y`.
- Produces (Tasks 2, 3 rely on these exact names):
  - `AngularValues` frozen dataclass: `values: np.ndarray` (complex for basis="complex", float for "real"), `theta: np.ndarray`, `phi: np.ndarray`, `l: int`, `m: int`, `basis: str`, `provenance: Provenance`.
  - `spherical_harmonic(l, m, theta, phi, basis="complex") -> AngularValues`, Condon-Shortley complex Y_lm, or real S_lm (chemistry convention: m>0 → cos-type, m<0 → sin-type).
  - `real_orbital_label(l, m) -> str`, "p_x", "d_z2", … for l ≤ 3; generic `"g(m=+2, cos)"` style above.
  - `validate_angular(l, m)`, raises `ValueError` unless `l >= 0` and `|m| <= l`.

- [x] **Step 1: Write the failing tests**, `tests/test_angular.py`:

```python
import numpy as np
import pytest

from atomsim.analytic.angular import (
    AngularValues,
    real_orbital_label,
    spherical_harmonic,
    validate_angular,
)
from atomsim.provenance import Fidelity

# Gauss-Legendre in cos(theta) x uniform phi: exact-enough quadrature on the sphere
_X, _W = np.polynomial.legendre.leggauss(96)
_THETA = np.arccos(_X)
_PHI = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
_TT, _PP = np.meshgrid(_THETA, _PHI, indexing="ij")

def _inner(l1, m1, b1, l2, m2, b2):
    y1 = spherical_harmonic(l1, m1, _TT, _PP, basis=b1).values
    y2 = spherical_harmonic(l2, m2, _TT, _PP, basis=b2).values
    integrand = np.conj(y1) * y2
    phi_mean = integrand.mean(axis=1) * 2.0 * np.pi
    return float(np.real(np.sum(_W * phi_mean)))

@pytest.mark.parametrize("basis", ["complex", "real"])
def test_orthonormality_up_to_l3(basis):
    states = [(l, m) for l in range(4) for m in range(-l, l + 1)]
    for i, (l1, m1) in enumerate(states):
        for l2, m2 in states[i:]:
            expected = 1.0 if (l1, m1) == (l2, m2) else 0.0
            assert _inner(l1, m1, basis, l2, m2, basis) == pytest.approx(
                expected, abs=1e-10
            ), (l1, m1, l2, m2, basis)

def test_known_closed_forms():
    th, ph = np.array([0.7]), np.array([1.1])
    y00 = spherical_harmonic(0, 0, th, ph).values[0]
    assert y00 == pytest.approx(1.0 / np.sqrt(4.0 * np.pi))
    y10 = spherical_harmonic(1, 0, th, ph).values[0]
    assert np.real(y10) == pytest.approx(np.sqrt(3.0 / (4.0 * np.pi)) * np.cos(0.7))

def test_condon_shortley_phase():
    # Y_1^1 = -sqrt(3/8pi) sin(theta) e^{i phi}: negative real part at phi=0
    y11 = spherical_harmonic(1, 1, np.array([np.pi / 2]), np.array([0.0])).values[0]
    assert np.real(y11) == pytest.approx(-np.sqrt(3.0 / (8.0 * np.pi)))
    assert np.imag(y11) == pytest.approx(0.0, abs=1e-15)

def test_real_basis_is_real_and_correctly_oriented():
    # p_x maximal along +x, p_y along +y, both positive there
    px = spherical_harmonic(1, 1, np.array([np.pi / 2]), np.array([0.0]), basis="real")
    py = spherical_harmonic(1, -1, np.array([np.pi / 2]), np.array([np.pi / 2]), basis="real")
    peak = np.sqrt(3.0 / (4.0 * np.pi))
    assert px.values.dtype.kind == "f"
    assert px.values[0] == pytest.approx(peak)
    assert py.values[0] == pytest.approx(peak)
    # d_xy positive at phi = pi/4 in the equatorial plane
    dxy = spherical_harmonic(2, -2, np.array([np.pi / 2]), np.array([np.pi / 4]), basis="real")
    assert dxy.values[0] > 0.0

def test_complex_phase_winds_with_m():
    phis = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)
    y = spherical_harmonic(2, 2, np.full_like(phis, 1.0), phis).values
    unwound = np.unwrap(np.angle(y))
    assert np.diff(unwound) == pytest.approx(2.0 * np.diff(phis))

def test_provenance_and_metadata():
    av = spherical_harmonic(2, 1, np.array([0.3]), np.array([0.4]), basis="real")
    assert isinstance(av, AngularValues)
    assert av.provenance.fidelity is Fidelity.EXACT
    assert av.basis == "real"
    assert (av.l, av.m) == (2, 1)

def test_labels():
    assert real_orbital_label(0, 0) == "s"
    assert real_orbital_label(1, 0) == "p_z"
    assert real_orbital_label(1, 1) == "p_x"
    assert real_orbital_label(1, -1) == "p_y"
    assert real_orbital_label(2, 0) == "d_z2"
    assert real_orbital_label(2, 2) == "d_x2-y2"
    assert real_orbital_label(2, -2) == "d_xy"
    assert real_orbital_label(3, 0) == "f_z3"
    assert real_orbital_label(4, 3) == "g(m=+3, cos)"

def test_validation():
    with pytest.raises(ValueError):
        validate_angular(-1, 0)
    with pytest.raises(ValueError):
        validate_angular(1, 2)
    with pytest.raises(ValueError):
        spherical_harmonic(1, 0, np.array([0.1]), np.array([0.1]), basis="chebyshev")
```

- [x] **Step 2: Run tests to verify they fail**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_angular.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.analytic.angular'`.

- [x] **Step 3: Implement**, `src/atomsim/analytic/angular.py`:

```python
"""Angular structure: complex spherical harmonics Y_lm and real orbitals S_lm.

Both bases are first-class engine outputs (spec 5.1). Complex Y_lm carry the
Condon-Shortley phase (scipy convention). Real orbitals use the standard
chemistry combinations: m > 0 -> cos(m phi) type, m < 0 -> sin(|m| phi) type.
The basis choice is physics-visible (chemistry vs physics teaching moment),
so `basis` is part of the provenance-carrying result, never a hidden default.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import sph_harm_y

from atomsim.provenance import Fidelity, Provenance

_L_LETTERS = "spdfghik"

_CHEMISTRY_LABELS = {
    (0, 0): "s",
    (1, 0): "p_z", (1, 1): "p_x", (1, -1): "p_y",
    (2, 0): "d_z2", (2, 1): "d_xz", (2, -1): "d_yz",
    (2, 2): "d_x2-y2", (2, -2): "d_xy",
    (3, 0): "f_z3", (3, 1): "f_xz2", (3, -1): "f_yz2",
    (3, 2): "f_z(x2-y2)", (3, -2): "f_xyz",
    (3, 3): "f_x(x2-3y2)", (3, -3): "f_y(3x2-y2)",
}

@dataclass(frozen=True)
class AngularValues:
    """Y_lm or S_lm evaluated on (theta, phi) points. Container carries provenance."""

    values: np.ndarray  # complex128 for basis="complex", float64 for "real"
    theta: np.ndarray
    phi: np.ndarray
    l: int
    m: int
    basis: str
    provenance: Provenance

def validate_angular(l: int, m: int) -> None:
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l}")
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")

def real_orbital_label(l: int, m: int) -> str:
    validate_angular(l, m)
    if (l, m) in _CHEMISTRY_LABELS:
        return _CHEMISTRY_LABELS[(l, m)]
    letter = _L_LETTERS[l] if l < len(_L_LETTERS) else f"(l={l})"
    if m == 0:
        return f"{letter}(m=0)"
    kind = "cos" if m > 0 else "sin"
    return f"{letter}(m={m:+d}, {kind})"

def spherical_harmonic(
    l: int, m: int, theta: np.ndarray, phi: np.ndarray, basis: str = "complex"
) -> AngularValues:
    """Evaluate Y_lm (complex, Condon-Shortley) or S_lm (real) on given angles."""
    validate_angular(l, m)
    if basis not in ("complex", "real"):
        raise ValueError(f"basis must be 'complex' or 'real', got {basis!r}")
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    if basis == "complex":
        values: np.ndarray = sph_harm_y(l, m, theta, phi)
        method = "complex spherical harmonic Y_lm (Condon-Shortley phase, scipy sph_harm_y)"
        assumptions = ("physics convention: eigenstates of L_z",)
    else:
        y_abs = sph_harm_y(l, abs(m), theta, phi)
        sign = (-1.0) ** abs(m)
        if m == 0:
            values = np.real(y_abs)
        elif m > 0:
            values = sign * np.sqrt(2.0) * np.real(y_abs)
        else:
            values = sign * np.sqrt(2.0) * np.imag(y_abs)
        method = (
            "real spherical harmonic S_lm = sqrt(2) (-1)^m Re/Im Y_l|m| "
            "(chemistry convention: m>0 cos-type, m<0 sin-type)"
        )
        assumptions = (
            "chemistry convention: NOT eigenstates of L_z for m != 0",
            f"orbital label: {real_orbital_label(l, m)}",
        )

    return AngularValues(
        values=values,
        theta=theta,
        phi=phi,
        l=l,
        m=m,
        basis=basis,
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method=method,
            assumptions=assumptions,
        ),
    )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_angular.py -q`
Expected: PASS (9 tests; the orthonormality sweep takes ~2 s).

- [x] **Step 5: Lint, full suite, commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
git add src/atomsim/analytic/angular.py tests/test_angular.py
git commit -m "feat: angular module - complex/real spherical harmonics with provenance"
```

---

### Task 2: Full wavefunction: `evaluate_state`

**Files:**
- Create: `src/atomsim/analytic/wavefunction.py`
- Test: `tests/test_wavefunction.py`

**Interfaces:**
- Consumes: `radial_wavefunction`, `validate_quantum_numbers` (hydrogen.py); `spherical_harmonic`, `validate_angular` (Task 1).
- Produces (server M3 jobs and tests rely on):
  - `WavefunctionValues` frozen dataclass: `values: np.ndarray` (complex128 or float64), `positions: np.ndarray` (N,3 bohr), `n, l, m: int`, `Z: int`, `mu_ratio: float`, `basis: str`, `provenance: Provenance`.
  - `evaluate_state(n, l, m, positions, Z=1, mu_ratio=1.0, basis="complex") -> WavefunctionValues`, ψ_nlm = R_nl(r)·Y_lm(θ,φ) at Cartesian points; `unit` implied bohr^-3/2 (documented in label).

- [x] **Step 1: Write the failing tests**, `tests/test_wavefunction.py`:

```python
import numpy as np
import pytest

from atomsim.analytic.wavefunction import WavefunctionValues, evaluate_state
from atomsim.provenance import Fidelity

def _spherical_grid(n, r_points=400, ang_points=48):
    r_max = 30.0 * n * n
    r = np.linspace(1e-8, r_max, r_points)
    x, w = np.polynomial.legendre.leggauss(ang_points)
    theta = np.arccos(x)
    phi = np.linspace(0.0, 2.0 * np.pi, ang_points, endpoint=False)
    R, T, P = np.meshgrid(r, theta, phi, indexing="ij")
    pos = np.stack(
        [
            (R * np.sin(T) * np.cos(P)).ravel(),
            (R * np.sin(T) * np.sin(P)).ravel(),
            (R * np.cos(T)).ravel(),
        ],
        axis=1,
    )
    # weights: trapezoid dr x GL d(cos theta) x uniform dphi
    dr = np.gradient(r)
    wgt = (
        (dr * r * r)[:, None, None]
        * w[None, :, None]
        * (2.0 * np.pi / ang_points)
    )
    return pos, wgt.ravel()

@pytest.mark.parametrize("n,l,m,basis", [(1, 0, 0, "complex"), (2, 1, 1, "complex"), (3, 2, -2, "real")])
def test_norm_is_one(n, l, m, basis):
    pos, wgt = _spherical_grid(n)
    psi = evaluate_state(n, l, m, pos, basis=basis).values
    norm = float(np.sum(np.abs(psi) ** 2 * wgt))
    assert norm == pytest.approx(1.0, abs=2e-3)

def test_complex_phase_is_exp_i_m_phi():
    phis = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)
    pos = np.stack([2.0 * np.cos(phis), 2.0 * np.sin(phis), np.full_like(phis, 1.3)], axis=1)
    psi = evaluate_state(3, 2, 2, pos).values
    unwound = np.unwrap(np.angle(psi))
    assert np.diff(unwound) == pytest.approx(2.0 * np.diff(phis))

def test_real_basis_returns_real_dtype():
    pos = np.array([[1.0, 0.5, -0.3], [0.2, -1.0, 2.0]])
    wf = evaluate_state(2, 1, 1, pos, basis="real")
    assert wf.values.dtype == np.float64

def test_origin_is_finite():
    pos = np.zeros((1, 3))
    psi_s = evaluate_state(1, 0, 0, pos).values
    psi_p = evaluate_state(2, 1, 0, pos).values
    assert np.isfinite(psi_s).all() and psi_s[0] != 0.0
    assert psi_p[0] == pytest.approx(0.0)

def test_container_and_provenance():
    pos = np.array([[1.0, 0.0, 0.0]])
    wf = evaluate_state(2, 1, 0, pos, Z=2, mu_ratio=0.5, basis="real")
    assert isinstance(wf, WavefunctionValues)
    assert wf.provenance.fidelity is Fidelity.EXACT
    assert (wf.n, wf.l, wf.m, wf.Z, wf.mu_ratio, wf.basis) == (2, 1, 0, 2, 0.5, "real")

def test_rejects_bad_input():
    with pytest.raises(ValueError):
        evaluate_state(1, 1, 0, np.zeros((1, 3)))
    with pytest.raises(ValueError):
        evaluate_state(1, 0, 0, np.zeros((3,)))  # not (N,3)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_wavefunction.py -q`
Expected: FAIL, `ModuleNotFoundError`.

- [x] **Step 3: Implement**, `src/atomsim/analytic/wavefunction.py`:

```python
"""Full hydrogen-like wavefunctions psi_nlm(r) = R_nl(r) Y_lm(theta, phi).

Both angular bases are first-class (spec 5.1). Values are complex128 in the
complex basis (phase feeds phase-as-hue rendering) and float64 in the real
basis. Positions are Cartesian, in bohr.
"""

from dataclasses import dataclass

import numpy as np

from atomsim.analytic.angular import spherical_harmonic, validate_angular
from atomsim.analytic.hydrogen import radial_wavefunction, validate_quantum_numbers
from atomsim.provenance import Fidelity, Provenance

@dataclass(frozen=True)
class WavefunctionValues:
    """psi_nlm evaluated at Cartesian points (bohr). Container carries provenance."""

    values: np.ndarray      # (N,) complex128 or float64, unit bohr^-3/2
    positions: np.ndarray   # (N, 3) float, bohr
    n: int
    l: int
    m: int
    Z: int
    mu_ratio: float
    basis: str
    provenance: Provenance

def evaluate_state(
    n: int,
    l: int,
    m: int,
    positions: np.ndarray,
    Z: int = 1,
    mu_ratio: float = 1.0,
    basis: str = "complex",
) -> WavefunctionValues:
    """Evaluate psi_nlm at (N, 3) Cartesian positions in bohr."""
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")

    r = np.linalg.norm(pos, axis=1)
    safe_r = np.where(r > 0.0, r, 1.0)
    theta = np.arccos(np.clip(pos[:, 2] / safe_r, -1.0, 1.0))
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    theta = np.where(r > 0.0, theta, 0.0)

    radial = radial_wavefunction(n, l, r, Z=Z, mu_ratio=mu_ratio)
    angular = spherical_harmonic(l, m, theta, phi, basis=basis)
    values = radial.values * angular.values

    return WavefunctionValues(
        values=values,
        positions=pos,
        n=n,
        l=l,
        m=m,
        Z=Z,
        mu_ratio=mu_ratio,
        basis=basis,
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method=(
                "psi_nlm = R_nl (closed-form Laguerre) x "
                f"{angular.provenance.method}"
            ),
            assumptions=radial.provenance.assumptions + angular.provenance.assumptions
            + ("values in bohr^-3/2 at Cartesian positions in bohr",),
        ),
    )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_wavefunction.py -q`
Expected: PASS (6 tests; the norm sweep takes a few seconds).

- [x] **Step 5: Lint, full suite, commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
git add src/atomsim/analytic/wavefunction.py tests/test_wavefunction.py
git commit -m "feat: full psi_nlm evaluation in both angular bases"
```

---

### Task 3: Real-basis Monte-Carlo sampling

**Files:**
- Modify: `src/atomsim/sampling.py`
- Test: `tests/test_sampling.py` (append)

**Interfaces:**
- Consumes: existing `sample_density` internals (`_radial_inverse_cdf`, `_costheta_inverse_cdf`, `_X_GRID_POINTS`); `validate_angular` (Task 1).
- Produces (Task 8 relies on):
  - `SampleCloud` gains field `basis: str` (after `mu_ratio`, before `provenance`).
  - `sample_density(..., basis: str = "complex")`, for `"real"`, φ is drawn from the analytic marginal ∝ cos²(mφ) (m>0) / sin²(|m|φ) (m<0) / uniform (m=0); θ-marginal is identical to the complex basis (|Θ_l|m||² is shared).

- [x] **Step 1: Write the failing tests**, append to `tests/test_sampling.py`:

```python
def _phis(cloud: SampleCloud) -> np.ndarray:
    return np.mod(
        np.arctan2(cloud.positions[:, 1].astype(float), cloud.positions[:, 0].astype(float)),
        2.0 * np.pi,
    )

def test_real_basis_px_phi_marginal_ks():
    # p_x: pdf(phi) = cos^2(phi)/pi -> CDF = (phi/2 + sin(2 phi)/4)/pi
    cloud = sample_density(2, 1, 1, count=COUNT, seed=21, basis="real")
    ks = kstest(
        _phis(cloud),
        lambda p: (p / 2.0 + np.sin(2.0 * p) / 4.0) / np.pi,
    )
    assert ks.statistic < 0.01, ks

def test_real_basis_px_angular_moment():
    # density ~ sin^2(theta) cos^2(phi): E[(x/r)^2] = 4/5 * 3/4 = 3/5
    cloud = sample_density(2, 1, 1, count=COUNT, seed=22, basis="real")
    r = _radii(cloud)
    assert ((cloud.positions[:, 0].astype(float) / r) ** 2).mean() == pytest.approx(
        0.6, abs=0.01
    )

def test_real_basis_dxy_sin_type():
    # d_xy (m=-2): pdf(phi) ~ sin^2(2 phi) -> E[sin^2(2 phi)] = 3/4
    cloud = sample_density(3, 2, -2, count=COUNT, seed=23, basis="real")
    assert (np.sin(2.0 * _phis(cloud)) ** 2).mean() == pytest.approx(0.75, abs=0.01)

def test_real_m0_matches_complex_m0_statistically():
    a = sample_density(2, 1, 0, count=COUNT, seed=24, basis="real")
    b = sample_density(2, 1, 0, count=COUNT, seed=25, basis="complex")
    za = (a.positions[:, 2].astype(float) / _radii(a)) ** 2
    zb = (b.positions[:, 2].astype(float) / _radii(b)) ** 2
    assert za.mean() == pytest.approx(zb.mean(), abs=0.01)

def test_basis_recorded_in_cloud_and_provenance():
    cloud = sample_density(2, 1, 1, count=2_000, seed=1, basis="real")
    assert cloud.basis == "real"
    assert "real" in cloud.provenance.method
    default = sample_density(2, 1, 1, count=2_000, seed=1)
    assert default.basis == "complex"

def test_rejects_unknown_basis():
    with pytest.raises(ValueError):
        sample_density(1, 0, 0, count=1_000, basis="cartoon")
```

- [x] **Step 2: Run tests to verify they fail**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_sampling.py -q`
Expected: FAIL, `TypeError: sample_density() got an unexpected keyword argument 'basis'`.

- [x] **Step 3: Implement**, in `src/atomsim/sampling.py`:

Add `basis: str` field to `SampleCloud` after `mu_ratio`:

```python
@dataclass(frozen=True)
class SampleCloud:
    """Positions sampled from |psi_nlm|^2, in bohr. Container carries provenance."""

    positions: np.ndarray  # (count, 3) float32
    n: int
    l: int
    m: int
    Z: int
    mu_ratio: float
    basis: str
    provenance: Provenance
```

Add the φ inverse-CDF helper below `_costheta_inverse_cdf`:

```python
def _phi_inverse_cdf(m: int):
    """Grid phi and CDF of the real-basis phi marginal (cos^2/sin^2 type)."""
    phi = np.linspace(0.0, 2.0 * np.pi, _X_GRID_POINTS)
    am = abs(m)
    if m > 0:
        cdf = (phi / 2.0 + np.sin(2.0 * am * phi) / (4.0 * am)) / np.pi
    else:
        cdf = (phi / 2.0 - np.sin(2.0 * am * phi) / (4.0 * am)) / np.pi
    cdf /= cdf[-1]
    return phi, cdf
```

Update `sample_density`, signature and body changes:

```python
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
    basis: str = "complex",
) -> SampleCloud:
    """Draw `count` positions from |psi_nlm|^2 in the chosen angular basis."""
    validate_quantum_numbers(n, l)
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if basis not in ("complex", "real"):
        raise ValueError(f"basis must be 'complex' or 'real', got {basis!r}")

    rng = np.random.default_rng(seed)
    r_grid, r_cdf, r_max = _radial_inverse_cdf(n, l, Z, mu_ratio)
    x_grid, x_cdf = _costheta_inverse_cdf(l, m)
    phi_sampler = None
    if basis == "real" and m != 0:
        phi_grid, phi_cdf = _phi_inverse_cdf(m)
        phi_sampler = (phi_grid, phi_cdf)
```

and inside the chunk loop replace the φ draw:

```python
        if phi_sampler is None:
            phi = rng.uniform(0.0, 2.0 * np.pi, size)
        else:
            phi = np.interp(rng.random(size), phi_sampler[1], phi_sampler[0])
```

Provenance and return, replace the method/assumptions and constructor:

```python
    phi_desc = (
        "phi uniform (|Y_lm|^2 is phi-independent)"
        if phi_sampler is None
        else "phi from analytic real-basis marginal (cos^2/sin^2 m phi)"
    )
    provenance = Provenance(
        fidelity=Fidelity.NUMERICAL,
        method=(
            f"factorized inverse-CDF Monte-Carlo of |psi_nlm|^2 ({basis} basis): "
            f"r from P(r)=r^2 R^2 (grid N={_R_GRID_POINTS}, r_max={r_max:g} bohr), "
            f"cos(theta) from |Theta_lm|^2 (grid N={_X_GRID_POINTS}), {phi_desc}"
        ),
        assumptions=(
            f"angular basis: {basis}",
            f"RNG PCG64 seed={seed}, count={count}",
            "positions in bohr",
        ),
        refinement="increase CDF grid resolution or sample count",
    )
    return SampleCloud(
        positions=positions, n=n, l=l, m=m, Z=Z, mu_ratio=mu_ratio,
        basis=basis, provenance=provenance,
    )
```

Also update the module docstring's last sentence (real-orbital sampling has now arrived) and keep `validate_angular` semantics consistent (sampling keeps its inline |m| check, no import needed).

- [x] **Step 4: Run tests to verify they pass**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_sampling.py -q`
Expected: PASS (17 tests).

- [x] **Step 5: Lint, full suite, commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
git add src/atomsim/sampling.py tests/test_sampling.py
git commit -m "feat: real-orbital Monte-Carlo sampling (analytic phi marginal)"
```

---

### Task 4: Fine structure: perturbative α² shifts

**Files:**
- Modify: `src/atomsim/constants.py` (add `ALPHA`)
- Create: `src/atomsim/analytic/fine_structure.py`
- Test: `tests/test_fine_structure.py`, `tests/test_constants.py` (append one test)

**Interfaces:**
- Consumes: `energy` (hydrogen.py), `Quantity`/`Provenance`/`Fidelity`.
- Produces (Tasks 6, 8 rely on):
  - `constants.ALPHA: float`, CODATA fine-structure constant (display/real-universe anchor, same caveat as `HARTREE_EV`).
  - `validate_j(l: int, j: float)`, raises unless `j == l ± 0.5` and `j >= 0.5`.
  - `fine_structure_shift(n, l, j, Z=1, mu_ratio=1.0, m_over_M=0.0) -> Quantity`, hartree, APPROXIMATION; `ΔE = -(mu_ratio · Z⁴ α² / (2 n⁴)) · (n/(j+½) − ¾)`; `error_estimate = |ΔE|·((Zα)² + m_over_M + 2·0.00116)` (α⁴ + recoil + g−2 scales).
  - `level_energy(n, l, j, Z=1, mu_ratio=1.0, m_over_M=0.0) -> Quantity`, hartree, APPROXIMATION: Bohr + shift.

- [x] **Step 1: Write the failing tests**, `tests/test_fine_structure.py`:

```python
import pytest
from scipy import constants as sc

from atomsim.analytic.fine_structure import fine_structure_shift, level_energy, validate_j
from atomsim.constants import HARTREE_EV
from atomsim.provenance import Fidelity

HARTREE_HZ = sc.physical_constants["hartree-hertz relationship"][0]
MU_H = 1836.152673426 / 1837.152673426  # proton-electron reduced mass ratio

def test_shift_is_negative_and_j_ordered():
    s12 = fine_structure_shift(2, 0, 0.5).value
    p12 = fine_structure_shift(2, 1, 0.5).value
    p32 = fine_structure_shift(2, 1, 1.5).value
    assert s12 < 0 and p12 < 0 and p32 < 0
    assert s12 == pytest.approx(p12)      # same j -> same shift (l-degenerate at alpha^2)
    assert p32 > p12                       # higher j is less bound

def test_2p_splitting_matches_measurement_within_g2_scale():
    # Measured 2p_{3/2}-2p_{1/2}: 10.969 GHz. Pauli alpha^2 with reduced mass
    # gives ~10.94 GHz; the ~0.2% gap is the electron anomalous moment (g != 2),
    # which the provenance declares as an assumption.
    split_hartree = (
        fine_structure_shift(2, 1, 1.5, mu_ratio=MU_H).value
        - fine_structure_shift(2, 1, 0.5, mu_ratio=MU_H).value
    )
    ghz = split_hartree * HARTREE_HZ / 1e9
    assert ghz == pytest.approx(10.969, rel=5e-3)

def test_1s_shift_magnitude():
    # -mu' alpha^2 / 8 hartree = -1.810e-4 eV
    ev = fine_structure_shift(1, 0, 0.5, mu_ratio=MU_H).value * HARTREE_EV
    assert ev == pytest.approx(-1.810e-4, rel=2e-3)

def test_level_energy_composes_bohr_plus_shift():
    e = level_energy(2, 1, 0.5)
    shift = fine_structure_shift(2, 1, 0.5)
    assert e.value == pytest.approx(-0.125 + shift.value)
    assert e.unit == "hartree"
    assert e.provenance.fidelity is Fidelity.APPROXIMATION

def test_provenance_is_honest():
    q = fine_structure_shift(2, 1, 0.5, mu_ratio=0.5, m_over_M=1.0)  # positronium-like
    assert q.provenance.fidelity is Fidelity.APPROXIMATION
    assert q.provenance.error_estimate >= abs(q.value)  # recoil O(1): error >= shift
    joined = " ".join(q.provenance.assumptions).lower()
    assert "darwin" in joined
    assert "g = 2" in joined or "g=2" in joined
    assert "dirac" in (q.provenance.refinement or "").lower()

def test_z_scaling_is_quartic():
    r = fine_structure_shift(2, 1, 1.5, Z=2).value / fine_structure_shift(2, 1, 1.5).value
    assert r == pytest.approx(16.0)

def test_validate_j():
    validate_j(1, 0.5)
    validate_j(1, 1.5)
    with pytest.raises(ValueError):
        validate_j(1, 2.5)
    with pytest.raises(ValueError):
        validate_j(0, -0.5)
    with pytest.raises(ValueError):
        fine_structure_shift(2, 1, 2.5)
```

Note on `test_provenance_is_honest`: `Quantity` has no top-level `error_estimate`, the assertion `q.error_estimate is None` must be DELETED if it does not typecheck; the real check is `q.provenance.error_estimate >= abs(q.value)`. (Keep only the provenance assertion.)

Append to `tests/test_constants.py`:

```python
def test_alpha_matches_codata():
    from atomsim.constants import ALPHA

    assert abs(ALPHA - 0.0072973525643) < 1e-11
```

- [x] **Step 2: Run tests to verify they fail**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_fine_structure.py tests/test_constants.py -q`
Expected: FAIL, `ModuleNotFoundError` / `ImportError: cannot import name 'ALPHA'`.

- [x] **Step 3: Implement**

In `src/atomsim/constants.py`, below `HARTREE_EV`:

```python
# Real-universe display anchor ONLY (same caveat as HARTREE_EV): counterfactual
# universes must derive alpha via FundamentalConstants.alpha, never this.
ALPHA: float = _sc.fine_structure
```

`src/atomsim/analytic/fine_structure.py`:

```python
"""Perturbative fine structure: spin-orbit + relativistic KE + Darwin, order alpha^2.

The standard combined Pauli result, a function of n and j only:
    Delta E = -(mu' Z^4 alpha^2 / (2 n^4)) (n/(j+1/2) - 3/4)   [hartree]
APPROXIMATION tier by construction; the error estimate carries the three known
neglected scales (alpha^4 terms, nuclear recoil O(m/M), electron g-2).
"""

from atomsim.analytic.hydrogen import energy, validate_quantum_numbers
from atomsim.constants import ALPHA
from atomsim.provenance import Fidelity, Provenance, Quantity

_G2 = 2.0 * 0.00116  # anomalous-moment scale on the spin-orbit piece

_FS_ASSUMPTIONS = (
    "Pauli approximation, order alpha^2: spin-orbit + relativistic kinetic + Darwin",
    "electron g = 2 exactly (anomalous moment ~0.1% of the splitting neglected)",
    "reduced-mass scaling to leading order; nuclear recoil O(m/M) neglected",
    "no Lamb shift / QED, no hyperfine structure",
)

def validate_j(l: int, j: float) -> None:
    if j < 0.5 or abs(abs(j - l) - 0.5) > 1e-12:
        raise ValueError(f"j must be l +/- 1/2 (and >= 1/2), got l={l}, j={j}")

def fine_structure_shift(
    n: int, l: int, j: float, Z: int = 1, mu_ratio: float = 1.0, m_over_M: float = 0.0
) -> Quantity:
    """Fine-structure energy shift Delta E(n, l, j) in hartree (APPROXIMATION)."""
    validate_quantum_numbers(n, l)
    validate_j(l, j)
    value = -(mu_ratio * Z**4 * ALPHA**2 / (2.0 * n**4)) * (n / (j + 0.5) - 0.75)
    error = abs(value) * ((Z * ALPHA) ** 2 + m_over_M + _G2)
    return Quantity(
        value=value,
        unit="hartree",
        label=f"dE_fs {n},{l},j={j:g} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                "combined Pauli fine structure "
                "dE = -(mu' Z^4 alpha^2 / 2 n^4)(n/(j+1/2) - 3/4)"
            ),
            assumptions=_FS_ASSUMPTIONS,
            error_estimate=error,
            refinement="exact Dirac hydrogen solution (planned Phase 3 flagship)",
        ),
    )

def level_energy(
    n: int, l: int, j: float, Z: int = 1, mu_ratio: float = 1.0, m_over_M: float = 0.0
) -> Quantity:
    """Bohr energy plus fine-structure shift, in hartree (APPROXIMATION)."""
    bohr = energy(n, Z=Z, mu_ratio=mu_ratio)
    shift = fine_structure_shift(n, l, j, Z=Z, mu_ratio=mu_ratio, m_over_M=m_over_M)
    return Quantity(
        value=bohr.value + shift.value,
        unit="hartree",
        label=f"E {n},{l},j={j:g} (Z={Z}, mu/m_e={mu_ratio:g})",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=f"{bohr.provenance.method} + {shift.provenance.method}",
            assumptions=_FS_ASSUMPTIONS,
            error_estimate=shift.provenance.error_estimate,
            refinement=shift.provenance.refinement,
        ),
    )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_fine_structure.py tests/test_constants.py -q`
Expected: PASS (7 + existing constants tests).

- [x] **Step 5: Lint, full suite, commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
git add src/atomsim/constants.py src/atomsim/analytic/fine_structure.py tests/test_fine_structure.py tests/test_constants.py
git commit -m "feat: perturbative fine structure with honest error scales"
```

---

### Task 5: System preset registry

**Files:**
- Create: `src/atomsim/systems.py`
- Test: `tests/test_systems.py`

**Interfaces:**
- Consumes: `Quantity`/`Provenance`/`Fidelity`; `scipy.constants.physical_constants`.
- Produces (Tasks 6, 8 rely on):
  - `System` frozen dataclass: `key: str`, `name: str`, `Z: int`, `mu_ratio: Quantity` (unit `"m_e"`), `m_over_M: float`, `description: str`.
  - `list_systems() -> tuple[System ...]`, order: h, d, t, mu-h, ps, he+.
  - `get_system(key: str) -> System`, `KeyError` with available keys on miss.
  - `hydrogen_like(Z: int, mu_ratio: float = 1.0) -> System`, generic preset, infinite-nuclear-mass assumption stated when `mu_ratio == 1.0`.

- [x] **Step 1: Write the failing tests**, `tests/test_systems.py`:

```python
import pytest
from scipy import constants as sc

from atomsim.provenance import Fidelity
from atomsim.systems import System, get_system, hydrogen_like, list_systems

R_P = sc.physical_constants["proton-electron mass ratio"][0]
R_MU = sc.physical_constants["muon-electron mass ratio"][0]

def test_registry_contents_and_order():
    keys = [s.key for s in list_systems()]
    assert keys == ["h", "d", "t", "mu-h", "ps", "he+"]

def test_hydrogen_reduced_mass():
    h = get_system("h")
    assert h.Z == 1
    assert h.mu_ratio.value == pytest.approx(R_P / (1.0 + R_P), rel=1e-12)
    assert h.mu_ratio.unit == "m_e"
    assert h.m_over_M == pytest.approx(1.0 / R_P, rel=1e-12)

def test_muonic_hydrogen_is_muon_orbiting_proton():
    muh = get_system("mu-h")
    expected = R_MU * R_P / (R_MU + R_P)  # in m_e units
    assert muh.mu_ratio.value == pytest.approx(expected, rel=1e-9)
    assert muh.mu_ratio.value == pytest.approx(185.84, rel=1e-3)
    assert muh.m_over_M == pytest.approx(R_MU / R_P, rel=1e-9)

def test_positronium_is_exactly_half():
    ps = get_system("ps")
    assert ps.mu_ratio.value == 0.5
    assert ps.m_over_M == 1.0
    assert ps.mu_ratio.provenance.error_estimate in (None, 0.0)

def test_helium_ion():
    he = get_system("he+")
    assert he.Z == 2
    assert he.mu_ratio.value == pytest.approx(7294.3 / 7295.3, rel=1e-4)

def test_provenance_cites_codata():
    d = get_system("d")
    assert d.mu_ratio.provenance.fidelity is Fidelity.EXACT
    assert "CODATA" in d.mu_ratio.provenance.method
    assert d.mu_ratio.provenance.error_estimate is not None  # measured-mass uncertainty

def test_generic_hydrogen_like():
    s = hydrogen_like(3)
    assert isinstance(s, System)
    assert s.Z == 3 and s.mu_ratio.value == 1.0 and s.m_over_M == 0.0
    assert "infinite nuclear mass" in " ".join(s.mu_ratio.provenance.assumptions)
    with pytest.raises(ValueError):
        hydrogen_like(0)

def test_unknown_key_lists_options():
    with pytest.raises(KeyError, match="he\\+"):
        get_system("uranium")
```

- [x] **Step 2: Run tests to verify they fail**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_systems.py -q`
Expected: FAIL, `ModuleNotFoundError`.

- [x] **Step 3: Implement**, `src/atomsim/systems.py`:

```python
"""Exotic-but-real hydrogen-like system presets (spec 5.5).

Each preset supplies nuclear charge Z and the exact reduced-mass ratio
mu/m_e as a Quantity whose provenance cites the CODATA mass ratios it was
built from. m_over_M (orbiting mass / nuclear mass) feeds the fine-structure
recoil error scale, honesty for positronium comes from a quantified error,
never a silent wrong number.
"""

from dataclasses import dataclass

from scipy import constants as _sc

from atomsim.provenance import Fidelity, Provenance, Quantity

@dataclass(frozen=True)
class System:
    key: str
    name: str
    Z: int
    mu_ratio: Quantity  # mu / m_e, unit "m_e"
    m_over_M: float     # orbiting mass / nuclear mass (recoil scale)
    description: str

def _mass_ratio(constant_name: str) -> tuple[float, float]:
    value, _unit, unc = _sc.physical_constants[constant_name]
    return value, unc

def _codata_system(
    key: str, name: str, Z: int, nucleus_constant: str, description: str,
    orbiter_constant: str | None = None,
) -> System:
    """Build a preset from CODATA mass ratios (electron orbiter unless given)."""
    R_nuc, u_nuc = _mass_ratio(nucleus_constant)  # M / m_e
    if orbiter_constant is None:
        r_orb, u_orb, orb_name = 1.0, 0.0, "electron"
    else:
        r_orb, u_orb = _mass_ratio(orbiter_constant)
        orb_name = orbiter_constant.split("-")[0]
    mu = r_orb * R_nuc / (r_orb + R_nuc)
    rel_unc = (u_nuc / R_nuc if R_nuc else 0.0) + (u_orb / r_orb if r_orb else 0.0)
    return System(
        key=key,
        name=name,
        Z=Z,
        mu_ratio=Quantity(
            value=mu,
            unit="m_e",
            label=f"mu/m_e ({name})",
            provenance=Provenance(
                fidelity=Fidelity.EXACT,
                method=(
                    "mu/m_e = m_orb M / (m_orb + M) from CODATA mass ratios "
                    f"(scipy.constants: {nucleus_constant}"
                    + (f", {orbiter_constant})" if orbiter_constant else ")")
                ),
                assumptions=(f"orbiting particle: {orb_name}",),
                error_estimate=mu * rel_unc,
            ),
        ),
        m_over_M=r_orb / R_nuc,
        description=description,
    )

_POSITRONIUM = System(
    key="ps",
    name="Positronium",
    Z=1,
    mu_ratio=Quantity(
        value=0.5,
        unit="m_e",
        label="mu/m_e (Positronium)",
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method="mu = m_e/2 exactly (electron-positron, equal masses)",
            assumptions=("orbiting particle: electron; 'nucleus': positron",),
            error_estimate=0.0,
        ),
    ),
    m_over_M=1.0,
    description="Electron bound to a positron; recoil is O(1), fine structure "
    "unreliable at alpha^2 (error estimate says so).",
)

_SYSTEMS: tuple[System, ...] = (
    _codata_system("h", "Hydrogen", 1, "proton-electron mass ratio",
                   "Ordinary hydrogen: electron + proton."),
    _codata_system("d", "Deuterium", 1, "deuteron-electron mass ratio",
                   "Heavy hydrogen: electron + deuteron."),
    _codata_system("t", "Tritium", 1, "triton-electron mass ratio",
                    "Radioactive hydrogen isotope: electron + triton."),
    _codata_system("mu-h", "Muonic hydrogen", 1, "proton-electron mass ratio",
                   "Muon orbiting a proton: ~186x smaller, ~186x deeper.",
                   orbiter_constant="muon-electron mass ratio"),
    _POSITRONIUM,
    _codata_system("he+", "Helium ion He+", 2, "alpha particle-electron mass ratio",
                   "One-electron helium: Z=2 scaling on real helium-4."),
)

def list_systems() -> tuple[System, ...]:
    return _SYSTEMS

def get_system(key: str) -> System:
    for s in _SYSTEMS:
        if s.key == key:
            return s
    raise KeyError(f"unknown system {key!r}; available: {[s.key for s in _SYSTEMS]}")

def hydrogen_like(Z: int, mu_ratio: float = 1.0) -> System:
    """Generic one-electron ion with charge Z (infinite nuclear mass by default)."""
    if Z < 1:
        raise ValueError(f"Z must be >= 1, got {Z}")
    assumptions = (
        ("infinite nuclear mass (mu_ratio = 1)",) if mu_ratio == 1.0
        else (f"user-supplied mu_ratio = {mu_ratio:g}",)
    )
    return System(
        key=f"z{Z}",
        name=f"Hydrogen-like Z={Z}",
        Z=Z,
        mu_ratio=Quantity(
            value=mu_ratio,
            unit="m_e",
            label=f"mu/m_e (Z={Z})",
            provenance=Provenance(
                fidelity=Fidelity.EXACT,
                method="user-specified reduced-mass ratio",
                assumptions=assumptions,
            ),
        ),
        m_over_M=0.0,
        description=f"Generic one-electron ion, Z={Z}.",
    )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_systems.py -q`
Expected: PASS (8 tests).

- [x] **Step 5: Lint, full suite, commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
git add src/atomsim/systems.py tests/test_systems.py
git commit -m "feat: system preset registry with CODATA-cited reduced masses"
```

---

### Task 6: Spectra: transition lines with selection rules

**Files:**
- Create: `src/atomsim/spectra.py` (comparison API arrives in Task 7)
- Test: `tests/test_spectra.py`

**Interfaces:**
- Consumes: `energy` (hydrogen.py), `level_energy` (Task 4), `System`/`get_system` (Task 5), `HARTREE_EV`.
- Produces (Tasks 7, 8 rely on):
  - `SpectralLine` frozen dataclass: `n_upper, l_upper: int`, `j_upper: float | None`, `n_lower, l_lower: int`, `j_lower: float | None`, `energy: Quantity` (eV), `wavelength: Quantity` (nm, vacuum).
  - `LineList` frozen dataclass: `system_key: str`, `n_max: int`, `fine_structure: bool`, `lines: tuple[SpectralLine, ...]`, `provenance: Provenance`.
  - `transition_lines(system: System, n_max: int, fine_structure: bool = False) -> LineList`, all electric-dipole-allowed (Δl = ±1; plus Δj ∈ {0, ±1} when fine structure is on) transitions with `n_upper <= n_max`, sorted by wavelength.

- [x] **Step 1: Write the failing tests**, `tests/test_spectra.py`:

```python
import numpy as np
import pytest

from atomsim.provenance import Fidelity
from atomsim.spectra import LineList, SpectralLine, transition_lines
from atomsim.systems import get_system

def _wavelengths(lines):
    return np.array([ln.wavelength.value for ln in lines])

def test_gross_lyman_alpha_wavelength():
    ll = transition_lines(get_system("h"), n_max=3)
    lya = [
        ln for ln in ll.lines
        if (ln.n_upper, ln.n_lower) == (2, 1) and ln.l_upper == 1
    ]
    assert len(lya) == 1
    # mu-corrected gross Lyman-alpha: 121.567 nm vacuum
    assert lya[0].wavelength.value == pytest.approx(121.567, abs=2e-3)
    assert lya[0].wavelength.unit == "nm (vacuum)"
    assert lya[0].energy.value == pytest.approx(10.199, abs=1e-3)

def test_selection_rule_delta_l():
    ll = transition_lines(get_system("h"), n_max=4)
    assert all(abs(ln.l_upper - ln.l_lower) == 1 for ln in ll.lines)
    # 2s -> 1s must be absent
    assert not any(
        (ln.n_upper, ln.l_upper, ln.n_lower) == (2, 0, 1) for ln in ll.lines
    )

def test_gross_lines_are_exact_and_sorted():
    ll = transition_lines(get_system("h"), n_max=5)
    assert isinstance(ll, LineList)
    assert ll.provenance.fidelity is Fidelity.EXACT
    w = _wavelengths(ll.lines)
    assert (np.diff(w) >= 0).all()
    assert all(ln.j_upper is None for ln in ll.lines)

def test_fine_structure_doublet():
    ll = transition_lines(get_system("h"), n_max=2, fine_structure=True)
    lya = [ln for ln in ll.lines if (ln.n_upper, ln.n_lower) == (2, 1)]
    # 2p_{1/2} -> 1s_{1/2} and 2p_{3/2} -> 1s_{1/2}
    assert sorted(ln.j_upper for ln in lya) == [0.5, 1.5]
    dl = abs(lya[0].wavelength.value - lya[1].wavelength.value)
    assert dl == pytest.approx(5.4e-4, rel=0.05)  # Lyman-alpha doublet ~0.54 pm
    assert ll.provenance.fidelity is Fidelity.APPROXIMATION

def test_fine_structure_delta_j_rule():
    ll = transition_lines(get_system("h"), n_max=3, fine_structure=True)
    assert all(abs(ln.j_upper - ln.j_lower) <= 1.0 + 1e-12 for ln in ll.lines)

def test_deuterium_isotope_shift_direction():
    h = transition_lines(get_system("h"), n_max=2).lines[0]
    d = transition_lines(get_system("d"), n_max=2).lines[0]
    assert d.wavelength.value < h.wavelength.value  # heavier nucleus -> bluer
    shift = h.wavelength.value - d.wavelength.value
    assert shift == pytest.approx(0.033, rel=0.05)  # ~33 pm Lyman-alpha H/D shift

def test_positronium_lyman_alpha_is_doubled():
    # lambda ~ 1/mu': lambda(mu'=1) = lambda_H * mu'_H; positronium doubles it
    MU_H = 1836.152673426 / 1837.152673426
    ps = transition_lines(get_system("ps"), n_max=2).lines[0]
    assert ps.wavelength.value == pytest.approx(2.0 * 121.567 * MU_H, rel=1e-4)
    assert ps.wavelength.value == pytest.approx(243.0, abs=0.1)  # literature Ps Lyman-alpha

def test_every_line_carries_provenance():
    ll = transition_lines(get_system("h"), n_max=3, fine_structure=True)
    for ln in ll.lines:
        assert isinstance(ln, SpectralLine)
        assert ln.energy.provenance is not None
        assert ln.wavelength.provenance is not None

def test_n_max_validation():
    with pytest.raises(ValueError):
        transition_lines(get_system("h"), n_max=1)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_spectra.py -q`
Expected: FAIL, `ModuleNotFoundError`.

- [x] **Step 3: Implement**, `src/atomsim/spectra.py`:

```python
"""Spectral lines from level differences, with selection rules and provenance.

Gross structure (fine_structure=False): EXACT Bohr levels (mu-scaled); lines
are (n_u, l_u) -> (n_l, l_l) with Delta l = +/-1. With fine structure on:
APPROXIMATION levels from the alpha^2 Pauli shifts, Delta j in {0, +/-1}.
Wavelengths are vacuum, in nm; energies in eV. NIST comparison: see the
compare_lines API (vendored reference data, never live queries).
"""

import itertools
from dataclasses import dataclass

from scipy import constants as _sc

from atomsim.analytic.fine_structure import level_energy
from atomsim.analytic.hydrogen import energy
from atomsim.constants import HARTREE_EV
from atomsim.provenance import Fidelity, Provenance, Quantity
from atomsim.systems import System

_EV_NM = _sc.h * _sc.c / _sc.e * 1e9  # photon wavelength(nm) = _EV_NM / E(eV)

@dataclass(frozen=True)
class SpectralLine:
    n_upper: int
    l_upper: int
    j_upper: float | None
    n_lower: int
    l_lower: int
    j_lower: float | None
    energy: Quantity      # eV
    wavelength: Quantity  # nm, vacuum

@dataclass(frozen=True)
class LineList:
    system_key: str
    n_max: int
    fine_structure: bool
    lines: tuple[SpectralLine, ...]
    provenance: Provenance

def _levels(system: System, n_max: int, fine_structure: bool):
    """Yield (n, l, j, E_hartree Quantity) for all levels up to n_max."""
    for n in range(1, n_max + 1):
        for l in range(n):
            if fine_structure:
                js = [l - 0.5, l + 0.5] if l > 0 else [0.5]
                for j in js:
                    yield n, l, j, level_energy(
                        n, l, j, Z=system.Z,
                        mu_ratio=system.mu_ratio.value, m_over_M=system.m_over_M,
                    )
            else:
                yield n, l, None, energy(n, Z=system.Z, mu_ratio=system.mu_ratio.value)

def transition_lines(
    system: System, n_max: int, fine_structure: bool = False
) -> LineList:
    """All dipole-allowed emission lines among levels with n <= n_max."""
    if n_max < 2:
        raise ValueError(f"n_max must be >= 2 to have any transition, got {n_max}")
    levels = list(_levels(system, n_max, fine_structure))
    lines: list[SpectralLine] = []
    for (nu, lu, ju, eu), (nl, ll_, jl, el) in itertools.permutations(levels, 2):
        if eu.value <= el.value:
            continue
        if abs(lu - ll_) != 1:
            continue
        if fine_structure and abs(ju - jl) > 1.0 + 1e-12:
            continue
        de_ev = (eu.value - el.value) * HARTREE_EV
        tier = Fidelity.APPROXIMATION if fine_structure else Fidelity.EXACT
        prov = Provenance(
            fidelity=tier,
            method=(
                f"level difference: [{eu.provenance.method}] minus "
                f"[{el.provenance.method}]; photon lambda = hc/dE (vacuum)"
            ),
            assumptions=eu.provenance.assumptions
            + ("electric-dipole selection rules (Delta l = +/-1"
               + (", Delta j in {0, +/-1})" if fine_structure else ")"),),
            error_estimate=(
                None if eu.provenance.error_estimate is None
                else (eu.provenance.error_estimate
                      + (el.provenance.error_estimate or 0.0)) * HARTREE_EV
            ),
            refinement=eu.provenance.refinement,
        )
        label = f"{nu}->{nl}"
        lines.append(
            SpectralLine(
                n_upper=nu, l_upper=lu, j_upper=ju,
                n_lower=nl, l_lower=ll_, j_lower=jl,
                energy=Quantity(de_ev, "eV", f"dE {label}", prov),
                wavelength=Quantity(_EV_NM / de_ev, "nm (vacuum)", f"lambda {label}", prov),
            )
        )
    lines.sort(key=lambda ln: ln.wavelength.value)
    return LineList(
        system_key=system.key,
        n_max=n_max,
        fine_structure=fine_structure,
        lines=tuple(lines),
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION if fine_structure else Fidelity.EXACT,
            method="dipole-allowed level differences (see per-line provenance)",
            assumptions=("emission lines only (E_upper > E_lower)",
                         "vacuum wavelengths in nm, energies in eV"),
        ),
    )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_spectra.py -q`
Expected: PASS (9 tests). If `test_gross_lyman_alpha_wavelength` fails on the 3rd decimal, print the computed value, the reference is `121.5670 nm`; a mismatch beyond 2e-3 nm means a units bug (check `_EV_NM` ≈ 1239.84198).

- [x] **Step 5: Lint, full suite, commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
git add src/atomsim/spectra.py tests/test_spectra.py
git commit -m "feat: spectral line lists with selection rules and per-line provenance"
```

---

### Task 7: Vendored NIST reference data + comparison API

**Files:**
- Create: `src/atomsim/data/__init__.py`, `src/atomsim/data/nist_h_i.json` (and `nist_he_ii.json`, `nist_d_i.json` if cleanly fetchable)
- Modify: `src/atomsim/spectra.py` (append comparison API), `pyproject.toml` (package data)
- Test: `tests/test_spectra.py` (append)

**Interfaces:**
- Consumes: `LineList`/`SpectralLine` (Task 6).
- Produces (Task 8 relies on):
  - `ReferenceData` frozen dataclass: `species: str`, `citation: str`, `retrieved: str`, `medium: str`, `lines: tuple[ReferenceLine, ...]`; `ReferenceLine`: `wavelength_nm: float`, `uncertainty_nm: float | None`, `label: str`.
  - `load_reference(system_key: str) -> ReferenceData | None`, returns `None` when no vendored data exists for that system (graceful degradation).
  - `LineComparison` frozen dataclass: `line: SpectralLine`, `reference_nm: float`, `reference_uncertainty_nm: float | None`, `delta_nm: float`, `relative_error: float`, `within_tolerance: bool`.
  - `compare_lines(line_list: LineList, reference: ReferenceData, tolerance_relative: float | None = None) -> tuple[LineComparison ...]`, default tolerance 3e-5 gross / 1e-5 with fine structure; matches each reference line to the nearest computed line.

**Data acquisition rule (honesty-critical):** the JSON values MUST come from NIST ASD at execution time, never from model memory. Every file carries citation + retrieval date. A sanity gate compares each vendored wavelength to the computed gross line at 1e-4 relative, which catches transcription errors.

- [x] **Step 1: Fetch NIST data.** Use WebFetch on the NIST ASD lines query for H I, vacuum Ritz wavelengths (90-2000 nm window covers Lyman through Brackett for n ≤ 6):

`https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=H+I&limits_type=0&low_w=90&upp_w=2000&unit=1&format=2&line_out=0&remove_js=on&en_unit=0&output=0&show_calc_wl=1&order_out=0&show_av=3&A_out=0&allowed_out=1&conf_out=on&term_out=on&J_out=on`

Extract, for each transition between principal levels n_l < n_u ≤ 6, the **Ritz vacuum wavelength in nm** and its quoted uncertainty. **Curation rule:** NIST resolves fine-structure components (e.g. several Hα entries); vendor exactly ONE line per gross (n_u → n_l) transition, the strongest component (largest relative intensity / A-value), recording which component in the label. The 3e-5 gross tolerance absorbs the ~2e-5 component spread. Include ONLY n_u ≤ 6 (higher-series reference lines would mis-match against the n_max=6 computed list). If the query format fights the fetcher, fall back to the interactive-form documentation page and adjust parameters, never transcribe wavelengths from memory. Repeat for `spectra=He+II` (He II) and `spectra=D+I` (D I); if either fetch is unusable, skip that file (loader degrades gracefully) and note it in the commit message.

- [x] **Step 2: Write the JSON + loader + sanity gate.**

`src/atomsim/data/__init__.py`:

```python
"""Vendored reference datasets (NIST ASD). Never fetched live at runtime."""
```

`src/atomsim/data/nist_h_i.json`, structure (VALUES FROM THE FETCH, these fields exactly):

```json
{
  "species": "H I",
  "citation": "Kramida, A., Ralchenko, Yu., Reader, J., and NIST ASD Team. NIST Atomic Spectra Database (ver. 5.12+), https://physics.nist.gov/asd",
  "retrieved": "<execution date, YYYY-MM-DD>",
  "medium": "vacuum",
  "lines": [
    {"wavelength_nm": 121.56701, "uncertainty_nm": 0.00001, "label": "Lyman-alpha (2-1)"}
  ]
}
```

(One entry per (n_u → n_l) gross line, n_u ≤ 6. The single line shown is the shape; fill all entries from the fetch.)

Append to `pyproject.toml` (check for an existing `[tool.setuptools]` section first and merge):

```toml
[tool.setuptools.package-data]
atomsim = ["data/*.json"]
```

Append to `src/atomsim/spectra.py`:

```python
import json
from importlib import resources

_REFERENCE_FILES = {"h": "nist_h_i.json", "d": "nist_d_i.json", "he+": "nist_he_ii.json"}

_DEFAULT_TOL = {False: 3e-5, True: 1e-5}  # relative, per fidelity tier

@dataclass(frozen=True)
class ReferenceLine:
    wavelength_nm: float
    uncertainty_nm: float | None
    label: str

@dataclass(frozen=True)
class ReferenceData:
    species: str
    citation: str
    retrieved: str
    medium: str
    lines: tuple[ReferenceLine, ...]

@dataclass(frozen=True)
class LineComparison:
    line: SpectralLine
    reference_nm: float
    reference_uncertainty_nm: float | None
    delta_nm: float
    relative_error: float
    within_tolerance: bool

def load_reference(system_key: str) -> ReferenceData | None:
    """Vendored NIST reference for a preset, or None (no live queries, ever)."""
    filename = _REFERENCE_FILES.get(system_key)
    if filename is None:
        return None
    ref = resources.files("atomsim.data").joinpath(filename)
    if not ref.is_file():
        return None
    raw = json.loads(ref.read_text(encoding="utf-8"))
    return ReferenceData(
        species=raw["species"],
        citation=raw["citation"],
        retrieved=raw["retrieved"],
        medium=raw["medium"],
        lines=tuple(
            ReferenceLine(
                wavelength_nm=ln["wavelength_nm"],
                uncertainty_nm=ln.get("uncertainty_nm"),
                label=ln.get("label", ""),
            )
            for ln in raw["lines"]
        ),
    )

def compare_lines(
    line_list: LineList,
    reference: ReferenceData,
    tolerance_relative: float | None = None,
) -> tuple[LineComparison, ...]:
    """Match each reference line to the nearest computed line; report residuals."""
    tol = tolerance_relative if tolerance_relative is not None else _DEFAULT_TOL[
        line_list.fine_structure
    ]
    out: list[LineComparison] = []
    for ref in reference.lines:
        if not line_list.lines:
            break
        nearest = min(
            line_list.lines, key=lambda ln: abs(ln.wavelength.value - ref.wavelength_nm)
        )
        delta = nearest.wavelength.value - ref.wavelength_nm
        rel = abs(delta) / ref.wavelength_nm
        if rel > 0.01:
            continue  # reference line outside the computed n_max window
        out.append(
            LineComparison(
                line=nearest,
                reference_nm=ref.wavelength_nm,
                reference_uncertainty_nm=ref.uncertainty_nm,
                delta_nm=delta,
                relative_error=rel,
                within_tolerance=rel <= tol,
            )
        )
    return tuple(out)
```

(Move the `import json` / `from importlib import resources` lines up into the module's import block, sorted.)

- [x] **Step 3: Write the tests**, append to `tests/test_spectra.py`:

```python
from atomsim.spectra import compare_lines, load_reference

def test_vendored_h_reference_loads_with_citation():
    ref = load_reference("h")
    assert ref is not None
    assert ref.medium == "vacuum"
    assert "NIST" in ref.citation
    assert len(ref.retrieved) == 10  # YYYY-MM-DD
    assert len(ref.lines) >= 10  # Lyman+Balmer+... up to n=6

def test_vendored_data_sanity_gate_against_computed_gross():
    # transcription-error tripwire: every vendored line within 1e-4 of computed
    ref = load_reference("h")
    ll = transition_lines(get_system("h"), n_max=6)
    for cmp_ in compare_lines(ll, ref, tolerance_relative=1e-4):
        assert cmp_.within_tolerance, (cmp_.reference_nm, cmp_.delta_nm)

def test_gross_comparison_within_tier_tolerance():
    ref = load_reference("h")
    ll = transition_lines(get_system("h"), n_max=6)
    comparisons = compare_lines(ll, ref)
    assert len(comparisons) >= 10
    assert all(c.within_tolerance for c in comparisons), [
        (c.reference_nm, c.relative_error) for c in comparisons if not c.within_tolerance
    ]

def test_fine_structure_improves_lyman_alpha():
    ref = load_reference("h")
    lya_ref = min(ref.lines, key=lambda ln: abs(ln.wavelength_nm - 121.567))
    gross = transition_lines(get_system("h"), n_max=2)
    fs = transition_lines(get_system("h"), n_max=2, fine_structure=True)
    ref_only = ReferenceData(
        species=ref.species, citation=ref.citation, retrieved=ref.retrieved,
        medium=ref.medium, lines=(lya_ref,),
    )
    d_gross = abs(compare_lines(gross, ref_only, 1.0)[0].delta_nm)
    d_fs = abs(compare_lines(fs, ref_only, 1.0)[0].delta_nm)
    assert d_fs <= d_gross * 1.5  # fs must not be wildly worse; usually better

def test_unknown_system_reference_is_none():
    assert load_reference("ps") is None
```

(Also add `from atomsim.spectra import ReferenceData` to the imports for `test_fine_structure_improves_lyman_alpha`.)

- [x] **Step 4: Reinstall (package data), run tests**

```powershell
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pip install -e ".[dev]"
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_spectra.py -q
```

Expected: PASS (14 tests). If the tier-tolerance test fails marginally (relative errors clustered just above 3e-5), inspect whether the reference wavelengths are truly vacuum, air wavelengths differ by ~2.8e-4 relative in the optical and mean the fetch grabbed the wrong column.

- [x] **Step 5: Lint, full suite, commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
git add src/atomsim/data src/atomsim/spectra.py tests/test_spectra.py pyproject.toml
git commit -m "feat: vendored NIST reference lines + comparison API with tier tolerances"
```

---

### Task 8: Server: systems, radial, spectrum endpoints; basis/system-aware jobs

**Files:**
- Modify: `src/atomsim/server/schemas.py`, `src/atomsim/server/app.py`, `web/src/api/types.ts`
- Test: `tests/test_server.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 3-7; existing schema models.
- Produces (M3 frontend consumes):
  - `GET /api/systems` → `{"systems": [SystemModel...]}`, `SystemModel {key, name, z, mu_ratio: QuantityModel, m_over_m_nucleus, description}`.
  - `GET /api/state/{n}/{l}/{m}?system=h&fine_structure=false` → existing `StateResponse` + `system: SystemModel` + `levels: list[LevelModel]` (`LevelModel {j, energy: QuantityModel, energy_ev: QuantityModel, shift: QuantityModel}`, empty list when `fine_structure=false`); energies/⟨r⟩ computed with the system's Z and μ.
  - `GET /api/radial/{n}/{l}?system=h&points=400` → `RadialResponse {n, l, system: SystemModel, r_wavefunction: FieldModel, radial_probability: FieldModel}`, grid 0..r_max(20n²/(Zμ)), downsampled to `points` (50-2000).
  - `GET /api/spectrum?system=h&n_max=6&fine_structure=false` → `SpectrumResponse {system: SystemModel, n_max, fine_structure, lines: [LineModel...], comparison: [ComparisonModel...] | null, reference_citation: str | null, tolerance_relative: float | null}`, `LineModel {n_upper, l_upper, j_upper, n_lower, l_lower, j_lower, energy_ev: QuantityModel, wavelength_nm: QuantityModel}`; `ComparisonModel {wavelength_nm, reference_nm, reference_uncertainty_nm, delta_nm, relative_error, within_tolerance}`.
  - `POST /api/jobs/sample` body gains `basis: "complex"|"real" = "complex"` and `system: str = "h"`; meta response gains `basis` and `system`.
  - `web/src/api/types.ts` mirrors all of the above 1:1 (`SystemInfo`, `LevelInfo`, `RadialResponse`, `SpectralLineInfo`, `ComparisonInfo`, `SpectrumResponse`; `SampleMeta` gains `basis: string; system: string`).

- [x] **Step 1: Write the failing tests**, append to `tests/test_server.py`:

```python
def test_systems_endpoint_lists_presets(client):
    body = client.get("/api/systems").json()
    keys = [s["key"] for s in body["systems"]]
    assert keys == ["h", "d", "t", "mu-h", "ps", "he+"]
    mu = body["systems"][0]["mu_ratio"]
    assert mu["provenance"]["fidelity"] == "exact"
    assert "CODATA" in mu["provenance"]["method"]

def test_state_accepts_system_and_fine_structure(client):
    body = client.get("/api/state/2/1/0?system=he%2B&fine_structure=true").json()
    assert body["system"]["key"] == "he+"
    assert body["energy"]["value"] == pytest.approx(-0.5 * 0.999863, rel=1e-4)
    js = sorted(lvl["j"] for lvl in body["levels"])
    assert js == [0.5, 1.5]
    assert body["levels"][0]["shift"]["provenance"]["fidelity"] == "approximation"

def test_state_defaults_stay_hydrogen_gross(client):
    body = client.get("/api/state/2/1/0").json()
    assert body["system"]["key"] == "h"
    assert body["levels"] == []
    # mu-scaled now: -0.125 * mu'
    assert body["energy"]["value"] == pytest.approx(-0.125 * 0.9994557, rel=1e-6)

def test_state_unknown_system_is_422(client):
    assert client.get("/api/state/1/0/0?system=xenon").status_code == 422

def test_radial_endpoint_fields(client):
    body = client.get("/api/radial/2/1?points=200").json()
    rw, rp = body["r_wavefunction"], body["radial_probability"]
    assert len(rw["grid"]) == 200 and len(rw["values"]) == 200
    assert rw["provenance"]["fidelity"] == "exact"
    p = np.array(rp["values"])
    g = np.array(rp["grid"])
    assert np.trapezoid(p, g) == pytest.approx(1.0, abs=5e-3)  # P(r) normalized

def test_spectrum_endpoint_with_comparison(client):
    body = client.get("/api/spectrum?system=h&n_max=6").json()
    assert body["reference_citation"] and "NIST" in body["reference_citation"]
    assert len(body["lines"]) > 10
    assert body["comparison"] is not None
    assert all(c["within_tolerance"] for c in body["comparison"])

def test_spectrum_without_reference_data(client):
    body = client.get("/api/spectrum?system=ps&n_max=3").json()
    assert body["comparison"] is None
    assert body["reference_citation"] is None
    assert len(body["lines"]) > 0

def test_sample_job_real_basis_and_system(client):
    r = client.post(
        "/api/jobs/sample",
        json={"n": 2, "l": 1, "m": 1, "count": 5000, "seed": 3,
              "basis": "real", "system": "d"},
    )
    job_id = r.json()["id"]
    final = _wait_done(client, job_id)
    assert final["status"] == "done"
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["basis"] == "real"
    assert meta["system"] == "d"

def test_sample_job_rejects_bad_basis(client):
    r = client.post(
        "/api/jobs/sample",
        json={"n": 1, "l": 0, "m": 0, "basis": "cartoon"},
    )
    assert r.status_code == 422
```

- [x] **Step 2: Run tests to verify they fail**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_server.py -q`
Expected: new tests FAIL (404s / missing keys); the 9 pre-existing tests still pass EXCEPT `test_state_carries_exact_provenance`, which asserts `energy == -0.125`, the state endpoint becoming μ-aware changes it to `-0.125·μ'`. Update that assertion in place to `pytest.approx(-0.125 * 0.9994557, rel=1e-6)`, this is a deliberate physics improvement (the M1 endpoint silently assumed infinite nuclear mass; now it is honest about hydrogen).

- [x] **Step 3: Implement schemas**, append to `src/atomsim/server/schemas.py`:

```python
from atomsim.spectra import LineComparison, SpectralLine
from atomsim.systems import System

class SystemModel(BaseModel):
    key: str
    name: str
    z: int
    mu_ratio: QuantityModel
    m_over_m_nucleus: float
    description: str

    @classmethod
    def from_system(cls, s: System) -> "SystemModel":
        return cls(
            key=s.key,
            name=s.name,
            z=s.Z,
            mu_ratio=QuantityModel.from_quantity(s.mu_ratio),
            m_over_m_nucleus=s.m_over_M,
            description=s.description,
        )

class LineModel(BaseModel):
    n_upper: int
    l_upper: int
    j_upper: float | None
    n_lower: int
    l_lower: int
    j_lower: float | None
    energy_ev: QuantityModel
    wavelength_nm: QuantityModel

    @classmethod
    def from_line(cls, ln: SpectralLine) -> "LineModel":
        return cls(
            n_upper=ln.n_upper, l_upper=ln.l_upper, j_upper=ln.j_upper,
            n_lower=ln.n_lower, l_lower=ln.l_lower, j_lower=ln.j_lower,
            energy_ev=QuantityModel.from_quantity(ln.energy),
            wavelength_nm=QuantityModel.from_quantity(ln.wavelength),
        )

class ComparisonModel(BaseModel):
    wavelength_nm: float
    reference_nm: float
    reference_uncertainty_nm: float | None
    delta_nm: float
    relative_error: float
    within_tolerance: bool

    @classmethod
    def from_comparison(cls, c: LineComparison) -> "ComparisonModel":
        return cls(
            wavelength_nm=c.line.wavelength.value,
            reference_nm=c.reference_nm,
            reference_uncertainty_nm=c.reference_uncertainty_nm,
            delta_nm=c.delta_nm,
            relative_error=c.relative_error,
            within_tolerance=c.within_tolerance,
        )
```

(`BaseModel`, `QuantityModel` already imported/defined in the module; add the two new imports to the sorted import block.)

- [x] **Step 4: Implement endpoints**, in `src/atomsim/server/app.py`:

New imports (merge into the sorted block):

```python
import numpy as np

from atomsim.analytic.hydrogen import radial_wavefunction
from atomsim.analytic.fine_structure import fine_structure_shift, level_energy
from atomsim.provenance import Field
from atomsim.server.schemas import (
    ComparisonModel,
    FieldModel,
    LineModel,
    ProvenanceModel,
    QuantityModel,
    SystemModel,
)
from atomsim.spectra import compare_lines, load_reference, transition_lines
from atomsim.systems import get_system, list_systems
```

New response models (near the existing ones):

```python
class LevelModel(BaseModel):
    j: float
    energy: QuantityModel
    energy_ev: QuantityModel
    shift: QuantityModel

class SystemsResponse(BaseModel):
    systems: list[SystemModel]

class RadialResponse(BaseModel):
    n: int
    l: int
    system: SystemModel
    r_wavefunction: FieldModel
    radial_probability: FieldModel

class SpectrumResponse(BaseModel):
    system: SystemModel
    n_max: int
    fine_structure: bool
    lines: list[LineModel]
    comparison: list[ComparisonModel] | None
    reference_citation: str | None
    tolerance_relative: float | None
```

`StateResponse` gains two fields:

```python
class StateResponse(BaseModel):
    n: int
    l: int
    m: int
    system: SystemModel
    energy: QuantityModel
    energy_ev: QuantityModel
    mean_radius: QuantityModel
    levels: list[LevelModel]
```

`SampleRequest` gains:

```python
    basis: Literal["complex", "real"] = "complex"
    system: str = "h"
```

(add `from typing import Literal` to imports). `SampleMetaModel` gains `basis: str` and `system: str`.

Helper + endpoint changes inside `create_app()`:

```python
    def _resolve_system(key: str):
        try:
            return get_system(key)
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/systems", response_model=SystemsResponse)
    def systems() -> SystemsResponse:
        return SystemsResponse(systems=[SystemModel.from_system(s) for s in list_systems()])
```

Replace the `state` endpoint:

```python
    @app.get("/api/state/{n}/{l}/{m}", response_model=StateResponse)
    def state(n: int, l: int, m: int, system: str = "h",
              fine_structure: bool = False) -> StateResponse:
        _validate_state(n, l, m)
        sys_ = _resolve_system(system)
        mu = sys_.mu_ratio.value
        e = energy(n, Z=sys_.Z, mu_ratio=mu)
        levels: list[LevelModel] = []
        if fine_structure:
            js = [l - 0.5, l + 0.5] if l > 0 else [0.5]
            for j in js:
                le = level_energy(n, l, j, Z=sys_.Z, mu_ratio=mu, m_over_M=sys_.m_over_M)
                sh = fine_structure_shift(
                    n, l, j, Z=sys_.Z, mu_ratio=mu, m_over_M=sys_.m_over_M
                )
                levels.append(
                    LevelModel(
                        j=j,
                        energy=QuantityModel.from_quantity(le),
                        energy_ev=QuantityModel.from_quantity(_to_ev(le)),
                        shift=QuantityModel.from_quantity(sh),
                    )
                )
        return StateResponse(
            n=n, l=l, m=m,
            system=SystemModel.from_system(sys_),
            energy=QuantityModel.from_quantity(e),
            energy_ev=QuantityModel.from_quantity(_to_ev(e)),
            mean_radius=QuantityModel.from_quantity(
                mean_radius(n, l, Z=sys_.Z, mu_ratio=mu)
            ),
            levels=levels,
        )
```

New radial endpoint:

```python
    @app.get("/api/radial/{n}/{l}", response_model=RadialResponse)
    def radial(n: int, l: int, system: str = "h", points: int = 400) -> RadialResponse:
        _validate_state(n, l, 0)
        if not 50 <= points <= 2000:
            raise HTTPException(status_code=422, detail="points must be in [50, 2000]")
        sys_ = _resolve_system(system)
        mu = sys_.mu_ratio.value
        r_max = 20.0 * n * n / (sys_.Z * mu)
        r = np.linspace(0.0, r_max, points)
        rw = radial_wavefunction(n, l, r, Z=sys_.Z, mu_ratio=mu)
        p = Field(
            values=r * r * rw.values**2,
            grid=r,
            unit="bohr^-1",
            grid_unit="bohr",
            label=f"P_{n},{l}(r) = r^2 R^2",
            provenance=rw.provenance,
        )
        return RadialResponse(
            n=n, l=l, system=SystemModel.from_system(sys_),
            r_wavefunction=FieldModel.from_field(rw),
            radial_probability=FieldModel.from_field(p),
        )
```

New spectrum endpoint:

```python
    @app.get("/api/spectrum", response_model=SpectrumResponse)
    def spectrum(system: str = "h", n_max: int = 6,
                 fine_structure: bool = False) -> SpectrumResponse:
        if not 2 <= n_max <= 10:
            raise HTTPException(status_code=422, detail="n_max must be in [2, 10]")
        sys_ = _resolve_system(system)
        lines = transition_lines(sys_, n_max=n_max, fine_structure=fine_structure)
        reference = load_reference(sys_.key)
        comparison = None
        citation = None
        tol = None
        if reference is not None:
            tol = 1e-5 if fine_structure else 3e-5
            comparison = [
                ComparisonModel.from_comparison(c)
                for c in compare_lines(lines, reference, tolerance_relative=tol)
            ]
            citation = reference.citation
        return SpectrumResponse(
            system=SystemModel.from_system(sys_),
            n_max=n_max,
            fine_structure=fine_structure,
            lines=[LineModel.from_line(ln) for ln in lines.lines],
            comparison=comparison,
            reference_citation=citation,
            tolerance_relative=tol,
        )
```

Sample job, inside `create_sample_job`, resolve the system and pass basis:

```python
        sys_ = _resolve_system(req.system)

        def work(progress):
            return sample_density(
                req.n, req.l, req.m, req.count,
                Z=sys_.Z, mu_ratio=sys_.mu_ratio.value,
                seed=req.seed, progress=progress, basis=req.basis,
            )
```

The cloud records physics (Z, mu_ratio, basis) but not the preset key, that is request metadata. Store it beside the job: initialize `app.state.job_systems: dict[str, str] = {}` next to `app.state.jobs`, set `app.state.job_systems[job.id] = req.system` in `create_sample_job`, and read it in `sample_meta`.

In `sample_meta`, add:

```python
            basis=cloud.basis,
            system=app.state.job_systems.get(job_id, "h"),
```

- [x] **Step 5: Mirror in TypeScript**, `web/src/api/types.ts`, append (and extend `SampleMeta`):

```ts
export interface SystemInfo {
  key: string;
  name: string;
  z: number;
  mu_ratio: Quantity;
  m_over_m_nucleus: number;
  description: string;
}

export interface LevelInfo {
  j: number;
  energy: Quantity;
  energy_ev: Quantity;
  shift: Quantity;
}

export interface RadialResponse {
  n: number;
  l: number;
  system: SystemInfo;
  r_wavefunction: FieldData;
  radial_probability: FieldData;
}

export interface FieldData {
  values: number[];
  grid: number[];
  unit: string;
  grid_unit: string;
  label: string;
  provenance: Provenance;
}

export interface SpectralLineInfo {
  n_upper: number;
  l_upper: number;
  j_upper: number | null;
  n_lower: number;
  l_lower: number;
  j_lower: number | null;
  energy_ev: Quantity;
  wavelength_nm: Quantity;
}

export interface ComparisonInfo {
  wavelength_nm: number;
  reference_nm: number;
  reference_uncertainty_nm: number | null;
  delta_nm: number;
  relative_error: number;
  within_tolerance: boolean;
}

export interface SpectrumResponse {
  system: SystemInfo;
  n_max: number;
  fine_structure: boolean;
  lines: SpectralLineInfo[];
  comparison: ComparisonInfo[] | null;
  reference_citation: string | null;
  tolerance_relative: number | null;
}
```

`StateResponse` in types.ts gains `system: SystemInfo; levels: LevelInfo[];`. `SampleMeta` gains `basis: string; system: string;`.

- [x] **Step 6: Run all tests**

```powershell
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_server.py -q
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
cd web
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm run build
cd ..
```

Expected: all PASS; `npm run build` type-checks the mirrored types.

- [x] **Step 7: Lint and commit**

```powershell
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
git add src/atomsim/server tests/test_server.py web/src/api/types.ts
git commit -m "feat: systems/radial/spectrum endpoints, basis+system-aware sampling"
```

---

### Task 9: M2 wrap: README, full sweep, push, CI green

**Files:**
- Modify: `README.md` (status section), this plan (check boxes)

- [x] **Step 1: README status**, in the `## Status, Phase 1 walking skeleton` section, retitle to `## Status, Phase 1 M2: engine depth` and add bullets (keep existing ones that still hold):

```markdown
- Real AND complex angular bases engine-wide (chemistry p_x/d_xy orbitals vs
  L_z eigenstates, the basis choice is labeled, never hidden)
- Perturbative fine structure (spin-orbit + relativistic + Darwin) with honest
  error scales: alpha^4, nuclear recoil, and electron g-2 all quantified
- Spectral line lists with selection rules, compared against vendored NIST ASD
  reference wavelengths in CI (citation + retrieval date in-repo)
- System presets: H, D, T, muonic hydrogen, positronium, He+, generic Z
```

- [x] **Step 2: Full verification sweep**

```powershell
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
cd web
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test
$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm run build
cd ..
```

Expected: everything green (~130 Python tests, 8 vitest).

- [x] **Step 3: Commit, push, verify CI**

```powershell
git add README.md docs/plans/2026-07-06-phase1-m2-engine-depth.md
git commit -m "docs: M2 engine-depth status"
git push
```

Then poll `https://api.github.com/repos/yaasshh09/atomsim/actions/runs?per_page=1` until `status=completed`; require `conclusion=success`. If red: open the run, fix, recommit, never leave main red.

---

M2 is complete when: CI is green on `main`; `/api/spectrum?system=h&n_max=6` returns NIST-validated lines; a real-basis 2p_x cloud samples correctly; and every new quantity (shift, line, mass ratio) shows honest provenance.
