# Phase 7 — Numerical Wavefunction Sampling for Screened Atoms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give screened atoms (He–Ar) real 3-D point-cloud (`CloudView`) and 2-D cross-section (`PlaneView`) renders by sampling/evaluating their numerical GSZ/GJG orbitals, replacing the Phase 6 honest placeholders.

**Architecture:** Both sampling and the plane factorize into a radial part × an angular part; the angular part of a central-field atom is identical to hydrogen's. So we (1) reuse the existing angular CDF/`Y_lm` helpers untouched, (2) swap the radial source from analytic `radial_wavefunction` to the numerical `screened_radial`, and (3) relabel fidelity from `EXACT`/`NUMERICAL` to `APPROXIMATION`. The job/WebSocket/binary transport and all response schemas are reused verbatim; only the two job *producers* in `server/app.py` change.

**Tech Stack:** Python 3.12 (conda env `atomsim`), NumPy, SciPy (`cumulative_trapezoid`, `lpmv`, `kstest`), FastAPI; React/TypeScript + Vitest frontend.

## Global Constraints

- **Prime directive:** every value crossing a module boundary carries a `Provenance` with an honest `Fidelity`. Both new renders are `Fidelity.APPROXIMATION` (GSZ/GJG model error dominates the Monte-Carlo/grid error). A test asserts the tier to guard against a stray `EXACT`/`NUMERICAL` copied from the hydrogen path.
- **Engine-internal math is in Hartree atomic units**; screened atoms use `mu_ratio = 1.0` (infinite-nucleus, as Phase 6 established).
- **No protocol/schema change.** Reuse `SampleCloud`, `WavefunctionValues`, `PlaneGrid`, `SampleJobResult`, and every `meta`/binary endpoint. `evaluate_screened_state` MUST return a `WavefunctionValues` (the server reads `res.psi.values` and `res.psi.provenance`).
- **No new store state or URL schema** on the frontend. `(n, l, m, system, view)` already round-trips.
- **Config independence:** a single orbital's radial shape depends only on `(Z, N)`; the `config` query param is irrelevant to cloud/plane and is neither required nor consumed by them.
- **Lint:** `ruff check .` must stay clean (line-length 100; E741 ignored). Do **not** assign lambdas to names (E731) — use nested `def`.
- **Import direction:** `sampling.py` and `plane.py` may import from `screened_atom.py`; `screened_atom.py` must NOT import `sampling`/`plane` (avoids a cycle).
- **Run tests (Windows PowerShell)** with the MKL fix:
  `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest -q`
  Lint: `& "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check .`

---

## Task 1: Screened cloud sampling (`sample_screened_density`)

**Files:**
- Modify: `src/atomsim/sampling.py`
- Test: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `screened_radial(z, n_electrons, n, l, points) -> tuple[Field, Field]` (first `Field` has `.grid` and `.values` = numerical `R_nl`, `.provenance.error_estimate`); `screening_provenance(z, n_electrons) -> Provenance`; existing `_costheta_inverse_cdf(l, m)`, `_phi_inverse_cdf(m)`.
- Produces: `sample_screened_density(z, n_electrons, n, l, m, count, *, seed=0, progress=None, n_chunks=10, basis="complex") -> SampleCloud` with `provenance.fidelity == Fidelity.APPROXIMATION`. Also the private helpers `_radial_inverse_cdf_tabulated(r_grid, R_values)` and `_draw_positions(...)` used by both samplers.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sampling.py`:

```python
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import lpmv
from scipy.stats import kstest

from atomsim.provenance import Fidelity
from atomsim.sampling import sample_screened_density
from atomsim.screened_atom import screened_radial


def test_screened_radial_marginal_matches_numerical_cdf():
    # Na 3s: sampled radial marginal must match the numerical P(r)=r^2 R^2 CDF.
    cloud = sample_screened_density(11, 11, 3, 0, 0, 20000, seed=1)
    r = np.linalg.norm(cloud.positions, axis=1)
    r_field, _ = screened_radial(11, 11, 3, 0, points=8192)
    grid, R = r_field.grid, r_field.values
    p = grid**2 * R**2
    cdf = cumulative_trapezoid(p, grid, initial=0.0)
    cdf /= cdf[-1]
    _, pval = kstest(r, lambda x: np.interp(x, grid, cdf))
    assert pval > 0.01


def test_screened_angular_marginal_matches_legendre():
    # Na 3p, m=0: cos(theta) marginal must follow |Theta_10|^2 (central field).
    cloud = sample_screened_density(11, 11, 3, 1, 0, 20000, seed=2)
    cos_t = cloud.positions[:, 2] / np.linalg.norm(cloud.positions, axis=1)
    x = np.linspace(-1.0, 1.0, 4096)
    p = lpmv(0, 1, x) ** 2
    cdf = cumulative_trapezoid(p, x, initial=0.0)
    cdf /= cdf[-1]
    _, pval = kstest(cos_t, lambda v: np.interp(v, x, cdf))
    assert pval > 0.01


def test_screened_cloud_is_approximation_and_sane():
    cloud = sample_screened_density(11, 11, 3, 0, 0, 5000, seed=3)
    assert cloud.positions.shape == (5000, 3)
    assert cloud.provenance.fidelity is Fidelity.APPROXIMATION
    assert "GSZ" in cloud.provenance.method or "screen" in cloud.provenance.method.lower()
    r = np.linalg.norm(cloud.positions, axis=1)
    assert np.all(np.isfinite(r))
    assert 1.0 < float(r.mean()) < 20.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_sampling.py -k screened -q`
Expected: FAIL — `ImportError: cannot import name 'sample_screened_density'`.

- [ ] **Step 3: Refactor the radial CDF + chunk loop into reusable helpers**

In `src/atomsim/sampling.py`, add a tabulated-radial helper and rewrite `_radial_inverse_cdf` to delegate (behavior identical — `r_grid[-1] == r_max`):

```python
def _radial_inverse_cdf_tabulated(r_grid: np.ndarray, R_values: np.ndarray):
    """Grid r and CDF of P(r) = r^2 R^2 from a tabulated R_nl (grid, values)."""
    p = r_grid * r_grid * R_values * R_values
    cdf = cumulative_trapezoid(p, r_grid, initial=0.0)
    cdf /= cdf[-1]
    return r_grid, cdf, float(r_grid[-1])


def _radial_inverse_cdf(n: int, l: int, Z: int, mu_ratio: float):
    """Grid r and CDF of P(r) = r^2 R_nl^2 for inverse-CDF sampling (analytic)."""
    r_max = 20.0 * n * n / (Z * mu_ratio)  # P(r_max)/P_peak < 1e-15 for all l < n
    r = np.linspace(0.0, r_max, _R_GRID_POINTS)
    R = radial_wavefunction(n, l, r, Z=Z, mu_ratio=mu_ratio).values
    return _radial_inverse_cdf_tabulated(r, R)
```

Extract the chunked draw loop (currently the body of `sample_density`) into a shared helper:

```python
def _draw_positions(count, r_grid, r_cdf, x_grid, x_cdf, phi_sampler, seed, n_chunks, progress):
    """Inverse-CDF draw of `count` Cartesian positions (bohr) from factorized CDFs."""
    rng = np.random.default_rng(seed)
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
        if phi_sampler is None:
            phi = rng.uniform(0.0, 2.0 * np.pi, size)
        else:
            phi = np.interp(rng.random(size), phi_sampler[1], phi_sampler[0])
        xyz = np.stack(
            [r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t], axis=1
        )
        chunks.append(xyz.astype(np.float32))
        done += int(size)
        if progress is not None:
            progress(done / count)
    return np.concatenate(chunks)
```

Now replace the body of `sample_density` (from the `rng = ...` line through the `positions = np.concatenate(chunks)` line) with a single delegation, leaving its validation, provenance, and `SampleCloud` return unchanged:

```python
    phi_sampler = _phi_inverse_cdf(m) if (basis == "real" and m != 0) else None
    r_grid, r_cdf, r_max = _radial_inverse_cdf(n, l, Z, mu_ratio)
    x_grid, x_cdf = _costheta_inverse_cdf(l, m)
    positions = _draw_positions(
        count, r_grid, r_cdf, x_grid, x_cdf, phi_sampler, seed, n_chunks, progress
    )
```

(Keep the existing `phi_desc`, `provenance`, and `return SampleCloud(...)` block exactly as-is below this.)

- [ ] **Step 4: Add `sample_screened_density`**

Add near the top of `src/atomsim/sampling.py` (with the other imports):

```python
from atomsim.numerics.screening import screening_provenance
from atomsim.screened_atom import screened_radial
```

Add the new public function:

```python
def sample_screened_density(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    count: int,
    *,
    seed: int = 0,
    progress: Callable[[float], None] | None = None,
    n_chunks: int = 10,
    basis: str = "complex",
) -> SampleCloud:
    """Draw `count` positions from |psi_nlm|^2 for a screened GSZ/GJG atom.

    Radial source is the numerical screened R_nl; the angular part is the same
    central-field Y_lm as hydrogen. Fidelity is APPROXIMATION (model error).
    """
    validate_quantum_numbers(n, l)
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if basis not in ("complex", "real"):
        raise ValueError(f"basis must be 'complex' or 'real', got {basis!r}")

    r_field, _ = screened_radial(z, n_electrons, n, l, points=_R_GRID_POINTS)
    r_grid, r_cdf, r_max = _radial_inverse_cdf_tabulated(r_field.grid, r_field.values)
    x_grid, x_cdf = _costheta_inverse_cdf(l, m)
    phi_sampler = _phi_inverse_cdf(m) if (basis == "real" and m != 0) else None
    positions = _draw_positions(
        count, r_grid, r_cdf, x_grid, x_cdf, phi_sampler, seed, n_chunks, progress
    )

    base = screening_provenance(z, n_electrons)
    phi_desc = (
        "phi uniform (|Y_lm|^2 is phi-independent)"
        if phi_sampler is None
        else "phi from analytic real-basis marginal (cos^2/sin^2 m phi)"
    )
    provenance = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=(
            f"factorized inverse-CDF Monte-Carlo of |psi_nlm|^2 over a numerical "
            f"screened R_nl ({basis} basis): r from P(r)=r^2 R^2 (grid N={r_grid.size}, "
            f"r_max={r_max:g} bohr), cos(theta) from |Theta_lm|^2, {phi_desc}; "
            f"{base.method}"
        ),
        assumptions=base.assumptions
        + (
            f"angular basis: {basis}",
            f"RNG PCG64 seed={seed}, count={count}",
            "positions in bohr",
        ),
        error_estimate=r_field.provenance.error_estimate,
        refinement="increase CDF grid resolution, sample count, or radial solver resolution",
    )
    return SampleCloud(
        positions=positions, n=n, l=l, m=m, Z=z, mu_ratio=1.0,
        basis=basis, provenance=provenance,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_sampling.py -q`
Expected: PASS (new screened tests + all existing hydrogen sampling tests, which the refactor must not change).

- [ ] **Step 6: Lint + commit**

Run: `& "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check src/atomsim/sampling.py tests/test_sampling.py`
Expected: clean.

```bash
git add src/atomsim/sampling.py tests/test_sampling.py
git commit -m "Sample screened-atom orbital clouds from the numerical radial function"
```

---

## Task 2: Screened state evaluation (`evaluate_screened_state`)

**Files:**
- Modify: `src/atomsim/screened_atom.py`
- Test: `tests/test_screened_atom.py`

**Interfaces:**
- Consumes: `screened_radial(...)`; `spherical_harmonic(l, m, theta, phi, basis) -> AngularValues` (`.values`, `.provenance`); `WavefunctionValues` dataclass from `atomsim.analytic.wavefunction`; `screening_provenance(z, n_electrons)`.
- Produces: `evaluate_screened_state(z, n_electrons, n, l, m, positions, *, basis="complex") -> WavefunctionValues` (`.values` complex ndarray, `.provenance.fidelity == APPROXIMATION`). Also module constant `_SCREENED_EVAL_POINTS = 4096`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_screened_atom.py`:

```python
import numpy as np

from atomsim.provenance import Fidelity
from atomsim.screened_atom import evaluate_screened_state


def test_evaluate_screened_state_is_real_on_y0_plane():
    # Central-field orbital is real on y=0 (e^{i m phi} = +/-1 there).
    x = np.linspace(0.1, 20.0, 50)
    pos = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    psi = evaluate_screened_state(11, 11, 3, 1, 0, pos, basis="complex")
    assert psi.values.shape == (50,)
    assert np.max(np.abs(psi.values.imag)) < 1e-9
    assert psi.provenance.fidelity is Fidelity.APPROXIMATION


def test_evaluate_screened_state_factorizes_R_times_Y():
    from atomsim.analytic.angular import spherical_harmonic
    from atomsim.screened_atom import screened_radial

    rng = np.random.default_rng(0)
    pos = rng.normal(size=(200, 3)) * 3.0
    psi = evaluate_screened_state(11, 11, 3, 0, 0, pos, basis="complex")
    r = np.linalg.norm(pos, axis=1)
    theta = np.arccos(np.clip(pos[:, 2] / np.where(r > 0, r, 1.0), -1.0, 1.0))
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    r_field, _ = screened_radial(11, 11, 3, 0, points=4096)
    R = np.interp(r, r_field.grid, r_field.values, right=0.0)
    Y = spherical_harmonic(0, 0, theta, phi, basis="complex").values
    assert np.allclose(psi.values, R * Y, atol=1e-8)


def test_evaluate_screened_state_node_count_along_ray():
    # Na 3s has n-l-1 = 2 radial nodes.
    z = np.linspace(0.05, 40.0, 4000)
    pos = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    psi = evaluate_screened_state(11, 11, 3, 0, 0, pos, basis="complex").values.real
    nz = psi[np.abs(psi) > 1e-6]
    sign_changes = int(np.sum(np.diff(np.sign(nz)) != 0))
    assert sign_changes == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_screened_atom.py -k evaluate -q`
Expected: FAIL — `ImportError: cannot import name 'evaluate_screened_state'`.

- [ ] **Step 3: Implement `evaluate_screened_state`**

Add imports at the top of `src/atomsim/screened_atom.py`:

```python
from atomsim.analytic.angular import spherical_harmonic
from atomsim.analytic.wavefunction import WavefunctionValues
```

Add the module constant near the other module-level constants:

```python
_SCREENED_EVAL_POINTS = 4096
```

Add the function:

```python
def evaluate_screened_state(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    positions: np.ndarray,
    *,
    basis: str = "complex",
) -> WavefunctionValues:
    """psi_nlm = numerical screened R_nl(|r|) x hydrogenic Y_lm, at (N, 3) positions."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")

    r = np.linalg.norm(pos, axis=1)
    safe_r = np.where(r > 0.0, r, 1.0)
    theta = np.arccos(np.clip(pos[:, 2] / safe_r, -1.0, 1.0))
    theta = np.where(r > 0.0, theta, 0.0)
    phi = np.arctan2(pos[:, 1], pos[:, 0])

    r_field, _ = screened_radial(z, n_electrons, n, l, points=_SCREENED_EVAL_POINTS)
    R = np.interp(r, r_field.grid, r_field.values, left=r_field.values[0], right=0.0)
    angular = spherical_harmonic(l, m, theta, phi, basis=basis)
    values = R * angular.values

    base = screening_provenance(z, n_electrons)
    prov = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=(
            f"psi_nlm = numerical screened R_nl (u/r) x {angular.provenance.method}; "
            f"{base.method}"
        ),
        assumptions=base.assumptions
        + angular.provenance.assumptions
        + ("values in bohr^-3/2 at Cartesian positions in bohr",),
        error_estimate=r_field.provenance.error_estimate,
    )
    return WavefunctionValues(
        values=values, positions=pos, n=n, l=l, m=m, Z=z, mu_ratio=1.0,
        basis=basis, provenance=prov,
    )
```

Note: `screening_provenance` is already imported in `screened_atom.py` (used by `screened_radial`). If not, add `from atomsim.numerics.screening import screening_provenance`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_screened_atom.py -q`
Expected: PASS (new evaluate tests + existing screened-atom tests).

- [ ] **Step 5: Lint + commit**

Run: `& "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check src/atomsim/screened_atom.py tests/test_screened_atom.py`
Expected: clean.

```bash
git add src/atomsim/screened_atom.py tests/test_screened_atom.py
git commit -m "Evaluate screened orbitals at arbitrary positions for cloud/plane"
```

---

## Task 3: Screened plane cross-section (`screened_plane_grid`)

**Files:**
- Modify: `src/atomsim/plane.py`
- Test: `tests/test_plane.py`

**Interfaces:**
- Consumes: `evaluate_screened_state(...)` (Task 2), `evaluate_state(...)`, `default_half_extent(n, Z, mu_ratio)`, `validate_quantum_numbers`, `validate_angular`, `_ROW_CHUNKS`.
- Produces: `screened_plane_grid(z, n_electrons, n, l, m, quantity="density", basis="complex", resolution=512, half_extent=None, progress=None) -> PlaneGrid` (`provenance.fidelity == APPROXIMATION`). Also the private helper `_plane_values(evaluator, quantity, resolution, half_extent, progress) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]`. Hydrogen `plane_grid` output is unchanged (still `EXACT`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plane.py`:

```python
import numpy as np

from atomsim.plane import plane_grid, screened_plane_grid
from atomsim.provenance import Fidelity


def test_screened_plane_is_approximation_and_real_signed():
    pg = screened_plane_grid(11, 11, 3, 1, 0, quantity="psi", resolution=64)
    assert pg.provenance.fidelity is Fidelity.APPROXIMATION
    assert pg.values.shape == (64, 64)
    assert np.isrealobj(pg.values)
    assert np.isfinite(pg.values).all()
    assert pg.Z == 11


def test_screened_plane_node_count_along_axis():
    # Na 3s density: 2 radial nodes -> the signed psi along +z axis changes sign twice.
    pg = screened_plane_grid(11, 11, 3, 0, 0, quantity="psi", resolution=401)
    mid = pg.values.shape[1] // 2           # x = 0 column
    col = pg.values[pg.values.shape[0] // 2 :, mid]  # z >= 0 half-ray
    nz = col[np.abs(col) > np.max(np.abs(col)) * 1e-3]
    sign_changes = int(np.sum(np.diff(np.sign(nz)) != 0))
    assert sign_changes == 2


def test_hydrogen_plane_unchanged_exact():
    pg = plane_grid(2, 1, 0, quantity="psi", resolution=32)
    assert pg.provenance.fidelity is Fidelity.EXACT
    assert pg.values.shape == (32, 32)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_plane.py -k screened -q`
Expected: FAIL — `ImportError: cannot import name 'screened_plane_grid'`.

- [ ] **Step 3: Extract `_plane_values` and route `plane_grid` through it**

In `src/atomsim/plane.py`, add the shared grid-evaluation helper (this is the current loop body of `plane_grid`, lines ~69–88, with the evaluator abstracted out):

```python
def _plane_values(evaluator, quantity, resolution, half_extent, progress):
    """Evaluate `quantity` on a (resolution x resolution) y=0 grid via `evaluator`.

    `evaluator(pos)` takes (N, 3) Cartesian positions (bohr) and returns a
    WavefunctionValues; returns (values 2-D, axis 1-D, psi assumptions).
    """
    axis = np.linspace(-half_extent, half_extent, resolution)
    values = np.zeros((resolution, resolution))
    psi_assumptions: tuple[str, ...] = ()
    starts = np.linspace(0, resolution, _ROW_CHUNKS + 1).astype(int)
    for k in range(_ROW_CHUNKS):
        i0, i1 = int(starts[k]), int(starts[k + 1])
        if i1 == i0:
            continue
        zz, xx = np.meshgrid(axis[i0:i1], axis, indexing="ij")
        pos = np.stack([xx.ravel(), np.zeros(xx.size), zz.ravel()], axis=1)
        psi = evaluator(pos)
        psi_assumptions = psi.provenance.assumptions
        block = psi.values.reshape(i1 - i0, resolution)
        if quantity == "density":
            values[i0:i1] = np.abs(block) ** 2
        else:
            values[i0:i1] = np.real(block)
        if progress is not None:
            progress(i1 / resolution)
    return values, axis, psi_assumptions
```

Replace the loop in `plane_grid` (the `axis = ...` through the `progress(i1 / resolution)` block) with a nested evaluator + delegation, leaving its validation above and its provenance/`PlaneGrid` return below unchanged:

```python
    def evaluator(pos):
        return evaluate_state(n, l, m, pos, Z=Z, mu_ratio=mu_ratio, basis=basis)

    values, axis, psi_assumptions = _plane_values(
        evaluator, quantity, resolution, he, progress
    )
```

- [ ] **Step 4: Add `screened_plane_grid`**

Add the import at the top of `src/atomsim/plane.py`:

```python
from atomsim.screened_atom import evaluate_screened_state
```

Add the function:

```python
def screened_plane_grid(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    quantity: str = "density",
    basis: str = "complex",
    resolution: int = 512,
    half_extent: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> PlaneGrid:
    """|psi|^2 or signed psi for a screened GSZ/GJG atom on the y=0 plane."""
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    if quantity not in ("density", "psi"):
        raise ValueError(f"quantity must be 'density' or 'psi', got {quantity!r}")
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    z_net = max(z - n_electrons + 1, 1)  # asymptotic core charge sets display extent
    he = default_half_extent(n, z_net, 1.0) if half_extent is None else float(half_extent)
    if he <= 0.0:
        raise ValueError(f"half_extent must be positive, got {he}")

    def evaluator(pos):
        return evaluate_screened_state(z, n_electrons, n, l, m, pos, basis=basis)

    values, axis, psi_assumptions = _plane_values(
        evaluator, quantity, resolution, he, progress
    )

    if quantity == "density":
        unit = "bohr^-3"
        label = f"|psi_{n},{l},{m}|^2 on the y=0 plane"
        qdesc = "|psi|^2 (probability density)"
        extra = ("plane y=0 contains the z quantization axis",)
    else:
        unit = "bohr^-3/2"
        label = f"psi_{n},{l},{m} on the y=0 plane"
        qdesc = "signed psi"
        extra = (
            "plane y=0 contains the z quantization axis",
            "psi is real on y=0 (e^{i m phi} = +/-1 there), so a signed plot is honest",
        )
    provenance = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=(
            f"{qdesc} from a numerical screened psi_nlm on a {resolution}x{resolution} "
            f"y=0 grid, half-extent {he:g} bohr"
        ),
        assumptions=psi_assumptions + extra,
        refinement="increase resolution, extent, or radial solver resolution",
    )
    return PlaneGrid(
        values=values, axis=axis, quantity=quantity, unit=unit, label=label,
        n=n, l=l, m=m, Z=z, mu_ratio=1.0, basis=basis, provenance=provenance,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_plane.py -q`
Expected: PASS (screened tests + all existing hydrogen plane tests unchanged).

- [ ] **Step 6: Lint + commit**

Run: `& "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check src/atomsim/plane.py tests/test_plane.py`
Expected: clean.

```bash
git add src/atomsim/plane.py tests/test_plane.py
git commit -m "Render screened-atom orbitals on the y=0 plane"
```

---

## Task 4: Server wiring — sample/plane jobs accept screened atoms

**Files:**
- Modify: `src/atomsim/server/app.py:640-689` (the two job producers)
- Test: `tests/test_server.py:526-541` (rewrite the two rejection tests)

**Interfaces:**
- Consumes: `sample_screened_density(...)`, `evaluate_screened_state(...)`, `screened_plane_grid(...)`, existing `_is_screened(key)`, `_validate_state(...)`, `atom_for_key(key)`, `SampleJobResult`.
- Produces: `POST /api/jobs/sample` and `POST /api/jobs/plane` return `200` + a running job for screened systems; downstream `meta` reports `fidelity == "approximation"`.

- [ ] **Step 1: Rewrite the failing tests**

Replace `test_cloud_job_rejects_screened_atom` and `test_plane_job_rejects_screened_atom` in `tests/test_server.py` with end-to-end acceptance tests (reuse the existing `_wait_done` helper):

```python
def test_cloud_job_screened_atom_end_to_end(client):
    r = client.post(
        "/api/jobs/sample",
        json={"n": 3, "l": 0, "m": 0, "system": "na", "count": 2000, "seed": 4},
    )
    assert r.status_code == 200
    job_id = r.json()["id"]
    final = _wait_done(client, job_id)
    assert final["status"] == "done"
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["count"] == 2000
    assert meta["provenance"]["fidelity"] == "approximation"
    raw = client.get(f"/api/jobs/{job_id}/data")
    positions = np.frombuffer(raw.content, dtype=np.float32).reshape(-1, 3)
    assert positions.shape == (2000, 3)
    assert np.isfinite(positions).all()


def test_plane_job_screened_atom_end_to_end(client):
    r = client.post(
        "/api/jobs/plane",
        json={"n": 3, "l": 0, "m": 0, "system": "na", "quantity": "psi", "resolution": 64},
    )
    assert r.status_code == 200
    job_id = r.json()["id"]
    final = _wait_done(client, job_id)
    assert final["status"] == "done"
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["quantity"] == "psi"
    assert meta["provenance"]["fidelity"] == "approximation"
    raw = client.get(f"/api/jobs/{job_id}/data")
    grid = np.frombuffer(raw.content, dtype=np.float32)
    assert grid.size == 64 * 64
    assert np.isfinite(grid).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_server.py -k "screened_atom_end_to_end" -q`
Expected: FAIL — POST returns `422` ("arrive in a later phase"), so the `200` assertion fails.

- [ ] **Step 3: Add imports and swap the sample-job branch**

In `src/atomsim/server/app.py`, extend the existing engine imports:

```python
from atomsim.sampling import SampleCloud, sample_density, sample_screened_density
from atomsim.screened_atom import (
    evaluate_screened_state,
    screened_radial,
    solve_screened_atom,
)
from atomsim.plane import PlaneGrid, plane_grid, screened_plane_grid
```

(Merge into the existing `from atomsim.sampling import ...`, `from atomsim.screened_atom import ...`, and `from atomsim.plane import ...` lines rather than duplicating them.)

Replace the `create_sample_job` refusal (currently lines 641-645) so screened systems build a screened cloud:

```python
    async def create_sample_job(req: SampleRequest) -> JobModel:
        _validate_state(req.n, req.l, req.m)
        job = jobs.create()
        app.state.job_systems[job.id] = req.system

        if _is_screened(req.system):
            atom = atom_for_key(req.system)

            def work(progress):
                cloud = sample_screened_density(
                    atom.z, atom.z, req.n, req.l, req.m, req.count,
                    seed=req.seed, progress=lambda f: progress(0.9 * f), basis=req.basis,
                )
                psi = evaluate_screened_state(
                    atom.z, atom.z, req.n, req.l, req.m,
                    cloud.positions.astype(np.float64), basis=req.basis,
                )
                progress(1.0)
                return SampleJobResult(cloud=cloud, psi=psi)

            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, jobs.run, job.id, work)
            return _job_model(job)

        sys_ = _resolve_system(req.system)

        def work(progress):
            cloud = sample_density(
                req.n, req.l, req.m, req.count,
                Z=sys_.Z, mu_ratio=sys_.mu_ratio.value,
                seed=req.seed, progress=lambda f: progress(0.9 * f), basis=req.basis,
            )
            psi = evaluate_state(
                req.n, req.l, req.m, cloud.positions.astype(np.float64),
                Z=sys_.Z, mu_ratio=sys_.mu_ratio.value, basis=req.basis,
            )
            progress(1.0)
            return SampleJobResult(cloud=cloud, psi=psi)

        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, jobs.run, job.id, work)
        return _job_model(job)
```

(`atom_for_key` is already imported/used by the levels/radial/spectrum branches; confirm the name matches what those branches call and reuse it.)

- [ ] **Step 4: Swap the plane-job branch**

Replace the `create_plane_job` refusal (currently lines 670-673) so screened systems build a screened plane:

```python
    async def create_plane_job(req: PlaneRequest) -> JobModel:
        _validate_state(req.n, req.l, req.m)
        job = jobs.create()
        app.state.job_systems[job.id] = req.system

        if _is_screened(req.system):
            atom = atom_for_key(req.system)

            def work(progress):
                return screened_plane_grid(
                    atom.z, atom.z, req.n, req.l, req.m,
                    quantity=req.quantity, basis=req.basis,
                    resolution=req.resolution, progress=progress,
                )

            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, jobs.run, job.id, work)
            return _job_model(job)

        sys_ = _resolve_system(req.system)

        def work(progress):
            return plane_grid(
                req.n, req.l, req.m, quantity=req.quantity, basis=req.basis,
                Z=sys_.Z, mu_ratio=sys_.mu_ratio.value,
                resolution=req.resolution, progress=progress,
            )

        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, jobs.run, job.id, work)
        return _job_model(job)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_server.py -q`
Expected: PASS, including the two new end-to-end screened tests. (No other server test asserted the old 422 for screened clouds/planes — verify none remain by searching for `rejects_screened`.)

- [ ] **Step 6: Lint + commit**

Run: `& "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check src/atomsim/server/app.py tests/test_server.py`
Expected: clean.

```bash
git add src/atomsim/server/app.py tests/test_server.py
git commit -m "Serve screened-atom clouds and planes from the sample/plane jobs"
```

---

## Task 5: Frontend — remove Cloud/Plane placeholders for screened atoms

**Files:**
- Modify: `web/src/components/CloudView.tsx:90-102` (remove screened early-return)
- Modify: `web/src/components/PlaneView.tsx:13-32` (remove screened early-return + gate)
- Modify: `web/src/lib/liberties.ts` (remove `SCREENED_ORBITAL_PLACEHOLDER` only if now unused)
- Test: `web/src/components/PlaneView.test.tsx` (new, minimal render assertion)

**Interfaces:**
- Consumes: existing store `(n, l, m, system, systems, basis, plane, planeStatus, loadPlane, ...)`.
- Produces: screened systems render the normal Cloud/Plane path; the `Badge` shows the job's `APPROXIMATION` provenance automatically.

- [ ] **Step 1: Remove the CloudView placeholder**

In `web/src/components/CloudView.tsx`, delete the entire `if (isScreened) { return ( ... ); }` block (lines ~90-102). Then delete the now-unused `isScreened` line (~66) **only if** `isScreened` is referenced nowhere else in the file — otherwise leave it. Remove the `SCREENED_ORBITAL_PLACEHOLDER` import if it is no longer referenced in this file.

- [ ] **Step 2: Remove the PlaneView placeholder and its fetch gate**

In `web/src/components/PlaneView.tsx`:
- Delete the `if (isScreened) { return ( ... ); }` block (lines ~19-32). This is required for correctness: it currently sits before a second `useEffect`, so removing it also fixes a conditional-hook order issue.
- Simplify the fetch effect to drop the `isScreened` guard:

```tsx
  useEffect(() => {
    if (!plane && planeStatus === "idle") void loadPlane();
  }, [plane, planeStatus, loadPlane, n, l, m, system, basis, planeQuantity]);
```

- Delete the now-unused `isScreened` line (~13), the `systems` destructure entry if unused elsewhere, and the `SCREENED_ORBITAL_PLACEHOLDER` import.

- [ ] **Step 3: Drop the now-dead liberty if unused**

Run: `cd web; npx grep -rn "SCREENED_ORBITAL_PLACEHOLDER" src` (or use ripgrep). If the only remaining hit is the definition in `src/lib/liberties.ts`, delete that export. If any component still imports it, leave it.

- [ ] **Step 4: Write a minimal render test**

Create `web/src/components/PlaneView.test.tsx` asserting a screened system no longer shows the placeholder copy:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PlaneView } from "./PlaneView";
import { useAppStore } from "../state/store";

describe("PlaneView screened", () => {
  it("does not show the placeholder for a screened atom", () => {
    useAppStore.setState({
      n: 3, l: 0, m: 0, system: "na", basis: "complex",
      systems: [{ key: "na", kind: "screened" } as never],
      plane: null, planeStatus: "idle",
    });
    render(<PlaneView />);
    expect(screen.queryByText(/arrives in a later phase/i)).toBeNull();
  });
});
```

(If the repo's existing component tests use a different render/setup helper, mirror that pattern — check a sibling `*.test.tsx` first. Keep the assertion: the placeholder text is gone.)

- [ ] **Step 5: Run the frontend suite + typecheck/build**

Run: `cd web; npm test`
Expected: PASS (new test + existing suite).

Run: `cd web; npm run build`
Expected: `tsc --noEmit` clean (catches any unused imports/vars left behind) + vite build succeeds.

- [ ] **Step 6: Commit**

```bash
git add web/src/components/CloudView.tsx web/src/components/PlaneView.tsx web/src/components/PlaneView.test.tsx web/src/lib/liberties.ts
git commit -m "Render real Cloud and Plane views for screened atoms"
```

---

## Task 6: Full verification + live smoke test

**Files:** none (verification only).

- [ ] **Step 1: Backend suite + lint**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest -q; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check .`
Expected: all green; ruff clean.

- [ ] **Step 2: Frontend suite + build**

Run: `cd web; npm test; npm run build`
Expected: all green; build clean; `web/dist` regenerated.

- [ ] **Step 3: Live smoke test** — start the server and exercise screened Cloud + Plane.

Run (background): `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\Scripts\atomsim.exe" serve --port 8013 --no-browser`

Then confirm the two jobs run to completion with APPROXIMATION provenance:

```bash
# Screened cloud (Na 3s): POST, poll, then meta must report approximation.
curl -s -X POST "http://127.0.0.1:8013/api/jobs/sample" -H "Content-Type: application/json" \
  -d '{"n":3,"l":0,"m":0,"system":"na","count":2000,"seed":4}'   # expect 200 + job id
# (poll /api/jobs/{id} until status=done, then:)
# curl -s "http://127.0.0.1:8013/api/jobs/{id}/meta" -> provenance.fidelity == "approximation"

# Screened plane (Na 3s, signed psi):
curl -s -X POST "http://127.0.0.1:8013/api/jobs/plane" -H "Content-Type: application/json" \
  -d '{"n":3,"l":0,"m":0,"system":"na","quantity":"psi","resolution":128}' # expect 200 + job id

# Hydrogen still works (regression): expect 200
curl -s -o /dev/null -w "%{http_code}\n" -X POST "http://127.0.0.1:8013/api/jobs/sample" \
  -H "Content-Type: application/json" -d '{"n":2,"l":1,"m":0,"count":1000}'
```

Expected: both screened POSTs return `200`; their completed `meta` reports `fidelity == "approximation"`; hydrogen sample still `200`. Stop the server (`TaskStop`) when done.

- [ ] **Step 4: Final commit (docs/tidy if any)**

```bash
git add -A
git commit -m "Phase 7: verify screened-atom cloud and plane end-to-end" --allow-empty
```

---

## Self-Review

**Spec coverage:**
- §1 both Cloud + Plane, no protocol/store/URL change → Tasks 1–5. ✅
- §2.1 sample `|psi|²` from numerical `R_nl` × hydrogenic `Y_lm` → Tasks 1, 2. ✅
- §2.2 both `APPROXIMATION`; tier asserted against stray `EXACT`/`NUMERICAL` → Task 1 Step 1, Task 3 `test_hydrogen_plane_unchanged_exact` + `test_screened_plane_is_approximation`. ✅
- §2.3 config independence (cloud/plane take `(n,l,m,system)`, ignore config) → Task 4 branches pass no config. ✅
- §3.1 factor out radial source + `sample_screened_density` → Task 1. ✅
- §3.2 `evaluate_screened_state` returns `WavefunctionValues` → Task 2. ✅
- §3.3 generalize `plane_grid` + `screened_plane_grid` → Task 3. ✅
- §4 swap two 422 branches, no schema change → Task 4. ✅
- §5 remove placeholders, Badge auto-shows provenance, no store/URL change → Task 5. ✅
- §6 validation: radial KS (T1), angular KS (T1), normalization (T1), plane real/nodes/provenance (T3), server 200+provenance (T4), frontend render (T5). ✅
- §7 file plan matches Tasks 1–5. ✅

**Placeholder scan:** no "TBD"/"add error handling"/"similar to Task N" — every code step carries full code; test bodies are concrete with real atoms (Na Z=11, N=11).

**Type consistency:** `sample_screened_density(z, n_electrons, n, l, m, count, *, seed, progress, n_chunks, basis)`, `evaluate_screened_state(z, n_electrons, n, l, m, positions, *, basis)`, `screened_plane_grid(z, n_electrons, n, l, m, quantity, basis, resolution, half_extent, progress)`, and `_plane_values(evaluator, quantity, resolution, half_extent, progress)` are used identically across Tasks 1–4. `evaluate_screened_state` returns `WavefunctionValues` (fields `values, positions, n, l, m, Z, mu_ratio, basis, provenance`) exactly as the server's `res.psi` consumer and `_sample_meta` require. `_radial_inverse_cdf_tabulated` / `_draw_positions` signatures match between `sample_density` and `sample_screened_density`.

**One deliberate scope note:** the screened radial tabulation for sampling uses `points=_R_GRID_POINTS` (8192) and for evaluation `_SCREENED_EVAL_POINTS` (4096); the radial self-consistency KS test (Task 1) compares against the *same* tabulation, so it validates the sampler faithfully, while the tabulation's physical accuracy stays Phase 6's already-tested concern.
