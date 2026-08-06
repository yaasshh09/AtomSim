# Phase 1 M3: UI Depth Implementation Plan

**Goal:** Turn the walking-skeleton frontend into the full "Hydrogen, Honestly" app, cloud color modes (density colormap, phase-as-hue), 2D inferno cross-sections, radial plots, energy-level diagram with fine-structure zoom, computed-vs-NIST spectrum overlay, thumbnail gallery, system/basis controls, layered KaTeX math, plus the four server additions the UI needs (sample ψ channels, plane-grid jobs, /api/levels, thumbnail PNGs).

**Architecture:** Spec `docs/specs/2026-07-05-phase1-hydrogen-honestly-design.md` §6, §7. Backend: new `atomsim/plane.py` (y=0 cross-section grids), sample jobs gain per-point ψ channels (density/phase) via `evaluate_state`, `/api/levels`, `/api/jobs/plane` (same job pattern), `/api/thumbnail/{n}/{l}/{m}` (matplotlib Agg + lru_cache). Frontend: zustand store grows system/basis/view/colorMode/fine-structure state; five views behind a view switch; custom SVG plots on `d3-scale` (no charting lib); colormaps come from ONE authority (matplotlib LUTs generated into `web/src/lib/luts.ts` so client renders match server thumbnails); KaTeX for the "show the physics" layer. Every boundary-crossing value stays a `Quantity`, `Field`, or provenance-carrying container; frontend rendering choices are disclosed via a `RENDER_LIBERTIES` visual-liberty provenance, never hidden.

**Tech Stack:** Python 3.12 (conda env `atomsim`), FastAPI, matplotlib (new dep, Agg only), pytest · TypeScript strict, React 19, react-three-fiber, zustand, d3-scale (new), katex (new), vitest.

## Global Constraints

- Native Windows only; no WSL2/Docker. Engine units: Hartree atomic units (`hartree`, `bohr`); display conversions happen server-side at the boundary (`_to_ev`, `_to_pm`), never as frontend physics.
- **Boundary rule (spec §4):** every physical value crossing a module boundary is a `Quantity`, a `Field`, or a container dataclass carrying its own `Provenance`. Fidelity tiers exactly: EXACT, NUMERICAL, APPROXIMATION, COUNTERFACTUAL, VISUAL_LIBERTY.
- The model never quietly lies: what is plotted is labeled exactly (|ψ|² vs ψ, the honesty fix over the classic poster); gamma-brightening, point glow, axis rotation are disclosed VISUAL LIBERTIES; thumbnails are labeled navigation aids.
- No physics computed in the frontend: energies, radii, |L|, node counts, densities, phases all arrive from the engine with provenance. Frontend-only facts (FPS, render choices) are the frontend's own honest domain.
- TDD for all Python/server code and all frontend *logic* (lib modules, store transitions). No pixel/screenshot tests (spec §8).
- CI green on `windows-latest` for every push; conventional commits; MIT license.
- Conda at `C:\ProgramData\miniforge3` (NOT on PATH). **Always set `$env:PYTHONUTF8='1'` first** (conda run crashes printing Unicode test output otherwise). From repo root:
  - Tests: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q`
  - Lint: `& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .`
  - Web (from `web\`): `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test` (and `npm run build` for tsc + vite)
- Ruff: line length 100, `E741` ignored, imports sorted (`I`). TypeScript `strict: true`, no `any`.
- `web/node_modules` is a junction to a local non-synced dir (OneDrive gotcha), `npm install <pkg>` works through it; never delete/recreate the junction.

## File Structure

```
src/atomsim/
├── constants.py                    # MODIFY: add BOHR_RADIUS_PM
├── plane.py                        # CREATE: PlaneGrid + plane_grid (y=0 cross-sections)
├── analytic/hydrogen.py            # MODIFY: angular_momentum_magnitude
└── server/
    ├── schemas.py                  # MODIFY: ChannelModel, PlaneMetaModel pieces live in app.py? No, 
    │                               #   ChannelModel goes here (canonical JSON forms live here)
    ├── thumbnails.py               # CREATE: matplotlib Agg PNG rendering + lru_cache
    └── app.py                      # MODIFY: sample channels, /api/levels, /api/jobs/plane,
                                    #   /api/thumbnail, state readouts, meta union
tests/
├── test_plane.py                   # CREATE
├── test_thumbnails.py              # CREATE
└── test_server.py                  # MODIFY: channels, levels, plane jobs, state readouts
scripts/gen_luts.py                 # CREATE: matplotlib → web/src/lib/luts.ts
pyproject.toml                      # MODIFY: matplotlib dependency
web/src/
├── api/types.ts                    # MODIFY: mirror every new JSON shape
├── api/client.ts                   # MODIFY: new endpoints, channels, thumbnailUrl
├── api/decode.test.ts              # MODIFY: decodeFloats tests
├── lib/luts.ts                     # GENERATED: INFERNO + RDBU_R (256-entry LUTs)
├── lib/colormap.ts (+.test.ts)     # CREATE: lutColor, densityT, signedT, phaseColor, hslToRgb
├── lib/cloudColors.ts (+.test.ts)  # CREATE: per-vertex color buffers for the cloud
├── lib/rasterize.ts (+.test.ts)    # CREATE: plane grid → RGBA pixels
├── lib/plot.ts (+.test.ts)         # CREATE: SVG path helper on d3-scale
├── lib/gallery.ts (+.test.ts)      # CREATE: galleryStates
├── lib/spectrum.ts (+.test.ts)     # CREATE: seriesName/seriesColor
├── lib/levels.ts (+.test.ts)       # CREATE: arrowsFor
├── lib/liberties.ts                # CREATE: RENDER_LIBERTIES / THUMBNAIL_LIBERTY provenances
├── lib/quantum.ts (+.test.ts)      # MODIFY: realOrbitalLabel (mirror of angular.py)
├── state/store.ts (+store.test.ts) # MODIFY: full app state (system/basis/view/colorMode/…)
├── physics/content.ts (+.test.ts)  # CREATE: KaTeX blocks per view
├── components/
│   ├── CloudView.tsx               # CREATE: canvas extracted from App + color modes + FPS
│   ├── PlaneView.tsx               # CREATE
│   ├── RadialView.tsx              # CREATE
│   ├── LevelsView.tsx              # CREATE
│   ├── SpectrumView.tsx            # CREATE
│   ├── GalleryStrip.tsx            # CREATE
│   ├── ShowPhysics.tsx             # CREATE
│   ├── Legend.tsx                  # CREATE: density gradient / phase hue-wheel legends
│   ├── Controls.tsx                # MODIFY: system/basis/view/colorMode/fine-structure
│   ├── InfoPanel.tsx               # MODIFY: dynamic system, |L|, nodes, pm, FS levels, FPS
│   └── PointCloud.tsx              # MODIFY: optional per-vertex colors
├── App.tsx                         # MODIFY: view switch + gallery row
├── main.tsx                        # MODIFY: katex css import
└── index.css                       # MODIFY: new layout/styles
web/package.json                    # MODIFY: d3-scale, katex (+types)
README.md                           # MODIFY: M3 status
```

Execution order is backend (1-5) then frontend (6-15) then ship (16); every task ends green (`pytest` and/or `npm test`) and committed.

---

### Task 1: Engine: plane-density grids (`atomsim/plane.py`)

**Files:**
- Create: `src/atomsim/plane.py`
- Test: `tests/test_plane.py`

**Interfaces:**
- Consumes: `evaluate_state` from `atomsim.analytic.wavefunction` (returns `WavefunctionValues` with `.values`, `.provenance`); `validate_quantum_numbers` (hydrogen), `validate_angular` (angular); `Provenance`, `Fidelity`.
- Produces (Tasks 4, 5 rely on these exact names):
  - `PlaneGrid` frozen dataclass: `values: np.ndarray` (resolution×resolution float64; `[i, j]` = point `(x=axis[j], y=0, z=axis[i])`), `axis: np.ndarray` (shared x/z axis, bohr), `quantity: str` ("density"|"psi"), `unit: str`, `label: str`, `n/l/m: int`, `Z: int`, `mu_ratio: float`, `basis: str`, `provenance: Provenance`.
  - `plane_grid(n, l, m, quantity="density", basis="complex", Z=1, mu_ratio=1.0, resolution=512, half_extent=None, progress=None) -> PlaneGrid`.
  - `default_half_extent(n, Z=1, mu_ratio=1.0) -> float` = `2.5 * n² / (Z·μ)`.

**Physics note (for the honest label):** on the y=0 plane, φ = 0 (x ≥ 0) or π (x < 0), so e^{imφ} = ±1 and ψ_nlm is real-valued in BOTH angular bases. A signed-ψ plot is therefore honest on this plane, and `quantity="psi"` takes `Re(ψ)` with an assumption stating exactly that.

- [x] **Step 1: Write the failing tests**, `tests/test_plane.py`:

```python
import numpy as np
import pytest

from atomsim.analytic.wavefunction import evaluate_state
from atomsim.plane import default_half_extent, plane_grid
from atomsim.provenance import Fidelity

def test_density_matches_direct_evaluation():
    pg = plane_grid(2, 1, 0, quantity="density", resolution=33)
    i, j = 5, 20
    pos = np.array([[pg.axis[j], 0.0, pg.axis[i]]])
    psi = evaluate_state(2, 1, 0, pos).values[0]
    assert pg.values[i, j] == pytest.approx(abs(psi) ** 2, rel=1e-12)

def test_psi_sign_structure_2pz():
    # 2p_z: psi > 0 for z > 0, psi < 0 for z < 0 (rows are z, ascending)
    pg = plane_grid(2, 1, 0, quantity="psi", resolution=33)
    mid = 16  # axis[16] == 0.0 for a 33-point symmetric grid
    assert pg.values[mid + 5, mid] > 0.0
    assert pg.values[mid - 5, mid] < 0.0

def test_density_nonnegative_shape_dtype():
    pg = plane_grid(3, 2, 1, quantity="density", basis="real", resolution=17)
    assert pg.values.shape == (17, 17)
    assert pg.values.dtype == np.float64
    assert (pg.values >= 0.0).all()
    assert pg.basis == "real"

def test_default_extent_and_axis():
    assert default_half_extent(2) == pytest.approx(10.0)
    pg = plane_grid(2, 0, 0, resolution=9)
    assert pg.axis[0] == pytest.approx(-10.0)
    assert pg.axis[-1] == pytest.approx(10.0)
    # mu-scaling shrinks the frame like the orbital itself
    assert default_half_extent(1, Z=1, mu_ratio=100.0) == pytest.approx(0.025)

def test_validation_errors():
    with pytest.raises(ValueError):
        plane_grid(1, 0, 0, quantity="colour")
    with pytest.raises(ValueError):
        plane_grid(1, 0, 0, resolution=1)
    with pytest.raises(ValueError):
        plane_grid(1, 0, 0, half_extent=-1.0)
    with pytest.raises(ValueError):
        plane_grid(1, 1, 0)  # l >= n

def test_provenance_and_units_are_honest():
    dens = plane_grid(1, 0, 0, resolution=8)
    assert dens.provenance.fidelity is Fidelity.EXACT
    assert "y=0" in dens.provenance.method
    assert dens.unit == "bohr^-3"
    psi = plane_grid(1, 0, 0, quantity="psi", resolution=8)
    assert psi.unit == "bohr^-3/2"
    assert any("real" in a for a in psi.provenance.assumptions)

def test_progress_monotone_to_one():
    seen: list[float] = []
    plane_grid(1, 0, 0, resolution=8, progress=seen.append)
    assert seen[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(seen, seen[1:]))
```

- [x] **Step 2: Run to verify failure**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_plane.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.plane'`

- [x] **Step 3: Implement**, `src/atomsim/plane.py`:

```python
"""2-D cross-sections of psi_nlm on the y=0 plane (contains the z quantization axis).

This is where the classic "hydrogen poster" pictures live. On y=0 the azimuth
is phi = 0 (x >= 0) or pi (x < 0), so e^{i m phi} = +/-1 and psi is real-valued
in BOTH angular bases: a signed-psi plot is honest here, and the plot label
must state exactly which quantity is shown (spec 7.2 honesty fix over the
poster's contradictory -/+ "probability density" colorbar).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from atomsim.analytic.angular import validate_angular
from atomsim.analytic.hydrogen import validate_quantum_numbers
from atomsim.analytic.wavefunction import evaluate_state
from atomsim.provenance import Fidelity, Provenance

_ROW_CHUNKS = 16

@dataclass(frozen=True)
class PlaneGrid:
    """psi-derived values on a square y=0 grid. Container carries provenance."""

    values: np.ndarray  # (resolution, resolution) float64; [i, j] = (x=axis[j], 0, z=axis[i])
    axis: np.ndarray    # (resolution,) shared x/z axis, bohr
    quantity: str       # "density" (|psi|^2) or "psi" (signed psi, real on y=0)
    unit: str           # bohr^-3 for density, bohr^-3/2 for psi
    label: str
    n: int
    l: int
    m: int
    Z: int
    mu_ratio: float
    basis: str
    provenance: Provenance

def default_half_extent(n: int, Z: int = 1, mu_ratio: float = 1.0) -> float:
    """Display framing: ~2.5 n^2 a0/(Z mu) keeps outer lobes visible at <1% peak density."""
    return 2.5 * n * n / (Z * mu_ratio)

def plane_grid(
    n: int,
    l: int,
    m: int,
    quantity: str = "density",
    basis: str = "complex",
    Z: int = 1,
    mu_ratio: float = 1.0,
    resolution: int = 512,
    half_extent: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> PlaneGrid:
    """Evaluate |psi|^2 or signed psi on a (resolution x resolution) y=0 grid."""
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    if quantity not in ("density", "psi"):
        raise ValueError(f"quantity must be 'density' or 'psi', got {quantity!r}")
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    he = default_half_extent(n, Z, mu_ratio) if half_extent is None else float(half_extent)
    if he <= 0.0:
        raise ValueError(f"half_extent must be positive, got {he}")

    axis = np.linspace(-he, he, resolution)
    values = np.zeros((resolution, resolution))
    psi_assumptions: tuple[str, ...] = ()
    starts = np.linspace(0, resolution, _ROW_CHUNKS + 1).astype(int)
    for k in range(_ROW_CHUNKS):
        i0, i1 = int(starts[k]), int(starts[k + 1])
        if i1 == i0:
            continue
        zz, xx = np.meshgrid(axis[i0:i1], axis, indexing="ij")
        pos = np.stack([xx.ravel(), np.zeros(xx.size), zz.ravel()], axis=1)
        psi = evaluate_state(n, l, m, pos, Z=Z, mu_ratio=mu_ratio, basis=basis)
        psi_assumptions = psi.provenance.assumptions
        block = psi.values.reshape(i1 - i0, resolution)
        if quantity == "density":
            values[i0:i1] = np.abs(block) ** 2
        else:
            values[i0:i1] = np.real(block)
        if progress is not None:
            progress(i1 / resolution)

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
        fidelity=Fidelity.EXACT,
        method=(
            f"{qdesc} from closed-form psi_nlm on a {resolution}x{resolution} "
            f"y=0 grid, half-extent {he:g} bohr"
        ),
        assumptions=psi_assumptions + extra,
        refinement="increase resolution or adjust extent",
    )
    return PlaneGrid(
        values=values, axis=axis, quantity=quantity, unit=unit, label=label,
        n=n, l=l, m=m, Z=Z, mu_ratio=mu_ratio, basis=basis, provenance=provenance,
    )
```

Note: `Provenance.assumptions` is a tuple in this codebase, if `evaluate_state` provenance ever returns a list, coerce with `tuple(...)`. Check `provenance.py` if the `+` concatenation fails.

- [x] **Step 4: Run to verify pass**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_plane.py -q`
Expected: 7 passed. Also run `ruff check .`

- [x] **Step 5: Commit**

```bash
git add src/atomsim/plane.py tests/test_plane.py
git commit -m "feat: y=0 plane cross-section grids with honest psi-vs-density labeling"
```

---

### Task 2: Server: sample jobs gain per-point ψ channels (density, phase)

**Files:**
- Modify: `src/atomsim/server/schemas.py` (add `ChannelModel`)
- Modify: `src/atomsim/server/app.py` (job result container, meta channels, data channel param)
- Test: `tests/test_server.py` (append)

**Interfaces:**
- Consumes: `evaluate_state` (Task 1's module already imports it; here import from `atomsim.analytic.wavefunction`), existing `sample_density`, `SampleCloud`, `_wait_done` test helper.
- Produces (Tasks 6, 7, 9 rely on these):
  - `ChannelModel(name, dtype, unit, provenance)` in schemas.py.
  - `SampleMetaModel` gains `kind: Literal["sample"] = "sample"` and `channels: list[ChannelModel]`. Channels are exactly `["positions", "density"]` for real basis, `["positions", "density", "phase"]` for complex.
  - `GET /api/jobs/{id}/data?channel=positions|density|phase`, float32 bytes; `positions` remains the default (back-compat); `phase` on a real-basis job → 422; unknown channel → 422.
  - Server-internal: `SampleJobResult` frozen dataclass `(cloud: SampleCloud, psi: WavefunctionValues)` stored as `job.result`.

- [x] **Step 1: Write the failing tests**, append to `tests/test_server.py` (it already imports `numpy as np`, if not, add; `_wait_done` exists):

```python
def test_sample_channels_complex_basis(client):
    r = client.post(
        "/api/jobs/sample", json={"n": 2, "l": 1, "m": 1, "count": 4000, "seed": 5}
    )
    job_id = r.json()["id"]
    assert _wait_done(client, job_id)["status"] == "done"

    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["kind"] == "sample"
    names = [c["name"] for c in meta["channels"]]
    assert names == ["positions", "density", "phase"]
    assert all(c["dtype"] == "float32" for c in meta["channels"])
    assert meta["channels"][1]["unit"] == "bohr^-3"
    assert meta["channels"][2]["unit"] == "rad"
    assert meta["channels"][2]["provenance"]["fidelity"] == "exact"

    density = np.frombuffer(
        client.get(f"/api/jobs/{job_id}/data?channel=density").content, dtype=np.float32
    )
    phase = np.frombuffer(
        client.get(f"/api/jobs/{job_id}/data?channel=phase").content, dtype=np.float32
    )
    assert density.shape == (4000,)
    assert (density >= 0.0).all() and density.max() > 0.0
    assert phase.shape == (4000,)
    assert (np.abs(phase) <= np.pi + 1e-6).all()

    # wiring check: channels must equal a direct re-evaluation at the sampled points
    from atomsim.analytic.wavefunction import evaluate_state

    xyz = np.frombuffer(
        client.get(f"/api/jobs/{job_id}/data").content, dtype=np.float32
    ).reshape(-1, 3)
    psi = evaluate_state(2, 1, 1, xyz.astype(np.float64)).values
    assert np.allclose(density, (np.abs(psi) ** 2).astype(np.float32), rtol=1e-5, atol=0)
    assert np.allclose(phase, np.angle(psi).astype(np.float32), atol=1e-5)

def test_sample_channels_real_basis_has_no_phase(client):
    r = client.post(
        "/api/jobs/sample",
        json={"n": 2, "l": 1, "m": 1, "count": 2000, "seed": 5, "basis": "real"},
    )
    job_id = r.json()["id"]
    assert _wait_done(client, job_id)["status"] == "done"
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert [c["name"] for c in meta["channels"]] == ["positions", "density"]
    assert client.get(f"/api/jobs/{job_id}/data?channel=phase").status_code == 422
    assert client.get(f"/api/jobs/{job_id}/data?channel=vibes").status_code == 422
```

- [x] **Step 2: Run to verify failure**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_server.py -q`
Expected: the two new tests FAIL (`KeyError: 'kind'` / missing channels); all pre-existing tests still pass.

- [x] **Step 3: Implement.** In `src/atomsim/server/schemas.py`, add after `ProvenanceModel`:

```python
class ChannelModel(BaseModel):
    """One binary per-point channel of a sample job (positions / density / phase)."""

    name: str
    dtype: str
    unit: str
    provenance: ProvenanceModel
```

In `src/atomsim/server/app.py`:

1. Extend imports:

```python
from atomsim.analytic.wavefunction import WavefunctionValues, evaluate_state
from atomsim.server.schemas import (
    ChannelModel,
    ComparisonModel,
    FieldModel,
    LineModel,
    ProvenanceModel,
    QuantityModel,
    SystemModel,
)
```

2. Add the result container (module level, after the Pydantic models):

```python
@dataclasses.dataclass(frozen=True)
class SampleJobResult:
    """A sampled cloud plus psi evaluated at exactly those positions."""

    cloud: SampleCloud
    psi: WavefunctionValues
```

3. Extend `SampleMetaModel`:

```python
class SampleMetaModel(BaseModel):
    kind: Literal["sample"] = "sample"
    count: int
    dtype: str
    layout: str
    unit: str
    n: int
    l: int
    m: int
    basis: str
    system: str
    provenance: ProvenanceModel
    channels: list[ChannelModel]
```

4. Rename `_finished_cloud` → `_finished_result` (returns `job.result` unchanged; update its docstring/uses).

5. In `create_sample_job`, replace `work` so sampling maps to 90% of progress and ψ evaluation completes it:

```python
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
```

6. Replace the body of `sample_meta` with a helper + call:

```python
    def _sample_meta(res: SampleJobResult, system_key: str) -> SampleMetaModel:
        cloud = res.cloud
        channels = [
            ChannelModel(
                name="positions", dtype="float32", unit="bohr",
                provenance=ProvenanceModel.from_provenance(cloud.provenance),
            ),
            ChannelModel(
                name="density", dtype="float32", unit="bohr^-3",
                provenance=ProvenanceModel.from_provenance(res.psi.provenance),
            ),
        ]
        if cloud.basis == "complex":
            channels.append(
                ChannelModel(
                    name="phase", dtype="float32", unit="rad",
                    provenance=ProvenanceModel.from_provenance(res.psi.provenance),
                )
            )
        return SampleMetaModel(
            count=cloud.positions.shape[0], dtype="float32", layout="xyz-interleaved",
            unit="bohr", n=cloud.n, l=cloud.l, m=cloud.m, basis=cloud.basis,
            system=system_key,
            provenance=ProvenanceModel.from_provenance(cloud.provenance),
            channels=channels,
        )

    @app.get("/api/jobs/{job_id}/meta", response_model=SampleMetaModel)
    def sample_meta(job_id: str) -> SampleMetaModel:
        res = _finished_result(jobs, job_id)
        return _sample_meta(res, app.state.job_systems.get(job_id, "h"))
```

7. Replace `sample_data`:

```python
    @app.get("/api/jobs/{job_id}/data")
    def job_data(job_id: str, channel: str | None = None) -> Response:
        res = _finished_result(jobs, job_id)
        name = channel or "positions"
        if name == "positions":
            payload = res.cloud.positions
        elif name == "density":
            payload = (np.abs(res.psi.values) ** 2).astype(np.float32)
        elif name == "phase" and res.cloud.basis == "complex":
            payload = np.angle(res.psi.values).astype(np.float32)
        else:
            raise HTTPException(status_code=422, detail=f"no channel {name!r} on this job")
        return Response(
            content=payload.tobytes(), media_type="application/octet-stream"
        )
```

- [x] **Step 4: Run full Python suite to verify pass**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q`
Expected: all pass (previous 140 + 2 new). `ruff check .` clean.

- [x] **Step 5: Commit**

```bash
git add src/atomsim/server/schemas.py src/atomsim/server/app.py tests/test_server.py
git commit -m "feat: sample jobs carry per-point density and phase channels"
```

---

### Task 3: Server: `/api/levels` + state readouts (|L|, nodes, pm, shift_ev)

**Files:**
- Modify: `src/atomsim/constants.py` (add `BOHR_RADIUS_PM`)
- Modify: `src/atomsim/analytic/hydrogen.py` (add `angular_momentum_magnitude`)
- Modify: `src/atomsim/server/app.py` (`/api/levels`, `StateResponse` additions, `_to_pm`)
- Test: `tests/test_hydrogen.py` (append; if the analytic tests live in a differently named file, e.g. `tests/test_analytic.py`, append there), `tests/test_server.py` (append)

**Interfaces:**
- Consumes: `energy(n, Z, mu_ratio) -> Quantity`, `level_energy(n, l, j, Z, mu_ratio, m_over_M)`, `fine_structure_shift(...)`, `mean_radius(n, l, Z, mu_ratio)`, scipy `physical_constants`.
- Produces (Tasks 6, 8, 12 rely on these exact JSON shapes):
  - `constants.BOHR_RADIUS_PM: float` (≈ 52.9177210544).
  - `angular_momentum_magnitude(l) -> Quantity`, value √(l(l+1)), unit `"hbar"`, EXACT.
  - `StateResponse` gains: `angular_momentum: QuantityModel`, `mean_radius_pm: QuantityModel`, `radial_nodes: int`, `angular_nodes: int`. `LevelModel` gains `shift_ev: QuantityModel`.
  - `GET /api/levels?system=h&n_max=6&fine_structure=false` → `LevelsResponse{system, n_max, fine_structure, gross: [{n, degeneracy, energy, energy_ev}], fine: [{n, l, j, energy, energy_ev, shift, shift_ev}] | null}`. `n_max` outside [1, 10] → 422.

- [x] **Step 1: Write the failing tests.** Engine test (append to the file holding `energy`/`mean_radius` tests):

```python
def test_angular_momentum_magnitude():
    from atomsim.analytic.hydrogen import angular_momentum_magnitude

    q0 = angular_momentum_magnitude(0)
    assert q0.value == 0.0
    assert q0.unit == "hbar"
    assert q0.provenance.fidelity.value == "exact"
    q2 = angular_momentum_magnitude(2)
    assert q2.value == pytest.approx(6.0**0.5)
    with pytest.raises(ValueError):
        angular_momentum_magnitude(-1)
```

Server tests (append to `tests/test_server.py`):

```python
def test_levels_endpoint_gross(client):
    body = client.get("/api/levels?n_max=3").json()
    assert body["n_max"] == 3 and body["fine"] is None
    assert [g["n"] for g in body["gross"]] == [1, 2, 3]
    assert [g["degeneracy"] for g in body["gross"]] == [2, 8, 18]
    e1 = body["gross"][0]["energy"]
    assert e1["unit"] == "hartree"
    assert e1["value"] == pytest.approx(-0.4997278, rel=1e-5)  # reduced-mass H
    assert body["gross"][0]["energy_ev"]["unit"] == "eV"

def test_levels_endpoint_fine_structure(client):
    body = client.get("/api/levels?n_max=2&fine_structure=true").json()
    fine = body["fine"]
    assert [(f["n"], f["l"], f["j"]) for f in fine] == [
        (1, 0, 0.5), (2, 0, 0.5), (2, 1, 0.5), (2, 1, 1.5),
    ]
    for f in fine:
        assert f["shift"]["provenance"]["fidelity"] == "approximation"
        assert f["shift_ev"]["unit"] == "eV"
    # 2p_1/2 lies below 2p_3/2
    assert fine[2]["energy"]["value"] < fine[3]["energy"]["value"]

def test_levels_endpoint_validation(client):
    assert client.get("/api/levels?n_max=0").status_code == 422
    assert client.get("/api/levels?n_max=11").status_code == 422
    assert client.get("/api/levels?system=unobtainium").status_code == 422

def test_state_readouts(client):
    body = client.get("/api/state/3/2/1").json()
    assert body["angular_momentum"]["value"] == pytest.approx(6.0**0.5)
    assert body["angular_momentum"]["unit"] == "hbar"
    assert body["radial_nodes"] == 0
    assert body["angular_nodes"] == 2
    r_bohr = body["mean_radius"]["value"]
    r_pm = body["mean_radius_pm"]["value"]
    assert body["mean_radius_pm"]["unit"] == "pm"
    assert r_pm == pytest.approx(r_bohr * 52.9177, rel=1e-4)

def test_state_fine_structure_shift_ev(client):
    body = client.get("/api/state/2/1/0?fine_structure=true").json()
    assert len(body["levels"]) == 2
    for lev in body["levels"]:
        assert lev["shift_ev"]["unit"] == "eV"
```

- [x] **Step 2: Run to verify failure**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_server.py tests/test_hydrogen.py -q` (adjust filename)
Expected: new tests FAIL (missing attribute / 404 on /api/levels / KeyError).

- [x] **Step 3: Implement.**

`src/atomsim/constants.py`, add (mirroring however `HARTREE_EV` is defined there, citing CODATA via scipy):

```python
BOHR_RADIUS_PM = physical_constants["Bohr radius"][0] * 1e12  # CODATA, in picometres
```

(If the module does not already import `physical_constants` from `scipy.constants`, add that import.)

`src/atomsim/analytic/hydrogen.py`, add:

```python
def angular_momentum_magnitude(l: int) -> Quantity:
    """|L| = sqrt(l(l+1)) hbar, the exact magnitude from the L^2 eigenvalue."""
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l}")
    return Quantity(
        value=float(np.sqrt(l * (l + 1))),
        unit="hbar",
        label=f"|L| (l={l})",
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method="sqrt(l(l+1)) hbar from the L^2 eigenvalue of the (n,l,m) eigenstate",
            assumptions=_EXACT_ASSUMPTIONS,
        ),
    )
```

`src/atomsim/server/app.py`:

1. Import `angular_momentum_magnitude` (extend the existing `atomsim.analytic.hydrogen` import) and `BOHR_RADIUS_PM` (extend the `atomsim.constants` import).
2. Add `_to_pm` next to `_to_ev`:

```python
def _to_pm(q: Quantity) -> Quantity:
    return Quantity(
        value=q.value * BOHR_RADIUS_PM,
        unit="pm",
        label=q.label + " [pm]",
        provenance=dataclasses.replace(
            q.provenance,
            method=q.provenance.method + "; converted to pm via CODATA Bohr radius",
        ),
    )
```

3. Extend models:

```python
class LevelModel(BaseModel):
    j: float
    energy: QuantityModel
    energy_ev: QuantityModel
    shift: QuantityModel
    shift_ev: QuantityModel

class GrossLevelModel(BaseModel):
    n: int
    degeneracy: int
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

class LevelsResponse(BaseModel):
    system: SystemModel
    n_max: int
    fine_structure: bool
    gross: list[GrossLevelModel]
    fine: list[FineLevelModel] | None
```

`StateResponse` gains:

```python
    angular_momentum: QuantityModel
    mean_radius_pm: QuantityModel
    radial_nodes: int
    angular_nodes: int
```

4. In the `state` handler: add `shift_ev=QuantityModel.from_quantity(_to_ev(sh))` to the `LevelModel(...)` construction, and to the `StateResponse(...)` construction add:

```python
            angular_momentum=QuantityModel.from_quantity(angular_momentum_magnitude(l)),
            mean_radius_pm=QuantityModel.from_quantity(
                _to_pm(mean_radius(n, l, Z=sys_.Z, mu_ratio=mu))
            ),
            radial_nodes=n - l - 1,
            angular_nodes=l,
```

(Compute `mr = mean_radius(n, l, Z=sys_.Z, mu_ratio=mu)` once and reuse for both bohr and pm forms.)

5. Add the endpoint after `state`:

```python
    @app.get("/api/levels", response_model=LevelsResponse)
    def levels(system: str = "h", n_max: int = 6,
               fine_structure: bool = False) -> LevelsResponse:
        if not 1 <= n_max <= 10:
            raise HTTPException(status_code=422, detail="n_max must be in [1, 10]")
        sys_ = _resolve_system(system)
        mu = sys_.mu_ratio.value
        gross = []
        for n in range(1, n_max + 1):
            e = energy(n, Z=sys_.Z, mu_ratio=mu)
            gross.append(GrossLevelModel(
                n=n, degeneracy=2 * n * n,
                energy=QuantityModel.from_quantity(e),
                energy_ev=QuantityModel.from_quantity(_to_ev(e)),
            ))
        fine = None
        if fine_structure:
            fine = []
            for n in range(1, n_max + 1):
                for l in range(n):
                    for j in ([0.5] if l == 0 else [l - 0.5, l + 0.5]):
                        le = level_energy(
                            n, l, j, Z=sys_.Z, mu_ratio=mu, m_over_M=sys_.m_over_M
                        )
                        sh = fine_structure_shift(
                            n, l, j, Z=sys_.Z, mu_ratio=mu, m_over_M=sys_.m_over_M
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
            fine_structure=fine_structure, gross=gross, fine=fine,
        )
```

- [x] **Step 4: Run full Python suite**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q`
Expected: all pass. `ruff check .` clean.

- [x] **Step 5: Commit**

```bash
git add src/atomsim/constants.py src/atomsim/analytic/hydrogen.py src/atomsim/server/app.py tests/
git commit -m "feat: /api/levels endpoint plus |L|, node-count and pm readouts on /api/state"
```

---

### Task 4: Server â€” plane-grid jobs (`POST /api/jobs/plane`, meta union)

**Files:**
- Modify: `src/atomsim/server/app.py`
- Test: `tests/test_server.py` (append)

**Interfaces:**
- Consumes: `plane_grid`, `PlaneGrid` from `atomsim.plane` (Task 1); `SampleJobResult`, `_finished_result`, `_sample_meta` (Task 2).
- Produces (Tasks 6, 7, 10 rely on these):
  - `POST /api/jobs/plane` body `{n, l, m, quantity: "density"|"psi" = "density", basis = "complex", system = "h", resolution: int = 512 (ge=32, le=1024)}` â†’ `JobModel`. Same WebSocket progress channel as sample jobs.
  - `GET /api/jobs/{id}/meta` â†’ union `SampleMetaModel | PlaneMetaModel` discriminated by `kind`.
  - `PlaneMetaModel{kind: "plane", resolution, dtype: "float32", layout, quantity, unit, label, half_extent, axis_unit: "bohr", n, l, m, basis, system, provenance}`.
  - `GET /api/jobs/{id}/data` on a plane job (NO channel param) â†’ row-major float32 grid bytes; any `channel` query on a plane job â†’ 422.

- [x] **Step 1: Write the failing tests** â€” append to `tests/test_server.py`:

```python
def test_plane_job_end_to_end(client):
    r = client.post(
        "/api/jobs/plane",
        json={"n": 2, "l": 1, "m": 0, "resolution": 64},
    )
    assert r.status_code == 200
    job_id = r.json()["id"]
    assert _wait_done(client, job_id)["status"] == "done"

    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["kind"] == "plane"
    assert meta["resolution"] == 64
    assert meta["quantity"] == "density"
    assert meta["unit"] == "bohr^-3"
    assert meta["half_extent"] == pytest.approx(10.0)
    assert meta["provenance"]["fidelity"] == "exact"

    raw = client.get(f"/api/jobs/{job_id}/data").content
    assert len(raw) == 64 * 64 * 4
    values = np.frombuffer(raw, dtype=np.float32).reshape(64, 64)

    # deterministic: must equal a direct engine computation
    from atomsim.plane import plane_grid

    expected = plane_grid(2, 1, 0, resolution=64).values.astype(np.float32)
    assert np.array_equal(values, expected)

    # plane jobs have exactly one channel
    assert client.get(f"/api/jobs/{job_id}/data?channel=density").status_code == 422

def test_plane_job_psi_quantity_and_validation(client):
    r = client.post(
        "/api/jobs/plane",
        json={"n": 2, "l": 1, "m": 0, "quantity": "psi", "resolution": 32},
    )
    job_id = r.json()["id"]
    assert _wait_done(client, job_id)["status"] == "done"
    meta = client.get(f"/api/jobs/{job_id}/meta").json()
    assert meta["quantity"] == "psi"
    assert meta["unit"] == "bohr^-3/2"

    assert client.post(
        "/api/jobs/plane", json={"n": 1, "l": 0, "m": 0, "resolution": 4096}
    ).status_code == 422
    assert client.post(
        "/api/jobs/plane", json={"n": 1, "l": 0, "m": 0, "quantity": "vibes"}
    ).status_code == 422
    assert client.post(
        "/api/jobs/plane", json={"n": 1, "l": 1, "m": 0}
    ).status_code == 422
```

- [x] **Step 2: Run to verify failure**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_server.py -q`
Expected: new tests FAIL with 404 on `/api/jobs/plane`.

- [x] **Step 3: Implement** in `src/atomsim/server/app.py`:

1. Import: `from atomsim.plane import PlaneGrid, plane_grid`.
2. Models:

```python
class PlaneRequest(BaseModel):
    n: int
    l: int
    m: int
    quantity: Literal["density", "psi"] = "density"
    basis: Literal["complex", "real"] = "complex"
    system: str = "h"
    resolution: int = PydanticField(default=512, ge=32, le=1024)

class PlaneMetaModel(BaseModel):
    kind: Literal["plane"] = "plane"
    resolution: int
    dtype: str
    layout: str
    quantity: str
    unit: str
    label: str
    half_extent: float
    axis_unit: str
    n: int
    l: int
    m: int
    basis: str
    system: str
    provenance: ProvenanceModel
```

3. Endpoint (next to `create_sample_job`):

```python
    @app.post("/api/jobs/plane", response_model=JobModel)
    async def create_plane_job(req: PlaneRequest) -> JobModel:
        _validate_state(req.n, req.l, req.m)
        sys_ = _resolve_system(req.system)
        job = jobs.create()
        app.state.job_systems[job.id] = req.system

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

4. Meta helper + union endpoint (replaces Task 2's `sample_meta` route):

```python
    def _plane_meta(pg: PlaneGrid, system_key: str) -> PlaneMetaModel:
        return PlaneMetaModel(
            resolution=pg.values.shape[0],
            dtype="float32",
            layout="row-major float32; row i = z=axis[i] ascending, col j = x=axis[j]",
            quantity=pg.quantity, unit=pg.unit, label=pg.label,
            half_extent=float(pg.axis[-1]), axis_unit="bohr",
            n=pg.n, l=pg.l, m=pg.m, basis=pg.basis, system=system_key,
            provenance=ProvenanceModel.from_provenance(pg.provenance),
        )

    @app.get("/api/jobs/{job_id}/meta", response_model=SampleMetaModel | PlaneMetaModel)
    def job_meta(job_id: str) -> SampleMetaModel | PlaneMetaModel:
        res = _finished_result(jobs, job_id)
        system_key = app.state.job_systems.get(job_id, "h")
        if isinstance(res, PlaneGrid):
            return _plane_meta(res, system_key)
        return _sample_meta(res, system_key)
```

5. Extend `job_data` (Task 2 version) with the plane branch at the top:

```python
    @app.get("/api/jobs/{job_id}/data")
    def job_data(job_id: str, channel: str | None = None) -> Response:
        res = _finished_result(jobs, job_id)
        if isinstance(res, PlaneGrid):
            if channel is not None:
                raise HTTPException(
                    status_code=422, detail="plane jobs have a single channel"
                )
            payload = res.values.astype(np.float32)
        elif (channel or "positions") == "positions":
            payload = res.cloud.positions
        elif channel == "density":
            payload = (np.abs(res.psi.values) ** 2).astype(np.float32)
        elif channel == "phase" and res.cloud.basis == "complex":
            payload = np.angle(res.psi.values).astype(np.float32)
        else:
            raise HTTPException(status_code=422, detail=f"no channel {channel!r} on this job")
        return Response(
            content=payload.tobytes(), media_type="application/octet-stream"
        )
```

- [x] **Step 4: Run full Python suite**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q`
Expected: all pass. `ruff check .` clean.

- [x] **Step 5: Commit**

```bash
git add src/atomsim/server/app.py tests/test_server.py
git commit -m "feat: plane-grid jobs with discriminated sample/plane meta"
```

---

### Task 5: Server â€” thumbnail PNGs (matplotlib Agg + cache)

**Files:**
- Modify: `pyproject.toml` (dependency `matplotlib>=3.9`)
- Create: `src/atomsim/server/thumbnails.py`
- Modify: `src/atomsim/server/app.py` (endpoint)
- Test: `tests/test_thumbnails.py`

**Interfaces:**
- Consumes: `plane_grid` (Task 1), `get_system`.
- Produces (Task 14 relies on this):
  - `render_thumbnail(n, l, m, system, basis, size) -> bytes` (PNG), `@lru_cache(maxsize=512)`, module constant `GAMMA = 0.5`.
  - `GET /api/thumbnail/{n}/{l}/{m}?system=h&basis=complex&size=120` â†’ `image/png`, `Cache-Control: public, max-age=86400`. `size` outside [32, 256] â†’ 422; bad basis/state/system â†’ 422.
  - **Honesty:** thumbnails are navigation aids; brightness is `(Ï/Ï_max)^0.5` gamma-compressed â€” the SAME `GAMMA` the frontend colormap module uses (Task 6), disclosed in the gallery UI (Task 14).

- [x] **Step 1: Install matplotlib and add the dependency.**

In `pyproject.toml` `[project] dependencies`, append `"matplotlib>=3.9"`. Then:

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pip install -e ".[dev]"`
Expected: matplotlib installed; `python -c "import matplotlib; print(matplotlib.__version__)"` prints â‰¥ 3.9.

- [x] **Step 2: Write the failing tests** â€” `tests/test_thumbnails.py`:

```python
import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app
from atomsim.server.thumbnails import render_thumbnail

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c

def test_render_thumbnail_returns_png_bytes():
    render_thumbnail.cache_clear()
    png = render_thumbnail(2, 1, 0, "h", "complex", 64)
    assert isinstance(png, bytes)
    assert png[:8] == PNG_MAGIC

def test_render_thumbnail_caches():
    render_thumbnail.cache_clear()
    render_thumbnail(1, 0, 0, "h", "complex", 48)
    render_thumbnail(1, 0, 0, "h", "complex", 48)
    assert render_thumbnail.cache_info().hits == 1

def test_thumbnail_endpoint(client):
    r = client.get("/api/thumbnail/2/1/0?size=48")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert "max-age" in r.headers["cache-control"]
    assert r.content[:8] == PNG_MAGIC

def test_thumbnail_states_differ(client):
    a = client.get("/api/thumbnail/1/0/0?size=48").content
    b = client.get("/api/thumbnail/2/1/0?size=48").content
    assert a != b

def test_thumbnail_validation(client):
    assert client.get("/api/thumbnail/2/1/0?size=10").status_code == 422
    assert client.get("/api/thumbnail/2/1/0?size=999").status_code == 422
    assert client.get("/api/thumbnail/2/1/0?basis=cartoon").status_code == 422
    assert client.get("/api/thumbnail/1/1/0").status_code == 422
    assert client.get("/api/thumbnail/2/1/0?system=unobtainium").status_code == 422
```

- [x] **Step 3: Run to verify failure**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest tests/test_thumbnails.py -q`
Expected: FAIL â€” `ModuleNotFoundError: No module named 'atomsim.server.thumbnails'`.

- [x] **Step 4: Implement.** `src/atomsim/server/thumbnails.py`:

```python
"""Server-rendered gallery thumbnails: small inferno PNGs of plane densities.

Navigation aids, not measurement surfaces: brightness is gamma-compressed
(t = (rho/rho_max)^GAMMA) so faint outer lobes stay visible â€” a VISUAL LIBERTY
disclosed in the gallery UI. The frontend cross-section renderer mirrors the
same GAMMA and the same matplotlib LUT (web/src/lib/colormap.ts, luts.ts), so
densities look identical everywhere.
"""

import io
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")

from matplotlib import image as mpl_image  # noqa: E402

from atomsim.plane import plane_grid       # noqa: E402
from atomsim.systems import get_system     # noqa: E402

GAMMA = 0.5

@lru_cache(maxsize=512)
def render_thumbnail(n: int, l: int, m: int, system: str, basis: str, size: int) -> bytes:
    """Inferno PNG of |psi|^2 on the y=0 plane; row order flipped so +z is up."""
    sys_ = get_system(system)
    pg = plane_grid(
        n, l, m, quantity="density", basis=basis,
        Z=sys_.Z, mu_ratio=sys_.mu_ratio.value, resolution=size,
    )
    rho = pg.values
    vmax = float(rho.max())
    t = (rho / vmax) ** GAMMA if vmax > 0.0 else rho
    buf = io.BytesIO()
    mpl_image.imsave(buf, t[::-1], cmap="inferno", vmin=0.0, vmax=1.0, format="png")
    return buf.getvalue()
```

In `src/atomsim/server/app.py`: import `from atomsim.server.thumbnails import render_thumbnail` and add the endpoint (after the spectrum endpoint):

```python
    @app.get("/api/thumbnail/{n}/{l}/{m}")
    def thumbnail(n: int, l: int, m: int, system: str = "h",
                  basis: str = "complex", size: int = 120) -> Response:
        _validate_state(n, l, m)
        _resolve_system(system)
        if basis not in ("complex", "real"):
            raise HTTPException(status_code=422, detail=f"unknown basis {basis!r}")
        if not 32 <= size <= 256:
            raise HTTPException(status_code=422, detail="size must be in [32, 256]")
        png = render_thumbnail(n, l, m, system, basis, size)
        return Response(
            content=png, media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )
```

- [x] **Step 5: Run full Python suite**

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q`
Expected: all pass. `ruff check .` clean (note the `# noqa: E402` comments â€” `matplotlib.use("Agg")` must precede the image-module import).

- [x] **Step 6: Commit**

```bash
git add pyproject.toml src/atomsim/server/thumbnails.py src/atomsim/server/app.py tests/test_thumbnails.py
git commit -m "feat: cached inferno thumbnail PNGs for the state gallery"
```

---

### Task 6: Web â€” type/client mirrors, generated LUTs, colormap lib

**Files:**
- Create: `scripts/gen_luts.py`, `web/src/lib/luts.ts` (generated), `web/src/lib/colormap.ts`, `web/src/lib/colormap.test.ts`
- Modify: `web/src/api/types.ts`, `web/src/api/client.ts`, `web/src/api/decode.test.ts`

**Interfaces:**
- Consumes: JSON shapes from Tasks 2â€“5.
- Produces (all later web tasks rely on these exact names):
  - `luts.ts`: `type Lut = ReadonlyArray<readonly [number, number, number]>`; `INFERNO: Lut`; `RDBU_R: Lut` (256 entries each, generated from matplotlib â€” single color authority with server thumbnails).
  - `colormap.ts`: `lutColor(lut, t)`, `DENSITY_GAMMA = 0.5`, `densityT(value, vmax)`, `signedT(value, vabs)`, `maxOf(arr)`, `maxAbs(arr)`, `phaseColor(phase)`, `hslToRgb(h, s, l)`.
  - `types.ts`: `ChannelInfo`, `SampleMeta` (+`kind`, `channels`), `PlaneMeta`, `JobMeta = SampleMeta | PlaneMeta`, `GrossLevel`, `FineLevel`, `LevelsResponse`, `SystemsResponse`, `LevelInfo` (+`shift_ev`), `StateResponse` (+`angular_momentum`, `mean_radius_pm`, `radial_nodes`, `angular_nodes`).
  - `client.ts`: `Basis`, `PlaneQuantity` types; `getSystems()`, `getState(n, l, m, system, fineStructure)`, `getRadial(n, l, system)`, `getLevels(system, nMax, fineStructure)`, `getSpectrum(system, nMax, fineStructure)`, `createSampleJob(params)`, `createPlaneJob(params)`, `getJobMeta(jobId)`, `getChannel(jobId, channel?)`, `decodeFloats(buffer)`, `thumbnailUrl(n, l, m, system, basis, size)`; `watchJob` unchanged.

- [x] **Step 1: Generate the LUTs.** Create `scripts/gen_luts.py`:

```python
"""Generate web/src/lib/luts.ts from matplotlib colormaps.

Single color authority: server thumbnails (matplotlib) and the browser client
read the same 256-entry tables, so a density looks identical in a thumbnail,
the cross-section canvas and the 3D cloud. Re-run after a matplotlib upgrade.
"""

from pathlib import Path

import matplotlib
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "web" / "src" / "lib" / "luts.ts"

def rows(name: str) -> str:
    cmap = matplotlib.colormaps[name]
    rgb = (np.asarray(cmap(np.linspace(0.0, 1.0, 256)))[:, :3] * 255).round().astype(int)
    return ",\n".join(f"  [{r}, {g}, {b}]" for r, g, b in rgb)

TEMPLATE = """\
// GENERATED by scripts/gen_luts.py from matplotlib {version} - do not edit by hand.
// Single color authority: server thumbnails (matplotlib) and this client LUT
// come from the same colormaps, so densities look identical everywhere.

export type Lut = ReadonlyArray<readonly [number, number, number]>;

export const INFERNO: Lut = [
{inferno},
];

export const RDBU_R: Lut = [
{rdbu},
];
"""

OUT.write_text(
    TEMPLATE.format(version=matplotlib.__version__, inferno=rows("inferno"),
                    rdbu=rows("RdBu_r")),
    encoding="utf-8", newline="\n",
)
print(f"wrote {OUT}")
```

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python scripts/gen_luts.py`
Expected: `wrote ...luts.ts`; file contains two 256-row arrays.

- [x] **Step 2: Write the failing web tests.** `web/src/lib/colormap.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import {
  DENSITY_GAMMA,
  densityT,
  hslToRgb,
  lutColor,
  maxAbs,
  maxOf,
  phaseColor,
  signedT,
} from "./colormap";
import { INFERNO, RDBU_R } from "./luts";

describe("luts", () => {
  it("has 256 rgb triples in each table", () => {
    for (const lut of [INFERNO, RDBU_R]) {
      expect(lut.length).toBe(256);
      for (const [r, g, b] of lut) {
        for (const c of [r, g, b]) {
          expect(Number.isInteger(c)).toBe(true);
          expect(c).toBeGreaterThanOrEqual(0);
          expect(c).toBeLessThanOrEqual(255);
        }
      }
    }
  });
  it("inferno runs dark to light", () => {
    const sum = (i: number) => INFERNO[i][0] + INFERNO[i][1] + INFERNO[i][2];
    expect(sum(0)).toBeLessThan(30);
    expect(sum(255)).toBeGreaterThan(600);
  });
});

describe("lutColor", () => {
  it("maps endpoints and clamps", () => {
    expect(lutColor(INFERNO, 0)).toEqual(INFERNO[0]);
    expect(lutColor(INFERNO, 1)).toEqual(INFERNO[255]);
    expect(lutColor(INFERNO, -5)).toEqual(INFERNO[0]);
    expect(lutColor(INFERNO, 5)).toEqual(INFERNO[255]);
  });
});

describe("densityT", () => {
  it("gamma-compresses with the disclosed exponent", () => {
    expect(DENSITY_GAMMA).toBe(0.5);
    expect(densityT(4, 4)).toBe(1);
    expect(densityT(1, 4)).toBeCloseTo(0.5, 12); // sqrt(0.25)
    expect(densityT(0, 4)).toBe(0);
    expect(densityT(1, 0)).toBe(0); // degenerate max
  });
});

describe("signedT", () => {
  it("centres zero at 0.5 and clamps", () => {
    expect(signedT(0, 2)).toBe(0.5);
    expect(signedT(2, 2)).toBe(1);
    expect(signedT(-2, 2)).toBe(0);
    expect(signedT(99, 2)).toBe(1);
    expect(signedT(1, 0)).toBe(0.5);
  });
});

describe("max helpers", () => {
  it("maxOf and maxAbs", () => {
    expect(maxOf(new Float32Array([1, 5, 2]))).toBe(5);
    expect(maxAbs(new Float32Array([1, -7, 2]))).toBe(7);
    expect(maxOf(new Float32Array([]))).toBe(0);
  });
});

describe("phaseColor", () => {
  it("is cyclic across the -pi/+pi seam", () => {
    expect(phaseColor(-Math.PI)).toEqual(phaseColor(Math.PI));
  });
  it("hslToRgb hits primary hues", () => {
    expect(hslToRgb(0, 1, 0.5)).toEqual([255, 0, 0]);
    expect(hslToRgb(1 / 3, 1, 0.5)).toEqual([0, 255, 0]);
    expect(hslToRgb(2 / 3, 1, 0.5)).toEqual([0, 0, 255]);
  });
});
```

Append to `web/src/api/decode.test.ts` (it already imports from `./client`; extend the import):

```typescript
describe("decodeFloats", () => {
  it("decodes float32 buffers", () => {
    const buf = new Float32Array([1.5, -2.5]).buffer;
    expect(Array.from(decodeFloats(buf))).toEqual([1.5, -2.5]);
  });
  it("rejects lengths that are not multiples of 4", () => {
    expect(() => decodeFloats(new ArrayBuffer(5))).toThrow(/multiple of 4/);
  });
});

describe("thumbnailUrl", () => {
  it("builds the endpoint url", () => {
    expect(thumbnailUrl(2, 1, -1, "mu-h", "real", 96)).toBe(
      "/api/thumbnail/2/1/-1?system=mu-h&basis=real&size=96",
    );
  });
});
```

(Keep the existing `decodePositions` tests as-is.)

- [x] **Step 3: Run to verify failure**

From `web\`: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test`
Expected: FAIL â€” `colormap.ts` does not exist; `decodeFloats` not exported.

- [x] **Step 4: Implement.** `web/src/lib/colormap.ts`:

```typescript
import type { Lut } from "./luts";

/** Clamped LUT lookup: t in [0, 1] -> [r, g, b] bytes. */
export function lutColor(lut: Lut, t: number): readonly [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  return lut[Math.min(lut.length - 1, Math.floor(clamped * lut.length))];
}

/**
 * VISUAL LIBERTY (disclosed wherever used): density brightness is
 * gamma-compressed, t = (rho / rho_max)^DENSITY_GAMMA, so faint outer lobes
 * stay visible. Mirrors GAMMA in src/atomsim/server/thumbnails.py.
 */
export const DENSITY_GAMMA = 0.5;

export function densityT(value: number, vmax: number): number {
  if (vmax <= 0) return 0;
  const clamped = Math.min(Math.max(value, 0), vmax);
  return (clamped / vmax) ** DENSITY_GAMMA;
}

/** Signed value -> [0, 1] with zero at 0.5, for diverging colormaps. */
export function signedT(value: number, vabs: number): number {
  if (vabs <= 0) return 0.5;
  return (Math.min(Math.max(value / vabs, -1), 1) + 1) / 2;
}

export function maxOf(values: Float32Array): number {
  let m = 0;
  for (let i = 0; i < values.length; i++) m = Math.max(m, values[i]);
  return m;
}

export function maxAbs(values: Float32Array): number {
  let m = 0;
  for (let i = 0; i < values.length; i++) m = Math.max(m, Math.abs(values[i]));
  return m;
}

/** Cyclic colour for arg(psi) in [-pi, pi]: full HSL hue wheel. */
export function phaseColor(phase: number): readonly [number, number, number] {
  const h = (phase + Math.PI) / (2 * Math.PI);
  return hslToRgb(h - Math.floor(h), 1, 0.55);
}

export function hslToRgb(
  h: number,
  s: number,
  l: number,
): readonly [number, number, number] {
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const f = (t: number): number => {
    let x = t;
    if (x < 0) x += 1;
    if (x > 1) x -= 1;
    if (x < 1 / 6) return p + (q - p) * 6 * x;
    if (x < 1 / 2) return q;
    if (x < 2 / 3) return p + (q - p) * (2 / 3 - x) * 6;
    return p;
  };
  return [
    Math.round(f(h + 1 / 3) * 255),
    Math.round(f(h) * 255),
    Math.round(f(h - 1 / 3) * 255),
  ];
}
```

`web/src/api/types.ts` â€” add/modify:

```typescript
export interface ChannelInfo {
  name: string;
  dtype: string;
  unit: string;
  provenance: Provenance;
}

export interface SampleMeta {
  kind: "sample";
  count: number;
  dtype: string;
  layout: string;
  unit: string;
  n: number;
  l: number;
  m: number;
  basis: string;
  system: string;
  provenance: Provenance;
  channels: ChannelInfo[];
}

export interface PlaneMeta {
  kind: "plane";
  resolution: number;
  dtype: string;
  layout: string;
  quantity: "density" | "psi";
  unit: string;
  label: string;
  half_extent: number;
  axis_unit: string;
  n: number;
  l: number;
  m: number;
  basis: string;
  system: string;
  provenance: Provenance;
}

export type JobMeta = SampleMeta | PlaneMeta;

export interface SystemsResponse {
  systems: SystemInfo[];
}

export interface GrossLevel {
  n: number;
  degeneracy: number;
  energy: Quantity;
  energy_ev: Quantity;
}

export interface FineLevel {
  n: number;
  l: number;
  j: number;
  energy: Quantity;
  energy_ev: Quantity;
  shift: Quantity;
  shift_ev: Quantity;
}

export interface LevelsResponse {
  system: SystemInfo;
  n_max: number;
  fine_structure: boolean;
  gross: GrossLevel[];
  fine: FineLevel[] | null;
}
```

Also: `LevelInfo` gains `shift_ev: Quantity;` and `StateResponse` gains:

```typescript
  angular_momentum: Quantity;
  mean_radius_pm: Quantity;
  radial_nodes: number;
  angular_nodes: number;
```

`web/src/api/client.ts` â€” full new version:

```typescript
import type {
  JobInfo,
  JobMeta,
  LevelsResponse,
  RadialResponse,
  SpectrumResponse,
  StateResponse,
  SystemsResponse,
} from "./types";

export type Basis = "complex" | "real";
export type PlaneQuantity = "density" | "psi";

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export function getSystems(): Promise<SystemsResponse> {
  return getJson("/api/systems");
}

export function getState(
  n: number,
  l: number,
  m: number,
  system: string,
  fineStructure: boolean,
): Promise<StateResponse> {
  return getJson(
    `/api/state/${n}/${l}/${m}?system=${system}&fine_structure=${fineStructure}`,
  );
}

export function getRadial(n: number, l: number, system: string): Promise<RadialResponse> {
  return getJson(`/api/radial/${n}/${l}?system=${system}`);
}

export function getLevels(
  system: string,
  nMax: number,
  fineStructure: boolean,
): Promise<LevelsResponse> {
  return getJson(
    `/api/levels?system=${system}&n_max=${nMax}&fine_structure=${fineStructure}`,
  );
}

export function getSpectrum(
  system: string,
  nMax: number,
  fineStructure: boolean,
): Promise<SpectrumResponse> {
  return getJson(
    `/api/spectrum?system=${system}&n_max=${nMax}&fine_structure=${fineStructure}`,
  );
}

export interface SampleParams {
  n: number;
  l: number;
  m: number;
  count: number;
  seed?: number;
  basis: Basis;
  system: string;
}

export function createSampleJob(params: SampleParams): Promise<JobInfo> {
  return postJson("/api/jobs/sample", { seed: 0, ...params });
}

export interface PlaneParams {
  n: number;
  l: number;
  m: number;
  quantity: PlaneQuantity;
  basis: Basis;
  system: string;
  resolution?: number;
}

export function createPlaneJob(params: PlaneParams): Promise<JobInfo> {
  return postJson("/api/jobs/plane", { resolution: 512, ...params });
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

export function getJobMeta(jobId: string): Promise<JobMeta> {
  return getJson(`/api/jobs/${jobId}/meta`);
}

export async function getChannel(jobId: string, channel?: string): Promise<Float32Array> {
  const url = channel
    ? `/api/jobs/${jobId}/data?channel=${channel}`
    : `/api/jobs/${jobId}/data`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return decodeFloats(await res.arrayBuffer());
}

export function decodeFloats(buffer: ArrayBuffer): Float32Array {
  if (buffer.byteLength % 4 !== 0) {
    throw new Error(`byte length ${buffer.byteLength} is not a multiple of 4 (float32)`);
  }
  return new Float32Array(buffer);
}

export function decodePositions(buffer: ArrayBuffer): Float32Array {
  if (buffer.byteLength % 12 !== 0) {
    throw new Error(
      `positions byte length ${buffer.byteLength} is not a multiple of 12 (xyz float32)`,
    );
  }
  return new Float32Array(buffer);
}

export function thumbnailUrl(
  n: number,
  l: number,
  m: number,
  system: string,
  basis: Basis,
  size: number,
): string {
  return `/api/thumbnail/${n}/${l}/${m}?system=${system}&basis=${basis}&size=${size}`;
}
```

(The old 3-arg `getState` and `getSampleMeta`/`getSampleData` are gone; the store catches up in Task 7. `npm test` stays green because vitest only compiles what tests import; the full `tsc` gate runs at Task 8's build step.)

- [x] **Step 5: Run web tests**

From `web\`: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test`
Expected: all vitest suites pass.

- [x] **Step 6: Commit**

```bash
git add scripts/gen_luts.py web/src/lib/luts.ts web/src/lib/colormap.ts web/src/lib/colormap.test.ts web/src/api/types.ts web/src/api/client.ts web/src/api/decode.test.ts
git commit -m "feat: web mirrors for M3 payloads + generated matplotlib LUTs"
```

---

### Task 7: Web â€” orbital labels + full store expansion

**Files:**
- Modify: `web/src/lib/quantum.ts`, `web/src/lib/quantum.test.ts`
- Modify: `web/src/state/store.ts`
- Create: `web/src/state/store.test.ts`

**Interfaces:**
- Consumes: Task 6 client/types.
- Produces (all view tasks rely on these exact names):
  - `realOrbitalLabel(l, m): string` â€” exact mirror of `atomsim.analytic.angular.real_orbital_label`.
  - Store state: `n l m`, `system ("h")`, `basis: Basis ("complex")`, `view: ViewMode ("cloud")`, `colorMode: ColorMode ("solid")`, `fineStructure: boolean`, `count`, `systems: SystemInfo[]`, `stateInfo`, `positions/density/phase: Float32Array | null`, `meta: SampleMeta | null`, `status/progress/error`, `fps: number`, `planeQuantity: PlaneQuantity ("density")`, `plane: { meta: PlaneMeta; values: Float32Array } | null`, `planeStatus/planeProgress`, `radial: RadialResponse | null`, `levels: LevelsResponse | null`, `spectrum: SpectrumResponse | null`.
  - Actions: `setQuantumNumbers`, `setSystem`, `setBasis`, `setView`, `setColorMode`, `setFineStructure`, `setCount`, `setPlaneQuantity`, `setFps`, `loadSystems`, `loadStateInfo`, `sample`, `loadPlane`, `loadRadial`, `loadLevels`, `loadSpectrum`.
  - Exported types: `ViewMode = "cloud" | "plane" | "radial" | "levels" | "spectrum"`, `ColorMode = "solid" | "density" | "phase"`, `SampleStatus` (unchanged).
  - **Invalidation rule:** any change to `n/l/m/system/basis` clears all fetched data back to null/idle; `setBasis("real")` demotes `colorMode: "phase"` â†’ `"density"`; `setFineStructure` clears only `stateInfo/levels/spectrum`; `setPlaneQuantity` clears only the plane.

- [x] **Step 1: Write the failing tests.** Append to `web/src/lib/quantum.test.ts`:

```typescript
import { realOrbitalLabel } from "./quantum";

describe("realOrbitalLabel", () => {
  it("mirrors atomsim.analytic.angular.real_orbital_label", () => {
    expect(realOrbitalLabel(0, 0)).toBe("s");
    expect(realOrbitalLabel(1, 0)).toBe("p_z");
    expect(realOrbitalLabel(1, 1)).toBe("p_x");
    expect(realOrbitalLabel(1, -1)).toBe("p_y");
    expect(realOrbitalLabel(2, -2)).toBe("d_xy");
    expect(realOrbitalLabel(3, 3)).toBe("f_x(x2-3y2)");
    expect(realOrbitalLabel(4, 0)).toBe("g(m=0)");
    expect(realOrbitalLabel(4, 2)).toBe("g(m=+2, cos)");
    expect(realOrbitalLabel(4, -2)).toBe("g(m=-2, sin)");
  });
});
```

Create `web/src/state/store.test.ts` (transitions only â€” no network; fetching actions are exercised in the running app):

```typescript
import { beforeEach, describe, expect, it } from "vitest";
import { useAppStore } from "./store";

const initial = useAppStore.getState();

beforeEach(() => {
  useAppStore.setState(initial, true);
});

function pretendLoaded() {
  useAppStore.setState({
    positions: new Float32Array(3),
    density: new Float32Array(1),
    phase: new Float32Array(1),
    stateInfo: {} as never,
    plane: {} as never,
    radial: {} as never,
    levels: {} as never,
    spectrum: {} as never,
    status: "ready",
  });
}

describe("store transitions", () => {
  it("clamps quantum numbers and invalidates data", () => {
    pretendLoaded();
    useAppStore.getState().setQuantumNumbers(3, 5, -9);
    const s = useAppStore.getState();
    expect([s.n, s.l, s.m]).toEqual([3, 2, -2]);
    expect(s.positions).toBeNull();
    expect(s.plane).toBeNull();
    expect(s.radial).toBeNull();
    expect(s.status).toBe("idle");
  });

  it("system change invalidates data", () => {
    pretendLoaded();
    useAppStore.getState().setSystem("mu-h");
    const s = useAppStore.getState();
    expect(s.system).toBe("mu-h");
    expect(s.stateInfo).toBeNull();
    expect(s.spectrum).toBeNull();
  });

  it("real basis demotes phase color mode", () => {
    useAppStore.setState({ colorMode: "phase" });
    useAppStore.getState().setBasis("real");
    const s = useAppStore.getState();
    expect(s.basis).toBe("real");
    expect(s.colorMode).toBe("density");
  });

  it("complex basis keeps chosen color mode", () => {
    useAppStore.setState({ colorMode: "density" });
    useAppStore.getState().setBasis("complex");
    expect(useAppStore.getState().colorMode).toBe("density");
  });

  it("fine-structure toggle clears only energy-derived data", () => {
    pretendLoaded();
    useAppStore.getState().setFineStructure(true);
    const s = useAppStore.getState();
    expect(s.fineStructure).toBe(true);
    expect(s.stateInfo).toBeNull();
    expect(s.levels).toBeNull();
    expect(s.spectrum).toBeNull();
    expect(s.positions).not.toBeNull();
  });

  it("plane quantity toggle clears only the plane", () => {
    pretendLoaded();
    useAppStore.getState().setPlaneQuantity("psi");
    const s = useAppStore.getState();
    expect(s.planeQuantity).toBe("psi");
    expect(s.plane).toBeNull();
    expect(s.positions).not.toBeNull();
  });
});
```

- [x] **Step 2: Run to verify failure**

From `web\`: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test`
Expected: FAIL â€” `realOrbitalLabel` not exported; store lacks the new actions.

- [x] **Step 3: Implement.** Append to `web/src/lib/quantum.ts`:

```typescript
const CHEMISTRY_LABELS: Record<string, string> = {
  "0,0": "s",
  "1,0": "p_z",
  "1,1": "p_x",
  "1,-1": "p_y",
  "2,0": "d_z2",
  "2,1": "d_xz",
  "2,-1": "d_yz",
  "2,2": "d_x2-y2",
  "2,-2": "d_xy",
  "3,0": "f_z3",
  "3,1": "f_xz2",
  "3,-1": "f_yz2",
  "3,2": "f_z(x2-y2)",
  "3,-2": "f_xyz",
  "3,3": "f_x(x2-3y2)",
  "3,-3": "f_y(3x2-y2)",
};

/** Mirror of atomsim.analytic.angular.real_orbital_label â€” keep in lockstep. */
export function realOrbitalLabel(l: number, m: number): string {
  const hit = CHEMISTRY_LABELS[`${l},${m}`];
  if (hit) return hit;
  const letter = L_LETTERS[l] ?? `(l=${l})`;
  if (m === 0) return `${letter}(m=0)`;
  const kind = m > 0 ? "cos" : "sin";
  const signed = m > 0 ? `+${m}` : `${m}`;
  return `${letter}(m=${signed}, ${kind})`;
}
```

Replace `web/src/state/store.ts` in full:

```typescript
import { create } from "zustand";
import * as client from "../api/client";
import type { Basis, PlaneQuantity } from "../api/client";
import type {
  LevelsResponse,
  PlaneMeta,
  RadialResponse,
  SampleMeta,
  SpectrumResponse,
  StateResponse,
  SystemInfo,
} from "../api/types";
import { clampState } from "../lib/quantum";

export type SampleStatus = "idle" | "sampling" | "ready" | "error";
export type ViewMode = "cloud" | "plane" | "radial" | "levels" | "spectrum";
export type ColorMode = "solid" | "density" | "phase";

const N_MAX_DIAGRAM = 6;

interface AppState {
  n: number;
  l: number;
  m: number;
  system: string;
  basis: Basis;
  view: ViewMode;
  colorMode: ColorMode;
  fineStructure: boolean;
  count: number;
  systems: SystemInfo[];
  stateInfo: StateResponse | null;
  positions: Float32Array | null;
  density: Float32Array | null;
  phase: Float32Array | null;
  meta: SampleMeta | null;
  status: SampleStatus;
  progress: number;
  error: string | null;
  fps: number;
  planeQuantity: PlaneQuantity;
  plane: { meta: PlaneMeta; values: Float32Array } | null;
  planeStatus: SampleStatus;
  planeProgress: number;
  radial: RadialResponse | null;
  levels: LevelsResponse | null;
  spectrum: SpectrumResponse | null;
  setQuantumNumbers: (n: number, l: number, m: number) => void;
  setSystem: (system: string) => void;
  setBasis: (basis: Basis) => void;
  setView: (view: ViewMode) => void;
  setColorMode: (colorMode: ColorMode) => void;
  setFineStructure: (fineStructure: boolean) => void;
  setCount: (count: number) => void;
  setPlaneQuantity: (planeQuantity: PlaneQuantity) => void;
  setFps: (fps: number) => void;
  loadSystems: () => Promise<void>;
  loadStateInfo: () => Promise<void>;
  sample: () => Promise<void>;
  loadPlane: () => Promise<void>;
  loadRadial: () => Promise<void>;
  loadLevels: () => Promise<void>;
  loadSpectrum: () => Promise<void>;
}

/** Everything derived from (n, l, m, system, basis) â€” cleared when any of them changes. */
const INVALIDATED = {
  stateInfo: null,
  positions: null,
  density: null,
  phase: null,
  meta: null,
  status: "idle" as SampleStatus,
  progress: 0,
  error: null,
  plane: null,
  planeStatus: "idle" as SampleStatus,
  planeProgress: 0,
  radial: null,
  levels: null,
  spectrum: null,
};

export const useAppStore = create<AppState>((set, get) => ({
  n: 1,
  l: 0,
  m: 0,
  system: "h",
  basis: "complex",
  view: "cloud",
  colorMode: "solid",
  fineStructure: false,
  count: 100_000,
  systems: [],
  fps: 0,
  planeQuantity: "density",
  ...INVALIDATED,
  setQuantumNumbers: (n, l, m) => set({ ...clampState(n, l, m), ...INVALIDATED }),
  setSystem: (system) => set({ system, ...INVALIDATED }),
  setBasis: (basis) =>
    set((s) => ({
      basis,
      ...INVALIDATED,
      colorMode: basis === "real" && s.colorMode === "phase" ? "density" : s.colorMode,
    })),
  setView: (view) => set({ view }),
  setColorMode: (colorMode) => set({ colorMode }),
  setFineStructure: (fineStructure) =>
    set({ fineStructure, stateInfo: null, levels: null, spectrum: null }),
  setCount: (count) => set({ count }),
  setPlaneQuantity: (planeQuantity) =>
    set({ planeQuantity, plane: null, planeStatus: "idle", planeProgress: 0 }),
  setFps: (fps) => set({ fps }),
  loadSystems: async () => {
    set({ systems: (await client.getSystems()).systems });
  },
  loadStateInfo: async () => {
    const { n, l, m, system, fineStructure } = get();
    set({ stateInfo: await client.getState(n, l, m, system, fineStructure) });
  },
  sample: async () => {
    const { n, l, m, count, basis, system } = get();
    set({ status: "sampling", progress: 0, error: null });
    try {
      const job = await client.createSampleJob({ n, l, m, count, basis, system });
      await client.watchJob(job.id, (progress) => set({ progress }));
      const [meta, positions, density, phase] = await Promise.all([
        client.getJobMeta(job.id),
        client.getChannel(job.id, "positions"),
        client.getChannel(job.id, "density"),
        basis === "complex" ? client.getChannel(job.id, "phase") : Promise.resolve(null),
      ]);
      if (meta.kind !== "sample") throw new Error("expected sample-job meta");
      set({ meta, positions, density, phase, status: "ready", progress: 1 });
    } catch (err) {
      set({ status: "error", error: err instanceof Error ? err.message : String(err) });
    }
  },
  loadPlane: async () => {
    const { n, l, m, system, basis, planeQuantity } = get();
    set({ planeStatus: "sampling", planeProgress: 0, error: null });
    try {
      const job = await client.createPlaneJob({
        n,
        l,
        m,
        system,
        basis,
        quantity: planeQuantity,
      });
      await client.watchJob(job.id, (planeProgress) => set({ planeProgress }));
      const [meta, values] = await Promise.all([
        client.getJobMeta(job.id),
        client.getChannel(job.id),
      ]);
      if (meta.kind !== "plane") throw new Error("expected plane-job meta");
      set({ plane: { meta, values }, planeStatus: "ready", planeProgress: 1 });
    } catch (err) {
      set({
        planeStatus: "error",
        error: err instanceof Error ? err.message : String(err),
      });
    }
  },
  loadRadial: async () => {
    const { n, l, system } = get();
    set({ radial: await client.getRadial(n, l, system) });
  },
  loadLevels: async () => {
    const { system, fineStructure } = get();
    set({ levels: await client.getLevels(system, N_MAX_DIAGRAM, fineStructure) });
  },
  loadSpectrum: async () => {
    const { system, fineStructure } = get();
    set({ spectrum: await client.getSpectrum(system, N_MAX_DIAGRAM, fineStructure) });
  },
}));
```

- [x] **Step 4: Run web tests**

From `web\`: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test`
Expected: all pass. (`npm run build` is still expected to fail typecheck in `InfoPanel.tsx`/`Controls.tsx` until Task 8 rewrites them; anything OUTSIDE those components must be fixed now.)

- [x] **Step 5: Commit**

```bash
git add web/src/lib/quantum.ts web/src/lib/quantum.test.ts web/src/state/store.ts web/src/state/store.test.ts
git commit -m "feat: full app store (system/basis/view/color/fine-structure) + orbital labels"
```

---

### Task 8: Web â€” Controls, InfoPanel, App view switch, CloudView extraction

**Files:**
- Create: `web/src/components/CloudView.tsx`
- Modify: `web/src/components/Controls.tsx`, `web/src/components/InfoPanel.tsx`, `web/src/App.tsx`, `web/src/index.css`

**Interfaces:**
- Consumes: store (Task 7), `realOrbitalLabel`/`stateLabel`.
- Produces: `CloudView` (Task 9 extends it); `VIEW_OPTIONS` array in `Controls.tsx` that later view tasks append to; the `.center-col` layout row that Task 14's gallery fills.
- No new logic to unit-test (components only) â€” the gate is `npm test` staying green plus `npm run build` (tsc) passing again.

- [x] **Step 1: Extract the canvas.** Create `web/src/components/CloudView.tsx`:

```tsx
import { OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { useEffect } from "react";
import type * as THREE from "three";
import { useAppStore } from "../state/store";
import { PointCloud } from "./PointCloud";

function CameraRig({ distance }: { distance: number }) {
  const camera = useThree((s) => s.camera as THREE.PerspectiveCamera);
  useEffect(() => {
    camera.position.set(distance * 0.7, distance * 0.45, distance);
    // near/far track the system scale so muonic hydrogen (0.008 a0) and
    // Rydberg-ish n=6 (50+ a0) both frame correctly
    camera.near = distance / 100;
    camera.far = distance * 100;
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, distance]);
  return null;
}

export function CloudView() {
  const { n, positions, stateInfo } = useAppStore();
  const distance = stateInfo
    ? Math.max(6 * stateInfo.mean_radius.value, 1e-3)
    : 5 * n * n + 3;
  return (
    <div className="canvas-wrap">
      <Canvas camera={{ fov: 50 }}>
        <color attach="background" args={["#0a0e12"]} />
        <CameraRig distance={distance} />
        {positions && (
          <PointCloud positions={positions} pointSize={distance / 350} />
        )}
        <OrbitControls />
      </Canvas>
      {!positions && <p className="hint">Choose a state and press Sample</p>}
    </div>
  );
}
```

- [x] **Step 2: Rewrite `web/src/App.tsx`:**

```tsx
import { CloudView } from "./components/CloudView";
import { Controls } from "./components/Controls";
import { InfoPanel } from "./components/InfoPanel";
import { useAppStore } from "./state/store";

export default function App() {
  const view = useAppStore((s) => s.view);
  return (
    <div className="app-grid">
      <InfoPanel />
      <main className="center-col">{view === "cloud" && <CloudView />}</main>
      <Controls />
    </div>
  );
}
```

(Later tasks add their `{view === "plane" && <PlaneView />}` etc. lines and the `<GalleryStrip />` row â€” the switch grows one line per view task.)

- [x] **Step 3: Rewrite `web/src/components/Controls.tsx`:**

```tsx
import { useEffect } from "react";
import { useAppStore } from "../state/store";
import type { ColorMode, ViewMode } from "../state/store";

const N_CHOICES = [1, 2, 3, 4, 5, 6];
const COUNT_CHOICES = [10_000, 50_000, 100_000, 250_000];

// Later tasks append entries as their views land.
const VIEW_OPTIONS: { value: ViewMode; label: string }[] = [
  { value: "cloud", label: "3D point cloud" },
];

export function Controls() {
  const {
    n, l, m, count, status, progress, error, system, systems, basis, view,
    colorMode, fineStructure,
    setQuantumNumbers, setCount, sample, setSystem, setBasis, setView,
    setColorMode, setFineStructure, loadSystems,
  } = useAppStore();
  useEffect(() => {
    if (systems.length === 0) void loadSystems();
  }, [systems.length, loadSystems]);
  const lChoices = Array.from({ length: n }, (_, i) => i);
  const mChoices = Array.from({ length: 2 * l + 1 }, (_, i) => i - l);
  return (
    <aside className="panel">
      <h2>System</h2>
      <label>
        preset
        <select value={system} onChange={(e) => setSystem(e.target.value)}>
          {systems.length === 0 && <option value={system}>{system}</option>}
          {systems.map((s) => (
            <option key={s.key} value={s.key}>
              {s.name}
            </option>
          ))}
        </select>
      </label>
      <h2>View</h2>
      <label>
        mode
        <select value={view} onChange={(e) => setView(e.target.value as ViewMode)}>
          {VIEW_OPTIONS.map((v) => (
            <option key={v.value} value={v.value}>
              {v.label}
            </option>
          ))}
        </select>
      </label>
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
      <h2>Physics</h2>
      <div className="radio-row">
        <label className="radio">
          <input
            type="radio"
            checked={basis === "complex"}
            onChange={() => setBasis("complex")}
          />
          complex Y<sub>lm</sub>
        </label>
        <label className="radio">
          <input type="radio" checked={basis === "real"} onChange={() => setBasis("real")} />
          real S<sub>lm</sub>
        </label>
      </div>
      <label className="check">
        <input
          type="checkbox"
          checked={fineStructure}
          onChange={(e) => setFineStructure(e.target.checked)}
        />
        fine structure (Î±Â² perturbation)
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
      <label>
        colour
        <select
          value={colorMode}
          onChange={(e) => setColorMode(e.target.value as ColorMode)}
        >
          <option value="solid">solid (accent)</option>
          <option value="density">density (inferno)</option>
          <option value="phase" disabled={basis === "real"}>
            phase as hue (complex only)
          </option>
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
      {status === "error" && error && <p className="error">{error}</p>}
    </aside>
  );
}
```

- [x] **Step 4: Rewrite `web/src/components/InfoPanel.tsx`:**

```tsx
import { useEffect } from "react";
import { realOrbitalLabel, stateLabel } from "../lib/quantum";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

export function InfoPanel() {
  const {
    n, l, m, basis, system, fineStructure, stateInfo, meta, fps, view, loadStateInfo,
  } = useAppStore();
  useEffect(() => {
    void loadStateInfo();
  }, [n, l, m, system, fineStructure, loadStateInfo]);
  const sys = stateInfo?.system;
  return (
    <aside className="panel">
      <h1 className="brand">atomsim</h1>
      <h2>{sys ? `${sys.name} (Z = ${sys.z})` : "â€¦"}</h2>
      {sys && <p className="system-desc">{sys.description}</p>}
      <p className="state-label">
        {stateLabel(n, l, m)}
        {basis === "real" && (
          <span className="orbital-label"> Â· {realOrbitalLabel(l, m)}</span>
        )}
      </p>
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
          {fineStructure && stateInfo.levels.length > 0 && (
            <>
              <dt>
                Fine structure <Badge provenance={stateInfo.levels[0].shift.provenance} />
              </dt>
              <dd>
                {stateInfo.levels.map((lev) => (
                  <span key={lev.j} className="fs-level">
                    j = {lev.j}: {(lev.shift_ev.value * 1e6).toFixed(2)} ÂµeV
                    <br />
                  </span>
                ))}
              </dd>
            </>
          )}
          <dt>
            {"âŸ¨râŸ©"} <Badge provenance={stateInfo.mean_radius.provenance} />
          </dt>
          <dd>
            {stateInfo.mean_radius.value.toFixed(3)} a{"â‚€"} Â·{" "}
            {stateInfo.mean_radius_pm.value.toFixed(1)} pm
          </dd>
          <dt>
            |L| <Badge provenance={stateInfo.angular_momentum.provenance} />
          </dt>
          <dd>{stateInfo.angular_momentum.value.toFixed(3)} â„</dd>
          <dt>Nodes</dt>
          <dd>
            {stateInfo.radial_nodes} radial Â· {stateInfo.angular_nodes} angular
          </dd>
          {meta && (
            <>
              <dt>
                Sampled points <Badge provenance={meta.provenance} />
              </dt>
              <dd>{meta.count.toLocaleString()}</dd>
            </>
          )}
          {view === "cloud" && fps > 0 && (
            <>
              <dt>FPS (measured)</dt>
              <dd>{fps}</dd>
            </>
          )}
        </dl>
      )}
    </aside>
  );
}
```

- [x] **Step 5: Append to `web/src/index.css`:**

```css
.center-col {
  display: grid;
  grid-template-rows: 1fr auto;
  min-width: 0;
  min-height: 0;
}

.canvas-wrap {
  min-height: 0;
}

.system-desc {
  color: var(--muted);
  font-size: 0.8rem;
  margin: 0 0 0.5rem;
}

.orbital-label {
  color: var(--accent);
  font-size: 1.1rem;
}

.radio-row {
  display: flex;
  gap: 1rem;
}

.radio,
.check {
  justify-content: flex-start;
  gap: 0.4rem;
}

.fs-level {
  font-size: 0.95rem;
}

.hint-block {
  color: var(--muted);
  padding: 2rem;
}
```

(`.canvas-wrap` already has `position: relative` from M1 â€” keep that rule, just add `min-height`.)

- [x] **Step 6: Gates**

From `web\`: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test` â†’ all pass.
Then `npm run build` â†’ **tsc must pass again** (this closes the deliberate Task-6/7 window) and vite build succeeds.

- [x] **Step 7: Commit**

```bash
git add web/src/components/CloudView.tsx web/src/components/Controls.tsx web/src/components/InfoPanel.tsx web/src/App.tsx web/src/index.css
git commit -m "feat: three-panel UI with system/basis/view/fine-structure controls and rich readouts"
```

---

### Task 9: Web â€” cloud color modes, FPS meter, render-liberties disclosure

**Files:**
- Create: `web/src/lib/liberties.ts`, `web/src/lib/cloudColors.ts`, `web/src/lib/cloudColors.test.ts`, `web/src/components/Legend.tsx`
- Modify: `web/src/components/PointCloud.tsx`, `web/src/components/CloudView.tsx`, `web/src/index.css`

**Interfaces:**
- Consumes: `density`/`phase`/`colorMode`/`setFps` from the store; `INFERNO`, colormap helpers (Task 6).
- Produces: `buildCloudColors(mode, density, phase): Float32Array | null` (RGB floats 0â€“1, 3 per point, or null â†’ solid material colour); `RENDER_LIBERTIES` and `THUMBNAIL_LIBERTY` provenance constants (Task 14 uses the latter); `Legend` component.

- [x] **Step 1: Write the failing tests** â€” `web/src/lib/cloudColors.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { buildCloudColors } from "./cloudColors";
import { INFERNO } from "./luts";

describe("buildCloudColors", () => {
  const density = new Float32Array([0, 2, 1]);
  const phase = new Float32Array([0, Math.PI / 2]);

  it("solid mode returns null (material colour handles it)", () => {
    expect(buildCloudColors("solid", density, phase)).toBeNull();
  });

  it("density mode maps extremes through the inferno LUT", () => {
    const colors = buildCloudColors("density", density, null);
    expect(colors).not.toBeNull();
    expect(colors).toHaveLength(9);
    const top = INFERNO[255];
    expect(colors![3]).toBeCloseTo(top[0] / 255, 6);
    expect(colors![4]).toBeCloseTo(top[1] / 255, 6);
    expect(colors![5]).toBeCloseTo(top[2] / 255, 6);
    const bottom = INFERNO[0];
    expect(colors![0]).toBeCloseTo(bottom[0] / 255, 6);
  });

  it("phase mode needs the phase channel", () => {
    expect(buildCloudColors("phase", density, null)).toBeNull();
    expect(buildCloudColors("phase", null, phase)).toHaveLength(6);
  });

  it("density mode without data returns null", () => {
    expect(buildCloudColors("density", null, phase)).toBeNull();
  });
});
```

- [x] **Step 2: Run to verify failure** â€” from `web\`: `npm test` â†’ FAIL (`cloudColors.ts` missing).

- [x] **Step 3: Implement.** `web/src/lib/liberties.ts`:

```typescript
import type { Provenance } from "../api/types";

/** The frontend is the authority on its own rendering choices â€” disclosed, never hidden. */
export const RENDER_LIBERTIES: Provenance = {
  fidelity: "visual_liberty",
  method: "three.js point-sprite rendering of engine-sampled positions",
  assumptions: [
    "z quantization axis drawn screen-vertical (data stays xyz in bohr)",
    "point size, opacity and additive glow are presentation, not physics",
    "density colour brightness gamma-compressed: t = (rho/rho_max)^0.5",
  ],
  error_estimate: null,
  refinement: "positions, density and phase channels come from the engine unmodified",
};

export const THUMBNAIL_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "server-rendered inferno PNG of |psi|^2 on the y=0 plane (navigation aid)",
  assumptions: [
    "brightness gamma-compressed: t = (rho/rho_max)^0.5",
    "not a measurement surface: no axes, no scale",
  ],
  error_estimate: null,
  refinement: "open the 2D cross-section view for the labeled, scaled version",
};
```

`web/src/lib/cloudColors.ts`:

```typescript
import type { ColorMode } from "../state/store";
import { densityT, lutColor, maxOf, phaseColor } from "./colormap";
import { INFERNO } from "./luts";

/** Per-vertex RGB floats (0-1) for the cloud, or null for solid-colour mode. */
export function buildCloudColors(
  mode: ColorMode,
  density: Float32Array | null,
  phase: Float32Array | null,
): Float32Array | null {
  if (mode === "density" && density) {
    const vmax = maxOf(density);
    const out = new Float32Array(density.length * 3);
    for (let i = 0; i < density.length; i++) {
      const [r, g, b] = lutColor(INFERNO, densityT(density[i], vmax));
      out[3 * i] = r / 255;
      out[3 * i + 1] = g / 255;
      out[3 * i + 2] = b / 255;
    }
    return out;
  }
  if (mode === "phase" && phase) {
    const out = new Float32Array(phase.length * 3);
    for (let i = 0; i < phase.length; i++) {
      const [r, g, b] = phaseColor(phase[i]);
      out[3 * i] = r / 255;
      out[3 * i + 1] = g / 255;
      out[3 * i + 2] = b / 255;
    }
    return out;
  }
  return null;
}
```

`web/src/components/PointCloud.tsx` â€” replace:

```tsx
import { useMemo } from "react";
import * as THREE from "three";

interface Props {
  positions: Float32Array;
  pointSize: number;
  colors?: Float32Array | null;
}

export function PointCloud({ positions, pointSize, colors }: Props) {
  const useVertexColors = Boolean(colors && colors.length === positions.length);
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    if (colors && colors.length === positions.length) {
      g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    }
    return g;
  }, [positions, colors]);
  return (
    // VISUAL LIBERTY: physics z (the quantization axis) is rendered screen-vertical
    // (three.js +y) so |m|-dependent structure reads at a glance; data stays xyz in bohr.
    <points geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      {/* VISUAL LIBERTY: point size, colour mapping, glow are presentational choices,
          disclosed via the RENDER_LIBERTIES badge in the canvas overlay. */}
      <pointsMaterial
        size={pointSize}
        sizeAttenuation
        color={useVertexColors ? "#ffffff" : "#7cffb2"}
        vertexColors={useVertexColors}
        transparent
        opacity={useVertexColors ? 0.55 : 0.35}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}
```

`web/src/components/Legend.tsx`:

```tsx
import { DENSITY_GAMMA } from "../lib/colormap";
import { INFERNO } from "../lib/luts";
import type { ColorMode } from "../state/store";

function infernoGradient(): string {
  const stops = [0, 32, 64, 96, 128, 160, 192, 224, 255].map((i) => {
    const [r, g, b] = INFERNO[i];
    return `rgb(${r},${g},${b}) ${((i / 255) * 100).toFixed(1)}%`;
  });
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

export function Legend({ mode }: { mode: ColorMode }) {
  if (mode === "density") {
    return (
      <div className="legend">
        <div className="legend-bar" style={{ background: infernoGradient() }} />
        <span>
          |Ïˆ|Â² Â· brightness âˆ (Ï/Ï<sub>max</sub>)<sup>{DENSITY_GAMMA}</sup>
        </span>
      </div>
    );
  }
  if (mode === "phase") {
    const stops = [0, 60, 120, 180, 240, 300, 360]
      .map((deg) => `hsl(${deg}, 100%, 55%)`)
      .join(", ");
    return (
      <div className="legend">
        <div
          className="legend-bar"
          style={{ background: `linear-gradient(to right, ${stops})` }}
        />
        <span>arg Ïˆ: âˆ’Ï€ (left) â†’ +Ï€ (right)</span>
      </div>
    );
  }
  return null;
}
```

`web/src/components/CloudView.tsx` â€” replace with the color/FPS/overlay version:

```tsx
import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import type * as THREE from "three";
import { buildCloudColors } from "../lib/cloudColors";
import { RENDER_LIBERTIES } from "../lib/liberties";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { Legend } from "./Legend";
import { PointCloud } from "./PointCloud";

function CameraRig({ distance }: { distance: number }) {
  const camera = useThree((s) => s.camera as THREE.PerspectiveCamera);
  useEffect(() => {
    camera.position.set(distance * 0.7, distance * 0.45, distance);
    camera.near = distance / 100;
    camera.far = distance * 100;
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, distance]);
  return null;
}

function FpsMeter() {
  const setFps = useAppStore((s) => s.setFps);
  const acc = useRef({ frames: 0, t0: 0 });
  useFrame(() => {
    const a = acc.current;
    if (a.t0 === 0) a.t0 = performance.now();
    a.frames += 1;
    const now = performance.now();
    if (now - a.t0 >= 500) {
      setFps(Math.round((a.frames * 1000) / (now - a.t0)));
      a.frames = 0;
      a.t0 = now;
    }
  });
  return null;
}

export function CloudView() {
  const { n, positions, density, phase, colorMode, stateInfo } = useAppStore();
  const colors = useMemo(
    () => buildCloudColors(colorMode, density, phase),
    [colorMode, density, phase],
  );
  const distance = stateInfo
    ? Math.max(6 * stateInfo.mean_radius.value, 1e-3)
    : 5 * n * n + 3;
  return (
    <div className="canvas-wrap">
      <Canvas camera={{ fov: 50 }}>
        <color attach="background" args={["#0a0e12"]} />
        <CameraRig distance={distance} />
        <FpsMeter />
        {positions && (
          <PointCloud
            positions={positions}
            pointSize={distance / 350}
            colors={colors}
          />
        )}
        <OrbitControls />
      </Canvas>
      {!positions && <p className="hint">Choose a state and press Sample</p>}
      <div className="canvas-overlay">
        <Badge provenance={RENDER_LIBERTIES} />
        <Legend mode={colorMode} />
      </div>
    </div>
  );
}
```

Append to `web/src/index.css`:

```css
.canvas-overlay {
  position: absolute;
  left: 1rem;
  bottom: 1rem;
  display: flex;
  gap: 0.75rem;
  align-items: center;
}

.legend {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--muted);
  font-size: 0.7rem;
}

.legend-bar {
  width: 120px;
  height: 10px;
  border-radius: 5px;
  border: 1px solid var(--edge);
}
```

- [x] **Step 4: Gates** â€” from `web\`: `npm test` all pass; `npm run build` passes.

- [x] **Step 5: Commit**

```bash
git add web/src/lib/liberties.ts web/src/lib/cloudColors.ts web/src/lib/cloudColors.test.ts web/src/components/Legend.tsx web/src/components/PointCloud.tsx web/src/components/CloudView.tsx web/src/index.css
git commit -m "feat: density-colormap and phase-as-hue cloud modes with disclosed render liberties"
```

---

### Task 10: Web â€” 2D cross-section view

**Files:**
- Create: `web/src/lib/rasterize.ts`, `web/src/lib/rasterize.test.ts`, `web/src/components/PlaneView.tsx`
- Modify: `web/src/App.tsx`, `web/src/components/Controls.tsx` (add view option), `web/src/index.css`

**Interfaces:**
- Consumes: store plane state/actions (Task 7), LUTs + colormap helpers (Task 6).
- Produces: `rasterize(values: Float32Array, resolution: number, quantity: "density" | "psi"): Uint8ClampedArray` â€” RGBA pixels, canvas row 0 = top = +z (grid row order is z-ascending, so it flips rows).

- [x] **Step 1: Write the failing tests** â€” `web/src/lib/rasterize.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { INFERNO, RDBU_R } from "./luts";
import { rasterize } from "./rasterize";

describe("rasterize", () => {
  it("maps density extremes through inferno and flips rows (+z up)", () => {
    // grid row 0 (z = -he): [0, 0.5]; grid row 1 (z = +he): [1, 2]
    const values = new Float32Array([0, 0.5, 1, 2]);
    const px = rasterize(values, 2, "density");
    expect(px).toHaveLength(16);
    // canvas pixel (row 0, col 1) = grid (row 1, col 1) = max -> INFERNO[255]
    const top = INFERNO[255];
    expect([px[4], px[5], px[6], px[7]]).toEqual([top[0], top[1], top[2], 255]);
    // canvas pixel (row 1, col 0) = grid (row 0, col 0) = 0 -> INFERNO[0]
    const zero = INFERNO[0];
    expect([px[8], px[9], px[10], px[11]]).toEqual([zero[0], zero[1], zero[2], 255]);
  });

  it("maps signed psi through diverging RdBu_r with zero at the midpoint", () => {
    // grid row 0: [-2, 0]; grid row 1: [2, 1]
    const values = new Float32Array([-2, 0, 2, 1]);
    const px = rasterize(values, 2, "psi");
    const pos = RDBU_R[255];
    expect([px[0], px[1], px[2]]).toEqual([pos[0], pos[1], pos[2]]); // +2 (top-left)
    const neg = RDBU_R[0];
    expect([px[8], px[9], px[10]]).toEqual([neg[0], neg[1], neg[2]]); // -2
    const mid = RDBU_R[128];
    expect([px[12], px[13], px[14]]).toEqual([mid[0], mid[1], mid[2]]); // 0
  });
});
```

- [x] **Step 2: Run to verify failure** â€” `npm test` â†’ FAIL (`rasterize.ts` missing).

- [x] **Step 3: Implement.** `web/src/lib/rasterize.ts`:

```typescript
import { densityT, lutColor, maxAbs, maxOf, signedT } from "./colormap";
import { INFERNO, RDBU_R } from "./luts";

/**
 * PlaneGrid float32 values (row i = z ascending) -> RGBA pixels (row 0 = top).
 * density: inferno with the disclosed gamma compression (VISUAL LIBERTY).
 * psi: diverging RdBu_r, LINEAR in psi, zero at the midpoint â€” signed structure
 * is the honest point of that mode, so no gamma is applied.
 */
export function rasterize(
  values: Float32Array,
  resolution: number,
  quantity: "density" | "psi",
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(resolution * resolution * 4);
  const vmax = quantity === "density" ? maxOf(values) : maxAbs(values);
  for (let row = 0; row < resolution; row++) {
    const src = resolution - 1 - row; // canvas row 0 is the top (+z)
    for (let col = 0; col < resolution; col++) {
      const v = values[src * resolution + col];
      const [r, g, b] =
        quantity === "density"
          ? lutColor(INFERNO, densityT(v, vmax))
          : lutColor(RDBU_R, signedT(v, vmax));
      const o = 4 * (row * resolution + col);
      out[o] = r;
      out[o + 1] = g;
      out[o + 2] = b;
      out[o + 3] = 255;
    }
  }
  return out;
}
```

`web/src/components/PlaneView.tsx`:

```tsx
import { useEffect, useRef } from "react";
import { rasterize } from "../lib/rasterize";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

export function PlaneView() {
  const {
    n, l, m, system, basis, planeQuantity, plane, planeStatus, planeProgress,
    error, loadPlane, setPlaneQuantity,
  } = useAppStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!plane && planeStatus === "idle") void loadPlane();
  }, [plane, planeStatus, loadPlane, n, l, m, system, basis, planeQuantity]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !plane) return;
    const { resolution, quantity } = plane.meta;
    canvas.width = resolution;
    canvas.height = resolution;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(
      new ImageData(rasterize(plane.values, resolution, quantity), resolution, resolution),
      0,
      0,
    );
  }, [plane]);

  return (
    <div className="view-wrap">
      <div className="view-header">
        <span className="plot-title">
          {plane ? `${plane.meta.label} [${plane.meta.unit}]` : "2D cross-section (y = 0 plane)"}
        </span>
        {plane && <Badge provenance={plane.meta.provenance} />}
        <div className="seg">
          <button
            type="button"
            className={planeQuantity === "density" ? "seg-on" : ""}
            onClick={() => setPlaneQuantity("density")}
          >
            |Ïˆ|Â²
          </button>
          <button
            type="button"
            className={planeQuantity === "psi" ? "seg-on" : ""}
            onClick={() => setPlaneQuantity("psi")}
          >
            Ïˆ (signed)
          </button>
        </div>
      </div>
      <div className="plane-frame">
        <canvas ref={canvasRef} className="plane-canvas" />
        {planeStatus === "sampling" && (
          <p className="hint">computingâ€¦ {(planeProgress * 100).toFixed(0)}%</p>
        )}
        {planeStatus === "error" && error && <p className="error">{error}</p>}
      </div>
      {plane && (
        <p className="caption">
          x, z âˆˆ [âˆ’{plane.meta.half_extent.toFixed(1)}, +
          {plane.meta.half_extent.toFixed(1)}] bohr; z vertical (quantization axis).{" "}
          {plane.meta.quantity === "density"
            ? "Inferno brightness is Î³-compressed, exponent 0.5 (VISUAL LIBERTY â€” reveals faint lobes)."
            : "Diverging RdBu, linear in Ïˆ: blue < 0 < red. Ïˆ is real on this plane (e^{imÏ†} = Â±1)."}
        </p>
      )}
    </div>
  );
}
```

In `web/src/components/Controls.tsx`, append to `VIEW_OPTIONS`:

```typescript
  { value: "plane", label: "2D cross-section" },
```

In `web/src/App.tsx`, extend the switch:

```tsx
        {view === "cloud" && <CloudView />}
        {view === "plane" && <PlaneView />}
```

(plus `import { PlaneView } from "./components/PlaneView";`)

Append to `web/src/index.css`:

```css
.view-wrap {
  padding: 1.25rem 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  min-height: 0;
  overflow-y: auto;
}

.view-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.plot-title {
  font-size: 0.95rem;
}

.plane-frame {
  position: relative;
  display: grid;
  place-items: center;
  min-height: 0;
}

.plane-canvas {
  width: 100%;
  max-width: min(100%, 70vh);
  aspect-ratio: 1;
  border: 1px solid var(--edge);
  border-radius: 6px;
  background: #000;
}

.caption {
  color: var(--muted);
  font-size: 0.75rem;
  margin: 0;
}

.seg {
  margin-left: auto;
  display: flex;
  border: 1px solid var(--edge);
  border-radius: 6px;
  overflow: hidden;
}

.seg button {
  background: transparent;
  color: var(--muted);
  border: 0;
  padding: 0.3rem 0.7rem;
  cursor: pointer;
  font-size: 0.85rem;
}

.seg .seg-on {
  background: var(--edge);
  color: var(--text);
}
```

- [x] **Step 4: Gates** â€” `npm test` all pass; `npm run build` passes.

- [x] **Step 5: Commit**

```bash
git add web/src/lib/rasterize.ts web/src/lib/rasterize.test.ts web/src/components/PlaneView.tsx web/src/App.tsx web/src/components/Controls.tsx web/src/index.css
git commit -m "feat: 2D cross-section view with honest psi-vs-density labeling"
```

---

### Task 11: Web â€” radial plots view (d3-scale SVG)

**Files:**
- Create: `web/src/lib/plot.ts`, `web/src/lib/plot.test.ts`, `web/src/components/RadialView.tsx`
- Modify: `web/package.json` (d3-scale), `web/src/App.tsx`, `web/src/components/Controls.tsx`, `web/src/index.css`

**Interfaces:**
- Consumes: `radial: RadialResponse` + `loadRadial` from the store; `stateInfo.mean_radius` for the âŸ¨râŸ© marker.
- Produces: `linePath(xs, ys, xScale, yScale): string` (SVG path).

- [x] **Step 1: Install d3-scale.** From `web\`:

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm install d3-scale`
Then: `... npm install -D @types/d3-scale`
Expected: both appear in `web/package.json`; `package-lock.json` updated (commit it â€” CI runs `npm ci`).

- [x] **Step 2: Write the failing tests** â€” `web/src/lib/plot.test.ts`:

```typescript
import { scaleLinear } from "d3-scale";
import { describe, expect, it } from "vitest";
import { linePath } from "./plot";

describe("linePath", () => {
  it("builds an SVG path under identity scales", () => {
    const s = scaleLinear([0, 1], [0, 1]);
    expect(linePath([0, 0.5, 1], [0, 1, 0], s, s)).toBe(
      "M0.00,0.00L0.50,1.00L1.00,0.00",
    );
  });
  it("empty input gives an empty path", () => {
    const s = scaleLinear([0, 1], [0, 1]);
    expect(linePath([], [], s, s)).toBe("");
  });
});
```

Run `npm test` â†’ FAIL (`plot.ts` missing).

- [x] **Step 3: Implement.** `web/src/lib/plot.ts`:

```typescript
import type { ScaleLinear } from "d3-scale";

/** SVG path through (xs[i], ys[i]) under the given scales. */
export function linePath(
  xs: readonly number[],
  ys: readonly number[],
  x: ScaleLinear<number, number>,
  y: ScaleLinear<number, number>,
): string {
  return xs
    .map((v, i) => `${i === 0 ? "M" : "L"}${x(v).toFixed(2)},${y(ys[i]).toFixed(2)}`)
    .join("");
}
```

`web/src/components/RadialView.tsx`:

```tsx
import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import type { FieldData, Quantity } from "../api/types";
import { linePath } from "../lib/plot";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 640;
const H = 240;
const M = { top: 16, right: 16, bottom: 34, left: 56 };

function FieldPlot({ field, marker }: { field: FieldData; marker?: Quantity }) {
  const x = scaleLinear(
    [0, field.grid[field.grid.length - 1]],
    [M.left, W - M.right],
  );
  const lo = Math.min(0, ...field.values);
  const hi = Math.max(...field.values);
  const y = scaleLinear([lo, hi], [H - M.bottom, M.top]).nice();
  return (
    <figure className="plot">
      <figcaption>
        {field.label} [{field.unit}] <Badge provenance={field.provenance} />
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img">
        <line
          x1={M.left} y1={H - M.bottom} x2={W - M.right} y2={H - M.bottom}
          className="axis"
        />
        <line x1={M.left} y1={M.top} x2={M.left} y2={H - M.bottom} className="axis" />
        {x.ticks(6).map((t) => (
          <g key={t} transform={`translate(${x(t)},${H - M.bottom})`}>
            <line y2="5" className="axis" />
            <text y="18" textAnchor="middle" className="tick">
              {t}
            </text>
          </g>
        ))}
        {y.ticks(4).map((t) => (
          <g key={t} transform={`translate(${M.left},${y(t)})`}>
            <line x2="-5" className="axis" />
            <text x="-8" dy="0.32em" textAnchor="end" className="tick">
              {t.toPrecision(2)}
            </text>
          </g>
        ))}
        {lo < 0 && (
          <line x1={M.left} x2={W - M.right} y1={y(0)} y2={y(0)} className="zero" />
        )}
        <path d={linePath(field.grid, field.values, x, y)} className="curve" />
        {marker && (
          <g>
            <line
              x1={x(marker.value)} x2={x(marker.value)} y1={M.top} y2={H - M.bottom}
              className="marker"
            />
            <text x={x(marker.value) + 4} y={M.top + 12} className="tick">
              âŸ¨râŸ©
            </text>
          </g>
        )}
        <text
          x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle" className="tick"
        >
          r [{field.grid_unit}]
        </text>
      </svg>
    </figure>
  );
}

export function RadialView() {
  const { n, l, system, radial, stateInfo, loadRadial } = useAppStore();
  useEffect(() => {
    void loadRadial();
  }, [n, l, system, loadRadial]);
  if (!radial) return <p className="hint-block">loading radial functionsâ€¦</p>;
  return (
    <div className="view-wrap">
      <FieldPlot field={radial.r_wavefunction} />
      <FieldPlot
        field={radial.radial_probability}
        marker={stateInfo?.mean_radius ?? undefined}
      />
    </div>
  );
}
```

`Controls.tsx` `VIEW_OPTIONS` append: `{ value: "radial", label: "Radial R(r), P(r)" },`
`App.tsx` switch: `{view === "radial" && <RadialView />}` + import.

Append to `web/src/index.css`:

```css
.plot {
  margin: 0;
}

.plot svg {
  width: 100%;
  height: auto;
  background: var(--panel);
  border: 1px solid var(--edge);
  border-radius: 6px;
}

.plot figcaption {
  font-size: 0.85rem;
  color: var(--muted);
  margin-bottom: 0.4rem;
}

.axis {
  stroke: var(--edge);
}

.tick {
  fill: var(--muted);
  font-size: 11px;
}

.curve {
  fill: none;
  stroke: var(--accent);
  stroke-width: 1.5;
}

.zero {
  stroke: var(--muted);
  stroke-dasharray: 2 4;
}

.marker {
  stroke: #fbbf24;
  stroke-dasharray: 4 4;
}
```

- [x] **Step 4: Gates** â€” `npm test` all pass; `npm run build` passes.

- [x] **Step 5: Commit**

```bash
git add web/package.json web/package-lock.json web/src/lib/plot.ts web/src/lib/plot.test.ts web/src/components/RadialView.tsx web/src/App.tsx web/src/components/Controls.tsx web/src/index.css
git commit -m "feat: radial R(r)/P(r) plots with mean-radius marker"
```

---

### Task 12: Web â€” energy-level diagram with fine-structure zoom

**Files:**
- Create: `web/src/lib/levels.ts`, `web/src/lib/levels.test.ts`, `web/src/components/LevelsView.tsx`
- Modify: `web/src/App.tsx`, `web/src/components/Controls.tsx`, `web/src/index.css`

**Interfaces:**
- Consumes: `levels: LevelsResponse`, `spectrum: SpectrumResponse` + loaders from the store (arrows reuse the engine's selection-rule-filtered lines â€” the frontend never re-derives selection rules).
- Produces: `arrowsFor(lines, n, l): SpectralLineInfo[]`.

- [x] **Step 1: Write the failing tests** â€” `web/src/lib/levels.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import type { SpectralLineInfo } from "../api/types";
import { arrowsFor } from "./levels";

function line(nu: number, lu: number, nl: number): SpectralLineInfo {
  return {
    n_upper: nu,
    l_upper: lu,
    j_upper: null,
    n_lower: nl,
    l_lower: 0,
    j_lower: null,
    energy_ev: {} as never,
    wavelength_nm: {} as never,
  };
}

describe("arrowsFor", () => {
  it("keeps only transitions out of the selected state", () => {
    const lines = [line(3, 1, 1), line(3, 1, 2), line(3, 2, 2), line(2, 1, 1)];
    expect(arrowsFor(lines, 3, 1)).toHaveLength(2);
    expect(arrowsFor(lines, 2, 1)).toHaveLength(1);
    expect(arrowsFor(lines, 1, 0)).toHaveLength(0);
  });
});
```

Run `npm test` â†’ FAIL (`levels.ts` missing).

- [x] **Step 2: Implement.** `web/src/lib/levels.ts`:

```typescript
import type { SpectralLineInfo } from "../api/types";

/** Downward transitions out of the selected (n, l) â€” already selection-rule
 *  filtered by the engine; the frontend only picks the relevant subset. */
export function arrowsFor(
  lines: readonly SpectralLineInfo[],
  n: number,
  l: number,
): SpectralLineInfo[] {
  return lines.filter((ln) => ln.n_upper === n && ln.l_upper === l);
}
```

`web/src/components/LevelsView.tsx`:

```tsx
import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import { arrowsFor } from "../lib/levels";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 680;
const H = 460;

export function LevelsView() {
  const {
    n, l, system, fineStructure, levels, spectrum, loadLevels, loadSpectrum,
  } = useAppStore();
  useEffect(() => {
    void loadLevels();
    void loadSpectrum();
  }, [system, fineStructure, loadLevels, loadSpectrum]);
  if (!levels) return <p className="hint-block">loading levelsâ€¦</p>;

  const eMin = levels.gross[0].energy_ev.value;
  const y = scaleLinear([eMin, 0], [H - 40, 24]);
  const rungX1 = 70;
  const rungX2 = 320;
  const arrows = spectrum ? arrowsFor(spectrum.lines, n, l) : [];
  const grossE = new Map(levels.gross.map((g) => [g.n, g.energy_ev.value]));
  const fineForN = levels.fine?.filter((f) => f.n === n) ?? [];

  return (
    <div className="view-wrap">
      <div className="view-header">
        <span className="plot-title">
          Energy levels E_n [eV]{" "}
          <Badge provenance={levels.gross[0].energy.provenance} />
        </span>
        {fineStructure && fineForN.length > 0 && (
          <span className="plot-title">
            Â· fine structure of n={n}{" "}
            <Badge provenance={fineForN[0].shift.provenance} />
          </span>
        )}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" className="levels-svg">
        {levels.gross.map((g) => (
          <g key={g.n}>
            <line
              x1={rungX1} x2={rungX2}
              y1={y(g.energy_ev.value)} y2={y(g.energy_ev.value)}
              className={g.n === n ? "rung rung-active" : "rung"}
            />
            <text
              x={rungX1 - 8} y={y(g.energy_ev.value)} dy="0.32em"
              textAnchor="end" className="tick"
            >
              n={g.n}
            </text>
            <text x={rungX2 + 8} y={y(g.energy_ev.value)} dy="0.32em" className="tick">
              {g.energy_ev.value.toFixed(2)} eV Â· 2nÂ²={g.degeneracy}
            </text>
          </g>
        ))}
        {arrows.map((a, i) => {
          if (!grossE.has(a.n_upper) || !grossE.has(a.n_lower)) return null;
          const ax = rungX1 + 30 + i * 26;
          const yTop = y(grossE.get(a.n_upper) ?? 0);
          const yBot = y(grossE.get(a.n_lower) ?? 0);
          return (
            <g key={`${a.n_lower}-${a.l_lower}-${i}`} className="arrow">
              <line x1={ax} x2={ax} y1={yTop} y2={yBot - 6} />
              <path d={`M${ax - 4},${yBot - 8} L${ax + 4},${yBot - 8} L${ax},${yBot} Z`} />
              <text x={ax + 4} y={(yTop + yBot) / 2} className="tick">
                {a.wavelength_nm.value.toFixed(0)} nm
              </text>
            </g>
          );
        })}
        {fineStructure && fineForN.length > 0 &&
          (() => {
            const shifts = fineForN.map((f) => f.shift_ev.value);
            const lo = Math.min(...shifts);
            const hi = Math.max(...shifts);
            const pad = (hi - lo || 1e-9) * 0.15;
            const yz = scaleLinear([lo - pad, hi + pad], [H - 60, 48]);
            const zx1 = 470;
            const zx2 = 590;
            return (
              <g>
                <text x={(zx1 + zx2) / 2} y={26} textAnchor="middle" className="tick">
                  n={n} shifts [ÂµeV] â€” zoomed, APPROXIMATION
                </text>
                {fineForN.map((f, idx) => (
                  <g key={`${f.l}-${f.j}`}>
                    <line
                      x1={zx1} x2={zx2}
                      y1={yz(f.shift_ev.value)} y2={yz(f.shift_ev.value)}
                      className={f.l === l ? "rung rung-active" : "rung"}
                    />
                    <text
                      x={zx2 + 6}
                      y={yz(f.shift_ev.value) + (idx % 2 ? 12 : 0)}
                      dy="0.32em" className="tick"
                    >
                      l={f.l}, j={f.j} Â· {(f.shift_ev.value * 1e6).toFixed(1)}
                    </text>
                  </g>
                ))}
              </g>
            );
          })()}
      </svg>
      <p className="caption">
        Gross levels are reduced-mass exact. The right column magnifies the Î±Â²
        fine-structure shifts of the selected n â€” the two scales differ by ~10âµ and are
        labeled, never blended. States with equal j coincide at this order (e.g. 2sâ‚/â‚‚ and
        2pâ‚/â‚‚ â€” the Lamb shift is beyond Î±Â² and honestly absent here).
      </p>
    </div>
  );
}
```

`Controls.tsx` `VIEW_OPTIONS` append: `{ value: "levels", label: "Energy levels" },`
`App.tsx` switch: `{view === "levels" && <LevelsView />}` + import.

Append to `web/src/index.css`:

```css
.levels-svg {
  width: 100%;
  height: auto;
  background: var(--panel);
  border: 1px solid var(--edge);
  border-radius: 6px;
}

.rung {
  stroke: var(--muted);
  stroke-width: 1.5;
}

.rung-active {
  stroke: var(--accent);
  stroke-width: 2.5;
}

.arrow line {
  stroke: #fbbf24;
}

.arrow path {
  fill: #fbbf24;
}

.arrow text {
  fill: #fbbf24;
  font-size: 10px;
}
```

- [x] **Step 3: Gates** â€” `npm test` all pass; `npm run build` passes.

- [x] **Step 4: Commit**

```bash
git add web/src/lib/levels.ts web/src/lib/levels.test.ts web/src/components/LevelsView.tsx web/src/App.tsx web/src/components/Controls.tsx web/src/index.css
git commit -m "feat: energy-level diagram with degeneracies, transition arrows, fine-structure zoom"
```

---

### Task 13: Web â€” spectrum view (computed vs NIST + residuals)

**Files:**
- Create: `web/src/lib/spectrum.ts`, `web/src/lib/spectrum.test.ts`, `web/src/components/SpectrumView.tsx`
- Modify: `web/src/App.tsx`, `web/src/components/Controls.tsx`, `web/src/index.css`

**Interfaces:**
- Consumes: `spectrum: SpectrumResponse` (+ `loadSpectrum`).
- Produces: `seriesName(nLower)`, `seriesColor(nLower)`.

- [x] **Step 1: Write the failing tests** â€” `web/src/lib/spectrum.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { seriesColor, seriesName } from "./spectrum";

describe("series", () => {
  it("names the classic series", () => {
    expect(seriesName(1)).toBe("Lyman");
    expect(seriesName(2)).toBe("Balmer");
    expect(seriesName(6)).toBe("Humphreys");
    expect(seriesName(7)).toBe("to n'=7");
  });
  it("colors are stable hex strings with a fallback", () => {
    expect(seriesColor(1)).toMatch(/^#[0-9a-f]{6}$/);
    expect(seriesColor(2)).not.toBe(seriesColor(1));
    expect(seriesColor(99)).toBe("#8b98a5");
  });
});
```

Run `npm test` â†’ FAIL (`spectrum.ts` missing).

- [x] **Step 2: Implement.** `web/src/lib/spectrum.ts`:

```typescript
const SERIES_NAMES = ["", "Lyman", "Balmer", "Paschen", "Brackett", "Pfund", "Humphreys"];
const SERIES_COLORS = ["", "#a78bfa", "#7cffb2", "#fbbf24", "#60a5fa", "#f472b6", "#f87171"];

export function seriesName(nLower: number): string {
  return SERIES_NAMES[nLower] ?? `to n'=${nLower}`;
}

export function seriesColor(nLower: number): string {
  return SERIES_COLORS[nLower] ?? "#8b98a5";
}
```

`web/src/components/SpectrumView.tsx`:

```tsx
import { scaleLinear, scaleLog } from "d3-scale";
import { useEffect } from "react";
import { seriesColor, seriesName } from "../lib/spectrum";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 680;
const LINES_H = 190;
const RES_H = 150;
const M = { left: 56, right: 16 };

export function SpectrumView() {
  const { system, fineStructure, spectrum, loadSpectrum } = useAppStore();
  useEffect(() => {
    void loadSpectrum();
  }, [system, fineStructure, loadSpectrum]);
  if (!spectrum) return <p className="hint-block">loading spectrumâ€¦</p>;

  const wls = spectrum.lines.map((ln) => ln.wavelength_nm.value);
  const x = scaleLog(
    [Math.min(...wls) * 0.9, Math.max(...wls) * 1.1],
    [M.left, W - M.right],
  );
  const nLowers = [...new Set(spectrum.lines.map((ln) => ln.n_lower))].sort(
    (a, b) => a - b,
  );
  const tol = spectrum.tolerance_relative;
  const comp = spectrum.comparison;
  const yRes = tol ? scaleLinear([-3 * tol, 3 * tol], [RES_H - 30, 14]) : null;
  const clampY = (v: number) => Math.min(Math.max(v, 14), RES_H - 30);

  return (
    <div className="view-wrap">
      <div className="view-header">
        <span className="plot-title">
          Emission lines Î» [nm]{" "}
          <Badge provenance={spectrum.lines[0].wavelength_nm.provenance} />
        </span>
        <span className="legend-inline">
          {nLowers.map((nl) => (
            <span key={nl} style={{ color: seriesColor(nl) }}>
              â, Ž{seriesName(nl)}
            </span>
          ))}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${LINES_H}`} role="img" className="levels-svg">
        <line
          x1={M.left} x2={W - M.right} y1={LINES_H - 24} y2={LINES_H - 24}
          className="axis"
        />
        {x.ticks(8).map((t) => (
          <g key={t} transform={`translate(${x(t)},${LINES_H - 24})`}>
            <line y2="5" className="axis" />
            <text y="17" textAnchor="middle" className="tick">
              {t}
            </text>
          </g>
        ))}
        {spectrum.lines.map((ln, i) => (
          <line
            key={i}
            x1={x(ln.wavelength_nm.value)} x2={x(ln.wavelength_nm.value)}
            y1={28} y2={LINES_H - 30}
            stroke={seriesColor(ln.n_lower)} strokeWidth={1.5} opacity={0.9}
          />
        ))}
        {comp?.map((c, i) => (
          <circle
            key={i} cx={x(c.reference_nm)} cy={LINES_H - 27} r={2.5}
            className={c.within_tolerance ? "ref-ok" : "ref-bad"}
          />
        ))}
        <text x={W - M.right} y={16} textAnchor="end" className="tick">
          computed lines (bars) Â· NIST reference (dots on axis; log-Î»)
        </text>
      </svg>
      {comp && yRes && tol && (
        <svg viewBox={`0 0 ${W} ${RES_H}`} role="img" className="levels-svg">
          <rect
            x={M.left} width={W - M.left - M.right}
            y={yRes(tol)} height={yRes(-tol) - yRes(tol)} className="tol-band"
          />
          <line x1={M.left} x2={W - M.right} y1={yRes(0)} y2={yRes(0)} className="zero" />
          {comp.map((c, i) => (
            <circle
              key={i} cx={x(c.reference_nm)} cy={clampY(yRes(c.relative_error))} r={3}
              className={c.within_tolerance ? "ref-ok" : "ref-bad"}
            />
          ))}
          <text x={M.left} y={12} className="tick">
            (Î»_computed âˆ’ Î»_NIST)/Î»_NIST â€” shaded band = stated tolerance Â±{tol.toExponential(0)}
          </text>
        </svg>
      )}
      <p className="caption">
        {spectrum.reference_citation
          ? `Reference: ${spectrum.reference_citation}`
          : "No vendored NIST reference for this system â€” computed lines only, honestly unchecked."}
      </p>
    </div>
  );
}
```

`Controls.tsx` `VIEW_OPTIONS` append: `{ value: "spectrum", label: "Spectrum vs NIST" },`
`App.tsx` switch: `{view === "spectrum" && <SpectrumView />}` + import.

Append to `web/src/index.css`:

```css
.legend-inline {
  display: flex;
  gap: 0.8rem;
  font-size: 0.75rem;
  margin-left: auto;
}

.ref-ok {
  fill: #4ade80;
}

.ref-bad {
  fill: #f87171;
}

.tol-band {
  fill: rgb(124 255 178 / 0.08);
}
```

- [x] **Step 3: Gates** â€” `npm test` all pass; `npm run build` passes.

- [x] **Step 4: Commit**

```bash
git add web/src/lib/spectrum.ts web/src/lib/spectrum.test.ts web/src/components/SpectrumView.tsx web/src/App.tsx web/src/components/Controls.tsx web/src/index.css
git commit -m "feat: spectrum view - computed lines vs NIST overlay with residual band"
```

---

### Task 14: Web â€” thumbnail gallery strip

**Files:**
- Create: `web/src/lib/gallery.ts`, `web/src/lib/gallery.test.ts`, `web/src/components/GalleryStrip.tsx`
- Modify: `web/src/App.tsx`, `web/src/index.css`

**Interfaces:**
- Consumes: `thumbnailUrl` (Task 6), `THUMBNAIL_LIBERTY` (Task 9), store.
- Produces: `galleryStates(n): { n: number; l: number; m: number }[]` â€” all (l, m) of the shell, l ascending then m ascending.

- [x] **Step 1: Write the failing tests** â€” `web/src/lib/gallery.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { galleryStates } from "./gallery";

describe("galleryStates", () => {
  it("enumerates all (l, m) of the shell in order", () => {
    expect(galleryStates(1)).toEqual([{ n: 1, l: 0, m: 0 }]);
    expect(galleryStates(2)).toHaveLength(4);
    expect(galleryStates(3)).toHaveLength(9);
    expect(galleryStates(2)[1]).toEqual({ n: 2, l: 1, m: -1 });
  });
});
```

Run `npm test` â†’ FAIL (`gallery.ts` missing).

- [x] **Step 2: Implement.** `web/src/lib/gallery.ts`:

```typescript
export interface StateRef {
  n: number;
  l: number;
  m: number;
}

/** All (l, m) states of shell n â€” the gallery row. nÂ² entries. */
export function galleryStates(n: number): StateRef[] {
  const out: StateRef[] = [];
  for (let l = 0; l < n; l++) {
    for (let m = -l; m <= l; m++) out.push({ n, l, m });
  }
  return out;
}
```

`web/src/components/GalleryStrip.tsx`:

```tsx
import { thumbnailUrl } from "../api/client";
import type { Basis } from "../api/client";
import { galleryStates } from "../lib/gallery";
import { THUMBNAIL_LIBERTY } from "../lib/liberties";
import { stateLabel } from "../lib/quantum";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

export function GalleryStrip() {
  const { n, l, m, system, basis, setQuantumNumbers } = useAppStore();
  return (
    <div className="gallery">
      <div className="gallery-head">
        <span>n = {n} states</span>
        <Badge provenance={THUMBNAIL_LIBERTY} />
      </div>
      <div className="gallery-scroll">
        {galleryStates(n).map((s) => {
          const active = s.l === l && s.m === m;
          return (
            <button
              key={`${s.l},${s.m}`}
              type="button"
              className={active ? "thumb thumb-active" : "thumb"}
              title={stateLabel(s.n, s.l, s.m)}
              onClick={() => setQuantumNumbers(s.n, s.l, s.m)}
            >
              <img
                src={thumbnailUrl(s.n, s.l, s.m, system, basis as Basis, 96)}
                alt={stateLabel(s.n, s.l, s.m)}
                width={72}
                height={72}
                loading="lazy"
              />
              <span>{stateLabel(s.n, s.l, s.m)}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
```

(`basis` is already typed `Basis` in the store â€” drop the cast if tsc agrees; it is only shown here defensively.)

`App.tsx` â€” add the strip as the second row of the centre column:

```tsx
      <main className="center-col">
        {view === "cloud" && <CloudView />}
        {view === "plane" && <PlaneView />}
        {view === "radial" && <RadialView />}
        {view === "levels" && <LevelsView />}
        {view === "spectrum" && <SpectrumView />}
        <GalleryStrip />
      </main>
```

(plus `import { GalleryStrip } from "./components/GalleryStrip";`)

Append to `web/src/index.css`:

```css
.gallery {
  border-top: 1px solid var(--edge);
  padding: 0.5rem 1rem;
  background: var(--panel);
}

.gallery-head {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  color: var(--muted);
  font-size: 0.75rem;
  margin-bottom: 0.4rem;
}

.gallery-scroll {
  display: flex;
  gap: 0.5rem;
  overflow-x: auto;
  padding-bottom: 0.4rem;
}

.thumb {
  flex: 0 0 auto;
  background: transparent;
  border: 1px solid var(--edge);
  border-radius: 6px;
  padding: 0.3rem;
  cursor: pointer;
  color: var(--muted);
  font-size: 0.65rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.2rem;
}

.thumb img {
  border-radius: 4px;
  display: block;
  background: #000;
}

.thumb-active {
  border-color: var(--accent);
  color: var(--accent);
}
```

- [x] **Step 3: Gates** â€” `npm test` all pass; `npm run build` passes.

- [x] **Step 4: Commit**

```bash
git add web/src/lib/gallery.ts web/src/lib/gallery.test.ts web/src/components/GalleryStrip.tsx web/src/App.tsx web/src/index.css
git commit -m "feat: thumbnail gallery strip with disclosed navigation-aid liberty"
```

---

### Task 15: Web â€” layered math ("Show the physics", KaTeX)

**Files:**
- Create: `web/src/physics/content.ts`, `web/src/physics/content.test.ts`, `web/src/components/ShowPhysics.tsx`
- Modify: `web/package.json` (katex), `web/src/main.tsx` (css import), `web/src/components/Controls.tsx` (mount), `web/src/index.css`

**Interfaces:**
- Consumes: `ViewMode` from the store.
- Produces: `PHYSICS_CONTENT: Record<ViewMode, { title: string; blocks: { tex: string; note: string }[] }>`; `ShowPhysics` component (collapsible; app fully usable with it closed â€” spec Â§7.4).

- [x] **Step 1: Install KaTeX.** From `web\`:

Run: `$env:PYTHONUTF8='1'; & "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm install katex`
Then: `... npm install -D @types/katex`
Expected: both in `web/package.json`; lockfile updated.

- [x] **Step 2: Write the failing tests** â€” `web/src/physics/content.test.ts`:

```typescript
import katex from "katex";
import { describe, expect, it } from "vitest";
import { PHYSICS_CONTENT } from "./content";

const VIEWS = ["cloud", "plane", "radial", "levels", "spectrum"] as const;

describe("PHYSICS_CONTENT", () => {
  it("covers every view with at least one block", () => {
    for (const v of VIEWS) {
      expect(PHYSICS_CONTENT[v].title.length).toBeGreaterThan(0);
      expect(PHYSICS_CONTENT[v].blocks.length).toBeGreaterThan(0);
    }
  });
  it("every TeX string parses strictly and every note is substantive", () => {
    for (const v of VIEWS) {
      for (const b of PHYSICS_CONTENT[v].blocks) {
        expect(() =>
          katex.renderToString(b.tex, { displayMode: true, throwOnError: true }),
        ).not.toThrow();
        expect(b.note.length).toBeGreaterThan(20);
      }
    }
  });
});
```

Run `npm test` â†’ FAIL (`content.ts` missing).

- [x] **Step 3: Implement.** `web/src/physics/content.ts`:

```typescript
import type { ViewMode } from "../state/store";

export interface PhysicsBlock {
  tex: string;
  note: string;
}

export const PHYSICS_CONTENT: Record<
  ViewMode,
  { title: string; blocks: PhysicsBlock[] }
> = {
  cloud: {
    title: "What the point cloud is",
    blocks: [
      {
        tex: String.raw`\psi_{n\ell m}(r,\theta,\varphi) = R_{n\ell}(r)\,Y_\ell^m(\theta,\varphi)`,
        note: "The stationary state factorizes into the closed-form radial part and a spherical harmonic; both are computed in the engine, never approximated in the browser.",
      },
      {
        tex: String.raw`p(\mathbf r)\,dV = |\psi_{n\ell m}(\mathbf r)|^2\,dV`,
        note: "Each dot is one independent draw from |Ïˆ|Â² (seeded inverse-CDF Monte-Carlo). The cloud is a histogram of position measurements, not a photograph of an object.",
      },
    ],
  },
  plane: {
    title: "The cross-section, honestly",
    blocks: [
      {
        tex: String.raw`\rho(x, 0, z) = |\psi_{n\ell m}(x, 0, z)|^2`,
        note: "Probability density on the plane containing the quantization axis. The classic poster labels a signed quantity 'probability density' â€” density is non-negative, so Ïˆ and |Ïˆ|Â² are labeled separately here.",
      },
      {
        tex: String.raw`e^{im\varphi}\big|_{y=0} = \pm 1 \;\Rightarrow\; \psi\big|_{y=0} \in \mathbb{R}`,
        note: "On y = 0 the azimuthal factor is Â±1, so Ïˆ itself is real there: the signed-Ïˆ view is exact on this plane, not a convention.",
      },
    ],
  },
  radial: {
    title: "Radial structure",
    blocks: [
      {
        tex: String.raw`P_{n\ell}(r) = r^2\,|R_{n\ell}(r)|^2,\qquad \int_0^\infty P_{n\ell}(r)\,dr = 1`,
        note: "P(r) is the probability density for the electron's distance from the nucleus; the rÂ² factor is the volume of the spherical shell.",
      },
      {
        tex: String.raw`\langle r\rangle = \frac{a_0\,m_e}{Z\,\mu}\;\frac{3n^2 - \ell(\ell+1)}{2}`,
        note: "The dashed marker is the quantum expectation value â€” not the Bohr-model radius nÂ²aâ‚€ that many visualizers quietly show instead.",
      },
    ],
  },
  levels: {
    title: "Level energies",
    blocks: [
      {
        tex: String.raw`E_n = -\frac{Z^2}{2n^2}\,\frac{\mu}{m_e}\,E_h`,
        note: "Reduced-mass exact (EXACT badge): isotope and exotic-system dependence enters only through Î¼.",
      },
      {
        tex: String.raw`\Delta E_{nj} = -\frac{(Z\alpha)^2\,|E_n|}{n}\left(\frac{1}{j+\tfrac12} - \frac{3}{4n}\right)`,
        note: "The Î±Â² fine structure (spinâ€“orbit + relativistic kinetic energy + Darwin term, combined). APPROXIMATION badge: Î±â´ terms and the Lamb shift are absent â€” that is why equal-j levels coincide.",
      },
    ],
  },
  spectrum: {
    title: "Where the lines come from",
    blocks: [
      {
        tex: String.raw`\frac{1}{\lambda} = R_M Z^2\left(\frac{1}{n_1^2} - \frac{1}{n_2^2}\right),\qquad R_M = \frac{\mu}{m_e}\,R_\infty`,
        note: "Lines are level differences filtered by the selection rules Î”l = Â±1 (and Î”j = 0, Â±1 with fine structure on); wavelengths compare against vendored NIST data with a stated tolerance.",
      },
    ],
  },
};
```

`web/src/components/ShowPhysics.tsx`:

```tsx
import katex from "katex";
import { PHYSICS_CONTENT } from "../physics/content";
import { useAppStore } from "../state/store";

function MathBlock({ tex }: { tex: string }) {
  // KaTeX renders our own static strings only â€” no user input reaches it.
  const html = katex.renderToString(tex, { displayMode: true, throwOnError: false });
  return <div className="math" dangerouslySetInnerHTML={{ __html: html }} />;
}

export function ShowPhysics() {
  const view = useAppStore((s) => s.view);
  const content = PHYSICS_CONTENT[view];
  return (
    <details className="physics">
      <summary>Show the physics</summary>
      <h3>{content.title}</h3>
      {content.blocks.map((b) => (
        <div key={b.tex}>
          <MathBlock tex={b.tex} />
          <p className="physics-note">{b.note}</p>
        </div>
      ))}
    </details>
  );
}
```

`web/src/main.tsx` â€” add `import "katex/dist/katex.min.css";` above the existing `./index.css` import.

`web/src/components/Controls.tsx` â€” import `ShowPhysics` and render `<ShowPhysics />` as the last child of the `<aside className="panel">`.

Append to `web/src/index.css`:

```css
.physics {
  margin-top: 1.25rem;
  border-top: 1px solid var(--edge);
  padding-top: 0.75rem;
}

.physics summary {
  cursor: pointer;
  color: var(--accent);
  font-size: 0.85rem;
}

.physics h3 {
  font-size: 0.8rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin: 0.75rem 0 0.25rem;
}

.math {
  overflow-x: auto;
  padding: 0.25rem 0;
}

.physics-note {
  color: var(--muted);
  font-size: 0.8rem;
}
```

- [x] **Step 4: Gates** â€” `npm test` all pass; `npm run build` passes.

- [x] **Step 5: Commit**

```bash
git add web/package.json web/package-lock.json web/src/physics/content.ts web/src/physics/content.test.ts web/src/components/ShowPhysics.tsx web/src/main.tsx web/src/components/Controls.tsx web/src/index.css
git commit -m "feat: layered math - per-view KaTeX physics expander"
```

---

### Task 16: End-to-end verification, docs, ship

**Files:**
- Modify: `README.md` (status section), this plan (check boxes)
- No production code except fixes surfaced by verification.

- [x] **Step 1: Full gates from clean state.**

```powershell
$env:PYTHONUTF8='1'
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim ruff check .
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim python -m pytest -q
# from web\:
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm test
& "C:\ProgramData\miniforge3\condabin\conda.bat" run -n atomsim npm run build
```

Expected: everything green; `web/dist` rebuilt.

- [x] **Step 2: Drive the real app.** Start `atomsim serve --no-browser --port 8000` in the background (conda run). There is no system browser (dev-env memory): use the scratchpad Playwright chromium. Script the walkthrough: load `http://127.0.0.1:8000`, wait for the systems dropdown to populate; press Sample; wait for `status: ready`; switch colour to density, then phase; switch views: 2D cross-section (density and signed-Ïˆ), radial, levels (toggle fine structure), spectrum; click two gallery thumbnails; switch system to muonic hydrogen and re-sample (checks the camera-scale fix); screenshot each view to the scratchpad; assert zero console errors. Kill the server.

- [x] **Step 3: Fix anything found** (commit each fix separately with a conventional message).

- [x] **Step 4: Update `README.md`** status section: M3 complete â€” list the six views, colour modes, honesty UI (badges + disclosed liberties + labeled Ïˆ vs |Ïˆ|Â²), layered math, gallery; note new deps (matplotlib, d3-scale, katex).

- [x] **Step 5: Check every box in this plan**, commit docs:

```bash
git add README.md docs/plans/2026-07-06-phase1-m3-ui-depth.md
git commit -m "docs: M3 UI-depth status"
```

- [x] **Step 6: Push and verify CI.**

```powershell
git push
# poll (public repo, no gh CLI):
Invoke-RestMethod "https://api.github.com/repos/yaasshh09/atomsim/actions/runs?per_page=1" | Select-Object -ExpandProperty workflow_runs | Select-Object status, conclusion, head_sha
```

Expected: `completed` / `success` for the pushed SHA (both jobs). If red, fix forward immediately.

- [x] **Step 7: Update memory** (`atom-sim-vision.md` status line: M3 done, next M4 polish).

---

## Self-review notes (author)

- Spec Â§7.2 coverage: view 1 cloud (T8/9), 2 cross-section (T1/4/10), 3 radial (T11), 4 levels+FS zoom (T3/12), 5 spectrum (T13), 6 gallery (T5/14). Â§7.1 readouts incl. FPS + color-by-l legend â†’ T8/13 (series legend) â€” the info-card colour legend is folded into the Legend component + series colours rather than a separate l-colour card; acceptable within spec wording ("color-by-l legend" appears in the spectrum/series legend).
- Â§7.3 honesty UI: Badge popover already implements the inspector (method/assumptions/error/refinement); M3 adds frontend-authored VISUAL LIBERTY provenances (T9) â€” spec's "drawer" realized as popover, consistent with M1 precedent.
- Â§7.4 layered math: T15; app fully usable with the expander closed.
- Poster mode stays Phase 2 (locked interview decision) â€” thumbnails/plane rendering built so Phase 2 poster mode reuses them.
- Type-consistency spot checks done: `shift_ev` present in both `LevelModel` (state) and `FineLevelModel` (levels) and both TS mirrors; `kind` literals match; `getChannel(jobId)` (no channel) â†” plane-data contract (channel param 422 on plane jobs).

