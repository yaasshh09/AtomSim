"""The atomsim local server: honest JSON + binary boundaries for the browser app."""

import asyncio
import dataclasses
import re
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic import Field as PydanticField

import atomsim
from atomsim.analytic.fine_structure import fine_structure_shift, level_energy
from atomsim.analytic.hydrogen import (
    angular_momentum_magnitude,
    energy,
    mean_radius,
    radial_wavefunction,
    validate_quantum_numbers,
)
from atomsim.analytic.wavefunction import WavefunctionValues, evaluate_state
from atomsim.atoms import (
    ATOM_KEYS,
    atom_for_key,
    aufbau_configuration,
    format_config,
    is_atom_key,
    parse_config,
    total_electrons,
    validate_config,
)
from atomsim.classical import classical_ghost
from atomsim.constants import ALPHA, BOHR_RADIUS_PM, HARTREE_EV
from atomsim.constants_lab import analyze_constants
from atomsim.numerics.force_law import PRESETS, force_law_levels
from atomsim.plane import PlaneGrid, plane_grid
from atomsim.provenance import Field, Quantity
from atomsim.sampling import SampleCloud, sample_density
from atomsim.screened_atom import screened_radial, solve_screened_atom
from atomsim.server.jobs import Job, JobStatus, JobStore
from atomsim.server.schemas import (
    ChannelModel,
    ClassicalGhostModel,
    ComparisonModel,
    ConstantsReportModel,
    FieldModel,
    ForceLawLevelModel,
    ForceLawModel,
    LineModel,
    PotentialCurveModel,
    ProvenanceModel,
    QuantityModel,
    ReferenceItemModel,
    ReferenceModel,
    ScreenedLevelsModel,
    ScreenedOrbitalModel,
    SystemModel,
)
from atomsim.server.thumbnails import render_thumbnail
from atomsim.spectra import (
    compare_lines,
    load_reference,
    screened_transition_lines,
    transition_lines,
)
from atomsim.systems import get_system, hydrogen_like, list_systems

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


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
    alpha: float
    gross: list[GrossLevelModel]
    fine: list[FineLevelModel] | None


class StateResponse(BaseModel):
    n: int
    l: int
    m: int
    system: SystemModel
    energy: QuantityModel
    energy_ev: QuantityModel
    mean_radius: QuantityModel
    mean_radius_pm: QuantityModel
    angular_momentum: QuantityModel
    radial_nodes: int
    angular_nodes: int
    levels: list[LevelModel]


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


class SampleRequest(BaseModel):
    n: int
    l: int
    m: int
    count: int = PydanticField(default=100_000, ge=1_000, le=1_000_000)
    seed: int = 0
    basis: Literal["complex", "real"] = "complex"
    system: str = "h"


class JobModel(BaseModel):
    id: str
    status: str
    progress: float
    error: str | None


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


@dataclasses.dataclass(frozen=True)
class SampleJobResult:
    """A sampled cloud plus psi evaluated at exactly those positions."""

    cloud: SampleCloud
    psi: WavefunctionValues


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


def _job_model(job: Job) -> JobModel:
    return JobModel(id=job.id, status=job.status.value, progress=job.progress, error=job.error)


def _finished_result(jobs: JobStore, job_id: str):
    """Return the finished job's result container (sample or, later, plane)."""
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
    app.state.job_systems = {}
    app.add_middleware(
        CORSMiddleware, allow_origins=_DEV_ORIGINS, allow_methods=["*"], allow_headers=["*"]
    )

    _Z_KEY = re.compile(r"^z(\d+)$")

    def _resolve_system(key: str):
        zmatch = _Z_KEY.match(key)
        if zmatch:
            Z = int(zmatch.group(1))
            if not 1 <= Z <= 10:
                raise HTTPException(
                    status_code=422,
                    detail=f"generic hydrogen-like Z must be in [1, 10], got {Z}",
                )
            return hydrogen_like(Z)
        try:
            return get_system(key)
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def _is_screened(key: str) -> bool:
        return is_atom_key(key)

    def _resolve_config(system_key: str, config: str | None):
        element = atom_for_key(system_key)
        if config is None:
            return aufbau_configuration(element.z)
        try:
            cfg = parse_config(config)
            validate_config(cfg)
        except (ValueError, IndexError) as exc:
            raise HTTPException(status_code=422, detail=f"bad config: {exc}") from exc
        if total_electrons(cfg) != element.z:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"config has {total_electrons(cfg)} electrons; "
                    f"{element.symbol} needs {element.z}"
                ),
            )
        return cfg

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": atomsim.__version__}

    @app.get("/api/systems", response_model=SystemsResponse)
    def systems() -> SystemsResponse:
        hydrogenic = [SystemModel.from_system(s) for s in list_systems()]
        screened = [
            SystemModel.from_atom(
                atom_for_key(k), n_electrons=atom_for_key(k).z,
                description=(
                    f"{atom_for_key(k).name}: GSZ screened central-field model "
                    "(APPROXIMATION)."
                ),
            )
            for k in ATOM_KEYS
        ]
        return SystemsResponse(systems=hydrogenic + screened)

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
                        shift_ev=QuantityModel.from_quantity(_to_ev(sh)),
                    )
                )
        mr = mean_radius(n, l, Z=sys_.Z, mu_ratio=mu)
        return StateResponse(
            n=n, l=l, m=m,
            system=SystemModel.from_system(sys_),
            energy=QuantityModel.from_quantity(e),
            energy_ev=QuantityModel.from_quantity(_to_ev(e)),
            mean_radius=QuantityModel.from_quantity(mr),
            mean_radius_pm=QuantityModel.from_quantity(_to_pm(mr)),
            angular_momentum=QuantityModel.from_quantity(angular_momentum_magnitude(l)),
            radial_nodes=n - l - 1,
            angular_nodes=l,
            levels=levels,
        )

    @app.get("/api/levels", response_model=LevelsResponse | ScreenedLevelsModel)
    def levels_endpoint(system: str = "h", n_max: int = 6,
                        fine_structure: bool = False,
                        alpha: float | None = None,
                        config: str | None = None):
        if _is_screened(system):
            element = atom_for_key(system)
            cfg = _resolve_config(system, config)
            result = solve_screened_atom(element.z, total_electrons(cfg), cfg)
            return ScreenedLevelsModel(
                system=SystemModel.from_atom(
                    element, element.z,
                    f"{element.name}: GSZ screened central-field model (APPROXIMATION).",
                ),
                config=format_config(cfg), is_ground=result.is_ground,
                orbitals=[
                    ScreenedOrbitalModel(
                        n=o.n, l=o.l, label=f"{o.n}{'spdfgh'[o.l]}",
                        occupancy=o.occupancy,
                        energy=QuantityModel.from_quantity(o.energy),
                        energy_ev=QuantityModel.from_quantity(_to_ev(o.energy)),
                    )
                    for o in result.orbitals
                ],
                total_energy=QuantityModel.from_quantity(result.total_energy),
                total_energy_ev=QuantityModel.from_quantity(_to_ev(result.total_energy)),
            )
        if not 1 <= n_max <= 10:
            raise HTTPException(status_code=422, detail="n_max must be in [1, 10]")
        if alpha is not None and not 0.0 < alpha <= 0.5:
            raise HTTPException(status_code=422, detail="alpha must be in (0, 0.5]")
        sys_ = _resolve_system(system)
        mu = sys_.mu_ratio.value
        alpha_used = ALPHA if alpha is None else alpha
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
        )

    @app.get("/api/constants", response_model=ConstantsReportModel)
    def constants_endpoint(hbar: float = 1.0, e: float = 1.0, m_e: float = 1.0,
                           eps0: float = 1.0, c: float = 1.0) -> ConstantsReportModel:
        for name, mult in (("hbar", hbar), ("e", e), ("m_e", m_e),
                           ("eps0", eps0), ("c", c)):
            if not 0.25 <= mult <= 4.0:
                raise HTTPException(
                    status_code=422,
                    detail=f"{name} multiplier must be in [0.25, 4], got {mult}",
                )
        report = analyze_constants(hbar=hbar, e=e, m_e=m_e, eps0=eps0, c=c)
        return ConstantsReportModel.from_report(report)

    @app.get("/api/classical", response_model=ClassicalGhostModel)
    def classical_endpoint(system: str = "h", n: int = 1) -> ClassicalGhostModel:
        if n < 1:
            raise HTTPException(status_code=422, detail=f"n must be >= 1, got {n}")
        sys_ = _resolve_system(system)
        return ClassicalGhostModel.from_ghost(classical_ghost(n=n, system=sys_))

    @app.get("/api/forcelaw", response_model=ForceLawModel)
    def forcelaw_endpoint(
        preset: str = "powerlaw",
        l: int = 0,
        system: str = "h",
        n_states: int = 4,
        p: float = 1.0,
        lambda_: float = Query(default=3.0, alias="lambda"),
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

    @app.get("/api/radial/{n}/{l}", response_model=RadialResponse)
    def radial(n: int, l: int, system: str = "h", points: int = 400) -> RadialResponse:
        _validate_state(n, l, 0)
        if not 50 <= points <= 2000:
            raise HTTPException(status_code=422, detail="points must be in [50, 2000]")
        if _is_screened(system):
            element = atom_for_key(system)
            rw, p = screened_radial(element.z, element.z, n, l, points=points)
            return RadialResponse(
                n=n, l=l,
                system=SystemModel.from_atom(
                    element, element.z,
                    f"{element.name}: GSZ screened central-field model (APPROXIMATION).",
                ),
                r_wavefunction=FieldModel.from_field(rw),
                radial_probability=FieldModel.from_field(p),
            )
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

    @app.get("/api/spectrum", response_model=SpectrumResponse)
    def spectrum(system: str = "h", n_max: int = 6,
                 fine_structure: bool = False,
                 config: str | None = None) -> SpectrumResponse:
        if _is_screened(system):
            element = atom_for_key(system)
            cfg = _resolve_config(system, config)
            result = solve_screened_atom(element.z, total_electrons(cfg), cfg)
            lines = screened_transition_lines(result)
            reference = load_reference(system)
            comparison = citation = tol = None
            if reference is not None:
                tol = 0.05  # 5% pass bar — disclosed, not hidden
                # Wide association window (25%): a GSZ valence line sits a few
                # percent off the real wavelength but is the correct transition,
                # so report it as a residual instead of silently dropping it.
                comparison = [
                    ComparisonModel.from_comparison(c)
                    for c in compare_lines(
                        lines, reference, tolerance_relative=tol, window_relative=0.25
                    )
                ]
                citation = reference.citation
            return SpectrumResponse(
                system=SystemModel.from_atom(
                    element, element.z,
                    f"{element.name}: GSZ screened central-field model (APPROXIMATION).",
                ),
                n_max=lines.n_max, fine_structure=False,
                lines=[LineModel.from_line(ln) for ln in lines.lines],
                comparison=comparison, reference_citation=citation, tolerance_relative=tol,
            )
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

    @app.post("/api/jobs/sample", response_model=JobModel)
    async def create_sample_job(req: SampleRequest) -> JobModel:
        if _is_screened(req.system):
            raise HTTPException(
                status_code=422,
                detail="screened-atom orbitals: 3-D cloud / 2-D plane arrive in a later phase",
            )
        _validate_state(req.n, req.l, req.m)
        sys_ = _resolve_system(req.system)
        job = jobs.create()
        app.state.job_systems[job.id] = req.system

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

    @app.post("/api/jobs/plane", response_model=JobModel)
    async def create_plane_job(req: PlaneRequest) -> JobModel:
        if _is_screened(req.system):
            raise HTTPException(
                status_code=422,
                detail="screened-atom orbitals: 3-D cloud / 2-D plane arrive in a later phase",
            )
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

    @app.get("/api/jobs/{job_id}", response_model=JobModel)
    def job_status(job_id: str) -> JobModel:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
        return _job_model(job)

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
