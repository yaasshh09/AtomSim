"""The atomsim local server: honest JSON + binary boundaries for the browser app."""

import asyncio
import dataclasses
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic import Field as PydanticField

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
