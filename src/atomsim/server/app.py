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
from pydantic import BaseModel, model_validator
from pydantic import Field as PydanticField

import atomsim
from atomsim.analytic.dirac import dirac_energy
from atomsim.analytic.fine_structure import fine_structure_shift, level_energy
from atomsim.analytic.hydrogen import (
    angular_momentum_magnitude,
    energy,
    mean_radius,
    radial_wavefunction,
    validate_quantum_numbers,
)
from atomsim.analytic.hyperfine import hyperfine_report
from atomsim.analytic.stark import stark_sublevels
from atomsim.analytic.wavefunction import WavefunctionValues, evaluate_state
from atomsim.analytic.zeeman import zeeman_sublevels
from atomsim.atoms import (
    ATOM_KEYS,
    SUBSHELL_LABELS,
    atom_for_key,
    aufbau_configuration,
    element_by_z,
    format_config,
    is_atom_key,
    parse_config,
    total_electrons,
    validate_config,
)
from atomsim.broadening import synthesize
from atomsim.classical import classical_ghost
from atomsim.constants import ALPHA, BOHR_RADIUS_PM, HARTREE_EV
from atomsim.constants_lab import analyze_constants
from atomsim.hf_atom import (
    HFResult,
    PauliCollapse,
    hf_exchange_energy,
    pauli_collapse,
    solve_hartree_fock,
)
from atomsim.numerics.expression import ExpressionError
from atomsim.numerics.force_law import PRESETS, force_law_levels, free_form_levels
from atomsim.plane import PlaneGrid, plane_grid, screened_plane_grid
from atomsim.populations import ThermalConditions
from atomsim.provenance import Field, Quantity
from atomsim.sampling import SampleCloud, sample_density, sample_screened_density
from atomsim.screened_atom import (
    evaluate_screened_state,
    screened_radial,
    solve_screened_atom,
)
from atomsim.server.jobs import Job, JobStatus, JobStore
from atomsim.server.schemas import (
    AbsorptionSpectrumModel,
    ChannelModel,
    ClassicalGhostModel,
    ComparisonModel,
    ConstantsReportModel,
    CurveOfGrowthModel,
    FieldModel,
    ForceLawLevelModel,
    ForceLawModel,
    HFOrbitalModel,
    HFResultModel,
    LineModel,
    PauliCollapseModel,
    PotentialCurveModel,
    ProfileModel,
    ProvenanceModel,
    QuantityModel,
    ReferenceItemModel,
    ReferenceModel,
    ScreenedLevelsModel,
    ScreenedOrbitalModel,
    SystemModel,
    ThermalModel,
)
from atomsim.server.thumbnails import render_thumbnail
from atomsim.spectra import (
    compare_lines,
    load_reference,
    screened_transition_lines,
    subshell_label,
    transition_lines,
)
from atomsim.systems import (
    element_emitter_mass,
    emitter_mass,
    get_system,
    hydrogen_like,
    list_systems,
)
from atomsim.transfer import absorb, curve_of_growth, default_columns

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


class LevelModel(BaseModel):
    j: float
    energy: QuantityModel
    energy_ev: QuantityModel
    shift: QuantityModel
    shift_ev: QuantityModel


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


class HyperfineLevelModel(BaseModel):
    F: float
    energy: QuantityModel
    energy_ev: QuantityModel
    shift: QuantityModel
    shift_ev: QuantityModel


class HyperfineShellModel(BaseModel):
    n: int
    available: bool
    nucleus: str | None = None
    I: float | None = None
    A: QuantityModel | None = None       # coupling constant, hartree
    A_ev: QuantityModel | None = None
    levels: list[HyperfineLevelModel] = []
    note: str | None = None              # e.g. spin-0: available but no split
    reason: str | None = None            # why hyperfine is unavailable


class LevelsResponse(BaseModel):
    system: SystemModel
    n_max: int
    fine_structure: bool
    alpha: float
    gross: list[GrossLevelModel]
    fine: list[FineLevelModel] | None
    dirac: bool = False
    b_field: float = 0.0
    e_field: float = 0.0
    hyperfine: bool = False
    hyperfine_shells: list[HyperfineShellModel] | None = None


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
    #: Why the lines carry no strengths, when they were asked for and withheld.
    intensity_note: str | None = None
    #: Present exactly when the lines carry an emissivity.
    thermal: ThermalModel | None = None
    #: The synthesized curve, when one was asked for and could be built. Null
    #: with `profile_note` set when the request was made but no mechanism gives
    #: the lines any width, since an invented width would be the lie.
    profile: ProfileModel | None = None
    profile_note: str | None = None


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


class HFRequest(BaseModel):
    z: int
    n_electrons: int | None = None  # defaults to neutral
    config: str | None = None       # defaults to the aufbau ground configuration
    # False solves the Hartree model instead: electrons that repel but are
    # distinguishable. Defaults to real physics, so a client that has never
    # heard of the toggle cannot accidentally ask for the counterfactual.
    exchange: bool = True
    # False lifts the occupancy cap too, and the configuration collapses to
    # 1s^N. Same default and the same reason.
    pauli: bool = True

    @model_validator(mode="after")
    def _pauli_off_implies_exchange_off(self) -> "HFRequest":
        """Refuse the combination rather than quietly flipping a flag.

        422 rather than 400 because this is a request that cannot be
        understood, not one the server is declining: pauli=False with
        exchange=True does not name a model this or any solver could run. A
        Slater determinant holding two electrons in the same spin-orbital is
        identically zero, so there is no wavefunction there for an exchange
        integral to act on. Correcting it for the client would hide that they
        asked for a state that does not exist.
        """
        if not self.pauli and self.exchange:
            raise ValueError(
                "pauli=false requires exchange=false: exchange energy is a "
                "consequence of antisymmetry and the exclusion principle IS "
                "antisymmetry, so with the principle off there is nothing for "
                "an exchange integral to act on"
            )
        return self


# The outermost principal quantum number the solver actually converges.
#
# Measured, and NOT a statement about Z. Argon-like ions converge cleanly all
# the way to Z = 36 (virial 2.000003 at every Z tried), and K+ and Ca2+ solve
# in about six seconds. What fails is the 4s channel of a neutral alkali: the
# starting potential falls back to the bare nucleus above argon, which puts
# potassium's 4s guess at -11.3 hartree against a true -0.15, and the SCF
# cannot screen a guess that far outward. LOBPCG then stagnates at a residual
# 28x over its ceiling, so this is stagnation rather than a budget that wants
# raising, and refusing up front beats letting a job die eight seconds in with
# an eigensolver message.
_HF_MAX_N = 3
# As far as the solver has been exercised. Beyond this the answer is untested,
# and a non-relativistic model is also getting thin: the neglected relativity
# is already 1.8% of the 1s energy at Z = 36 (hf_atom quantifies it in the
# provenance from Z = 9 up).
_HF_MAX_Z = 36


@dataclasses.dataclass(frozen=True)
class HFJobResult:
    """A finished solve, plus what exchange was worth if the job measured it.

    A wrapper rather than a second endpoint, because the exchange energy is a
    difference between two solves and the two have to be the same solves. If a
    view fetched the Hartree-Fock energy from one job and the Hartree energy
    from another, nothing would stop it differencing a warm result against a
    cold one, or a fine mesh against a coarse one, and reporting the gap
    between two calculations as the gap between two models.

    None when the job ran only the real model, which is the common case and
    costs one solve rather than two.
    """

    result: HFResult
    exchange_energy: Quantity | None = None
    #: The real atom beside the collapsed one, on a pauli=False ground solve.
    #: Same reasoning as exchange_energy: a comparison between two models has
    #: to be made from two solves that are known to match, so it is assembled
    #: here rather than left for a client to difference across two jobs.
    collapse: PauliCollapse | None = None


def _parse_config_or_422(text: str, pauli: bool = True):
    """Parse a hand-written configuration string, 422 on malformed input.

    Two status codes are in play here and the split is deliberate. 422 means
    the request could not be understood - "2s^9" is not a configuration. 400
    means it was understood perfectly and the server is declining it, which is
    what _validate_hf_request returns: a neutral potassium atom is a real,
    well-posed request that this solver cannot answer honestly.

    With pauli off, "1s10" IS a configuration, so the capacity check goes away
    here as well; n > l stays, because that one is not the exclusion principle.
    """
    try:
        cfg = parse_config(text)
        validate_config(cfg, pauli)
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=f"bad config: {exc}") from exc
    return cfg


def _validate_hf_request(z: int, n_electrons: int, config, pauli: bool = True) -> None:
    """Refuse what the solver cannot do, with the reason, before starting a job."""
    if not 1 <= z <= _HF_MAX_Z:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Z must be in [1, {_HF_MAX_Z}], got {z}; the Hartree-Fock "
                f"solver has not been exercised above {_HF_MAX_Z}, and a "
                f"non-relativistic model is a poor description of a heavier atom"
            ),
        )
    if not 1 <= n_electrons <= z + 1:
        raise HTTPException(
            status_code=400,
            detail=f"electron count must be in [1, Z+1] = [1, {z + 1}], got {n_electrons}",
        )
    try:
        validate_config(config, pauli)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if total_electrons(config) != n_electrons:
        raise HTTPException(
            status_code=400,
            detail=(
                f"configuration holds {total_electrons(config)} electrons, "
                f"not the {n_electrons} requested"
            ),
        )
    n_top = max(n for (n, _), _ in config)
    if n_top > _HF_MAX_N:
        raise HTTPException(
            status_code=400,
            detail=(
                f"this configuration occupies n = {n_top}, and the solver "
                f"converges only to n = {_HF_MAX_N}. The {n_top}s guess starts "
                f"from a bare-nucleus potential, which for a diffuse outer "
                f"shell is far too contracted for the self-consistent loop to "
                f"recover; the eigensolver stagnates rather than converging "
                f"slowly. Ions whose outermost shell is n <= {_HF_MAX_N} are "
                f"fine at any Z up to {_HF_MAX_Z}"
            ),
        )


def _hf_channel(n: int, l: int) -> str:
    return f"P_{n}{SUBSHELL_LABELS[l]}"


def _hf_symbol(z: int) -> str | None:
    """The element symbol, or None above the preset table.

    Hartree-Fock solves further up than the preset library reaches, and naming
    the element is a convenience for the view rather than part of the physics,
    so not knowing one is not an error.
    """
    try:
        return element_by_z(z).symbol
    except KeyError:
        return None


def _pauli_collapse_model(collapse: PauliCollapse) -> PauliCollapseModel:
    return PauliCollapseModel(
        binding_change=QuantityModel.from_quantity(collapse.binding_change),
        binding_change_ev=QuantityModel.from_quantity(_to_ev(collapse.binding_change)),
        real_total_energy=QuantityModel.from_quantity(collapse.real.total_energy),
        real_total_energy_ev=QuantityModel.from_quantity(
            _to_ev(collapse.real.total_energy)
        ),
        real_config=format_config(collapse.real.config),
        real_radius=QuantityModel.from_quantity(collapse.real_radius),
        collapsed_radius=QuantityModel.from_quantity(collapse.collapsed_radius),
        radius_ratio=QuantityModel.from_quantity(collapse.radius_ratio),
        variational_zeta=QuantityModel.from_quantity(collapse.variational_zeta),
        variational_energy=QuantityModel.from_quantity(collapse.variational_energy),
        variational_energy_ev=QuantityModel.from_quantity(
            _to_ev(collapse.variational_energy)
        ),
    )


def _hf_result_model(
    result: HFResult,
    exchange_energy: Quantity | None = None,
    collapse: PauliCollapse | None = None,
) -> HFResultModel:
    return HFResultModel(
        z=result.z,
        n_electrons=result.n_electrons,
        symbol=_hf_symbol(result.z),
        config=format_config(result.config),
        is_ground=result.is_ground,
        exchange=result.exchange,
        exchange_energy=(
            None if exchange_energy is None
            else QuantityModel.from_quantity(exchange_energy)
        ),
        exchange_energy_ev=(
            None if exchange_energy is None
            else QuantityModel.from_quantity(_to_ev(exchange_energy))
        ),
        pauli=result.pauli,
        collapse=None if collapse is None else _pauli_collapse_model(collapse),
        orbitals=[
            HFOrbitalModel(
                n=o.n, l=o.l, label=f"{o.n}{SUBSHELL_LABELS[o.l]}",
                occupancy=o.occupancy,
                energy=QuantityModel.from_quantity(o.energy),
                energy_ev=QuantityModel.from_quantity(_to_ev(o.energy)),
                channel=_hf_channel(o.n, o.l),
            )
            for o in result.orbitals
        ],
        total_energy=QuantityModel.from_quantity(result.total_energy),
        total_energy_ev=QuantityModel.from_quantity(_to_ev(result.total_energy)),
        kinetic=QuantityModel.from_quantity(result.kinetic),
        potential=QuantityModel.from_quantity(result.potential),
        virial_ratio=QuantityModel.from_quantity(result.virial_ratio),
        iterations=result.iterations,
        coarse_iterations=result.coarse_iterations,
        converged=result.converged,
        provenance=ProvenanceModel.from_provenance(result.provenance),
        grid_channel="grid",
        grid_points=len(result.orbitals[0].P.grid),
        channels=[
            ChannelModel(
                name="grid", dtype="float32", unit="bohr",
                # The mesh is a solver choice, not a measurement of the atom.
                provenance=ProvenanceModel.from_provenance(
                    result.orbitals[0].P.provenance
                ),
            ),
            *(
                ChannelModel(
                    name=_hf_channel(o.n, o.l), dtype="float32", unit=o.P.unit,
                    provenance=ProvenanceModel.from_provenance(o.P.provenance),
                )
                for o in result.orbitals
            ),
        ],
    )


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


def _hyperfine_shell_model(rep) -> "HyperfineShellModel":
    """Map an available HyperfineReport to its response model (with eV display)."""
    return HyperfineShellModel(
        n=rep.n,
        available=True,
        nucleus=rep.nucleus_name,
        I=rep.I,
        A=QuantityModel.from_quantity(rep.A) if rep.A is not None else None,
        A_ev=QuantityModel.from_quantity(_to_ev(rep.A)) if rep.A is not None else None,
        levels=[
            HyperfineLevelModel(
                F=lv.F,
                energy=QuantityModel.from_quantity(lv.energy),
                energy_ev=QuantityModel.from_quantity(_to_ev(lv.energy)),
                shift=QuantityModel.from_quantity(lv.shift),
                shift_ev=QuantityModel.from_quantity(_to_ev(lv.shift)),
            )
            for lv in rep.levels
        ],
        note=rep.note,
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

    def _resolve_thermal(
        temperature_k: float | None, electron_density_cm3: float | None
    ) -> ThermalConditions | None:
        """Both knobs or neither: half of Saha is not a state anyone can read.

        The bounds are display limits, not physics limits. Below ~100 K every
        excited level is empty and the spectrum is a single dark band; above
        ~10^6 K hydrogen is long gone. The formulas hold outside; the view has
        nothing to show there.
        """
        if temperature_k is None and electron_density_cm3 is None:
            return None
        if temperature_k is None or electron_density_cm3 is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "temperature_k and electron_density_cm3 must be given "
                    "together: ionization depends on both"
                ),
            )
        if not 1e2 <= temperature_k <= 1e6:
            raise HTTPException(
                status_code=422, detail="temperature_k must be in [1e2, 1e6]"
            )
        if not 1e4 <= electron_density_cm3 <= 1e22:
            raise HTTPException(
                status_code=422, detail="electron_density_cm3 must be in [1e4, 1e22]"
            )
        return ThermalConditions(temperature_k, electron_density_cm3)

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
                        config: str | None = None,
                        dirac: bool = False,
                        b_field: float = 0.0,
                        e_field: float = 0.0,
                        hyperfine: bool = False):
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
        if b_field < 0.0:
            raise HTTPException(status_code=422, detail="b_field must be >= 0")
        if e_field < 0.0:
            raise HTTPException(status_code=422, detail="e_field must be >= 0")
        sys_ = _resolve_system(system)
        mu = sys_.mu_ratio.value
        alpha_used = ALPHA if alpha is None else alpha
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
        fine = None
        if dirac or fine_structure:
            fine = []
            for n in range(1, n_max + 1):
                for l in range(n):
                    for j in ([0.5] if l == 0 else [l - 0.5, l + 0.5]):
                        if dirac:
                            try:
                                le = dirac_energy(
                                    n, j, Z=sys_.Z, mu_ratio=mu, alpha=alpha_used
                                )
                            except ValueError as exc:
                                raise HTTPException(status_code=422, detail=str(exc)) from exc
                            e_bohr = energy(n, Z=sys_.Z, mu_ratio=mu)
                            sh = dataclasses.replace(
                                le,
                                value=le.value - e_bohr.value,
                                label=f"dE_Dirac {n},{l},j={j:g}",
                            )
                        else:
                            le = level_energy(
                                n, l, j, Z=sys_.Z, mu_ratio=mu,
                                m_over_M=sys_.m_over_M, alpha=alpha_used,
                            )
                            sh = fine_structure_shift(
                                n, l, j, Z=sys_.Z, mu_ratio=mu,
                                m_over_M=sys_.m_over_M, alpha=alpha_used,
                            )
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
                                if z.j_label == j
                            ]
                        fine.append(FineLevelModel(
                            n=n, l=l, j=j,
                            energy=QuantityModel.from_quantity(le),
                            energy_ev=QuantityModel.from_quantity(_to_ev(le)),
                            shift=QuantityModel.from_quantity(sh),
                            shift_ev=QuantityModel.from_quantity(_to_ev(sh)),
                            sublevels=subs,
                        ))
        hf_shells = None
        if hyperfine:
            first = hyperfine_report(1, sys_)
            if not first.available:
                # availability is n-independent: one honest reason, not n_max copies.
                hf_shells = [HyperfineShellModel(
                    n=1, available=False, reason=first.reason,
                )]
            else:
                hf_shells = [
                    _hyperfine_shell_model(hyperfine_report(n, sys_))
                    for n in range(1, n_max + 1)
                ]
        return LevelsResponse(
            system=SystemModel.from_system(sys_), n_max=n_max,
            fine_structure=fine_structure, alpha=alpha_used, gross=gross, fine=fine,
            dirac=dirac, b_field=b_field, e_field=e_field,
            hyperfine=hyperfine, hyperfine_shells=hf_shells,
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
        expr: str | None = None,
    ) -> ForceLawModel:
        if l < 0:
            raise HTTPException(status_code=422, detail=f"l must be >= 0, got {l}")
        if not 1 <= n_states <= 8:
            raise HTTPException(
                status_code=422, detail=f"n_states must be in [1, 8], got {n_states}"
            )
        sys_ = _resolve_system(system)

        if preset == "custom":
            if not expr or not expr.strip():
                raise HTTPException(status_code=422, detail="custom preset requires 'expr'")
            try:
                result = free_form_levels(expr, l=l, system=sys_, n_states=n_states)
            except (ExpressionError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        else:
            if preset not in PRESETS:
                raise HTTPException(
                    status_code=422,
                    detail=f"unknown preset {preset!r}; known: {sorted(PRESETS)}",
                )
            supplied = {
                "p": p, "lambda": lambda_, "omega": omega, "v0": v0, "a": a, "core": core,
            }
            params = {spec.name: supplied[spec.name] for spec in PRESETS[preset].params}
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
                    trusted=c.trusted,
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
            expression=result.expression,
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

    def _resolve_zoom(
        lambda_min: float | None, lambda_max: float | None
    ) -> tuple[float, float] | None:
        """Both ends or neither, and the low end has to be real light."""
        if lambda_min is None and lambda_max is None:
            return None
        if lambda_min is None or lambda_max is None:
            raise HTTPException(
                status_code=422,
                detail="lambda_min and lambda_max must be given together",
            )
        if not 0.0 < lambda_min < lambda_max:
            raise HTTPException(
                status_code=422, detail="need 0 < lambda_min < lambda_max"
            )
        return (lambda_min, lambda_max)

    def _profile_window(lines) -> tuple[float, float] | None:
        """The wavelength span a synthesized curve should cover.

        Same structural rule the view uses for the bar axis: across-n lines set
        the range, because a fine-structure list also holds within-n components
        out at millimetres to metres, and stretching a synthesis over eleven
        decades of wavelength spends the whole point budget on empty space.
        None means "no split applies, use the lot".
        """
        across = [
            ln.wavelength.value for ln in lines if ln.n_upper != ln.n_lower
        ]
        if not across or len(across) == len(lines):
            return None
        return (min(across), max(across))

    def _synthesize_profile(
        lines, mass, hydrogenic: bool, resolving_power: float | None,
        full_range: bool, zoom: tuple[float, float] | None,
    ) -> tuple[ProfileModel | None, str | None]:
        """Build the curve, or say plainly why there is none.

        The failure mode is a feature: with no decay rate, no temperature and
        no instrument, every line has zero width, and the only way to draw a
        curve would be to invent one. The note names the knob instead.

        A `zoom` window is where the phase earns its keep: a profile only shows
        its shape when the axis is narrow enough to resolve it, and the whole
        point budget then lands on the one line being looked at.
        """
        window = zoom if zoom else (None if full_range else _profile_window(lines.lines))
        try:
            syn = synthesize(
                lines, emitter_mass=mass, hydrogenic=hydrogenic,
                resolving_power=resolving_power, window_nm=window,
                # A transport cap, not a physics one. Closure stays inside
                # 0.1% here, and the response reports what it actually was.
                max_points=6000,
            )
        except ValueError as exc:
            return None, str(exc)
        return ProfileModel.from_synthetic(syn), None

    @app.get("/api/spectrum", response_model=SpectrumResponse)
    def spectrum(system: str = "h", n_max: int = 6,
                 fine_structure: bool = False,
                 intensities: bool = False,
                 temperature_k: float | None = None,
                 electron_density_cm3: float | None = None,
                 profile: bool = False,
                 resolving_power: float | None = None,
                 full_range: bool = False,
                 lambda_min: float | None = None,
                 lambda_max: float | None = None,
                 config: str | None = None) -> SpectrumResponse:
        thermal = _resolve_thermal(temperature_k, electron_density_cm3)
        if resolving_power is not None and not 1e2 <= resolving_power <= 1e7:
            raise HTTPException(
                status_code=422, detail="resolving_power must be in [1e2, 1e7]"
            )
        zoom = _resolve_zoom(lambda_min, lambda_max)
        if _is_screened(system):
            element = atom_for_key(system)
            cfg = _resolve_config(system, config)
            result = solve_screened_atom(element.z, total_electrons(cfg), cfg)
            lines = screened_transition_lines(
                result, intensities=intensities, thermal=thermal
            )
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
            prof = note = None
            if profile:
                prof, note = _synthesize_profile(
                    lines, element_emitter_mass(element), hydrogenic=False,
                    resolving_power=resolving_power, full_range=full_range,
                    zoom=zoom,
                )
            return SpectrumResponse(
                system=SystemModel.from_atom(
                    element, element.z,
                    f"{element.name}: GSZ screened central-field model (APPROXIMATION).",
                ),
                n_max=lines.n_max, fine_structure=False,
                lines=[LineModel.from_line(ln) for ln in lines.lines],
                comparison=comparison, reference_citation=citation, tolerance_relative=tol,
                intensity_note=lines.intensity_note,
                thermal=(
                    None if lines.thermal is None
                    else ThermalModel.from_state(lines.thermal)
                ),
                profile=prof, profile_note=note,
            )
        if not 2 <= n_max <= 10:
            raise HTTPException(status_code=422, detail="n_max must be in [2, 10]")
        sys_ = _resolve_system(system)
        lines = transition_lines(
            sys_, n_max=n_max, fine_structure=fine_structure,
            intensities=intensities, thermal=thermal,
        )
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
        prof = note = None
        if profile:
            prof, note = _synthesize_profile(
                lines, emitter_mass(sys_), hydrogenic=True,
                resolving_power=resolving_power, full_range=full_range,
                zoom=zoom,
            )
        return SpectrumResponse(
            system=SystemModel.from_system(sys_),
            n_max=n_max,
            fine_structure=fine_structure,
            lines=[LineModel.from_line(ln) for ln in lines.lines],
            comparison=comparison,
            reference_citation=citation,
            tolerance_relative=tol,
            intensity_note=lines.intensity_note,
            thermal=(
                None if lines.thermal is None
                else ThermalModel.from_state(lines.thermal)
            ),
            profile=prof, profile_note=note,
        )

    def _check_resolving_power(resolving_power: float | None) -> None:
        if resolving_power is not None and not 1e2 <= resolving_power <= 1e7:
            raise HTTPException(
                status_code=422, detail="resolving_power must be in [1e2, 1e7]"
            )

    def _lines_with_strengths(
        system: str, n_max: int, fine_structure: bool, thermal, config: str | None
    ):
        """A line list carrying oscillator strengths and populations, plus the
        emitter mass its Doppler widths need.

        Both transfer endpoints want exactly this and want it identically: a
        curve of growth and an absorption spectrum that disagreed about which
        lines exist would be two answers about one gas.
        """
        if _is_screened(system):
            element = atom_for_key(system)
            cfg = _resolve_config(system, config)
            result = solve_screened_atom(element.z, total_electrons(cfg), cfg)
            lines = screened_transition_lines(result, intensities=True, thermal=thermal)
            return lines, element_emitter_mass(element), False
        if not 2 <= n_max <= 10:
            raise HTTPException(status_code=422, detail="n_max must be in [2, 10]")
        sys_ = _resolve_system(system)
        lines = transition_lines(
            sys_, n_max=n_max, fine_structure=fine_structure,
            intensities=True, thermal=thermal,
        )
        return lines, emitter_mass(sys_), True

    @app.get("/api/absorption", response_model=AbsorptionSpectrumModel)
    def absorption_endpoint(
        system: str = "h", n_max: int = 6, fine_structure: bool = False,
        temperature_k: float = 10000.0, electron_density_cm3: float = 1e13,
        column_density_m2: float = 1e20,
        resolving_power: float | None = None,
        lambda_min: float | None = None, lambda_max: float | None = None,
        config: str | None = None,
    ) -> AbsorptionSpectrumModel:
        """A whole line list in front of a flat continuum, and what survives.

        One column density for the element; each line's own lower-level
        fraction turns it into that line's absorbers. That is what makes the
        Lyman lines go black while the Balmer lines stay invisible in the same
        gas, and it is the fact the emission endpoint cannot represent.

        The window is left to the engine unless asked for, because sizing it
        by eye is how Phase 19 lost a third of an equivalent width without
        anything reporting a problem.
        """
        # Absorption has no meaning without populations, so unlike the emission
        # endpoint these conditions are not optional. They default rather than
        # refuse, and the gas they describe rides back on the response so a
        # spectrum is never a shape with no conditions attached.
        thermal = _resolve_thermal(temperature_k, electron_density_cm3)
        if not 0.0 <= column_density_m2 <= 1e30:
            raise HTTPException(
                status_code=422,
                detail="column_density_m2 must be in [0, 1e30]",
            )
        _check_resolving_power(resolving_power)
        window = _resolve_zoom(lambda_min, lambda_max)
        lines, mass, hydrogenic = _lines_with_strengths(
            system, n_max, fine_structure, thermal, config
        )
        try:
            spectrum = absorb(
                lines, column_density_m2, emitter_mass=mass,
                hydrogenic=hydrogenic, resolving_power=resolving_power,
                window_nm=window,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return AbsorptionSpectrumModel.from_spectrum(spectrum, lines.thermal)

    @app.get("/api/curve-of-growth", response_model=CurveOfGrowthModel)
    def curve_of_growth_endpoint(
        system: str = "h", n_max: int = 6, fine_structure: bool = False,
        temperature_k: float = 10000.0, electron_density_cm3: float = 1e13,
        lambda_nm: float = 656.28, resolving_power: float | None = None,
        config: str | None = None,
    ) -> CurveOfGrowthModel:
        """How much light one line removes, against how much gas is in the way.

        The line is picked by wavelength rather than by quantum numbers so the
        view can hand back whatever the user clicked. Widths come from the same
        Phase 18 synthesis that drew the profile, so the curve and the profile
        beside it describe the same line.
        """
        thermal = _resolve_thermal(temperature_k, electron_density_cm3)
        if lambda_nm <= 0.0:
            raise HTTPException(status_code=422, detail="lambda_nm must be > 0")
        _check_resolving_power(resolving_power)
        lines, mass, hydrogenic = _lines_with_strengths(
            system, n_max, fine_structure, thermal, config
        )

        syn = synthesize(
            lines, emitter_mass=mass, hydrogenic=hydrogenic,
            resolving_power=resolving_power, max_points=2000,
        )
        if not syn.profiles:
            raise HTTPException(status_code=404, detail="no lines in this spectrum")
        # Nearest computed wavelength to what was asked for, then the strongest
        # transition sitting on it. Both steps matter: "H-alpha" is three
        # lines at exactly 656.4696 nm with oscillator strengths of 0.014,
        # 0.435 and 0.696, so picking whichever comes first in the list draws a
        # curve of growth for the weakest of them and calls it H-alpha. The
        # pairing comes from `synthesize` rather than from a second wavelength
        # match, which could not tell those three apart at all.
        paired = list(zip(syn.profiles, syn.lines, strict=True))
        target = min(
            paired, key=lambda pl: abs(pl[0].wavelength_nm - lambda_nm)
        )[0].wavelength_nm
        width, line = max(
            (pl for pl in paired if pl[0].wavelength_nm == target),
            key=lambda pl: (
                pl[1].oscillator_strength.value
                if pl[1].oscillator_strength is not None else 0.0
            ),
        )
        if line.oscillator_strength is None or line.oscillator_strength.value <= 0.0:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"the {width.label} line has no oscillator strength, so it "
                    "has no absorption cross-section and no curve of growth"
                ),
            )
        f = line.oscillator_strength.value
        try:
            columns = default_columns(f, width.wavelength_nm, width.sigma_nm,
                                      width.gamma_nm)
            curve = curve_of_growth(
                f, width.wavelength_nm, width.sigma_nm, width.gamma_nm, columns
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return CurveOfGrowthModel.from_curve(
            curve, f"{width.label} ({subshell_label(line)})",
            width.sigma_nm, width.gamma_nm
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
        _validate_state(req.n, req.l, req.m)
        job = jobs.create()
        app.state.job_systems[job.id] = req.system

        if _is_screened(req.system):
            element = atom_for_key(req.system)

            def work(progress):
                cloud = sample_screened_density(
                    element.z, element.z, req.n, req.l, req.m, req.count,
                    seed=req.seed, progress=lambda f: progress(0.9 * f), basis=req.basis,
                )
                psi = evaluate_screened_state(
                    element.z, element.z, req.n, req.l, req.m,
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

    @app.post("/api/jobs/plane", response_model=JobModel)
    async def create_plane_job(req: PlaneRequest) -> JobModel:
        _validate_state(req.n, req.l, req.m)
        job = jobs.create()
        app.state.job_systems[job.id] = req.system

        if _is_screened(req.system):
            element = atom_for_key(req.system)

            def work(progress):
                return screened_plane_grid(
                    element.z, element.z, req.n, req.l, req.m,
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

    @app.post("/api/jobs/hf", response_model=JobModel)
    async def create_hf_job(req: HFRequest) -> JobModel:
        """Start a Hartree-Fock solve.

        A job rather than a plain GET because the solve takes seconds, not
        milliseconds - argon is about 5s cold and chlorine about 7s - which is
        long enough that a blocking request would be a bad answer even though
        it would be a correct one. The engine memoizes, so a repeat is free and
        the job simply finishes immediately.
        """
        n_electrons = req.z if req.n_electrons is None else req.n_electrons
        config = (
            aufbau_configuration(n_electrons, req.pauli)
            if req.config is None
            else _parse_config_or_422(req.config, req.pauli)
        )
        _validate_hf_request(req.z, n_electrons, config, req.pauli)
        # A collapsed solve is compared against the real atom only when it is
        # the ground configuration of its own rule. A hand-written 1s5 2s3 has
        # no "same atom with the cap on" to be measured against, and comparing
        # it to the Aufbau ground state would report the distance between two
        # different configurations as the cost of the exclusion principle.
        comparable = not req.pauli and config == aufbau_configuration(
            n_electrons, pauli=False
        )

        job = jobs.create()

        def work(progress):
            # solve_hartree_fock runs its own two-mesh loop with no progress
            # hook, so there is nothing honest to report between 0 and 1. A
            # synthetic ramp would look like information and be none.
            result = solve_hartree_fock(
                req.z, n_electrons, config, req.exchange, req.pauli
            )
            # The counterfactual solves are the only ones that owe the reader a
            # comparison, and they are the only ones that pay for a second
            # solve. Cheap in practice: a client reaches these by flipping a
            # switch on an atom it was just looking at, so the real solve is
            # already memoized and this is arithmetic on two cached results.
            delta = (
                None if req.exchange or not req.pauli
                else hf_exchange_energy(req.z, n_electrons, config)
            )
            collapse = pauli_collapse(req.z, n_electrons) if comparable else None
            progress(1.0)
            return HFJobResult(result, delta, collapse)

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

    @app.get(
        "/api/jobs/{job_id}/meta",
        response_model=SampleMetaModel | PlaneMetaModel | HFResultModel,
    )
    def job_meta(job_id: str) -> SampleMetaModel | PlaneMetaModel | HFResultModel:
        res = _finished_result(jobs, job_id)
        system_key = app.state.job_systems.get(job_id, "h")
        if isinstance(res, PlaneGrid):
            return _plane_meta(res, system_key)
        if isinstance(res, HFJobResult):
            # Unlike sample and plane, the whole scientific result is here: the
            # energies and their provenance. /data carries only the orbital
            # shapes, which are the part that is an array.
            return _hf_result_model(res.result, res.exchange_energy, res.collapse)
        return _sample_meta(res, system_key)

    def _hf_channel_payload(res: HFResult, channel: str | None) -> np.ndarray:
        if channel is None or channel == "grid":
            return res.orbitals[0].P.grid.astype(np.float32)
        for o in res.orbitals:
            if _hf_channel(o.n, o.l) == channel:
                return o.P.values.astype(np.float32)
        known = ", ".join(
            ["grid", *(_hf_channel(o.n, o.l) for o in res.orbitals)]
        )
        raise HTTPException(
            status_code=422,
            detail=f"no channel {channel!r} on this job; it has {known}",
        )

    @app.get("/api/jobs/{job_id}/data")
    def job_data(job_id: str, channel: str | None = None) -> Response:
        res = _finished_result(jobs, job_id)
        if isinstance(res, PlaneGrid):
            if channel is not None:
                raise HTTPException(
                    status_code=422, detail="plane jobs have a single channel"
                )
            payload = res.values.astype(np.float32)
        elif isinstance(res, HFJobResult):
            payload = _hf_channel_payload(res.result, channel)
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
