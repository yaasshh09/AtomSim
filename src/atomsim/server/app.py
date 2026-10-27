"""The local server: the honest JSON and binary boundary the app talks to."""

import asyncio
import dataclasses
import functools
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, model_validator
from pydantic import Field as PydanticField
from scipy import constants as _sc

import atomsim
from atomsim.analytic.dirac import dirac_energy, dirac_kappa, dirac_moments, dirac_radial
from atomsim.analytic.fine_structure import fine_structure_shift, level_energy
from atomsim.analytic.hydrogen import (
    angular_momentum_magnitude,
    energy,
    mean_radius,
    momentum_wavefunction,
    radial_wavefunction,
    validate_quantum_numbers,
)
from atomsim.analytic.hyperfine import hyperfine_report
from atomsim.analytic.nodes import nodes
from atomsim.analytic.stark import stark_sublevels
from atomsim.analytic.wavefunction import WavefunctionValues, evaluate_state
from atomsim.analytic.zeeman import zeeman_sublevels
from atomsim.atoms import (
    ATOM_KEYS,
    SUBSHELL_LABELS,
    Configuration,
    atom_for_key,
    aufbau_configuration,
    element_by_z,
    format_config,
    has_gsz_parameters,
    is_atom_key,
    parse_config,
    total_electrons,
    validate_config,
    why_no_gsz,
)
from atomsim.broadening import synthesize
from atomsim.cascade import decay_cascade
from atomsim.classical import classical_ghost
from atomsim.constants import ALPHA, AU_TIME_FS, BOHR_RADIUS_FM, BOHR_RADIUS_PM, HARTREE_EV
from atomsim.constants_lab import analyze_constants
from atomsim.density_compare import compare_total_densities
from atomsim.expectations import hydrogen_expectations, hydrogen_virial
from atomsim.hf_atom import (
    HFResult,
    PauliCollapse,
    evaluate_hf_state,
    hf_exchange_energy,
    hf_radial,
    hf_total_radial_density,
    pauli_collapse,
    solve_hartree_fock,
)
from atomsim.hf_reference import NIST_IONIZATION
from atomsim.hybrids import HYBRIDS, evaluate_hybrid, hybrid_plane_grid, sample_hybrid
from atomsim.isoelectronic import (
    NIST_ISOELECTRONIC,
    SEQUENCE_ELECTRONS,
    sequence_coefficients,
    sequence_reference,
)
from atomsim.isosurface import (
    GRID_SIZES,
    Isosurface,
    hf_isosurface,
    isosurface,
    screened_isosurface,
)
from atomsim.nuclear_size import finite_size_shift, load_lamb_reference, muonic_lamb_shift
from atomsim.numerics.expression import ExpressionError
from atomsim.numerics.force_law import PRESETS, force_law_levels, free_form_levels
from atomsim.plane import (
    PlaneGrid,
    current_through_plane,
    hf_plane_grid,
    plane_grid,
    screened_plane_grid,
)
from atomsim.populations import ThermalConditions
from atomsim.provenance import Fidelity, Field, Provenance, Quantity
from atomsim.rabi import rabi_drive
from atomsim.sampling import (
    SampleCloud,
    sample_density,
    sample_hf_density,
    sample_screened_density,
)
from atomsim.screened_atom import (
    evaluate_screened_state,
    screened_radial,
    screened_total_radial_density,
    solve_screened_atom,
)
from atomsim.server.jobs import Job, JobStatus, JobStore
from atomsim.server.ratelimit import DEFAULT_CAPACITY, DEFAULT_PERIOD, TokenBucket
from atomsim.server.result_cache import DEFAULT_MAX_BYTES, ResultCache
from atomsim.server.schemas import (
    AbsorptionSpectrumModel,
    CascadeModel,
    ChannelModel,
    ClassicalGhostModel,
    ComparisonModel,
    ConstantsReportModel,
    CurveOfGrowthModel,
    DensityComparisonModel,
    DiracComponentModel,
    DiracRadialModel,
    FieldModel,
    FiniteSizeLevelModel,
    FiniteSizeModel,
    ForceLawLevelModel,
    ForceLawModel,
    HFOrbitalModel,
    HFResultModel,
    IsoelectronicIonModel,
    IsoelectronicModel,
    LambPredictionModel,
    LineModel,
    NodeSetModel,
    PauliCollapseModel,
    PotentialCurveModel,
    ProfileModel,
    ProvenanceModel,
    QuantityModel,
    RabiModel,
    ReferenceItemModel,
    ReferenceModel,
    ScreenedLevelsModel,
    ScreenedOrbitalModel,
    SystemModel,
    ThermalModel,
    VirialModel,
)
from atomsim.server.thumbnails import (
    OG_HEIGHT,
    OG_WIDTH,
    render_og_card,
    render_thumbnail,
    state_title,
)
from atomsim.snippets import snippet
from atomsim.spectra import (
    compare_lines,
    load_reference,
    screened_transition_lines,
    subshell_label,
    transition_lines,
)
from atomsim.superposition import (
    MAX_COMPONENTS,
    Component,
    SuperpositionGrid,
    superposition_grid,
)
from atomsim.systems import (
    element_emitter_mass,
    emitter_mass,
    get_system,
    hydrogen_like,
    list_systems,
)
from atomsim.transfer import absorb, curve_of_growth, default_columns

_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
_HARTREE_HZ = _sc.physical_constants["hartree-hertz relationship"][0]

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Make this package's log lines audible under uvicorn.

    uvicorn installs handlers for its `uvicorn*` loggers and leaves the root
    logger alone. A module logger propagating to an unconfigured root is
    not merely unformatted, it is dropped: Python's last-resort handler emits
    WARNING and above and discards the rest. The mount disclosure below is an
    INFO line, so without this it exists in the code, passes its test against
    `caplog`, and prints nothing on the host where it matters.

    Guarded rather than unconditional. If anything has already configured
    logging, the host or a test harness, it knows better than this where the
    output should go.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")


def _web_dist() -> Path:
    """Where the built frontend is looked for.

    The default assumes a source checkout: `parents[3]` is the repo root, the
    directory holding both `src/` and `web/`. Installed into site-packages the
    same expression resolves into the Python library directory, where there is
    no `web/dist` and never will be, so the mount below is skipped. That failure
    is silent by construction, so the override exists to let a container say
    where it put the build rather than hope.
    """
    override = os.environ.get("ATOMSIM_WEB_DIST")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "web" / "dist"


def _job_worker_count() -> int:
    """How many jobs may hold a CPU at once.

    The jobs are numpy-bound and release the GIL, so this pool size is a
    statement about cores, not about concurrency. asyncio's default executor
    would allow up to 32 of them, which is harmless on a 14-core laptop and
    ruinous on a one-vCPU host: an isosurface (1.7 s here) and a Hartree-Fock
    solve (1.4 s here) are seconds of pinned CPU each, and thirty of them
    timesharing one core turns every request slow instead of making any fast.

    The floor of 2 is deliberate. One worker would strictly serialize, which
    parks an 18 ms sample behind whatever long solve arrived first. Two lets a
    cheap request overtake, and on a single core they simply timeshare.
    """
    override = os.environ.get("ATOMSIM_JOB_WORKERS")
    if override:
        return max(1, int(override))
    return max(2, min(4, os.cpu_count() or 2))


def _warm_enabled() -> bool:
    """Precompute the popular states at startup? Off unless asked for (see fly.toml).

    Off by default because a laptop session and the test suite have no first
    visitor to save a wait for, and the warm-up would only take CPU from them.
    """
    return os.environ.get("ATOMSIM_WARM", "").lower() in ("1", "on", "true")


def _cache_bytes() -> int:
    override = os.environ.get("ATOMSIM_CACHE_MB")
    return int(float(override) * 1024 * 1024) if override else DEFAULT_MAX_BYTES


@asynccontextmanager
async def _lifespan(app: FastAPI):
    warm = asyncio.create_task(app.state.warm()) if _warm_enabled() else None
    yield
    if warm is not None:
        warm.cancel()
    app.state.executor.shutdown(wait=False, cancel_futures=True)


def _build_rate_limiter() -> TokenBucket | None:
    """The job-endpoint limiter, or None when it has been switched off.

    It is on by default, because forgetting to enable it on a public host is a
    worse failure than tripping it on a laptop, and the default burst is wider
    than any honest click-storm the UI can produce. That claim costs
    something to keep, so the row it is measured against is derived on
    `DEFAULT_CAPACITY` and asserted in `test_ratelimit.py`.
    """
    if os.environ.get("ATOMSIM_RATE_LIMIT", "").lower() in ("off", "0", "false"):
        return None
    return TokenBucket(
        capacity=int(os.environ.get("ATOMSIM_RATE_LIMIT_BURST", DEFAULT_CAPACITY)),
        period=float(os.environ.get("ATOMSIM_RATE_LIMIT_PERIOD", DEFAULT_PERIOD)),
    )


def _client_key(request, header: str | None) -> str:
    """Who gets charged for this request.

    Behind a proxy every request carries the proxy's address, so without the
    header the whole internet shares one bucket and the first busy visitor locks
    out the rest. The header is named explicitly rather than assumed, because
    trusting a forwarded address a client can set is the same as having no
    limiter at all.

    The rightmost entry is the one charged, and the distinction is not cosmetic.
    A forwarding proxy appends, so a caller that sends its own
    `X-Forwarded-For: someone-else` arrives as `someone-else, <real address>`.
    Charging the leftmost entry charges a string the caller typed, and varying
    it per request buys an unlimited supply of full buckets. Only the last hop
    was written by the proxy that was named as trusted.

    This assumes exactly one trusted proxy in front, which is what the
    deployment has. Behind two, the rightmost entry is the inner proxy and every
    client would share its bucket: wrong in the safe direction, and worth
    re-deriving rather than inheriting if another hop is ever added.
    """
    if header:
        forwarded = request.headers.get(header)
        if forwarded:
            candidate = forwarded.split(",")[-1].strip()
            if candidate:
                return candidate
    return request.client.host if request.client else "unknown"


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


class ClientErrorReport(BaseModel):
    """A browser-side crash. Capped so a report cannot be used to fill the log."""

    message: str = PydanticField(max_length=500)
    stack: str | None = PydanticField(default=None, max_length=4000)
    #: The page path, without the query string: the app state is not needed to debug.
    page: str = PydanticField(default="", max_length=200)
    version: str = PydanticField(default="", max_length=40)


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
    #: Closed-form moments and <T>, <V> (EXACT), then the same <T>, <V> integrated
    #: from the wavefunction as a live virial check (NUMERICAL).
    expectations: list[QuantityModel] = []
    virial: VirialModel | None = None


class SystemsResponse(BaseModel):
    systems: list[SystemModel]


class RadialResponse(BaseModel):
    n: int
    l: int
    system: SystemModel
    r_wavefunction: FieldModel
    radial_probability: FieldModel
    #: D(r) for the whole cloud, summed over occupied subshells. Sent only
    #: under the Hartree-Fock model, which is the one that knows the occupancies
    #: from its own solve. Null elsewhere rather than omitted, so a client reads
    #: one response shape whichever model it asked for.
    total_density: FieldModel | None = None
    #: Both models' D(r) on one grid, sent only on request, because it
    #: costs a second solve. Null rather than omitted for the same reason as
    #: above: one response shape, asked for or not.
    density_comparison: DensityComparisonModel | None = None
    #: p^2 |F_nl(p)|^2, the same orbital's momentum distribution. Hydrogen-like
    #: systems only, where it has a closed form; null for atoms, same shape.
    momentum_probability: FieldModel | None = None
    #: sqrt(<p^2>) = Z mu / n, exact from the virial theorem.
    momentum_rms: QuantityModel | None = None


class SpectrumResponse(BaseModel):
    system: SystemModel
    n_max: int
    fine_structure: bool
    lines: list[LineModel]
    comparison: list[ComparisonModel] | None
    reference_citation: str | None
    tolerance_relative: float | None
    #: Why the lines carry no strengths, when they were asked for but withheld.
    intensity_note: str | None = None
    #: Sent exactly when the lines carry an emissivity.
    thermal: ThermalModel | None = None
    #: The synthesized curve, when one was asked for and could be built. Null
    #: with `profile_note` set when it was asked for but no mechanism gives the
    #: lines any width, since an invented width would be the lie.
    profile: ProfileModel | None = None
    profile_note: str | None = None


class ManyElectronRequest(BaseModel):
    """The four fields that pick which many-electron model draws the picture.

    The sample, plane and isosurface requests share this rather than repeating
    it, because the pauli/exchange rule below has to be the same rule in all
    three and a copied validator is a rule waiting to drift.

    Every default is the already-shipped screened behaviour, so a client that
    has never heard of any of these fields cannot accidentally ask for
    Hartree-Fock or for a counterfactual. This mirrors HFRequest.
    """

    model: Literal["gsz", "hf"] = "gsz"
    #: Electron configuration, e.g. "1s2 2s2 2p5 3s1". None means the
    #: ground configuration for the rule in force. It is ignored under
    #: model="gsz", which has no per-subshell solve for it to change.
    config: str | None = None
    exchange: bool = True
    pauli: bool = True

    @model_validator(mode="after")
    def _pauli_off_implies_exchange_off(self) -> "ManyElectronRequest":
        """Refuse the combination rather than quietly flipping a flag.

        Same rule and the same 422 as HFRequest: a Slater determinant holding
        two electrons in one spin-orbital is identically zero, so there is no
        wavefunction there for an exchange integral to act on. Correcting it
        would hide that the request names a state that does not exist.

        This is scoped to model="hf" because these two flags name Hartree-Fock's
        rules and the screened model has neither an exchange term nor an
        occupancy cap of its own. A stale switch left set while the model
        changes is not an incoherent request; it is a control that does not
        apply.
        """
        if self.model == "hf" and not self.pauli and self.exchange:
            raise ValueError(
                "pauli=false requires exchange=false: exchange energy is a "
                "consequence of antisymmetry and the exclusion principle IS "
                "antisymmetry, so with the principle off there is nothing for "
                "an exchange integral to act on"
            )
        return self


class SampleRequest(ManyElectronRequest):
    n: int
    l: int
    m: int
    count: int = PydanticField(default=100_000, ge=1_000, le=1_000_000)
    seed: int = 0
    basis: Literal["complex", "real"] = "complex"
    system: str = "h"
    #: An sp/sp2/sp3 hybrid key ("sp3-1"); when set it replaces (n, l, m).
    hybrid: str | None = None


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
    #: Which many-electron model drew this. It is echoed back from the
    #: request so the browser can name what it is looking at without parsing
    #: provenance prose, which is a thing a view that has to parse prose
    #: eventually gets wrong.
    model: str = "gsz"
    #: The hybrid drawn instead of (n, l, m), if one was asked for.
    hybrid: str | None = None
    provenance: ProvenanceModel
    channels: list[ChannelModel]


@dataclasses.dataclass(frozen=True)
class SampleJobResult:
    """A sampled cloud, plus psi evaluated at exactly those positions."""

    cloud: SampleCloud
    psi: WavefunctionValues


class PlaneRequest(ManyElectronRequest):
    n: int
    l: int
    m: int
    quantity: Literal["density", "psi", "current"] = "density"
    basis: Literal["complex", "real"] = "complex"
    system: str = "h"
    resolution: int = PydanticField(default=512, ge=32, le=1024)
    #: As on SampleRequest.
    hybrid: str | None = None


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
    #: One bohr in picometres (CODATA, via scipy), so the client can draw a
    #: real-size scale bar without carrying a physical constant of its own.
    bohr_pm: float = BOHR_RADIUS_PM
    n: int
    l: int
    m: int
    basis: str
    system: str
    #: Which many-electron model drew this; see SampleMetaModel.model.
    model: str = "gsz"
    #: The hybrid drawn instead of (n, l, m), if one was asked for.
    hybrid: str | None = None
    provenance: ProvenanceModel


class ComponentModel(BaseModel):
    """One state in a superposition: a relative population and a phase in radians."""

    n: int
    l: int
    m: int
    weight: float = PydanticField(ge=0.0)
    phase: float = 0.0


class SuperpositionRequest(BaseModel):
    components: list[ComponentModel] = PydanticField(
        min_length=1, max_length=MAX_COMPONENTS
    )
    basis: Literal["complex", "real"] = "complex"
    system: str = "h"
    resolution: int = PydanticField(default=256, ge=32, le=512)


class SuperpositionComponentMeta(BaseModel):
    n: int
    l: int
    m: int
    #: The normalized c_k, split for JSON. sum |c_k|^2 = 1.
    re: float
    im: float
    energy: QuantityModel


class SuperpositionMetaModel(BaseModel):
    kind: Literal["superposition"] = "superposition"
    resolution: int
    dtype: str
    layout: str
    unit: str
    half_extent: float
    axis_unit: str
    basis: str
    system: str
    components: list[SuperpositionComponentMeta]
    #: hbar / E_h in fs, so the browser's clock (which runs in atomic units,
    #: because E_k is in hartree) can be read out in femtoseconds.
    #: One bohr in picometres, as on PlaneMetaModel.
    bohr_pm: float = BOHR_RADIUS_PM
    time_unit_fs: float
    #: The distinct nonzero beat periods 2 pi hbar / |E_j - E_k|, ascending
    #: frequency. Empty means the mix is stationary.
    beat_periods: list[QuantityModel]
    provenance: ProvenanceModel


class IsoRequest(ManyElectronRequest):
    """An isosurface requested by enclosed fraction, never by a level.

    A raw contour value is meaningless without the grid it was measured on, and
    a client that could pass one would be choosing a picture rather than asking
    a question. `fraction` is the question, and the level comes back with the
    answer.

    The box is not exposed either. It is fitted to hold 99.9% of the electron
    with the remainder reported, so the only available surface is one whose
    escaped mass has been measured.
    """

    n: int
    l: int
    m: int
    fraction: float = PydanticField(default=0.9, gt=0.0, lt=1.0)
    resolution: int = 96
    basis: Literal["complex", "real"] = "complex"
    system: str = "h"

    @model_validator(mode="after")
    def _resolution_is_offered(self) -> "IsoRequest":
        if self.resolution not in GRID_SIZES:
            raise ValueError(
                f"resolution must be one of {GRID_SIZES}, got {self.resolution}"
            )
        return self


class IsoMetaModel(BaseModel):
    kind: Literal["isosurface"] = "isosurface"
    vertex_count: int
    triangle_count: int
    channels: list[ChannelModel]
    target_fraction: float
    enclosed_fraction: QuantityModel
    outside_fraction: float
    level: QuantityModel
    escaped_fraction: QuantityModel
    mesh_volume: QuantityModel
    voxel_volume: QuantityModel
    area: QuantityModel
    components: int
    half_width: float
    resolution: int
    axis_unit: str
    n: int
    l: int
    m: int
    basis: str
    system: str
    #: Which many-electron model drew this; see SampleMetaModel.model.
    model: str = "gsz"
    label: str
    provenance: ProvenanceModel


class HFRequest(BaseModel):
    z: int
    n_electrons: int | None = None  # defaults to neutral
    config: str | None = None       # defaults to the aufbau ground configuration
    # False solves the Hartree model instead: electrons that repel but are
    # distinguishable. The default is real physics, so a client that has never
    # heard of the toggle cannot accidentally ask for the counterfactual.
    exchange: bool = True
    # False lifts the occupancy cap too, and the configuration collapses to
    # 1s^N. Same default and the same reason.
    pauli: bool = True

    @model_validator(mode="after")
    def _pauli_off_implies_exchange_off(self) -> "HFRequest":
        """Refuse the combination rather than quietly flipping a flag.

        422 rather than 400 because this is a request that cannot be understood,
        not one being declined: pauli=False with exchange=True does not name a
        model this or any other solver could run. A Slater determinant holding
        two electrons in the same spin-orbital is identically zero, so there is
        no wavefunction there for an exchange integral to act on. Correcting it
        would hide that the request names a state that does not exist.
        """
        if not self.pauli and self.exchange:
            raise ValueError(
                "pauli=false requires exchange=false: exchange energy is a "
                "consequence of antisymmetry and the exclusion principle IS "
                "antisymmetry, so with the principle off there is nothing for "
                "an exchange integral to act on"
            )
        return self


# The outermost principal quantum number that actually converges.
#
# Measured, and NOT a statement about Z. Argon-like ions come out cleanly all
# the way to Z = 36 (virial 2.000003 at every Z tried), and K+ and Ca2+ solve in
# about six seconds. What used to defeat the solver was the 4s channel of a
# neutral alkali: the starting potential fell back to the bare nucleus above
# argon, which put potassium's 4s guess at -11.3 hartree against a true -0.15,
# and LOBPCG stagnated. The Thomas-Fermi-Latter starting field fixed that
# (hf_atom._thomas_fermi_latter), and every neutral atom to krypton now lands
# on its Hartree-Fock limit within the solver's own error bar. n = 5 has not
# been exercised, so it is refused up front rather than left to fail inside a job.
_HF_MAX_N = 4
# As far as this has been exercised. Beyond it the answer is untested, and a
# non-relativistic model is also getting thin: the neglected relativity is
# already 1.8% of the 1s energy at Z = 36 (hf_atom quantifies it in the
# provenance from Z = 9 up).
_HF_MAX_Z = 36

# Five sequences, each a few milliseconds of quadrature; no reason to redo them.
_sequence_coefficients = functools.cache(sequence_coefficients)


@dataclasses.dataclass(frozen=True)
class HFJobResult:
    """A finished solve, plus what exchange was worth when it was measured.

    A wrapper rather than a second endpoint, because the exchange energy is a
    difference between two solves and the two have to be the same solves. If a
    view fetched the Hartree-Fock energy from one job and the Hartree energy
    from another, nothing would stop it differencing a warm result
    against a cold one, or a fine mesh against a coarse one, and reporting the
    gap between two calculations as the gap between two models.

    None when the job ran only the real model, which is the common case and
    costs one solve rather than two.
    """

    result: HFResult
    exchange_energy: Quantity | None = None
    #: The real atom beside the collapsed one, on a pauli=False ground solve.
    #: Same reasoning as exchange_energy: a comparison between two models has to
    #: be made from two solves that are known to match, so it is assembled here
    #: rather than left to a client to difference across two jobs.
    collapse: PauliCollapse | None = None


def _parse_config_or_422(text: str, pauli: bool = True):
    """Parse a hand-written configuration string, 422 on malformed input.

    Two status codes are in play here and the split is deliberate. 422 means the
    request could not be understood, since "2s^9" is not a configuration. 400
    means it was understood perfectly and is being declined, which is what
    _validate_hf_request returns: a neutral potassium atom is a real, well-posed
    request that cannot be answered honestly.

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
    """Refuse what cannot be done, with the reason, before starting a job."""
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
                f"this configuration occupies n = {n_top}, and convergence "
                f"reaches only n = {_HF_MAX_N}. The {n_top}s guess starts from a "
                f"bare-nucleus potential, which for a diffuse outer shell is far "
                f"too contracted for the self-consistent loop to recover; the "
                f"eigensolver stagnates rather than converging slowly. Ions "
                f"whose outermost shell is n <= {_HF_MAX_N} are fine at any Z up "
                f"to {_HF_MAX_Z}"
            ),
        )


def _hf_channel(n: int, l: int) -> str:
    return f"P_{n}{SUBSHELL_LABELS[l]}"


def _hf_symbol(z: int) -> str | None:
    """The element symbol, or None above the preset table.

    Hartree-Fock solves further up than the preset library reaches, and naming
    the element is a convenience for the view rather than part of the physics,
    so not having one is not an error.
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
                # The mesh is a choice, not a measurement of the atom.
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


#: The app's n select stops at 6 and its URL parser clamps there, so a card for
#: n = 7 would preview a state the link does not open.
_OG_CARD_N_MAX = 6

#: The URL keys that name a state; a link carrying none of them is the bare
#: front page, which keeps its designed og.png.
_STATE_KEYS = ("n", "l", "m", "system", "basis")


def _og_card_state(n: str, l: str, m: str, system: str, basis: str):
    """(n, l, m, system, basis) if a preview card can be drawn for it, else None.

    Only the named hydrogen-like presets: a many-electron atom's orbital is not
    the closed-form density the card is labelled with.
    """
    try:
        n_, l_, m_ = int(n), int(l), int(m)
        validate_quantum_numbers(n_, l_)
        get_system(system)
    except (ValueError, KeyError):
        return None
    if n_ > _OG_CARD_N_MAX or abs(m_) > l_ or basis not in ("complex", "real"):
        return None
    return n_, l_, m_, system, basis


def _set_meta(html: str, key: str, value: str) -> str:
    """Replace the content of the <meta> whose property or name is `key`."""
    value = escape(value, quote=True)
    return re.sub(
        rf'((?:property|name)="{re.escape(key)}"\s+content=")[^"]*',
        lambda mt: mt.group(1) + value, html,
    )


def _with_state_preview(html: str, query) -> str:
    """index.html with its link-preview tags pointed at the state the URL opens.

    Scrapers never run the app's JavaScript, so this is the only place a shared
    deep link can say which orbital it is. Anything that is not a drawable state
    leaves the page exactly as built.
    """
    if not any(k in query for k in _STATE_KEYS):
        return html
    state = _og_card_state(
        query.get("n", "1"), query.get("l", "0"), query.get("m", "0"),
        query.get("system", "h"), query.get("basis", "complex"),
    )
    found = re.search(r'property="og:image"\s+content="([^"]*)/og\.png"', html)
    if state is None or found is None:
        return html
    n, l, m, system, basis = state
    image = f"{found.group(1)}/api/og/{n}/{l}/{m}?system={system}&basis={basis}"
    label = f"{get_system(system).name} {state_title(n, l, m, basis)}"
    alt = f"The {label} orbital's probability density on the y = 0 plane."
    for key, value in (
        ("og:title", f"{label} | AtomSim"), ("twitter:title", f"{label} | AtomSim"),
        ("og:image", image), ("twitter:image", image), ("og:image:alt", alt),
        ("og:image:width", str(OG_WIDTH)), ("og:image:height", str(OG_HEIGHT)),
    ):
        html = _set_meta(html, key, value)
    return html


def _rescaled(q: Quantity, factor: float, unit: str, via: str) -> Quantity:
    """A display conversion. The error bar is in the quantity's unit, so it moves too."""
    err = q.provenance.error_estimate
    return Quantity(
        value=q.value * factor,
        unit=unit,
        label=q.label + f" [{unit}]",
        provenance=dataclasses.replace(
            q.provenance,
            method=q.provenance.method + f"; converted to {unit} via {via}",
            error_estimate=None if err is None else err * abs(factor),
        ),
    )


def _to_ev(q: Quantity) -> Quantity:
    return _rescaled(q, HARTREE_EV, "eV", "CODATA Hartree-eV factor")


def _hyperfine_shell_model(rep) -> "HyperfineShellModel":
    """Map an available HyperfineReport to its response model, in eV as well."""
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
    return _rescaled(q, BOHR_RADIUS_PM, "pm", "CODATA Bohr radius")


def _job_model(job: Job) -> JobModel:
    return JobModel(id=job.id, status=job.status.value, progress=job.progress, error=job.error)


def _finished_result(jobs: JobStore, job_id: str):
    """A finished job's result container (sample or, later, plane)."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    if job.status is not JobStatus.DONE:
        raise HTTPException(status_code=409, detail=f"job is {job.status.value}, not done")
    return job.result


def create_app() -> FastAPI:
    _configure_logging()
    app = FastAPI(title="atomsim", version=atomsim.__version__, lifespan=_lifespan)
    app.state.job_systems = {}
    # Parallel to job_systems, and for the same reason: the meta endpoint sees
    # only the finished result object, and a SampleCloud does not know which
    # model produced it.
    app.state.job_models = {}
    #: Hybrid keys per job, for the same reason as job_models.
    app.state.job_hybrids = {}

    def _forget_job(job_id: str) -> None:
        """Keep the side tables in step with the store they are keyed against."""
        app.state.job_systems.pop(job_id, None)
        app.state.job_models.pop(job_id, None)
        app.state.job_hybrids.pop(job_id, None)

    jobs = JobStore(on_evict=_forget_job)
    app.state.jobs = jobs
    results = ResultCache(_cache_bytes())
    app.state.results = results
    app.state.executor = ThreadPoolExecutor(
        max_workers=_job_worker_count(), thread_name_prefix="atomsim-job"
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=_DEV_ORIGINS, allow_methods=["*"], allow_headers=["*"]
    )

    app.state.rate_limit = _build_rate_limiter()
    client_ip_header = os.environ.get("ATOMSIM_CLIENT_IP_HEADER")

    @app.middleware("http")
    async def _limit_job_creation(request, call_next):
        """Meter only the endpoints that buy CPU.

        Reads and static files are cheap and cacheable, and limiting them
        would throttle the page itself. Only a job POST commits the host to
        seconds of work, so only a job POST is charged.
        """
        limiter = app.state.rate_limit
        if (
            limiter is not None
            and request.method == "POST"
            and request.url.path.startswith("/api/jobs/")
        ):
            charged = _client_key(request, client_ip_header)
            wait = limiter.check(charged)
            if wait is not None:
                retry = max(1, math.ceil(wait))
                # Named here, because a refusal is the one moment the key matters
                # operationally: it says whether one visitor is hammering the
                # host, or whether the whole internet is sharing one bucket
                # because nobody configured the forwarded-address header.
                logger.warning("rate limit refused %s; retry in %ds", charged, retry)
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": str(retry)},
                    content={
                        "detail": (
                            f"too many compute jobs from this client; retry in "
                            f"{retry}s. Each job is seconds of solver time, "
                            f"so the rate is capped to stay responsive for everyone."
                        )
                    },
                )
        return await call_next(request)

    def _dispatch(job, work, req: BaseModel) -> JobModel:
        """Hand a job to the bounded pool and answer with its id immediately.

        Every job endpoint ends this way. Routing them all through one
        function is what keeps the pool bounded: a new endpoint that reached for
        `run_in_executor(None, ...)` would silently opt back into the 32-thread
        default, and nothing would look wrong until the host fell over.

        It is also where the result cache sits, keyed on the validated request,
        so no endpoint can forget it or key it differently. A hit still runs as
        a job and reports through the same websocket; it only finishes at once.
        """
        key = f"{type(req).__name__}:{req.model_dump_json()}"

        def cached(progress):
            hit = results.get(key)
            if hit is not None:
                progress(1.0)
                return hit
            out = work(progress)
            results.put(key, out)
            return out

        loop = asyncio.get_running_loop()
        loop.run_in_executor(app.state.executor, jobs.run, job.id, cached)
        return _job_model(job)

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
        except (KeyError, ValueError) as exc:  # ValueError: a mass key out of range
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def _is_screened(key: str) -> bool:
        """Whether this key names a many-electron atom, whatever model draws it."""
        return is_atom_key(key)

    def _gsz_element(key: str):
        """The element, or a refusal if GSZ has no parameters for it.

        This sits at the top of every screened branch, in place of the bare
        `atom_for_key` those branches used when the two lists were one. Sulfur
        and chlorine reach these endpoints now (they are real atoms and the
        picker offers them), and without this the fault would land somewhere
        inside the screening table with a ValueError that names no way forward.

        400 rather than 422: the request is well posed and understood, and it is
        being declined, which is the same split `_validate_hf_request`
        already draws for a neutral potassium atom.
        """
        element = atom_for_key(key)
        if not has_gsz_parameters(element.z):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{element.name} has no published GSZ screening parameters: "
                    f"{why_no_gsz(element.z)}. Inventing "
                    f"them would mean shipping physics with no source. Ask "
                    f"for model='hf' instead: Hartree-Fock builds its potential "
                    f"out of the orbitals it is solving for and needs no fitted "
                    f"table, which is why this atom is offered at all"
                ),
            )
        return element

    def _resolve_thermal(
        temperature_k: float | None, electron_density_cm3: float | None
    ) -> ThermalConditions | None:
        """Both knobs or neither: half of Saha is not a state anyone can read.

        These bounds are display limits, not physics limits. Below ~100 K
        every excited level is empty and the spectrum is a single dark band;
        above ~10^6 K hydrogen is long gone. The formulas hold outside; the view
        has nothing to show there.
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

    def _many_electron_target(
        system: str, config: str | None, pauli: bool
    ) -> tuple[int, int, Configuration]:
        """(Z, N, configuration) for any many-electron picture, or a refusal.

        This is the half of the old `_hf_view_target` that is about which atom
        is being solved rather than which orbital is being drawn. It was split
        out because a total density needs the atom and does not depend on (n, l)
        at all, so refusing a density comparison over an unoccupied subshell would
        refuse a legitimate request about an orbital nobody asked about.

        Every refusal here is synchronous and carries its reason, because the
        alternative is a job that dies several seconds in with an engine message
        the client has to guess at.
        """
        if not _is_screened(system):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"this model needs an atom with a known electron count, "
                    f"and {system!r} is a one-electron system; its exact "
                    f"wavefunction is already in the other views, so there "
                    f"is nothing a self-consistent field would add"
                ),
            )
        element = atom_for_key(system)
        n_electrons = element.z
        cfg = (
            aufbau_configuration(n_electrons, pauli)
            if config is None
            else _parse_config_or_422(config, pauli)
        )
        _validate_hf_request(element.z, n_electrons, cfg, pauli)
        return element.z, n_electrons, cfg

    def _hf_view_target(req) -> tuple[int, int, Configuration]:
        """The atom, plus the occupancy check only an orbital picture needs.

        That check costs no solve: which subshells exist is a property of the
        configuration, and the configuration is in the request.
        """
        z, n_electrons, config = _many_electron_target(
            req.system, req.config, req.pauli
        )
        if (req.n, req.l) not in [nl for nl, _ in config]:
            held = ", ".join(f"{n}{SUBSHELL_LABELS[l]}" for (n, l), _ in config)
            why = (
                "the occupancy cap is lifted, so every electron is in the 1s "
                "and there is no other orbital to draw"
                if not req.pauli
                else "Hartree-Fock builds one Fock operator per occupied "
                "subshell, so an empty one has no operator to be an "
                "eigenfunction of, and borrowing another subshell's operator "
                "would silently answer a different question"
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{req.n}{SUBSHELL_LABELS[req.l]} is not occupied in "
                    f"{atom_for_key(req.system).symbol} ({held}); {why}"
                ),
            )
        return z, n_electrons, config

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": atomsim.__version__}

    #: Its own small bucket: an error report costs no CPU, but a page stuck in an
    #: error loop must not be able to fill the host's log.
    error_reports = TokenBucket(capacity=5, period=300.0)

    @app.post("/api/client-error", status_code=204)
    def client_error(report: ClientErrorReport, request: Request) -> Response:
        """A browser-side crash, written to the log as one line and kept nowhere else."""
        if error_reports.check(_client_key(request, client_ip_header)) is not None:
            return Response(status_code=429)

        def flat(text: str) -> str:
            # One line per report, whatever the report contains: a newline in a
            # stack would otherwise let a client write lines that look like ours.
            return " | ".join(text.splitlines())

        logger.warning(
            "client error on %s (version %s): %s :: %s",
            flat(report.page), flat(report.version), flat(report.message),
            flat(report.stack or ""),
        )
        return Response(status_code=204)

    @app.get("/api/systems", response_model=SystemsResponse)
    def systems() -> SystemsResponse:
        """Every selectable system, and which models can speak for it.

        All nineteen atoms are listed, including the four GSZ has no parameters
        for. `has_gsz` is what tells them apart, so the client greys one control
        rather than hiding an atom that solves perfectly well; the description
        carries the same fact in words, because a greyed control with no
        sentence beside it is a dead end.
        """

        def describe(element) -> str:
            if has_gsz_parameters(element.z):
                return (
                    f"{element.name}: GSZ screened central-field model "
                    f"(APPROXIMATION), or self-consistent Hartree-Fock."
                )
            gap = (
                "Szydlik and Green's 1974 table stops at argon"
                if element.z > 18
                else "Szydlik and Green never published neutral GSZ screening "
                f"parameters for Z = {element.z}"
            )
            return (
                f"{element.name}: Hartree-Fock only (APPROXIMATION). "
                f"{gap}, and Hartree-Fock needs none."
            )

        hydrogenic = [SystemModel.from_system(s) for s in list_systems()]
        screened = [
            SystemModel.from_atom(
                el, n_electrons=el.z, description=describe(el),
            )
            for el in (atom_for_key(k) for k in ATOM_KEYS)
        ]
        return SystemsResponse(systems=hydrogenic + screened)

    @app.get("/api/system/{key}", response_model=SystemModel)
    def system_one(key: str) -> SystemModel:
        """One hydrogenic system by key, including the nuclear-mass family.

        The table above lists the presets only: a continuum of masses cannot be
        enumerated, so a client holding an `m<M>` key asks for it here.
        """
        return SystemModel.from_system(_resolve_system(key))

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
            expectations=[
                QuantityModel.from_quantity(q)
                for q in hydrogen_expectations(n, l, Z=sys_.Z, mu_ratio=mu)
            ],
            virial=VirialModel.from_check(hydrogen_virial(n, l, Z=sys_.Z, mu_ratio=mu)),
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
            element = _gsz_element(system)
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
                # availability does not depend on n: one honest reason,
                # not n_max copies of it.
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
                    virial=VirialModel.from_check(c.virial) if c.virial else None,
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
    def radial(
        n: int,
        l: int,
        system: str = "h",
        points: int = 400,
        model: Literal["gsz", "hf"] = "gsz",
        config: str | None = None,
        exchange: bool = True,
        pauli: bool = True,
        compare: bool = False,
    ) -> RadialResponse:
        _validate_state(n, l, 0)
        if not 50 <= points <= 2000:
            raise HTTPException(status_code=422, detail="points must be in [50, 2000]")

        def _comparison() -> DensityComparisonModel | None:
            """Both densities, or nothing, or a refusal with the reason.

            This calls the two resolvers that already know when a model cannot
            speak, so the wording and status codes cannot drift from the ones the
            model radio shows: 400 from `_gsz_element` for sulfur and chlorine,
            422 from `_many_electron_target` for a one-electron system. Nothing
            new is refused here.

            The configuration comes from the Hartree-Fock side and goes to
            both, so that with the occupancy cap off the overlay compares two
            models of the same altered atom rather than two different atoms.

            Order matters. `_many_electron_target` runs first because it is the
            one that refuses a one-electron system, and `_gsz_element` expects
            an atom key: asked about hydrogen first it would fail looking the
            element up rather than returning the 422 that says why.
            """
            if not compare:
                return None
            z, n_electrons, cfg = _many_electron_target(system, config, pauli)
            _gsz_element(system)
            return DensityComparisonModel.from_comparison(
                compare_total_densities(
                    z, n_electrons, config=cfg, exchange=exchange, pauli=pauli,
                )
            )

        if model == "hf":
            if not pauli and exchange:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "pauli=false requires exchange=false: exchange energy "
                        "is a consequence of antisymmetry and the exclusion "
                        "principle IS antisymmetry, so with the principle off "
                        "there is nothing for an exchange integral to act on"
                    ),
                )
            # An object, because _hf_view_target reads attributes, and
            # one resolver behind four views is what keeps the refusals from
            # drifting apart between the picture endpoints and this one.
            hf_z, hf_n, hf_config = _hf_view_target(
                SimpleNamespace(system=system, config=config, pauli=pauli, n=n, l=l)
            )
            rw, p = hf_radial(
                hf_z, hf_n, n, l, points=points,
                config=hf_config, exchange=exchange, pauli=pauli,
            )
            # Free beside the orbital: the same cached solve, summed rather
            # than picked from. Sent unasked because the two curves above
            # it carry a caption saying they are not observable, and shipping
            # that claim without the thing it points at is half a sentence.
            density = hf_total_radial_density(
                hf_z, hf_n, config=hf_config, exchange=exchange, pauli=pauli,
                points=points,
            )
            element = atom_for_key(system)
            return RadialResponse(
                n=n, l=l,
                system=SystemModel.from_atom(
                    element, element.z,
                    f"{element.name}: self-consistent Hartree-Fock "
                    f"(APPROXIMATION; COUNTERFACTUAL with a switch thrown).",
                ),
                r_wavefunction=FieldModel.from_field(rw),
                radial_probability=FieldModel.from_field(p),
                total_density=FieldModel.from_field(density),
                density_comparison=_comparison(),
            )
        if _is_screened(system):
            element = _gsz_element(system)
            rw, p = screened_radial(element.z, element.z, n, l, points=points)
            # The orbital above does not depend on the configuration (in a
            # central field it cannot, since (Z, N) alone fix the field), but
            # the density does, because the configuration is what says which
            # orbitals are occupied. So this is the first thing on this branch
            # that has to ask.
            density = screened_total_radial_density(
                element.z, element.z,
                config=_resolve_config(system, config), points=points,
            )
            return RadialResponse(
                n=n, l=l,
                system=SystemModel.from_atom(
                    element, element.z,
                    f"{element.name}: GSZ screened central-field model (APPROXIMATION).",
                ),
                r_wavefunction=FieldModel.from_field(rw),
                radial_probability=FieldModel.from_field(p),
                total_density=FieldModel.from_field(density),
                density_comparison=_comparison(),
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
        # Momentum scales as Z mu / n where r scales as n^2 / (Z mu), and the
        # s-state tail falls as p^-6, under 1e-4 of the peak by 12 Z mu / n.
        p_grid = np.linspace(0.0, 12.0 * sys_.Z * mu / n, points)
        fp = momentum_wavefunction(n, l, p_grid, Z=sys_.Z, mu_ratio=mu)
        momentum = Field(
            values=p_grid * p_grid * fp.values**2,
            grid=p_grid,
            unit="(hbar/bohr)^-1",
            grid_unit="hbar/bohr",
            label=f"Pi_{n},{l}(p) = p^2 F^2",
            provenance=fp.provenance,
        )
        p_rms = Quantity(
            value=sys_.Z * mu / n,
            unit="hbar/bohr",
            label="sqrt(<p^2>)",
            provenance=Provenance(
                fidelity=Fidelity.EXACT,
                method="virial theorem: <p^2> = 2 mu <T> = -2 mu E_n = (Z mu / n)^2",
                assumptions=fp.provenance.assumptions,
            ),
        )
        return RadialResponse(
            n=n, l=l, system=SystemModel.from_system(sys_),
            r_wavefunction=FieldModel.from_field(rw),
            radial_probability=FieldModel.from_field(p),
            density_comparison=_comparison(),
            momentum_probability=FieldModel.from_field(momentum),
            momentum_rms=QuantityModel.from_quantity(p_rms),
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

        Same structural rule the view uses for its bar axis: across-n lines set
        the range, because a fine-structure list also holds within-n components
        out at millimetres to metres, and stretching a synthesis over eleven
        decades of wavelength spends the whole point budget on empty space. None
        means "no split applies, use the lot".
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

        The failure mode here is a feature: with no decay rate, no temperature
        and no instrument, every line has zero width, and the only way to
        draw a curve would be to invent one. The note names the knob instead.

        A `zoom` window is where this phase earns its keep: a profile only shows
        its shape when the axis is narrow enough to resolve it, and the whole
        point budget then lands on the one line being looked at.
        """
        window = zoom if zoom else (None if full_range else _profile_window(lines.lines))
        try:
            syn = synthesize(
                lines, emitter_mass=mass, hydrogenic=hydrogenic,
                resolving_power=resolving_power, window_nm=window,
                # A transport cap, not a physics one. The closure stays
                # inside 0.1% here, and what it actually was gets reported.
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
            element = _gsz_element(system)
            cfg = _resolve_config(system, config)
            result = solve_screened_atom(element.z, total_electrons(cfg), cfg)
            lines = screened_transition_lines(
                result, intensities=intensities, thermal=thermal
            )
            reference = load_reference(system)
            comparison = citation = tol = None
            if reference is not None:
                tol = 0.05  # the 5% pass bar, disclosed rather than hidden
                # Wide association window (25%): a GSZ valence line sits
                # a few percent off the real wavelength but is the correct
                # transition, so it is reported as a residual rather than
                # silently dropped.
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

        Both transfer endpoints want exactly this and want it identically:
        a curve of growth and an absorption spectrum that disagreed about which
        lines exist would be two answers about one gas.
        """
        if _is_screened(system):
            element = _gsz_element(system)
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
        """A whole line list in front of a flat continuum, and what survives it.

        One column density for the element comes in; each line's own
        lower-level fraction turns it into that line's absorbers. That is what
        makes the Lyman lines go black while the Balmer lines stay invisible in
        the same gas, and it is the fact the emission endpoint cannot represent.

        The window is left to the synthesis unless asked for, because sizing
        it by eye is how a third of an equivalent width went missing in Phase 19
        without anything reporting a problem.
        """
        # Absorption has no meaning without populations, so unlike the emission
        # endpoint these conditions are not optional. They are defaulted
        # rather than refused, and the gas they describe rides back on the
        # response, so a spectrum is never a shape with no conditions
        # attached.
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

        The line is taken by wavelength rather than by quantum numbers so the
        view can hand back whatever was clicked. The widths come from the same
        Phase 18 synthesis that drew the profile, so this curve and the profile
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
        # Take the nearest computed wavelength to the one asked for, then
        # the strongest transition sitting on it. Both steps matter: "H-alpha"
        # is three lines at exactly 656.4696 nm with oscillator strengths of
        # 0.014, 0.435 and 0.696, so taking whichever comes first in the list
        # would draw a curve of growth for the weakest of them and call it
        # H-alpha. The pairing comes from `synthesize` rather than from a
        # second wavelength match, which could not tell those three apart.
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
                    f"the {width.label} line has no oscillator strength, so "
                    "there is no absorption cross-section for it and no curve "
                    "of growth"
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

    @app.get("/api/snippet")
    def snippet_endpoint(
        view: str = "cloud", n: int = 1, l: int = 0, m: int = 0, system: str = "h",
        basis: Literal["complex", "real"] = "complex", model: Literal["gsz", "hf"] = "gsz",
        hybrid: str | None = None, plane: Literal["density", "psi", "current"] = "density",
        count: int = Query(default=100_000, ge=1_000, le=1_000_000),
        sup: list[str] = Query(default=[]),
    ) -> dict[str, str]:
        """The Python that reproduces this view (see snippets.py, whose tests run it)."""
        _validate_state(n, l, m)
        if not is_atom_key(system):
            _resolve_system(system)  # a 422 now beats a snippet that cannot run
        if hybrid is not None and hybrid not in HYBRIDS:
            raise HTTPException(status_code=422, detail=f"unknown hybrid {hybrid!r}")
        try:
            code = snippet(
                view, n=n, l=l, m=m, system=system, basis=basis, model=model,
                hybrid=hybrid, plane=plane, count=count, sup=sup or None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"code": code}

    @app.get("/api/reference/ionization")
    def ionization_reference() -> dict:
        """The vendored NIST first ionization energies, He to Ca, citation and all.

        Reference data rather than a result, so it carries a citation and a
        retrieval date instead of a fidelity tier, like the NIST line lists.
        """
        return NIST_IONIZATION

    @app.get("/api/isoelectronic/{n_electrons}", response_model=IsoelectronicModel)
    def isoelectronic_endpoint(n_electrons: int) -> IsoelectronicModel:
        """One electron count, Z turned up: the 1/Z coefficients and NIST's ions.

        The Hartree-Fock energies themselves come from /api/jobs/hf, one ion at
        a time, so this stays a millisecond GET and every solve is memoized.
        """
        if n_electrons not in SEQUENCE_ELECTRONS:
            raise HTTPException(
                status_code=422,
                detail=f"no vendored sequence for N = {n_electrons}; "
                f"available: {list(SEQUENCE_ELECTRONS)}",
            )
        coeffs = _sequence_coefficients(n_electrons)
        ref = sequence_reference(n_electrons)
        return IsoelectronicModel(
            n_electrons=n_electrons,
            isoelectronic_with=ref["isoelectronic_with"],
            config=format_config(coeffs.config),
            e0=QuantityModel.from_quantity(coeffs.e0),
            e1=QuantityModel.from_quantity(coeffs.e1),
            ions=[
                IsoelectronicIonModel(
                    z=ion["z"], symbol=ion["spectrum"].split()[0], spectrum=ion["spectrum"],
                    ie_ev=ion["ie_ev"], uncertainty_ev=ion["uncertainty_ev"], kind=ion["kind"],
                )
                for ion in ref["ions"]
            ],
            citation=NIST_ISOELECTRONIC["citation"],
            retrieved=NIST_ISOELECTRONIC["retrieved"],
            note=NIST_ISOELECTRONIC["note"],
        )

    @app.get("/api/nodes/{n}/{l}/{m}", response_model=NodeSetModel)
    def nodes_endpoint(
        n: int, l: int, m: int, system: str = "h", basis: str = "complex"
    ) -> NodeSetModel:
        """The exact nodal surfaces, hydrogen-like systems only.

        A screened or Hartree-Fock orbital has nodes too, but they are zeros of a
        numerical function; reporting them through this EXACT endpoint would
        give them a tier they have not earned, so they are refused here.
        """
        _validate_state(n, l, m)
        if _is_screened(system):
            raise HTTPException(
                status_code=400,
                detail="exact nodes exist only for the hydrogen-like systems; a "
                "many-electron orbital's nodes are zeros of a numerical function",
            )
        if basis not in ("complex", "real"):
            raise HTTPException(status_code=422, detail=f"unknown basis {basis!r}")
        sys_ = _resolve_system(system)
        return NodeSetModel.from_nodes(
            nodes(n, l, m, Z=sys_.Z, mu_ratio=sys_.mu_ratio.value, basis=basis)
        )

    @app.get("/api/cascade/{n}/{l}", response_model=CascadeModel)
    def cascade_endpoint(n: int, l: int, system: str = "h") -> CascadeModel:
        """Where an excited hydrogen-like atom goes: branches, photons, populations.

        Hydrogenic rates only. A screened atom's cascade would need its rates
        over every lower orbital, including ones its own electrons fill, so it
        is refused rather than approximated with the hydrogen numbers.
        """
        _validate_state(n, l, 0)
        if n > 10:
            raise HTTPException(status_code=422, detail="n must be <= 10 for a cascade")
        if _is_screened(system):
            raise HTTPException(
                status_code=400,
                detail="the decay cascade is computed for hydrogen-like systems only",
            )
        sys_ = _resolve_system(system)
        return CascadeModel.from_cascade(
            decay_cascade(n, l, Z=sys_.Z, mu_ratio=sys_.mu_ratio.value)
        )

    @app.get("/api/dirac-radial/{n}/{l}", response_model=DiracRadialModel)
    def dirac_radial_endpoint(n: int, l: int, z: int = 1, points: int = 400) -> DiracRadialModel:
        """The exact Dirac radial functions of a hydrogen-like ion of charge z, every j.

        Any z from 1 to 118 is a real hydrogen-like ion (one electron left), so
        this takes z directly rather than a preset. The nucleus is infinitely
        heavy here, which every value's assumptions say: the recoil it drops is
        m_e/M, 5e-4 for hydrogen and smaller for every heavier ion.
        """
        heavy = (
            "an infinitely heavy nucleus (mu = m_e): the dropped recoil is m_e/M, "
            "5e-4 for hydrogen and smaller for heavier ions"
        )

        def _h(q):
            """The same value, with the heavy-nucleus assumption stated on it."""
            p = q.provenance
            return dataclasses.replace(
                q, provenance=dataclasses.replace(p, assumptions=p.assumptions + (heavy,))
            )

        _validate_state(n, l, 0)
        if n > 6:
            raise HTTPException(status_code=422, detail="n must be <= 6")
        if not 1 <= z <= 118:
            raise HTTPException(status_code=422, detail="z must be in [1, 118]")
        if not 50 <= points <= 2000:
            raise HTTPException(status_code=422, detail="points must be in [50, 2000]")
        r_max = 20.0 * n * n / z
        # r = 0 is left off: for j = 1/2 the point-nucleus functions diverge there.
        r = np.linspace(r_max / points, r_max, points)
        schr = _h(radial_wavefunction(n, l, r, Z=z))
        components = []
        for j in ([l - 0.5, l + 0.5] if l else [0.5]):
            g, f = (_h(x) for x in dirac_radial(n, l, j, r, Z=z))
            mean_r, small_w = (_h(x) for x in dirac_moments(n, l, j, Z=z))
            components.append(DiracComponentModel(
                j=j,
                kappa=dirac_kappa(l, j),
                energy=QuantityModel.from_quantity(_h(dirac_energy(n, j, Z=z))),
                large=FieldModel.from_field(g),
                small=FieldModel.from_field(f),
                probability=FieldModel.from_field(Field(
                    r * r * (g.values**2 + f.values**2), r, "bohr^-1", "bohr",
                    f"P_Dirac {n},l={l},j={j:g} = r^2 (g^2 + f^2)", g.provenance,
                )),
                mean_r=QuantityModel.from_quantity(mean_r),
                small_weight=QuantityModel.from_quantity(small_w),
            ))
        return DiracRadialModel(
            n=n, l=l, z=z,
            schrodinger=FieldModel.from_field(Field(
                r * r * schr.values**2, r, "bohr^-1", "bohr",
                f"P_{n},{l}(r) = r^2 R^2", schr.provenance,
            )),
            schrodinger_mean_r=QuantityModel.from_quantity(_h(mean_radius(n, l, Z=z))),
            schrodinger_energy=QuantityModel.from_quantity(_h(energy(n, Z=z))),
            components=components,
        )

    @app.get("/api/finite-size", response_model=FiniteSizeModel)
    def finite_size_endpoint(system: str = "h") -> FiniteSizeModel:
        """How much the nucleus' size moves the low levels, and the proton radius puzzle."""
        if _is_screened(system):
            raise HTTPException(
                status_code=400,
                detail="the finite-size shift is computed for hydrogen-like systems only",
            )
        sys_ = _resolve_system(system)
        radius = sys_.nuclear_radius
        if radius is None:
            raise HTTPException(
                status_code=400,
                detail=f"{sys_.name} has no nuclear charge radius: its nucleus is a point "
                "particle or unidentified, so there is no size to shift anything",
            )
        mu = sys_.mu_ratio.value
        levels = []
        for n, l in ((1, 0), (2, 0), (3, 0), (2, 1)):
            q = finite_size_shift(n, l, radius, Z=sys_.Z, mu_ratio=mu)
            levels.append(FiniteSizeLevelModel(
                n=n, l=l,
                shift_mhz=QuantityModel.from_quantity(
                    _rescaled(q, _HARTREE_HZ / 1e6, "MHz", "CODATA hartree-hertz relation")
                ),
                shift_mev=QuantityModel.from_quantity(
                    _rescaled(q, HARTREE_EV * 1e3, "meV", "CODATA Hartree-eV factor")
                ),
                fraction=q.value / abs(energy(n, Z=sys_.Z, mu_ratio=mu).value),
            ))
        ref = load_lamb_reference()
        one_fm = Quantity(1.0 / BOHR_RADIUS_FM, "bohr", "1 fm", Provenance(
            fidelity=Fidelity.EXACT, method="a unit radius, to read off the r^2 coefficient",
        ))
        coeff = finite_size_shift(2, 0, one_fm, mu_ratio=get_system("mu-h").mu_ratio.value)
        predictions = [
            LambPredictionModel(
                key=key, source=src, radius_fm=r, radius_uncertainty_fm=dr,
                lamb_shift=QuantityModel.from_quantity(muonic_lamb_shift(r, dr)),
            )
            for key, (r, dr, src) in ref.radii_fm.items()
        ]
        return FiniteSizeModel(
            system=system,
            radius_fm=QuantityModel.from_quantity(
                _rescaled(radius, BOHR_RADIUS_FM, "fm", "CODATA Bohr radius")
            ),
            levels=levels,
            coefficient_ours=QuantityModel.from_quantity(
                _rescaled(coeff, HARTREE_EV * 1e3, "meV", "CODATA Hartree-eV factor")
            ),
            coefficient_published=ref.r2_coefficient[0],
            coefficient_published_uncertainty=ref.r2_coefficient[1],
            predictions=predictions,
            citation=ref.citation,
            retrieved=ref.retrieved,
        )

    @app.get("/api/rabi", response_model=RabiModel)
    def rabi_endpoint(
        n1: int = 1, l1: int = 0, n2: int = 2, l2: int = 1, m: int = 0,
        intensity: float = 1e6, detuning: float = 0.0, decay: bool = True,
        system: str = "h",
    ) -> RabiModel:
        """One dipole transition driven by a z-polarized laser (intensity in W/m^2,
        detuning in MHz), hydrogen-like systems only."""
        if max(n1, n2) > 10:
            raise HTTPException(status_code=422, detail="n must be <= 10")
        if not math.isfinite(detuning) or abs(detuning) > 1e6:
            raise HTTPException(status_code=422, detail="|detuning| must be <= 1e6 MHz")
        if _is_screened(system):
            raise HTTPException(
                status_code=400,
                detail="laser driving is computed for hydrogen-like systems only",
            )
        sys_ = _resolve_system(system)
        try:
            drive = rabi_drive(
                (n1, l1, m), (n2, l2, m), intensity, detuning_mhz=detuning,
                Z=sys_.Z, mu_ratio=sys_.mu_ratio.value, decay=decay,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return RabiModel.from_drive(drive)

    @app.get("/api/og/{n}/{l}/{m}")
    def og_card(n: int, l: int, m: int, system: str = "h", basis: str = "complex") -> Response:
        if _og_card_state(str(n), str(l), str(m), system, basis) is None:
            raise HTTPException(
                status_code=422,
                detail=f"no preview card for that state; n <= {_OG_CARD_N_MAX}, "
                "named hydrogen-like systems only",
            )
        return Response(
            content=render_og_card(n, l, m, system, basis), media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    def _hybrid_system(req):
        """The system a hybrid request names, or a refusal before any job exists."""
        if req.hybrid not in HYBRIDS:
            raise HTTPException(
                status_code=422,
                detail=f"unknown hybrid {req.hybrid!r}; known: {sorted(HYBRIDS)}",
            )
        if _is_screened(req.system) or req.model == "hf":
            raise HTTPException(
                status_code=400,
                detail=(
                    "hybrids are built from hydrogen's degenerate 2s and 2p, so they are "
                    "exact stationary states only for the hydrogen-like systems; in a "
                    "many-electron atom 2s and 2p are not degenerate"
                ),
            )
        return _resolve_system(req.system)

    def _new_job(req) -> Job:
        job = jobs.create()
        app.state.job_systems[job.id] = req.system
        app.state.job_models[job.id] = req.model
        if req.hybrid is not None:
            app.state.job_hybrids[job.id] = req.hybrid
        return job

    @app.post("/api/jobs/sample", response_model=JobModel)
    async def create_sample_job(req: SampleRequest) -> JobModel:
        _validate_state(req.n, req.l, req.m)
        if req.hybrid is not None:
            sys_ = _hybrid_system(req)
            mu = sys_.mu_ratio.value
            job = _new_job(req)

            def work(progress):
                cloud = sample_hybrid(
                    req.hybrid, req.count, Z=sys_.Z, mu_ratio=mu, seed=req.seed,
                    progress=lambda f: progress(0.9 * f),
                )
                pos = cloud.positions.astype(np.float64)
                psi = WavefunctionValues(
                    values=evaluate_hybrid(req.hybrid, pos, sys_.Z, mu), positions=pos,
                    n=2, l=1, m=0, Z=sys_.Z, mu_ratio=mu, basis="real",
                    provenance=cloud.provenance,
                )
                progress(1.0)
                return SampleJobResult(cloud=cloud, psi=psi)

            return _dispatch(job, work, req)
        # Resolved before the job exists, so a refusal does not
        # leave an orphan job in the store for a client that never got an id.
        hf_target = _hf_view_target(req) if req.model == "hf" else None
        job = jobs.create()
        app.state.job_systems[job.id] = req.system
        app.state.job_models[job.id] = req.model

        if hf_target is not None:
            hf_z, hf_n, hf_config = hf_target

            def work(progress):
                cloud = sample_hf_density(
                    hf_z, hf_n, req.n, req.l, req.m, req.count,
                    seed=req.seed, progress=lambda f: progress(0.9 * f),
                    basis=req.basis, config=hf_config,
                    exchange=req.exchange, pauli=req.pauli,
                )
                psi = evaluate_hf_state(
                    hf_z, hf_n, req.n, req.l, req.m,
                    cloud.positions.astype(np.float64), basis=req.basis,
                    config=hf_config, exchange=req.exchange, pauli=req.pauli,
                )
                progress(1.0)
                return SampleJobResult(cloud=cloud, psi=psi)

            return _dispatch(job, work, req)

        if _is_screened(req.system):
            element = _gsz_element(req.system)

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

            return _dispatch(job, work, req)

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

        return _dispatch(job, work, req)

    @app.post("/api/jobs/plane", response_model=JobModel)
    async def create_plane_job(req: PlaneRequest) -> JobModel:
        _validate_state(req.n, req.l, req.m)
        if req.hybrid is not None:
            sys_ = _hybrid_system(req)
            job = _new_job(req)

            def work(progress):
                pg = hybrid_plane_grid(
                    req.hybrid, "density" if req.quantity == "current" else req.quantity,
                    Z=sys_.Z, mu_ratio=sys_.mu_ratio.value, resolution=req.resolution,
                    progress=progress,
                )
                # A hybrid is a real function, so its current is identically zero,
                # and current_through_plane says so in its provenance.
                return current_through_plane(pg) if req.quantity == "current" else pg

            return _dispatch(job, work, req)
        hf_target = _hf_view_target(req) if req.model == "hf" else None
        job = jobs.create()
        app.state.job_systems[job.id] = req.system
        app.state.job_models[job.id] = req.model
        # The current is a closed-form identity on the density, so every model
        # draws it from its own density grid and keeps that grid's tier.
        quantity = "density" if req.quantity == "current" else req.quantity

        def _dispatch_plane(job, work):
            if req.quantity != "current":
                return _dispatch(job, work, req)
            return _dispatch(
                job, lambda progress: current_through_plane(work(progress)), req
            )

        if hf_target is not None:
            hf_z, hf_n, hf_config = hf_target

            def work(progress):
                return hf_plane_grid(
                    hf_z, hf_n, req.n, req.l, req.m,
                    quantity=quantity, basis=req.basis,
                    resolution=req.resolution, progress=progress,
                    config=hf_config, exchange=req.exchange, pauli=req.pauli,
                )

            return _dispatch_plane(job, work)

        if _is_screened(req.system):
            element = _gsz_element(req.system)

            def work(progress):
                return screened_plane_grid(
                    element.z, element.z, req.n, req.l, req.m,
                    quantity=quantity, basis=req.basis,
                    resolution=req.resolution, progress=progress,
                )

            return _dispatch_plane(job, work)

        sys_ = _resolve_system(req.system)

        def work(progress):
            return plane_grid(
                req.n, req.l, req.m, quantity=quantity, basis=req.basis,
                Z=sys_.Z, mu_ratio=sys_.mu_ratio.value,
                resolution=req.resolution, progress=progress,
            )

        return _dispatch_plane(job, work)

    @app.post("/api/jobs/superposition", response_model=JobModel)
    async def create_superposition_job(req: SuperpositionRequest) -> JobModel:
        """Evaluate each psi_k of a superposition once; the browser runs the clock.

        Hydrogen-like systems only. A screened or Hartree-Fock orbital is not an
        eigenstate of the many-electron Hamiltonian, so a phase sum over them
        would wear the EXACT badge without earning it.
        """
        if _is_screened(req.system):
            raise HTTPException(
                status_code=422,
                detail=(
                    "time evolution needs exact stationary states, which only the "
                    "hydrogen-like systems have; a screened or Hartree-Fock orbital "
                    "is not an eigenstate of the many-electron atom"
                ),
            )
        for c in req.components:
            _validate_state(c.n, c.l, c.m)
        sys_ = _resolve_system(req.system)
        comps = [Component(c.n, c.l, c.m, c.weight, c.phase) for c in req.components]
        try:
            # The cheap checks (duplicates, all-zero weights) fail here as a 422
            # rather than later inside the worker as a failed job.
            superposition_grid(comps, Z=sys_.Z, mu_ratio=sys_.mu_ratio.value, resolution=2)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        job = jobs.create()
        app.state.job_systems[job.id] = req.system

        def work(progress):
            return superposition_grid(
                comps, basis=req.basis, Z=sys_.Z, mu_ratio=sys_.mu_ratio.value,
                resolution=req.resolution, progress=progress,
            )

        return _dispatch(job, work, req)

    @app.post("/api/jobs/isosurface", response_model=JobModel)
    async def create_iso_job(req: IsoRequest) -> JobModel:
        """Start an isosurface extraction.

        A job for the same reason the cloud is: 96^3 is nearly a million psi
        evaluations before a triangle is cut, which is about a second for a
        closed-form state and several for a screened one.
        """
        _validate_state(req.n, req.l, req.m)
        hf_target = _hf_view_target(req) if req.model == "hf" else None
        job = jobs.create()
        app.state.job_systems[job.id] = req.system
        app.state.job_models[job.id] = req.model

        if hf_target is not None:
            hf_z, hf_n, hf_config = hf_target

            def work(progress):
                return hf_isosurface(
                    hf_z, hf_n, req.n, req.l, req.m,
                    target_fraction=req.fraction, basis=req.basis,
                    resolution=req.resolution, progress=progress,
                    config=hf_config, exchange=req.exchange, pauli=req.pauli,
                )

        elif _is_screened(req.system):
            element = _gsz_element(req.system)

            def work(progress):
                return screened_isosurface(
                    element.z, element.z, req.n, req.l, req.m,
                    target_fraction=req.fraction, basis=req.basis,
                    resolution=req.resolution, progress=progress,
                )

        else:
            sys_ = _resolve_system(req.system)

            def work(progress):
                return isosurface(
                    req.n, req.l, req.m,
                    target_fraction=req.fraction, basis=req.basis,
                    Z=sys_.Z, mu_ratio=sys_.mu_ratio.value,
                    resolution=req.resolution, progress=progress,
                )

        return _dispatch(job, work, req)

    @app.post("/api/jobs/hf", response_model=JobModel)
    async def create_hf_job(req: HFRequest) -> JobModel:
        """Start a Hartree-Fock solve.

        A job rather than a plain GET because the solve takes seconds, not
        milliseconds (argon about 5s cold, chlorine about 7s), which is long
        enough that a blocking request would be a bad answer even though it
        would be a correct one. Results are memoized, so a repeat is free and
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
        # no "same atom with the cap on" to measure against, and comparing it to
        # the Aufbau ground state would report the distance between
        # two different configurations as the cost of the exclusion principle.
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
            # The counterfactual solves are the only ones that owe the reader
            # a comparison, and the only ones worth a second solve. Cheap
            # in practice: a client reaches these by flipping a switch on an
            # atom it was just looking at, so the real solve is already memoized
            # and this is arithmetic on two cached results.
            delta = (
                None if req.exchange or not req.pauli
                else hf_exchange_energy(req.z, n_electrons, config)
            )
            collapse = pauli_collapse(req.z, n_electrons) if comparable else None
            progress(1.0)
            return HFJobResult(result, delta, collapse)

        return _dispatch(job, work, req)

    @app.get("/api/jobs/{job_id}", response_model=JobModel)
    def job_status(job_id: str) -> JobModel:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
        return _job_model(job)

    def _sample_meta(
        res: SampleJobResult, system_key: str, model_key: str
    ) -> SampleMetaModel:
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
            system=system_key, model=model_key,
            provenance=ProvenanceModel.from_provenance(cloud.provenance),
            channels=channels,
        )

    def _plane_meta(
        pg: PlaneGrid, system_key: str, model_key: str
    ) -> PlaneMetaModel:
        return PlaneMetaModel(
            resolution=pg.values.shape[0],
            dtype="float32",
            layout="row-major float32; row i = z=axis[i] ascending, col j = x=axis[j]",
            quantity=pg.quantity, unit=pg.unit, label=pg.label,
            half_extent=float(pg.axis[-1]), axis_unit="bohr",
            n=pg.n, l=pg.l, m=pg.m, basis=pg.basis, system=system_key,
            model=model_key,
            provenance=ProvenanceModel.from_provenance(pg.provenance),
        )

    def _superposition_meta(
        sg: SuperpositionGrid, system_key: str
    ) -> SuperpositionMetaModel:
        method = (
            "2 pi hbar / |E_j - E_k| from the exact Bohr energies, converted to fs "
            "via hbar / E_h"
        )
        periods = [
            QuantityModel.from_quantity(Quantity(
                value=2 * np.pi / w * AU_TIME_FS, unit="fs",
                label=f"beat period at |dE| = {w * HARTREE_EV:.6g} eV",
                provenance=dataclasses.replace(sg.provenance, method=method),
            ))
            for w in sg.beat_frequencies()
        ]
        return SuperpositionMetaModel(
            resolution=sg.psi.shape[1],
            dtype="float32",
            layout=(
                "one channel psi{k} per component, row-major float32; "
                "row i = z=axis[i] ascending, col j = x=axis[j]"
            ),
            unit="bohr^-3/2", half_extent=float(sg.axis[-1]), axis_unit="bohr",
            basis=sg.basis, system=system_key,
            components=[
                SuperpositionComponentMeta(
                    n=c.n, l=c.l, m=c.m, re=float(a.real), im=float(a.imag),
                    energy=QuantityModel.from_quantity(e),
                )
                for c, a, e in zip(sg.components, sg.coefficients, sg.energies)
            ],
            time_unit_fs=AU_TIME_FS,
            beat_periods=periods,
            provenance=ProvenanceModel.from_provenance(sg.provenance),
        )

    def _iso_meta(
        surf: Isosurface, system_key: str, model_key: str
    ) -> IsoMetaModel:
        prov = ProvenanceModel.from_provenance(surf.provenance)
        return IsoMetaModel(
            vertex_count=int(surf.vertices.shape[0]),
            triangle_count=int(surf.triangles.shape[0]),
            channels=[
                ChannelModel(
                    name="vertices", dtype="float32", unit="bohr", provenance=prov
                ),
                ChannelModel(
                    name="triangles", dtype="uint32", unit="1", provenance=prov
                ),
                ChannelModel(
                    name="phase", dtype="float32", unit="rad",
                    provenance=ProvenanceModel.from_provenance(surf.level.provenance),
                ),
            ],
            target_fraction=surf.target_fraction,
            enclosed_fraction=QuantityModel.from_quantity(surf.enclosed_fraction),
            outside_fraction=surf.outside_fraction,
            level=QuantityModel.from_quantity(surf.level),
            escaped_fraction=QuantityModel.from_quantity(surf.escaped_fraction),
            mesh_volume=QuantityModel.from_quantity(surf.mesh_volume),
            voxel_volume=QuantityModel.from_quantity(surf.voxel_volume),
            area=QuantityModel.from_quantity(surf.area),
            components=surf.components,
            half_width=surf.half_width,
            resolution=surf.resolution,
            axis_unit="bohr",
            n=surf.n, l=surf.l, m=surf.m, basis=surf.basis, system=system_key,
            model=model_key,
            label=surf.label,
            provenance=prov,
        )

    @app.get(
        "/api/jobs/{job_id}/meta",
        response_model=(
            SampleMetaModel | PlaneMetaModel | IsoMetaModel | HFResultModel
            | SuperpositionMetaModel
        ),
    )
    def job_meta(
        job_id: str,
    ) -> (
        SampleMetaModel | PlaneMetaModel | IsoMetaModel | HFResultModel
        | SuperpositionMetaModel
    ):
        res = _finished_result(jobs, job_id)
        system_key = app.state.job_systems.get(job_id, "h")
        model_key = app.state.job_models.get(job_id, "gsz")
        hybrid_key = app.state.job_hybrids.get(job_id)
        if isinstance(res, PlaneGrid):
            return _plane_meta(res, system_key, model_key).model_copy(
                update={"hybrid": hybrid_key}
            )
        if isinstance(res, SuperpositionGrid):
            return _superposition_meta(res, system_key)
        if isinstance(res, Isosurface):
            return _iso_meta(res, system_key, model_key)
        if isinstance(res, HFJobResult):
            # Unlike the sample and plane, the whole scientific result is here:
            # the energies and their provenance. /data carries only the orbital
            # shapes, which are the part that is an array.
            return _hf_result_model(res.result, res.exchange_energy, res.collapse)
        return _sample_meta(res, system_key, model_key).model_copy(
            update={"hybrid": hybrid_key}
        )

    def _iso_channel_payload(surf: Isosurface, channel: str | None) -> np.ndarray:
        """Three channels, because a mesh is three arrays of different shapes.

        `triangles` goes out as uint32 rather than the internal int32: WebGL
        index buffers are unsigned, and converting here rather than in the
        browser keeps the client from having to know that.
        """
        if channel is None or channel == "vertices":
            return surf.vertices.astype(np.float32)
        if channel == "triangles":
            return surf.triangles.astype(np.uint32)
        if channel == "phase":
            return surf.vertex_phase.astype(np.float32)
        raise HTTPException(
            status_code=422,
            detail=f"no channel {channel!r} on this job; it has vertices, triangles, phase",
        )

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
        elif isinstance(res, SuperpositionGrid):
            names = [f"psi{k}" for k in range(res.psi.shape[0])]
            if channel not in names:
                raise HTTPException(
                    status_code=422,
                    detail=f"no channel {channel!r} on this job; it has {', '.join(names)}",
                )
            payload = res.psi[names.index(channel)].astype(np.float32)
        elif isinstance(res, Isosurface):
            payload = _iso_channel_payload(res, channel)
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

    def _popular_requests() -> list[tuple[str, BaseModel]]:
        """What a first visit, the tours and the atom picker open.

        Built as the client builds them (the HF request names n_electrons, as
        the store does), because a warm result only helps if its key is the key
        a real request will produce.
        """
        out: list[tuple[str, BaseModel]] = []
        for n in (1, 2, 3):
            for l in range(n):
                for m in range(-l, l + 1):
                    out.append(("sample", SampleRequest(n=n, l=l, m=m)))
                    out.append(("plane", PlaneRequest(n=n, l=l, m=m)))
        for key in ATOM_KEYS:
            z = atom_for_key(key).z
            out.append(("hf", HFRequest(z=z, n_electrons=z)))
        return out

    async def warm() -> None:
        """Run the popular states through the real endpoints, one at a time.

        One at a time so a first visitor never queues behind the warm-up for
        more than one job, and each warm job is discarded once it finishes so
        it cannot push a real visitor's finished job out of the store.
        """
        endpoints = {"sample": create_sample_job, "plane": create_plane_job, "hf": create_hf_job}
        for kind, req in app.state.popular:
            try:
                job = await endpoints[kind](req)
            except HTTPException:
                continue
            while (j := jobs.get(job.id)) is not None and j.status not in (
                JobStatus.DONE, JobStatus.ERROR,
            ):
                await asyncio.sleep(0.2)
            jobs.discard(job.id)
        logger.info(
            "Warmed %d results (%.0f MB) for the popular states",
            len(results), results.size_bytes / 2**20,
        )

    app.state.popular = _popular_requests()
    app.state.warm = warm

    web_dist = _web_dist()
    if web_dist.is_dir():
        @app.get("/", include_in_schema=False)
        def index(request: Request) -> HTMLResponse:
            html = (web_dist / "index.html").read_text(encoding="utf-8")
            return HTMLResponse(_with_state_preview(html, request.query_params))

        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")
        logger.info("Mounted the UI from %s", web_dist)
    else:
        logger.warning("No UI found at %s; serving the API only", web_dist)

    return app
