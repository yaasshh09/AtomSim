"""What-If force laws: energy levels under a counterfactual central potential.

A preset registry drives the numerical radial solver with different V(r) shapes,
pairs each with an honest per-preset reference (EXACT hydrogen, EXACT harmonic-
oscillator levels, or structural markers), and returns a sampled V(r) curve so
the view can draw the potential itself. See docs/specs/
2026-07-17-phase5-force-law-presets-design.md.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from atomsim.analytic.hydrogen import energy as hydrogen_energy
from atomsim.analytic.oscillator import oscillator_energy
from atomsim.numerics.expression import compile_potential
from atomsim.numerics.radial_solver import solve_radial_with_error
from atomsim.provenance import Fidelity, Field, Provenance, Quantity
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
    trusted: bool = True  # False when the free-form trust gate flags the level


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
    expression: str | None = None  # set for the free-form "custom" preset


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


PRESETS: dict[str, ForcePreset] = {
    POWERLAW.key: POWERLAW,
    YUKAWA.key: YUKAWA,
    COULOMBCORE.key: COULOMBCORE,
    HARMONIC.key: HARMONIC,
    FINITEWELL.key: FINITEWELL,
}


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


# ---- free-form V(r): user-typed potential + trust gate -----------------------

FREE_FORM_BOX_TOL = 5e-3   # hartree; box-doubling shift above this = not converged
FREE_FORM_GRID_FRAC = 1e-2  # grid error above this fraction of |E| = not converged


def _free_form_rmax(z: int, n_states: int) -> float:
    return 40.0 * (n_states + 1) ** 2 / max(z, 1)


def free_form_levels(
    expr: str,
    l: int,
    system: str | System = "h",
    n_states: int = 4,
) -> ForceLawResult:
    """Solve a user-typed radial potential V(r), gating each level for trust.

    A level is trusted only when it is box-converged (energy stable under
    doubling r_max, i.e. not continuum contamination) and grid-converged (the
    grid-halving error is small relative to |E|, i.e. not fall-to-center).
    Untrusted levels are returned and labeled, never silently dropped.
    """
    if l < 0:
        raise ValueError(f"orbital quantum number l must be >= 0, got {l}")
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")

    potential = compile_potential(expr)  # raises ExpressionError on bad input

    sys = system if isinstance(system, System) else get_system(system)
    z = sys.Z
    mu = sys.mu_ratio.value
    r_max = _free_form_rmax(z, n_states)

    # The expression must be finite on the interior grid before we trust a solve.
    r_probe = np.linspace(r_max / CURVE_POINTS, r_max, CURVE_POINTS)
    v_probe = np.asarray(potential(r_probe), dtype=float)
    if not np.all(np.isfinite(v_probe)):
        bad = r_probe[~np.isfinite(v_probe)][0]
        raise ValueError(f"V(r) is not finite at r = {bad:g} bohr")

    small = solve_radial_with_error(potential, l=l, mu_ratio=mu, n_states=n_states, r_max=r_max)
    big = solve_radial_with_error(
        potential, l=l, mu_ratio=mu, n_states=n_states, r_max=2.0 * r_max
    )
    threshold = float(potential(np.array([2.0 * r_max]))[0])

    note = f"; counterfactual free-form V(r) = {expr}"
    levels: list[ForceLawLevel] = []
    for k in range(n_states):
        e = small.energies[k]
        if e.value >= threshold:  # above the continuum floor: not a bound state
            continue
        box_shift = abs(e.value - big.energies[k].value)
        grid_err = e.provenance.error_estimate or 0.0
        trusted = (
            box_shift <= FREE_FORM_BOX_TOL
            and grid_err <= FREE_FORM_GRID_FRAC * max(abs(e.value), 1e-6)
        )
        reason = (
            "" if trusted
            else "; UNTRUSTED (not box/grid-converged, not a real bound state)"
        )
        levels.append(
            ForceLawLevel(
                radial_index=len(levels),
                energy=_tag(e, note + reason),
                trusted=trusted,
            )
        )

    bound_count = sum(1 for lvl in levels if lvl.trusted)
    reference = _hydrogen_reference({}, z, mu, l, n_states)
    curve = _sample_curve(potential, r_max, note)

    return ForceLawResult(
        preset_key="custom",
        params={},
        l=l,
        z=z,
        system_key=sys.key,
        counterfactual=tuple(levels),
        bound_count=bound_count,
        requested_count=n_states,
        reference=reference,
        potential_curve=curve,
        expression=expr,
    )
