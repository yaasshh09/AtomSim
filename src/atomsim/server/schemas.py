"""THE canonical JSON forms of Provenance, Quantity, and Field.

Defined exactly once here; web/src/api/types.ts mirrors these shapes.
Provenance reaches the browser by construction, never as an afterthought.
"""

import dataclasses
from typing import Literal

from pydantic import BaseModel

from atomsim.broadening import SyntheticSpectrum
from atomsim.classical import BohrOrbit, ClassicalGhost
from atomsim.constants import BOHR_RADIUS_FM
from atomsim.constants_lab import ConstantsReport, DerivedObservable
from atomsim.populations import ThermalState
from atomsim.provenance import Fidelity, Field, Provenance, Quantity
from atomsim.spectra import LineComparison, SpectralLine
from atomsim.systems import System
from atomsim.transfer import CurveOfGrowth

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


class ChannelModel(BaseModel):
    """One binary per-point channel of a sample job (positions / density / phase)."""

    name: str
    dtype: str
    unit: str
    provenance: ProvenanceModel


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


class DerivedObservableModel(BaseModel):
    quantity: QuantityModel
    ratio: float
    changed: bool

    @classmethod
    def from_observable(cls, o: DerivedObservable) -> "DerivedObservableModel":
        return cls(
            quantity=QuantityModel.from_quantity(o.quantity),
            ratio=o.ratio,
            changed=o.changed,
        )


class ConstantsReportModel(BaseModel):
    alpha: DerivedObservableModel
    bohr_radius_pm: DerivedObservableModel
    hartree_ev: DerivedObservableModel
    altered: bool

    @classmethod
    def from_report(cls, r: ConstantsReport) -> "ConstantsReportModel":
        return cls(
            alpha=DerivedObservableModel.from_observable(r.alpha),
            bohr_radius_pm=DerivedObservableModel.from_observable(r.bohr_radius_pm),
            hartree_ev=DerivedObservableModel.from_observable(r.hartree_ev),
            altered=r.altered,
        )


class BohrOrbitModel(BaseModel):
    n: int
    radius_bohr: QuantityModel
    radius_pm: QuantityModel

    @classmethod
    def from_orbit(cls, o: BohrOrbit) -> "BohrOrbitModel":
        return cls(
            n=o.n,
            radius_bohr=QuantityModel.from_quantity(o.radius_bohr),
            radius_pm=QuantityModel.from_quantity(o.radius_pm),
        )


class ClassicalGhostModel(BaseModel):
    n: int
    system_key: str
    z: int
    orbits: list[BohrOrbitModel]
    r0_bohr: QuantityModel
    collapse_time_s: QuantityModel
    orbital_period_s: QuantityModel
    orbit_count: QuantityModel

    @classmethod
    def from_ghost(cls, g: ClassicalGhost) -> "ClassicalGhostModel":
        return cls(
            n=g.n,
            system_key=g.system_key,
            z=g.z,
            orbits=[BohrOrbitModel.from_orbit(o) for o in g.orbits],
            r0_bohr=QuantityModel.from_quantity(g.r0_bohr),
            collapse_time_s=QuantityModel.from_quantity(g.collapse_time_s),
            orbital_period_s=QuantityModel.from_quantity(g.orbital_period_s),
            orbit_count=QuantityModel.from_quantity(g.orbit_count),
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


def _to_fm(q: Quantity) -> Quantity:
    """Display conversion bohr -> fm for nuclear radii (server boundary only)."""
    return Quantity(
        value=q.value * BOHR_RADIUS_FM,
        unit="fm",
        label=q.label + " [fm]",
        provenance=dataclasses.replace(
            q.provenance,
            method=q.provenance.method + "; converted to fm via CODATA Bohr radius",
            error_estimate=(
                None if q.provenance.error_estimate is None
                else q.provenance.error_estimate * BOHR_RADIUS_FM
            ),
        ),
    )


class SystemModel(BaseModel):
    key: str
    name: str
    z: int
    mu_ratio: QuantityModel
    m_over_m_nucleus: float
    description: str
    # None = honestly absent (point lepton or unidentified nucleus), never zero
    nuclear_radius: QuantityModel | None
    nuclear_radius_fm: QuantityModel | None
    # Hydrogenic presets stay exactly as before; screened atoms set kind/n_electrons.
    kind: Literal["hydrogenic", "screened"] = "hydrogenic"
    n_electrons: int | None = None

    @classmethod
    def from_system(cls, s: System) -> "SystemModel":
        r = s.nuclear_radius
        return cls(
            key=s.key,
            name=s.name,
            z=s.Z,
            mu_ratio=QuantityModel.from_quantity(s.mu_ratio),
            m_over_m_nucleus=s.m_over_M,
            description=s.description,
            nuclear_radius=None if r is None else QuantityModel.from_quantity(r),
            nuclear_radius_fm=None if r is None else QuantityModel.from_quantity(_to_fm(r)),
        )

    @classmethod
    def from_atom(cls, element, n_electrons: int, description: str) -> "SystemModel":
        mu = Quantity(
            1.0, "m_e", f"mu/m_e ({element.name})",
            Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method="infinite nuclear mass (screened-atom model)",
            ),
        )
        return cls(
            key=element.symbol.lower(), name=element.name, z=element.z,
            mu_ratio=QuantityModel.from_quantity(mu), m_over_m_nucleus=0.0,
            description=description, nuclear_radius=None, nuclear_radius_fm=None,
            kind="screened", n_electrons=n_electrons,
        )


class ScreenedOrbitalModel(BaseModel):
    n: int
    l: int
    label: str
    occupancy: int
    energy: QuantityModel
    energy_ev: QuantityModel


class ScreenedLevelsModel(BaseModel):
    system: SystemModel
    config: str
    is_ground: bool
    orbitals: list[ScreenedOrbitalModel]
    total_energy: QuantityModel
    total_energy_ev: QuantityModel


class ForceLawLevelModel(BaseModel):
    radial_index: int
    energy: QuantityModel
    energy_ev: QuantityModel
    trusted: bool = True


class ReferenceItemModel(BaseModel):
    label: str
    energy: QuantityModel
    energy_ev: QuantityModel


class ReferenceModel(BaseModel):
    kind: Literal["levels", "markers"]
    items: list[ReferenceItemModel]


class PotentialCurveModel(BaseModel):
    r: list[float]        # bohr
    v_ev: list[float]     # eV
    provenance: ProvenanceModel


class ForceLawModel(BaseModel):
    preset: str
    params: dict[str, float]
    l: int
    z: int
    system: SystemModel
    counterfactual: list[ForceLawLevelModel]
    bound_count: int
    requested_count: int
    reference: ReferenceModel
    potential_curve: PotentialCurveModel
    expression: str | None = None


class LineModel(BaseModel):
    n_upper: int
    l_upper: int
    j_upper: float | None
    n_lower: int
    l_lower: int
    j_lower: float | None
    energy_ev: QuantityModel
    wavelength_nm: QuantityModel
    #: Null unless intensities were asked for and can be given honestly; the
    #: line list's `intensity_note` then says which case applies.
    einstein_a_s: QuantityModel | None = None
    oscillator_strength: QuantityModel | None = None
    #: eV/s per atom of the element. Null unless thermal conditions were given.
    #: A modelled LTE emission rate, not a measured brightness: see the
    #: response's `thermal` block for the conditions and their assumptions.
    emissivity: QuantityModel | None = None

    @classmethod
    def from_line(cls, ln: SpectralLine) -> "LineModel":
        return cls(
            n_upper=ln.n_upper, l_upper=ln.l_upper, j_upper=ln.j_upper,
            n_lower=ln.n_lower, l_lower=ln.l_lower, j_lower=ln.j_lower,
            energy_ev=QuantityModel.from_quantity(ln.energy),
            wavelength_nm=QuantityModel.from_quantity(ln.wavelength),
            einstein_a_s=(
                None if ln.einstein_a is None
                else QuantityModel.from_quantity(ln.einstein_a)
            ),
            oscillator_strength=(
                None if ln.oscillator_strength is None
                else QuantityModel.from_quantity(ln.oscillator_strength)
            ),
            emissivity=(
                None if ln.emissivity is None
                else QuantityModel.from_quantity(ln.emissivity)
            ),
        )


class ThermalModel(BaseModel):
    """The LTE conditions a spectrum was computed at, and what they produced.

    Carried so the view can state what it is drawing. The ionized fraction in
    particular is not decoration: once it approaches 1 the whole spectrum is
    dim because there are no neutrals left, and a view that rescaled to the
    brightest remaining line without saying so would hide that entirely.
    """

    temperature_k: float
    electron_density_cm3: float
    ionized_fraction: QuantityModel
    partition_function: QuantityModel

    @classmethod
    def from_state(cls, state: ThermalState) -> "ThermalModel":
        return cls(
            temperature_k=state.conditions.temperature_k,
            electron_density_cm3=state.conditions.electron_density_cm3,
            ionized_fraction=QuantityModel.from_quantity(state.ionized_fraction),
            partition_function=QuantityModel.from_quantity(state.partition_function),
        )


class LineWidthModel(BaseModel):
    """The width budget of one line, so the view can say what set it.

    Kept separate from the curve: a user pointing at a line wants to know
    whether they are looking at temperature, lifetime, or the spectrograph,
    and only the breakdown answers that.
    """

    label: str
    wavelength_nm: float
    n_upper: int
    n_lower: int
    #: Gaussian sigma, nm (Doppler and instrument in quadrature).
    sigma_nm: float
    #: Lorentzian HWHM, nm (natural).
    gamma_nm: float
    fwhm_nm: float
    terms: list[str]


class ProfileModel(BaseModel):
    """A synthesized spectrum: the curve, its widths, and what it leaves out."""

    wavelength_nm: list[float]
    intensity: list[float]
    unit: str
    #: "emissivity" | "rate" | "uniform" — what the area under a line means.
    weight_kind: str
    resolving_power: float | None
    #: Curve integral over summed line strengths. The grid's own quadrature
    #: error, measured by the engine rather than assumed to be negligible.
    flux_closure: float
    widths: list[LineWidthModel]
    #: The collisional broadening that is NOT in the curve, sized.
    stark_span_nm: QuantityModel | None
    stark_note: str | None
    provenance: ProvenanceModel

    @classmethod
    def from_synthetic(cls, syn: SyntheticSpectrum) -> "ProfileModel":
        return cls(
            wavelength_nm=[float(x) for x in syn.spectrum.grid],
            intensity=[float(v) for v in syn.spectrum.values],
            unit=syn.spectrum.unit,
            weight_kind=syn.weight_kind,
            resolving_power=syn.resolving_power,
            flux_closure=syn.flux_closure,
            widths=[
                LineWidthModel(
                    label=p.label, wavelength_nm=p.wavelength_nm,
                    n_upper=p.n_upper, n_lower=p.n_lower,
                    sigma_nm=p.sigma_nm, gamma_nm=p.gamma_nm,
                    fwhm_nm=p.fwhm_nm, terms=list(p.terms),
                )
                for p in syn.profiles
            ],
            stark_span_nm=(
                None if syn.stark_span is None
                else QuantityModel.from_quantity(syn.stark_span)
            ),
            stark_note=syn.stark_note,
            provenance=ProvenanceModel.from_provenance(syn.spectrum.provenance),
        )


class CurveOfGrowthModel(BaseModel):
    """How a line's measured strength responds to adding more gas.

    The regime labels are the payload, not the curve. Which branch a line sits
    on decides whether its strength measures the amount of gas at all, and a
    plot without that answer would invite exactly the misreading the phase
    exists to prevent.
    """

    label: str
    wavelength_nm: float
    oscillator_strength: float
    #: The widths the curve was computed for; the knees sit where they put them.
    sigma_nm: float
    gamma_nm: float
    damping_parameter: float
    column_density_m2: list[float]
    equivalent_width_nm: list[float]
    #: "linear" | "saturated" | "damping", per point.
    regime: list[str]
    #: Local log-log slope: 1, then ~0, then 1/2. The visible signature of the
    #: branch, though not what the branch is decided by.
    slope: list[float]
    tau_centre: list[float]
    window_nm: float
    provenance: ProvenanceModel

    @classmethod
    def from_curve(
        cls, c: CurveOfGrowth, label: str, sigma_nm: float, gamma_nm: float
    ) -> "CurveOfGrowthModel":
        return cls(
            label=label,
            wavelength_nm=c.wavelength_nm,
            oscillator_strength=c.oscillator_strength,
            sigma_nm=sigma_nm,
            gamma_nm=gamma_nm,
            damping_parameter=c.damping_parameter,
            column_density_m2=[float(v) for v in c.column_density],
            equivalent_width_nm=[float(v) for v in c.equivalent_width],
            regime=list(c.regime),
            slope=[float(v) for v in c.slope],
            tau_centre=[float(v) for v in c.tau_centre],
            window_nm=c.window_nm,
            provenance=ProvenanceModel.from_provenance(c.provenance),
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
