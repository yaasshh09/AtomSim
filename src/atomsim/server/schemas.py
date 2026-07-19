"""THE canonical JSON forms of Provenance, Quantity, and Field.

Defined exactly once here; web/src/api/types.ts mirrors these shapes.
Provenance reaches the browser by construction, never as an afterthought.
"""

import dataclasses
from typing import Literal

from pydantic import BaseModel

from atomsim.classical import BohrOrbit, ClassicalGhost
from atomsim.constants import BOHR_RADIUS_FM
from atomsim.constants_lab import ConstantsReport, DerivedObservable
from atomsim.provenance import Fidelity, Field, Provenance, Quantity
from atomsim.spectra import LineComparison, SpectralLine
from atomsim.systems import System

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
