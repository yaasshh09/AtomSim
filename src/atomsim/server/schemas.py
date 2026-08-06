"""THE canonical JSON forms of Provenance, Quantity, and Field.

Defined exactly once here; web/src/api/types.ts mirrors these shapes.
Provenance reaches the browser by construction, never as an afterthought.
"""

import dataclasses
from typing import Literal

from pydantic import BaseModel

from atomsim.atoms import has_gsz_parameters
from atomsim.broadening import SyntheticSpectrum
from atomsim.classical import BohrOrbit, ClassicalGhost
from atomsim.constants import BOHR_RADIUS_FM
from atomsim.constants_lab import ConstantsReport, DerivedObservable
from atomsim.density_compare import DensityComparison
from atomsim.populations import ThermalState
from atomsim.provenance import Fidelity, Field, Provenance, Quantity
from atomsim.spectra import LineComparison, SpectralLine
from atomsim.systems import System
from atomsim.transfer import AbsorbingLine, AbsorptionSpectrum, CurveOfGrowth

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


class ShellPeakModel(BaseModel):
    """One shell under both models. A null radius means this model resolves none."""

    label: str
    gsz_radius: float | None
    hf_radius: float | None
    gsz_depth: float | None
    hf_depth: float | None


class DensityComparisonModel(BaseModel):
    gsz: FieldModel
    hf: FieldModel
    displaced_charge: QuantityModel
    shells: list[ShellPeakModel]
    provenance: ProvenanceModel

    @classmethod
    def from_comparison(cls, c: DensityComparison) -> "DensityComparisonModel":
        return cls(
            gsz=FieldModel.from_field(c.gsz),
            hf=FieldModel.from_field(c.hf),
            displaced_charge=QuantityModel.from_quantity(c.displaced_charge),
            shells=[
                ShellPeakModel(
                    label=s.label,
                    gsz_radius=s.gsz_radius, hf_radius=s.hf_radius,
                    gsz_depth=s.gsz_depth, hf_depth=s.hf_depth,
                )
                for s in c.shells
            ],
            provenance=ProvenanceModel.from_provenance(c.provenance),
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
    #: Whether the GSZ screened model has published parameters for this atom.
    #: False for sulfur and chlorine, which Hartree-Fock solves anyway, so the
    #: client can grey one model choice instead of hiding the whole element.
    #: Meaningless for hydrogenic presets, which no screened model touches.
    has_gsz: bool = True

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
            # Derived, never passed in: a caller that had to remember this flag
            # would eventually forget it, and the wrong answer here is the app
            # offering a model that has no parameters to run on.
            has_gsz=has_gsz_parameters(element.z),
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


class HFOrbitalModel(BaseModel):
    """One converged Hartree-Fock subshell.

    Mirrors ScreenedOrbitalModel field for field so the two models are
    swappable in a view, and adds `channel`: the orbital amplitude P(r) is an
    array, so it travels as raw float32 on /api/jobs/{id}/data like every other
    array in this API rather than being inflated into JSON.
    """

    n: int
    l: int
    label: str
    occupancy: int
    energy: QuantityModel
    energy_ev: QuantityModel
    channel: str


class PauliCollapseModel(BaseModel):
    """The real atom carried alongside the collapsed one, so the view can say
    what the exclusion principle was worth without fetching it separately.

    Both halves come off one comparison in the engine rather than two requests,
    for the reason the exchange energy does: a client free to difference two
    jobs is free to difference a warm solve against a cold one and report the
    gap between two calculations as the gap between two models.

    The variational fields are the closed-form check - N electrons in one 1s of
    exponent zeta - and they are here so the page can show that the collapsed
    number was tested against something outside this codebase.
    """

    #: E(collapsed) - E(real). Negative: the collapsed atom is far more bound.
    binding_change: QuantityModel
    binding_change_ev: QuantityModel
    real_total_energy: QuantityModel
    real_total_energy_ev: QuantityModel
    real_config: str
    real_radius: QuantityModel
    collapsed_radius: QuantityModel
    #: <r>(collapsed) / <r>(real), below 1 and falling with Z.
    radius_ratio: QuantityModel
    variational_zeta: QuantityModel
    variational_energy: QuantityModel
    variational_energy_ev: QuantityModel


class HFResultModel(BaseModel):
    """A finished Hartree-Fock solve, as the browser sees it.

    Carries z and n_electrons rather than a SystemModel. The screened models
    are keyed to a named neutral preset, but Hartree-Fock needs no fitted table
    and so also solves ions that have no preset - K+ and Ar-like Fe both
    converge - and inventing an Element for those to satisfy the schema would
    be a fiction in the one place this codebase least wants one.

    `iterations` and `virial_ratio` are convergence diagnostics. They describe
    the solve, not the atom, and their provenance says NUMERICAL rather than
    APPROXIMATION for exactly that reason; a view must not present the virial
    ratio as a physical result.
    """

    kind: Literal["hf"] = "hf"
    z: int
    n_electrons: int
    symbol: str | None
    config: str
    is_ground: bool
    # False means the exchange term was removed: the Hartree model, in which
    # electrons repel but are distinguishable. Sent as a field rather than left
    # for the client to infer from the provenance tier, because a view that has
    # to parse prose to find out which physics it is drawing will eventually
    # draw the wrong one.
    exchange: bool = True
    # E_HF - E_Hartree, present only on a solve that has both to compare, which
    # in practice means the exchange-energy endpoint rather than a plain solve.
    exchange_energy: QuantityModel | None = None
    exchange_energy_ev: QuantityModel | None = None
    # False means the occupancy cap was lifted as well and the configuration
    # collapsed to 1s^N. Its own field rather than inferred from `exchange`,
    # because exchange=False on its own is the weaker counterfactual in which
    # the cap is still enforced, and a view that conflated them would show the
    # wrong disclosure for whichever one it guessed.
    pauli: bool = True
    # The real atom next to the collapsed one. Present only on a pauli=False
    # solve of a ground configuration; a hand-written collapsed configuration
    # has no "same atom with the cap on" to be compared against, so nothing is
    # sent rather than a comparison against a different configuration.
    collapse: PauliCollapseModel | None = None
    orbitals: list[HFOrbitalModel]
    total_energy: QuantityModel
    total_energy_ev: QuantityModel
    kinetic: QuantityModel
    potential: QuantityModel
    virial_ratio: QuantityModel
    iterations: int
    coarse_iterations: int
    converged: bool
    provenance: ProvenanceModel
    # The shared radial grid every P channel is sampled on, in bohr.
    grid_channel: str
    grid_points: int
    channels: list[ChannelModel]


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
    #: "emissivity" | "rate" | "uniform", what the area under a line means.
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


class AbsorbingLineModel(BaseModel):
    """One line's share of a blended absorption spectrum."""

    wavelength_nm: float
    label: str
    oscillator_strength: float
    #: Column in *this line's* lower level. The reason one gas gives every
    #: line a different optical depth, and the number the view has to show
    #: for the Lyman/Balmer contrast to mean anything.
    lower_column_m2: float
    tau_centre: float
    regime: str
    thin_width_nm: float
    fwhm_nm: float

    @classmethod
    def from_line(cls, d: AbsorbingLine) -> "AbsorbingLineModel":
        return cls(
            wavelength_nm=d.wavelength_nm,
            label=d.label,
            oscillator_strength=d.oscillator_strength,
            lower_column_m2=d.lower_column_m2,
            tau_centre=d.tau_centre,
            regime=d.regime,
            thin_width_nm=d.thin_width_nm,
            fwhm_nm=d.fwhm_nm,
        )


class AbsorptionSpectrumModel(BaseModel):
    """A whole line list absorbing at once against a flat continuum.

    `saturation` is the payload rather than the curve. It is how much of the
    census the spectrum is losing, and without it a plot of transmission
    invites the reading this phase exists to prevent: that a deeper line means
    proportionally more gas.
    """

    wavelength_nm: list[float]
    transmission: list[float]
    optical_depth: list[float]
    lines: list[AbsorbingLineModel]
    #: The gas that produced these populations. Without it the spectrum is a
    #: shape with no conditions attached, and the whole Lyman/Balmer contrast
    #: below is unexplained.
    thermal: ThermalModel | None
    column_density_m2: float
    equivalent_width_nm: float
    thin_limit_width_nm: float
    saturation: float
    #: Pairs of line labels whose profiles overlap.
    blends: list[tuple[str, str]]
    flux_closure: float
    provenance: ProvenanceModel
    #: Provenance of the column density itself, which is a knob and says so.
    column_provenance: ProvenanceModel

    @classmethod
    def from_spectrum(
        cls, a: AbsorptionSpectrum, thermal: ThermalState | None = None
    ) -> "AbsorptionSpectrumModel":
        return cls(
            wavelength_nm=[float(v) for v in a.transmission.grid],
            transmission=[float(v) for v in a.transmission.values],
            optical_depth=[float(v) for v in a.optical_depth.values],
            lines=[AbsorbingLineModel.from_line(d) for d in a.lines],
            thermal=(
                None if thermal is None else ThermalModel.from_state(thermal)
            ),
            column_density_m2=a.column_density.value,
            equivalent_width_nm=a.equivalent_width.value,
            thin_limit_width_nm=a.thin_limit_width.value,
            saturation=a.saturation.value,
            blends=[tuple(b) for b in a.blends],
            flux_closure=a.flux_closure,
            provenance=ProvenanceModel.from_provenance(
                a.transmission.provenance
            ),
            column_provenance=ProvenanceModel.from_provenance(
                a.column_density.provenance
            ),
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
