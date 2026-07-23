// Mirrors src/atomsim/server/schemas.py exactly — the single canonical JSON contract.

export type Fidelity =
  | "exact"
  | "numerical"
  | "approximation"
  | "counterfactual"
  | "visual_liberty";

export interface Provenance {
  fidelity: Fidelity;
  method: string;
  assumptions: string[];
  error_estimate: number | null;
  refinement: string | null;
}

export interface Quantity {
  value: number;
  unit: string;
  label: string;
  provenance: Provenance;
}

export interface FieldData {
  values: number[];
  grid: number[];
  unit: string;
  grid_unit: string;
  label: string;
  provenance: Provenance;
}

export interface SystemInfo {
  key: string;
  name: string;
  z: number;
  mu_ratio: Quantity;
  m_over_m_nucleus: number;
  description: string;
  /** null = honestly absent (point lepton / unidentified nucleus), never zero */
  nuclear_radius: Quantity | null;
  nuclear_radius_fm: Quantity | null;
  /** Hydrogenic presets stay "hydrogenic"; He–Ar screened atoms are "screened". */
  kind: "hydrogenic" | "screened";
  /** Electron count for screened atoms; null for hydrogenic systems. */
  n_electrons: number | null;
}

export interface LevelInfo {
  j: number;
  energy: Quantity;
  energy_ev: Quantity;
  shift: Quantity;
  shift_ev: Quantity;
}

export interface StateResponse {
  n: number;
  l: number;
  m: number;
  system: SystemInfo;
  energy: Quantity;
  energy_ev: Quantity;
  mean_radius: Quantity;
  mean_radius_pm: Quantity;
  angular_momentum: Quantity;
  radial_nodes: number;
  angular_nodes: number;
  levels: LevelInfo[];
}

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
  alpha: number;
  gross: GrossLevel[];
  fine: FineLevel[] | null;
  dirac: boolean;
}

export interface ScreenedOrbital {
  n: number;
  l: number;
  label: string;
  occupancy: number;
  energy: Quantity;
  energy_ev: Quantity;
}

export interface ScreenedLevels {
  system: SystemInfo;
  config: string;
  is_ground: boolean;
  orbitals: ScreenedOrbital[];
  total_energy: Quantity;
  total_energy_ev: Quantity;
}

export interface DerivedObservable {
  quantity: Quantity;
  ratio: number;
  changed: boolean;
}

export interface ConstantsReport {
  alpha: DerivedObservable;
  bohr_radius_pm: DerivedObservable;
  hartree_ev: DerivedObservable;
  altered: boolean;
}

export interface RadialResponse {
  n: number;
  l: number;
  system: SystemInfo;
  r_wavefunction: FieldData;
  radial_probability: FieldData;
}

export interface SpectralLineInfo {
  n_upper: number;
  l_upper: number;
  j_upper: number | null;
  n_lower: number;
  l_lower: number;
  j_lower: number | null;
  energy_ev: Quantity;
  wavelength_nm: Quantity;
}

export interface ComparisonInfo {
  wavelength_nm: number;
  reference_nm: number;
  reference_uncertainty_nm: number | null;
  delta_nm: number;
  relative_error: number;
  within_tolerance: boolean;
}

export interface SpectrumResponse {
  system: SystemInfo;
  n_max: number;
  fine_structure: boolean;
  lines: SpectralLineInfo[];
  comparison: ComparisonInfo[] | null;
  reference_citation: string | null;
  tolerance_relative: number | null;
}

export interface BohrOrbit {
  n: number;
  radius_bohr: Quantity;
  radius_pm: Quantity;
}

export interface ClassicalGhost {
  n: number;
  system_key: string;
  z: number;
  orbits: BohrOrbit[];
  r0_bohr: Quantity;
  collapse_time_s: Quantity;
  orbital_period_s: Quantity;
  orbit_count: Quantity;
}

export interface ForceLawLevel {
  radial_index: number;
  energy: Quantity;
  energy_ev: Quantity;
  trusted: boolean;
}

export interface ReferenceItem {
  label: string;
  energy: Quantity;
  energy_ev: Quantity;
}

export interface Reference {
  kind: "levels" | "markers";
  items: ReferenceItem[];
}

export interface PotentialCurve {
  r: number[];
  v_ev: number[];
  provenance: Provenance;
}

export interface ForceLawResult {
  preset: string;
  params: Record<string, number>;
  l: number;
  z: number;
  system: SystemInfo;
  counterfactual: ForceLawLevel[];
  bound_count: number;
  requested_count: number;
  reference: Reference;
  potential_curve: PotentialCurve;
  expression: string | null;
}

export type JobStatus = "pending" | "running" | "done" | "error";

export interface JobInfo {
  id: string;
  status: JobStatus;
  progress: number;
  error: string | null;
}

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
