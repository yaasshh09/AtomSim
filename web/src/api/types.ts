// I mirror src/atomsim/server/schemas.py exactly here; it is my single
// canonical JSON contract.

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
  /** null means honestly absent (a point lepton or an unidentified nucleus), and never zero */
  nuclear_radius: Quantity | null;
  nuclear_radius_fm: Quantity | null;
  /** I keep hydrogenic presets as "hydrogenic"; He-Ar screened atoms I call "screened". */
  kind: "hydrogenic" | "screened";
  /** The electron count for screened atoms; null for hydrogenic systems. */
  n_electrons: number | null;
  /**
   * Whether the GSZ screened model has published parameters for this atom.
   *
   * False for sulfur and chlorine only: Szydlik and Green never fitted Z = 16
   * or 17, and my Hartree-Fock solve needs no fitted table, so I offer the atom
   * with one of its two models unavailable rather than withholding it
   * entirely. Always true for hydrogenic presets, where no screened model
   * applies at all.
   */
  has_gsz: boolean;
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

export interface StarkSublevel {
  n1: number;
  n2: number;
  m: number;
  k: number;
  energy: Quantity;
  energy_ev: Quantity;
}

export interface GrossLevel {
  n: number;
  degeneracy: number;
  energy: Quantity;
  energy_ev: Quantity;
  sublevels?: StarkSublevel[] | null;
}

export interface ZeemanSublevel {
  m_j: number;
  branch: string;
  j_label: number;
  high_field_label: string;
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
  sublevels?: ZeemanSublevel[] | null;
}

export interface HyperfineLevel {
  F: number;
  energy: Quantity;
  energy_ev: Quantity;
  shift: Quantity;
  shift_ev: Quantity;
}

export interface HyperfineShell {
  n: number;
  available: boolean;
  nucleus?: string | null;
  I?: number | null;
  A?: Quantity | null;
  A_ev?: Quantity | null;
  levels: HyperfineLevel[];
  note?: string | null;
  reason?: string | null;
}

export interface LevelsResponse {
  system: SystemInfo;
  n_max: number;
  fine_structure: boolean;
  alpha: number;
  gross: GrossLevel[];
  fine: FineLevel[] | null;
  dirac: boolean;
  b_field: number;
  e_field: number;
  hyperfine: boolean;
  hyperfine_shells?: HyperfineShell[] | null;
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
  /**
   * The total radial electron density, which I send only under Hartree-Fock.
   *
   * This is the observable, unlike the two curves above it: its peaks are the
   * shells, and the area under each is that shell's electron count. I send
   * null under the screened model rather than dropping the field, so I keep
   * one response shape.
   */
  total_density: FieldData | null;
  /**
   * Both models' total density on one grid, which I send only when asked.
   *
   * I send null rather than dropping the field when the toggle is off, so I
   * keep one response shape whichever model I draw the orbital plots under.
   */
  density_comparison: DensityComparison | null;
}

export interface ShellPeak {
  label: string;
  /** null means this model's density has no local maximum I could find for this shell. */
  gsz_radius: number | null;
  hf_radius: number | null;
  /** The relative drop into the minimum before the peak; null for the innermost. */
  gsz_depth: number | null;
  hf_depth: number | null;
}

export interface DensityComparison {
  gsz: FieldData;
  hf: FieldData;
  displaced_charge: Quantity;
  shells: ShellPeak[];
  provenance: Provenance;
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
  /** The spontaneous emission rate [s^-1]; null when nobody asked me for
   *  strengths or I cannot give them honestly, see
   *  SpectrumResponse.intensity_note. */
  einstein_a_s: Quantity | null;
  /** The absorption oscillator strength (dimensionless); null on the same terms. */
  oscillator_strength: Quantity | null;
  /** The LTE emission rate [eV/s per atom of the element]; null unless you
   *  gave me thermal conditions. A rate I modelled, not a brightness anyone
   *  measured. */
  emissivity: Quantity | null;
}

/** The LTE conditions I computed a spectrum at, and what they produced. */
export interface ThermalInfo {
  temperature_k: number;
  electron_density_cm3: number;
  /** The fraction of the element that is ionized. Once this nears 1 my whole
   *  spectrum is faint because there are no neutrals left to emit, which is
   *  why I show it rather than folding it silently into the scale. */
  ionized_fraction: Quantity;
  /** I truncate this at n_max, and I put the cutoff in its provenance assumptions. */
  partition_function: Quantity;
}

export interface ComparisonInfo {
  wavelength_nm: number;
  reference_nm: number;
  reference_uncertainty_nm: number | null;
  delta_nm: number;
  relative_error: number;
  within_tolerance: boolean;
}

/** The width budget of one line, so I can say what set it. */
export interface LineWidthInfo {
  label: string;
  wavelength_nm: number;
  n_upper: number;
  n_lower: number;
  /** The Gaussian sigma [nm]: I add Doppler and instrument in quadrature. */
  sigma_nm: number;
  /** The Lorentzian half-width at half maximum [nm]: the natural, lifetime width. */
  gamma_nm: number;
  fwhm_nm: number;
  /** Which mechanisms contributed, for example ["natural", "Doppler"]. */
  terms: string[];
}

/** A spectrum I synthesized: the curve, its widths, and what I left out. */
export interface ProfileInfo {
  wavelength_nm: number[];
  intensity: number[];
  unit: string;
  /** What the area under a line means; I mirror the bar quantity exactly. */
  weight_kind: "emissivity" | "rate" | "uniform";
  resolving_power: number | null;
  /** My curve integral over the summed line strengths: my own measured
   *  quadrature error rather than an assumption, where 1.0 means I lost
   *  nothing. */
  flux_closure: number;
  widths: LineWidthInfo[];
  /** The collisional broadening I have NOT put in this curve, sized. */
  stark_span_nm: Quantity | null;
  /** I set this only when that missing width rivals the one I did model. */
  stark_note: string | null;
  provenance: Provenance;
}

/** One regime of the curve of growth. Which one a line sits in decides
 *  whether its strength measures how much gas there is at all. */
export type GrowthRegime = "linear" | "saturated" | "damping";

/** Equivalent width against column density, with the branches I labelled. */
export interface CurveOfGrowthInfo {
  label: string;
  wavelength_nm: number;
  oscillator_strength: number;
  /** The widths I computed the curve for; the knees sit where they put them. */
  sigma_nm: number;
  gamma_nm: number;
  /** a = gamma / (sigma sqrt2). My damping branch starts at a*tau = 1. */
  damping_parameter: number;
  column_density_m2: number[];
  equivalent_width_nm: number[];
  regime: GrowthRegime[];
  /** The local log-log slope: 1, then ~0, then 1/2. */
  slope: number[];
  /** The optical depth at line centre. This is what decides the regime. */
  tau_centre: number[];
  window_nm: number;
  provenance: Provenance;
}

/** One line's share of a blended absorption spectrum. */
export interface AbsorbingLineInfo {
  wavelength_nm: number;
  /** I name the transition here, not the series: "3d->2p", not "3->2". Three
   *  lines share 656.4696 nm and they do not absorb alike. */
  label: string;
  oscillator_strength: number;
  /** The column in *this line's* lower level. It is why one gas gives every
   *  line a different optical depth. */
  lower_column_m2: number;
  tau_centre: number;
  regime: GrowthRegime;
  /** What this line would remove on its own with nothing saturating. */
  thin_width_nm: number;
  fwhm_nm: number;
}

/** A whole line list absorbing at once against a flat continuum. */
export interface AbsorptionInfo {
  wavelength_nm: number[];
  transmission: number[];
  /** I keep this beside the transmission because 1e-9 and 1e-30 both draw as
   *  black and are not the same gas. */
  optical_depth: number[];
  lines: AbsorbingLineInfo[];
  thermal: ThermalInfo | null;
  column_density_m2: number;
  equivalent_width_nm: number;
  thin_limit_width_nm: number;
  /** measured / thin. It sits below 1 by exactly how much the naive sum
   *  overstates the absorption, which is how much of the census I am losing. */
  saturation: number;
  blends: [string, string][];
  flux_closure: number;
  provenance: Provenance;
  /** The column is a knob you turned, and it says so. */
  column_provenance: Provenance;
}

export interface SpectrumResponse {
  system: SystemInfo;
  n_max: number;
  fine_structure: boolean;
  lines: SpectralLineInfo[];
  comparison: ComparisonInfo[] | null;
  reference_citation: string | null;
  tolerance_relative: number | null;
  /** I set this when you asked for strengths and I withheld them, and I say
   *  which case applies. */
  intensity_note: string | null;
  /** I send this exactly when my lines carry an emissivity. */
  thermal: ThermalInfo | null;
  /** The curve I synthesized, when you asked for one and I could build it. */
  profile: ProfileInfo | null;
  /** Why I have no curve, when you asked for one. I name the knob I am missing
   *  rather than inventing a width to draw. */
  profile_note: string | null;
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

/**
 * Which many-electron model a picture job asks for, and under which rules.
 *
 * Every field defaults on the server to the already-shipped screened
 * behaviour, so omitting the object entirely cannot ask for Hartree-Fock or
 * for a counterfactual.
 */
export interface ManyElectronParams {
  model: "gsz" | "hf";
  /** Electron configuration, e.g. "1s2 2s2 2p5 3s1". null is the ground one. */
  config: string | null;
  exchange: boolean;
  pauli: boolean;
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
  /**
   * Which many-electron model drew this: "gsz" or "hf".
   *
   * Read this rather than sniffing the provenance prose. A view that has to
   * parse a sentence to learn which physics it is drawing will eventually
   * parse it wrong and draw the other one.
   */
  model: string;
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
  /** Which many-electron model drew this; see SampleMeta.model. */
  model: string;
  provenance: Provenance;
}

/**
 * A contour of |psi|^2 and the numbers that make it a claim about an atom.
 *
 * `target_fraction` is what was asked for and `enclosed_fraction` is what the
 * grid delivered; they are close and they are not the same number, so both are
 * carried. `outside_fraction` is the sentence textbooks leave out, and it is
 * sent rather than derived in the browser so that the value on screen is the
 * one the engine measured.
 */
export interface IsoMeta {
  kind: "isosurface";
  vertex_count: number;
  triangle_count: number;
  channels: ChannelInfo[];
  target_fraction: number;
  enclosed_fraction: Quantity;
  outside_fraction: number;
  level: Quantity;
  escaped_fraction: Quantity;
  mesh_volume: Quantity;
  voxel_volume: Quantity;
  area: Quantity;
  components: number;
  half_width: number;
  resolution: number;
  axis_unit: string;
  n: number;
  l: number;
  m: number;
  basis: string;
  system: string;
  /** Which many-electron model drew this; see SampleMeta.model. */
  model: string;
  label: string;
  provenance: Provenance;
}

export interface HFOrbital {
  n: number;
  l: number;
  label: string;
  occupancy: number;
  energy: Quantity;
  energy_ev: Quantity;
  /** The /data channel carrying this orbital's P(r); shapes travel as binary. */
  channel: string;
}

/**
 * The atom with the exclusion principle, next to the atom without it.
 *
 * Both halves come from one server-side comparison rather than two jobs the
 * client differences, for the same reason the exchange energy does: two solves
 * fetched separately are free to differ by their mesh as well as by their
 * physics, and the gap between two calculations would get drawn as the gap
 * between two models.
 *
 * The `variational_*` fields are the closed-form check on the collapsed number,
 * N electrons in one 1s of exponent zeta, carried so the page can show what
 * the result was tested against instead of asking to be believed.
 */
export interface PauliCollapse {
  /** E(collapsed) − E(real). Negative, and large: the collapse binds hard. */
  binding_change: Quantity;
  binding_change_ev: Quantity;
  real_total_energy: Quantity;
  real_total_energy_ev: Quantity;
  real_config: string;
  real_radius: Quantity;
  collapsed_radius: Quantity;
  /** ⟨r⟩(collapsed) / ⟨r⟩(real): below 1, and falling with Z. */
  radius_ratio: Quantity;
  variational_zeta: Quantity;
  variational_energy: Quantity;
  variational_energy_ev: Quantity;
}

/**
 * A finished Hartree-Fock solve.
 *
 * Arrives on the job `meta` endpoint rather than as its own response, because
 * the solve takes seconds. Unlike a sample or plane job the meta IS the
 * result: /data carries only the orbital shapes.
 *
 * `symbol` is null above the preset table. Hartree-Fock needs no fitted
 * parameters and so solves ions the named-atom library has never heard of, and
 * the server declines to invent an element for them.
 */
export interface HFLevels {
  kind: "hf";
  z: number;
  n_electrons: number;
  symbol: string | null;
  config: string;
  is_ground: boolean;
  /**
   * False means exchange was removed: the Hartree model, in which electrons
   * repel but are distinguishable. Read this rather than sniffing the
   * provenance tier, a view that has to parse prose to learn which physics it
   * is drawing will eventually draw the wrong one.
   */
  exchange: boolean;
  /**
   * E_HF − E_Hartree, in hartree. Negative, because exchange stabilizes; zero
   * for an atom with no same-spin pair, which is the answer and not a missing
   * value. Present only on a counterfactual solve, since only that one ran
   * both models.
   */
  exchange_energy: Quantity | null;
  exchange_energy_ev: Quantity | null;
  /**
   * False means the occupancy cap went too and the configuration collapsed to
   * 1s^N. Its own flag rather than something to infer from `exchange`, because
   * exchange alone is the weaker counterfactual in which the cap still holds,
   * and the two carry opposite disclosures.
   */
  pauli: boolean;
  /**
   * The real atom beside the collapsed one. Present only on a `pauli: false`
   * solve of a ground configuration; a hand-written collapsed configuration has
   * no twin with the cap on, so nothing is sent rather than a comparison
   * against a different atom.
   */
  collapse: PauliCollapse | null;
  orbitals: HFOrbital[];
  total_energy: Quantity;
  total_energy_ev: Quantity;
  kinetic: Quantity;
  potential: Quantity;
  /** A diagnostic of the solve, NUMERICAL, never draw this as physics. */
  virial_ratio: Quantity;
  iterations: number;
  coarse_iterations: number;
  converged: boolean;
  provenance: Provenance;
  grid_channel: string;
  grid_points: number;
  channels: ChannelInfo[];
}

export type JobMeta = SampleMeta | PlaneMeta | IsoMeta | HFLevels;
