import { create } from "zustand";
import * as client from "../api/client";
import type { Basis, ConstMultipliers, PlaneQuantity } from "../api/client";
import type {
  ClassicalGhost,
  ConstantsReport,
  ForceLawResult,
  HFLevels,
  IsoMeta,
  LevelsResponse,
  PlaneMeta,
  RadialResponse,
  SampleMeta,
  AbsorptionInfo,
  CurveOfGrowthInfo,
  ScreenedLevels,
  SpectrumResponse,
  StateResponse,
  SystemInfo,
} from "../api/types";
import {
  DEFAULT_EXPR,
  PRESET_PARAMS,
  clampParam,
  defaultParams,
  type ForcePreset,
} from "../lib/forceLaw";
import { manyElectronParams, resolveModel } from "../lib/hfModel";
import type { NucleusMode } from "../lib/nucleus";
import { clampState } from "../lib/quantum";
import { isAlphaValid } from "../lib/whatif";

export type SampleStatus = "idle" | "sampling" | "ready" | "error";
export type ViewMode =
  | "cloud"
  | "plane"
  | "radial"
  | "levels"
  | "spectrum"
  | "whatif"
  | "forcelaw";
export type ColorMode = "solid" | "density" | "phase";

/**
 * What the 3-D view draws: the sampled cloud, the enclosing surface, or both.
 *
 * Purely a choice of representation over the same state, so it invalidates
 * nothing — the same rule the view and colour-mode toggles follow. The surface
 * data it selects is invalidated, but by (n, l, m, system, basis) like every
 * other derived payload, not by this.
 */
export type SurfaceMode = "cloud" | "surface" | "both";

/**
 * Fractions offered as one-click contours.
 *
 * 0.9 is the textbook lobe and the default, so the first surface a user sees is
 * the one they have seen in books, with the 10% it leaves out stated. 0.99 is
 * there because it looks nothing like a textbook lobe, which is the lesson.
 */
export const ISO_FRACTIONS = [0.5, 0.75, 0.9, 0.95, 0.99] as const;

/**
 * Which many-electron model an atom's levels come from.
 *
 * "gsz" is the fitted Green-Sellin-Zachor screened central field, "hf" the
 * self-consistent Hartree-Fock solve. Both are APPROXIMATION, but of different
 * things, and they disagree — so this is a physics input and it invalidates,
 * unlike the presentational toggles which deliberately invalidate nothing.
 */
export type AtomModel = "gsz" | "hf";

const N_MAX_DIAGRAM = 6;

interface AppState {
  n: number;
  l: number;
  m: number;
  system: string;
  basis: Basis;
  view: ViewMode;
  colorMode: ColorMode;
  fineStructure: boolean;
  dirac: boolean;
  bField: number;
  eField: number;
  hyperfine: boolean;
  /** Show real line strengths in the Spectrum view. On by default: uniform bars
   *  silently assert that every line is equally strong, which is false. */
  intensities: boolean;
  /** Weight the Spectrum view by LTE populations. Off by default: it is a
   *  model with a temperature in it, and it should be something you switch on
   *  deliberately rather than the picture you get without asking. */
  thermal: boolean;
  temperatureK: number;
  /** log10(n_e / cm^-3). The control is logarithmic, so the state is too. */
  logNe: number;
  /** Synthesize a line-profile curve instead of leaving lines as bars. Off by
   *  default: it costs a wider request and only means something once a width
   *  mechanism is switched on. */
  profile: boolean;
  /** log10 of the spectrograph resolving power R = lambda/dlambda, or null for
   *  no instrument at all. A model of a machine, never of the atom. */
  logResolvingPower: number | null;
  /** Wavelength window [nm] the profile is synthesized over, or null for the
   *  full across-n range. Set by clicking a line. */
  profileZoom: [number, number] | null;
  /** Show the curve of growth for the zoomed line. Off by default: it answers
   *  a different question from the profile (how the line responds to more gas,
   *  not what it looks like) and deserves to be asked for. */
  showCurveOfGrowth: boolean;
  curveOfGrowth: CurveOfGrowthInfo | null;
  /** Put the whole line list in front of a continuum instead of watching it
   *  emit. A different question from either panel above: not what the gas
   *  gives off, but what it takes out of light passing through. Off by
   *  default, and it needs populations, so it needs LTE on. */
  absorption: boolean;
  /** log10(column density of the element / m^-2). The knob the curve of growth
   *  sweeps, here held at one value for every line at once. */
  logColumn: number;
  absorptionData: AbsorptionInfo | null;
  nucleusMode: NucleusMode;
  count: number;
  systems: SystemInfo[];
  stateInfo: StateResponse | null;
  positions: Float32Array | null;
  density: Float32Array | null;
  phase: Float32Array | null;
  meta: SampleMeta | null;
  status: SampleStatus;
  progress: number;
  error: string | null;
  fps: number;
  planeQuantity: PlaneQuantity;
  plane: { meta: PlaneMeta; values: Float32Array } | null;
  planeStatus: SampleStatus;
  planeProgress: number;
  surfaceMode: SurfaceMode;
  /** The fraction of the electron the surface must enclose. The level follows. */
  isoFraction: number;
  iso: {
    meta: IsoMeta;
    vertices: Float32Array;
    triangles: Uint32Array;
    phase: Float32Array;
  } | null;
  isoStatus: SampleStatus;
  isoProgress: number;
  radial: RadialResponse | null;
  /** Hydrogenic keys return LevelsResponse; screened atoms return ScreenedLevels. */
  levels: LevelsResponse | ScreenedLevels | null;
  spectrum: SpectrumResponse | null;
  /** null = Aufbau ground config (server fills it); else an explicit config string. */
  config: string | null;
  model: AtomModel;
  /**
   * The finished Hartree-Fock solve, or null.
   *
   * Derived from (system, config) and from nothing else — an HF solve knows
   * about occupied subshells, not about the (n, l, m) state being drawn. So it
   * is reset explicitly by setSystem and setConfig rather than living in
   * INVALIDATED, exactly like classicalGhost, and a change of n does not throw
   * away seconds of solve.
   */
  hf: HFLevels | null;
  hfStatus: SampleStatus;
  /**
   * Whether the Hartree-Fock solve keeps its exchange term.
   *
   * False asks for the Hartree model: electrons that repel but are
   * distinguishable, returned COUNTERFACTUAL. A physics input, so it clears
   * `hf` exactly the way setConfig does — the two models are different atoms
   * and a stale one must never sit under a flipped switch.
   *
   * True by default, and deliberately not remembered across a system change:
   * altered physics should be something the user asked for on the atom in
   * front of them, not something inherited from the last one.
   */
  exchange: boolean;
  /**
   * Whether the Hartree-Fock solve keeps the occupancy cap.
   *
   * False is the stronger counterfactual: no cap, so every electron falls into
   * the 1s and the atom has no shells left to have. It contains the exchange
   * one — antisymmetry is what the exclusion principle IS — so the two flags
   * are coupled here rather than left free, and `pauli: false, exchange: true`
   * never leaves this store. The server would answer 422 for it, and building
   * a request the API defines as meaningless is not a state to pass through.
   *
   * True by default and reset by setSystem, exactly like `exchange`.
   */
  pauli: boolean;
  labConst: ConstMultipliers;
  labZ: number;
  whatif: {
    report: ConstantsReport;
    real: LevelsResponse;
    altered: LevelsResponse | null;
  } | null;
  whatifStatus: SampleStatus;
  ghost: boolean;
  classicalGhost: ClassicalGhost | null;
  classicalStatus: SampleStatus;
  forcePreset: ForcePreset;
  forceParams: Record<string, number>;
  forceL: number;
  forceExpr: string;
  forceViz: "well" | "ladder";
  forceLaw: ForceLawResult | null;
  forceStatus: SampleStatus;
  setForcePreset: (preset: ForcePreset) => void;
  setForceParam: (name: string, value: number) => void;
  setForceL: (l: number) => void;
  setForceExpr: (expr: string) => void;
  setForceViz: (viz: "well" | "ladder") => void;
  loadForceLaw: () => Promise<void>;
  setGhost: (on: boolean) => void;
  loadClassical: () => Promise<void>;
  setLabConst: (partial: Partial<ConstMultipliers>) => void;
  setLabZ: (labZ: number) => void;
  loadWhatIf: () => Promise<void>;
  setQuantumNumbers: (n: number, l: number, m: number) => void;
  setSystem: (system: string) => void;
  setConfig: (config: string | null) => void;
  setModel: (model: AtomModel) => void;
  setExchange: (exchange: boolean) => void;
  setPauli: (pauli: boolean) => void;
  loadHF: () => Promise<void>;
  /**
   * Solve the atom before drawing it, under Hartree-Fock only.
   *
   * The solve is what says which subshells exist, so a picture fired before it
   * lands is a picture that may be about to be refused. Resolves to whether a
   * solve is available; false means the reason is already on `error`.
   */
  ensureHF: () => Promise<boolean>;
  setBasis: (basis: Basis) => void;
  setView: (view: ViewMode) => void;
  setColorMode: (colorMode: ColorMode) => void;
  setFineStructure: (fineStructure: boolean) => void;
  setDirac: (dirac: boolean) => void;
  setBField: (bField: number) => void;
  setEField: (eField: number) => void;
  setHyperfine: (hyperfine: boolean) => void;
  setIntensities: (intensities: boolean) => void;
  setThermal: (thermal: boolean) => void;
  setTemperatureK: (temperatureK: number) => void;
  setLogNe: (logNe: number) => void;
  setProfile: (profile: boolean) => void;
  setLogResolvingPower: (logResolvingPower: number | null) => void;
  setProfileZoom: (profileZoom: [number, number] | null) => void;
  setShowCurveOfGrowth: (showCurveOfGrowth: boolean) => void;
  loadCurveOfGrowth: (lambdaNm: number) => Promise<void>;
  setAbsorption: (absorption: boolean) => void;
  setLogColumn: (logColumn: number) => void;
  loadAbsorption: () => Promise<void>;
  setNucleusMode: (nucleusMode: NucleusMode) => void;
  setCount: (count: number) => void;
  setPlaneQuantity: (planeQuantity: PlaneQuantity) => void;
  setSurfaceMode: (surfaceMode: SurfaceMode) => void;
  setIsoFraction: (isoFraction: number) => void;
  loadIso: () => Promise<void>;
  setFps: (fps: number) => void;
  loadSystems: () => Promise<void>;
  loadStateInfo: () => Promise<void>;
  sample: () => Promise<void>;
  loadPlane: () => Promise<void>;
  loadRadial: () => Promise<void>;
  loadLevels: () => Promise<void>;
  loadSpectrum: () => Promise<void>;
}

/** Everything derived from (n, l, m, system, basis) — cleared when any of them changes. */
const INVALIDATED = {
  stateInfo: null,
  positions: null,
  density: null,
  phase: null,
  meta: null,
  status: "idle" as SampleStatus,
  progress: 0,
  error: null,
  plane: null,
  planeStatus: "idle" as SampleStatus,
  planeProgress: 0,
  // A mesh is a contour of one particular |psi|^2, so it is as stale as the
  // cloud is the moment the state changes. The requested fraction is not in
  // here: that is a question the user asked, and it survives to be asked again
  // of the next orbital.
  iso: null,
  isoStatus: "idle" as SampleStatus,
  isoProgress: 0,
  radial: null,
  levels: null,
  spectrum: null,
  curveOfGrowth: null,
  absorptionData: null,
  // A zoom window names a wavelength, and a wavelength names a line of one
  // particular system. Carrying it across a system change would point the
  // profile at empty spectrum and quietly return nothing.
  profileZoom: null as [number, number] | null,
};

export const useAppStore = create<AppState>((set, get) => ({
  n: 1,
  l: 0,
  m: 0,
  system: "h",
  basis: "complex",
  view: "cloud",
  colorMode: "solid",
  fineStructure: false,
  dirac: false,
  bField: 0,
  eField: 0,
  hyperfine: false,
  intensities: true,
  thermal: false,
  temperatureK: 10000,
  logNe: 13,
  profile: false,
  logResolvingPower: null,
  showCurveOfGrowth: false,
  absorption: false,
  // 1e20 m^-2 of hydrogen at 10,000 K: Lyman-alpha black, Balmer-alpha barely
  // there. The default is chosen to open on the contrast the view exists to
  // show rather than on a flat line or an all-black one.
  logColumn: 20,
  // profileZoom's default lives in INVALIDATED, which is spread below.
  nucleusMode: "marker",
  count: 100_000,
  systems: [],
  fps: 0,
  planeQuantity: "density",
  surfaceMode: "cloud",
  isoFraction: 0.9,
  labConst: { hbar: 1, e: 1, m_e: 1, eps0: 1, c: 1 },
  labZ: 1,
  whatif: null,
  whatifStatus: "idle",
  ghost: false,
  classicalGhost: null,
  classicalStatus: "idle",
  forcePreset: "powerlaw",
  forceParams: defaultParams("powerlaw"),
  forceL: 0,
  forceExpr: DEFAULT_EXPR,
  forceViz: "well",
  forceLaw: null,
  forceStatus: "idle",
  config: null,
  // gsz, so that every deep link written before Hartree-Fock existed keeps
  // resolving to the physics it was written against.
  model: "gsz",
  hf: null,
  hfStatus: "idle",
  exchange: true,
  pauli: true,
  ...INVALIDATED,
  // classical ghost data depends on (n, system) but not (l, m, basis), so it is
  // reset explicitly here rather than living in INVALIDATED (basis changes keep it).
  setQuantumNumbers: (n, l, m) =>
    set({ ...clampState(n, l, m), ...INVALIDATED, classicalGhost: null, classicalStatus: "idle" }),
  setSystem: (system) =>
    set((state) => ({
      system,
      // Sulfur and chlorine have no GSZ parameters, so the screened model
      // cannot draw them at all. Selecting one moves to Hartree-Fock rather
      // than leaving a model selected under which every request is refused.
      model: resolveModel(state.systems, system, state.model),
      ...INVALIDATED,
      // selecting a system resets to the Aufbau ground config (server fills it)
      config: null,
      classicalGhost: null,
      classicalStatus: "idle",
      forceLaw: null,
      forceStatus: "idle",
      // a solve belongs to one atom; see the comment on `hf`
      hf: null,
      hfStatus: "idle" as SampleStatus,
      // and altered physics does not follow the user to the next atom
      exchange: true,
      pauli: true,
    })),
  // config is its own physics input: it clears everything derived but keeps
  // the selected system.
  //
  // The full INVALIDATED spread rather than the four level payloads it used to
  // clear. Since Phase 26 the configuration reaches the cloud, the plane, the
  // surface and the radial curve as well, because a Hartree-Fock picture is an
  // orbital of one particular configuration. A 2p drawn under 1s2 2s2 2p6
  // sitting beneath a picker that now reads 1s2 2s2 2p5 3s1 is exactly the
  // stale-physics render this block exists to make impossible.
  setConfig: (config) => set({ config, ...INVALIDATED, hf: null, hfStatus: "idle" }),
  // Switching model changes what the numbers mean, so everything derived under
  // the old one goes. The HF solve itself survives: it is keyed on the atom,
  // not on which model is being displayed, so switching away and back is free.
  setModel: (model) => set({ model, ...INVALIDATED }),
  // Same shape as setConfig: the solve under the old setting is a different
  // atom, so it goes rather than sitting stale beneath a flipped switch. The
  // status goes back to idle rather than to sampling — nothing is running yet,
  // and a view that reported "solving" before a request existed would be
  // describing work nobody started.
  // Turning exchange back ON also restores the cap, and that is physics rather
  // than tidiness: an exchange term exists because the wavefunction is
  // antisymmetric, and an antisymmetric wavefunction is what the exclusion
  // principle is. There is no state with one and not the other, so the store
  // cannot hold one.
  //
  // INVALIDATED goes with it since Phase 26: the switch reaches the cloud, the
  // plane, the surface and the radial curve now, and a Hartree orbital is a
  // different curve rather than the same curve at a different accuracy.
  setExchange: (exchange) =>
    set((s) => ({
      exchange,
      pauli: exchange ? true : s.pauli,
      ...INVALIDATED,
      hf: null,
      hfStatus: "idle",
    })),
  // Off takes exchange with it, for the same reason in the other direction.
  // Back on restores real physics rather than leaving the user in the weaker
  // counterfactual they never asked for.
  //
  // Both directions clear `config`, so the solve uses the ground configuration
  // of whichever rule is now in force: 1s^N with the cap off, Aufbau with it
  // on. A configuration carried across the switch would be a different atom on
  // one side of it, and the server withholds the comparison for exactly that
  // reason — leaving it set would silently cost the user the comparison.
  setPauli: (pauli) =>
    set({
      pauli,
      exchange: pauli,
      config: null,
      ...INVALIDATED,
      hf: null,
      hfStatus: "idle",
    }),
  setBasis: (basis) =>
    set((s) => ({
      basis,
      ...INVALIDATED,
      colorMode: basis === "real" && s.colorMode === "phase" ? "density" : s.colorMode,
    })),
  setView: (view) => set({ view }),
  setColorMode: (colorMode) => set({ colorMode }),
  setFineStructure: (fineStructure) =>
    set({ fineStructure, stateInfo: null, levels: null, spectrum: null }),
  setDirac: (dirac) => set({ dirac, levels: null }),
  setBField: (bField) => set({ bField, levels: null }),
  setEField: (eField) => set({ eField, levels: null }),
  setHyperfine: (hyperfine) => set({ hyperfine, levels: null }),
  setIntensities: (intensities) => set({ intensities, spectrum: null }),
  // Each of these changes what the engine is asked for, so the cached spectrum
  // is stale the instant they move. Same rule as setIntensities.
  setThermal: (thermal) => set({ thermal, spectrum: null }),
  setTemperatureK: (temperatureK) => set({ temperatureK, spectrum: null }),
  setLogNe: (logNe) => set({ logNe, spectrum: null }),
  // The profile is synthesized server-side, so every knob that changes its
  // shape invalidates the cached response exactly like the thermal ones.
  setProfile: (profile) => set({ profile, spectrum: null }),
  setLogResolvingPower: (logResolvingPower) =>
    set({ logResolvingPower, spectrum: null }),
  // The curve belongs to one line, so moving the window discards it too.
  setProfileZoom: (profileZoom) =>
    set({ profileZoom, spectrum: null, curveOfGrowth: null, absorptionData: null }),
  setShowCurveOfGrowth: (showCurveOfGrowth) => set({ showCurveOfGrowth }),
  // The absorption curve is computed for one column against one gas, so both
  // knobs discard it. Same rule as the thermal ones above.
  setAbsorption: (absorption) => set({ absorption, absorptionData: null }),
  setLogColumn: (logColumn) => set({ logColumn, absorptionData: null }),
  // pure render choice: nothing physical to invalidate
  setNucleusMode: (nucleusMode) => set({ nucleusMode }),
  setCount: (count) => set({ count }),
  setPlaneQuantity: (planeQuantity) =>
    set({ planeQuantity, plane: null, planeStatus: "idle", planeProgress: 0 }),
  setFps: (fps) => set({ fps }),
  // lab slice: independent of the main (n,l,m,system) physics — never in INVALIDATED
  setLabConst: (partial) =>
    set((s) => ({
      labConst: { ...s.labConst, ...partial },
      whatif: null,
      whatifStatus: "idle",
    })),
  setLabZ: (labZ) => set({ labZ, whatif: null, whatifStatus: "idle" }),
  // overlay visibility is presentational; the data itself carries provenance
  setGhost: (on) => {
    set({ ghost: on });
    if (on && get().classicalStatus === "idle") void get().loadClassical();
  },
  // force-law slice: its own axis (preset, params, l) — independent of the main
  // (n,l,m,system) physics, so never in INVALIDATED. Changing the preset, a
  // param, or l clears only the force-law data. forceViz is presentational and
  // clears nothing (store invariant). System changes clear it too (Z/mu change).
  setForcePreset: (preset) =>
    set({
      forcePreset: preset,
      forceParams: defaultParams(preset),
      forceLaw: null,
      forceStatus: "idle",
    }),
  setForceParam: (name, value) => {
    const { forcePreset, forceParams } = get();
    const spec = PRESET_PARAMS[forcePreset].find((s) => s.name === name);
    if (spec === undefined) return;
    set({
      forceParams: { ...forceParams, [name]: clampParam(spec, value) },
      forceLaw: null,
      forceStatus: "idle",
    });
  },
  setForceL: (l) =>
    set({ forceL: Math.max(0, Math.round(l)), forceLaw: null, forceStatus: "idle" }),
  setForceExpr: (expr) => set({ forceExpr: expr, forceLaw: null, forceStatus: "idle" }),
  setForceViz: (viz) => set({ forceViz: viz }),
  loadForceLaw: async () => {
    const { forcePreset, forceParams, forceL, forceExpr, system } = get();
    set({ forceStatus: "sampling", error: null });
    try {
      const expr = forcePreset === "custom" ? forceExpr : undefined;
      const forceLaw = await client.getForceLaw(
        system, forcePreset, forceParams, forceL, undefined, expr,
      );
      set({ forceLaw, forceStatus: "ready" });
    } catch (err) {
      set({ forceStatus: "error", error: err instanceof Error ? err.message : String(err) });
    }
  },
  loadClassical: async () => {
    const { n, system } = get();
    set({ classicalStatus: "sampling" });
    try {
      const classicalGhost = await client.getClassical(system, n);
      set({ classicalGhost, classicalStatus: "ready" });
    } catch (err) {
      set({
        classicalStatus: "error",
        error: err instanceof Error ? err.message : String(err),
      });
    }
  },
  loadWhatIf: async () => {
    const { labConst, labZ } = get();
    const sys = `z${labZ}`;
    set({ whatifStatus: "sampling", error: null });
    try {
      const report = await client.getConstants(labConst);
      const alpha = report.alpha.quantity.value;
      // What-If only uses hydrogenic z{N} systems; narrow the union defensively.
      const real = await client.getLevels(sys, N_MAX_DIAGRAM, true);
      if (client.isScreenedLevels(real)) throw new Error("what-if expects hydrogenic levels");
      // altered diagram only when the derived alpha stays in the perturbative range
      const alteredRaw =
        report.altered && isAlphaValid(alpha)
          ? await client.getLevels(sys, N_MAX_DIAGRAM, true, alpha)
          : null;
      if (alteredRaw !== null && client.isScreenedLevels(alteredRaw)) {
        throw new Error("what-if expects hydrogenic levels");
      }
      set({ whatif: { report, real, altered: alteredRaw }, whatifStatus: "ready" });
    } catch (err) {
      set({
        whatifStatus: "error",
        error: err instanceof Error ? err.message : String(err),
      });
    }
  },
  loadSystems: async () => {
    const systems = (await client.getSystems()).systems;
    // A deep link can name ?system=s&model=gsz, which is only knowably wrong
    // once the table saying sulfur has no GSZ parameters has arrived. This is
    // the moment it arrives, so this is where that link gets corrected.
    const { system, model } = get();
    set({ systems, model: resolveModel(systems, system, model) });
  },
  loadStateInfo: async () => {
    const { n, l, m, system, fineStructure } = get();
    set({ stateInfo: await client.getState(n, l, m, system, fineStructure) });
  },
  sample: async () => {
    if (!(await get().ensureHF())) return;
    const { n, l, m, count, basis, system } = get();
    set({ status: "sampling", progress: 0, error: null });
    try {
      const job = await client.createSampleJob({
        n, l, m, count, basis, system, ...manyElectronParams(get()),
      });
      await client.watchJob(job.id, (progress) => set({ progress }));
      const [meta, positions, density, phase] = await Promise.all([
        client.getJobMeta(job.id),
        client.getChannel(job.id, "positions"),
        client.getChannel(job.id, "density"),
        basis === "complex" ? client.getChannel(job.id, "phase") : Promise.resolve(null),
      ]);
      if (meta.kind !== "sample") throw new Error("expected sample-job meta");
      set({ meta, positions, density, phase, status: "ready", progress: 1 });
    } catch (err) {
      set({ status: "error", error: err instanceof Error ? err.message : String(err) });
    }
  },
  // Which of the two representations is drawn is a viewing choice over the same
  // physics, so it clears nothing. The fraction is not: it names a different
  // contour, so the mesh under the old one goes rather than sitting beneath a
  // moved slider claiming to enclose something it does not.
  setSurfaceMode: (surfaceMode) => set({ surfaceMode }),
  setIsoFraction: (isoFraction) =>
    set({
      isoFraction: Math.min(0.999, Math.max(0.001, isoFraction)),
      iso: null,
      isoStatus: "idle",
      isoProgress: 0,
    }),
  loadIso: async () => {
    if (get().isoStatus === "sampling") return;
    if (!(await get().ensureHF())) return;
    const { n, l, m, system, basis, isoFraction } = get();
    set({ isoStatus: "sampling", isoProgress: 0, error: null });
    try {
      const job = await client.createIsoJob({
        n,
        l,
        m,
        system,
        basis,
        fraction: isoFraction,
        ...manyElectronParams(get()),
      });
      await client.watchJob(job.id, (isoProgress) => set({ isoProgress }));
      const [meta, vertices, triangles, phase] = await Promise.all([
        client.getJobMeta(job.id),
        client.getChannel(job.id, "vertices"),
        client.getIndexChannel(job.id, "triangles"),
        client.getChannel(job.id, "phase"),
      ]);
      if (meta.kind !== "isosurface") throw new Error("expected isosurface-job meta");
      set({
        iso: { meta, vertices, triangles, phase },
        isoStatus: "ready",
        isoProgress: 1,
      });
    } catch (err) {
      set({
        isoStatus: "error",
        error: err instanceof Error ? err.message : String(err),
      });
    }
  },
  loadPlane: async () => {
    if (!(await get().ensureHF())) return;
    const { n, l, m, system, basis, planeQuantity } = get();
    set({ planeStatus: "sampling", planeProgress: 0, error: null });
    try {
      const job = await client.createPlaneJob({
        n,
        l,
        m,
        system,
        basis,
        quantity: planeQuantity,
        ...manyElectronParams(get()),
      });
      await client.watchJob(job.id, (planeProgress) => set({ planeProgress }));
      const [meta, values] = await Promise.all([
        client.getJobMeta(job.id),
        client.getChannel(job.id),
      ]);
      if (meta.kind !== "plane") throw new Error("expected plane-job meta");
      set({ plane: { meta, values }, planeStatus: "ready", planeProgress: 1 });
    } catch (err) {
      set({
        planeStatus: "error",
        error: err instanceof Error ? err.message : String(err),
      });
    }
  },
  loadRadial: async () => {
    if (!(await get().ensureHF())) return;
    const { n, l, system } = get();
    set({
      radial: await client.getRadial(n, l, system, undefined, manyElectronParams(get())),
    });
  },
  loadLevels: async () => {
    const { system, fineStructure, config, dirac, bField, eField, hyperfine } = get();
    set({
      levels: await client.getLevels(
        system, N_MAX_DIAGRAM, fineStructure, undefined, config, dirac,
        bField, eField, hyperfine,
      ),
    });
  },
  loadHF: async () => {
    const { system, config, systems, hfStatus, exchange, pauli } = get();
    if (hfStatus === "sampling") return;
    // Z and the electron count live on the system table, so a solve cannot be
    // requested before it has loaded. Ask for it rather than guessing them
    // from the key — the key is a label, the table is the authority.
    const table = systems.length === 0 ? (await client.getSystems()).systems : systems;
    if (systems.length === 0) set({ systems: table });
    const info = table.find((s) => s.key === system);
    if (info === undefined || info.n_electrons === null) {
      set({
        hfStatus: "error",
        error: `Hartree-Fock needs an atom with a known electron count; ${system} has none`,
      });
      return;
    }
    set({ hfStatus: "sampling", error: null });
    try {
      const job = await client.createHFJob({
        z: info.z,
        n_electrons: info.n_electrons,
        config,
        exchange,
        pauli,
      });
      // The solve reports no intermediate progress (the server says why), so
      // there is nothing to show between 0 and 1 and nothing is pretended.
      await client.watchJob(job.id, () => {});
      const meta = await client.getJobMeta(job.id);
      if (!client.isHFLevels(meta)) throw new Error("expected hartree-fock job meta");
      set({ hf: meta, hfStatus: "ready" });
    } catch (err) {
      set({ hfStatus: "error", error: err instanceof Error ? err.message : String(err) });
    }
  },
  ensureHF: async () => {
    // The table first, because it is what decides the model. A deep link can
    // say ?system=s&model=gsz, and firing that request before the table lands
    // spends a 400 on a question the client is one fetch away from answering
    // itself. Cheap: after the first load `systems` is populated and this is
    // a length check.
    if (get().systems.length === 0) await get().loadSystems();
    if (get().model !== "hf") return true;
    if (get().hf !== null) return true;
    await get().loadHF();
    return get().hf !== null;
  },
  loadCurveOfGrowth: async (lambdaNm) => {
    const {
      system, fineStructure, config, temperatureK, logNe, logResolvingPower,
    } = get();
    set({
      curveOfGrowth: await client.getCurveOfGrowth({
        system,
        nMax: N_MAX_DIAGRAM,
        fineStructure,
        lambdaNm,
        thermal: { temperatureK, electronDensityCm3: 10 ** logNe },
        resolvingPower: logResolvingPower === null ? null : 10 ** logResolvingPower,
        config,
      }),
    });
  },
  loadAbsorption: async () => {
    const {
      system, fineStructure, config, temperatureK, logNe, logResolvingPower,
      logColumn, profileZoom,
    } = get();
    set({
      absorptionData: await client.getAbsorption({
        system,
        nMax: N_MAX_DIAGRAM,
        fineStructure,
        columnDensityM2: 10 ** logColumn,
        thermal: { temperatureK, electronDensityCm3: 10 ** logNe },
        resolvingPower: logResolvingPower === null ? null : 10 ** logResolvingPower,
        // Same window as the profile panel. Across the full 90 to 7500 nm range
        // a line is narrower than a pixel, so the only place an absorption
        // line's actual shape exists is a zoom, exactly as for emission.
        window: profileZoom,
        config,
      }),
    });
  },
  loadSpectrum: async () => {
    const {
      system, fineStructure, config, intensities, thermal, temperatureK, logNe,
      profile, logResolvingPower, profileZoom,
    } = get();
    set({
      spectrum: await client.getSpectrum(
        system, N_MAX_DIAGRAM, fineStructure, config, intensities,
        thermal ? { temperatureK, electronDensityCm3: 10 ** logNe } : null,
        {
          on: profile,
          resolvingPower: logResolvingPower === null ? null : 10 ** logResolvingPower,
          window: profileZoom,
        },
      ),
    });
  },
}));
