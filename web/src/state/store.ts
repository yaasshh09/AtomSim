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
import { manyElectronParams, resolveCompare, resolveModel } from "../lib/hfModel";
import type { NucleusMode } from "../lib/nucleus";
import { clampState } from "../lib/quantum";
import { currentUrlState, type UrlState } from "../lib/urlState";
import { isAlphaValid } from "../lib/whatif";
import { tourReset } from "../tours/apply";
import { tourById } from "../tours/registry";
import { readMemory, rememberCompleted, rememberDismissed, shouldInvite } from "../tours/seen";
import { clampStep, stepState } from "../tours/step";

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
 * What I draw in the 3-D view: the sampled cloud, the enclosing surface, or
 * both.
 *
 * This is purely a choice of representation over the same state, so it
 * invalidates nothing, the same rule I hold my view and colour-mode toggles
 * to. I do invalidate the surface data it selects, but by (n, l, m, system,
 * basis) like every other derived payload, not by this.
 */
export type SurfaceMode = "cloud" | "surface" | "both";

/**
 * The fractions I offer as one-click contours.
 *
 * 0.9 is the textbook lobe and my default, so the first surface you see is the
 * one you have seen in books, with the 10% it leaves out stated. I keep 0.99
 * because it looks nothing like a textbook lobe, which is the lesson.
 */
export const ISO_FRACTIONS = [0.5, 0.75, 0.9, 0.95, 0.99] as const;

/**
 * Which many-electron model I take an atom's levels from.
 *
 * "gsz" is the fitted Green-Sellin-Zachor screened central field, "hf" the
 * self-consistent Hartree-Fock solve. Both are APPROXIMATION, but of different
 * things, and they disagree, so this is a physics input and I invalidate on
 * it, unlike my presentational toggles which deliberately invalidate nothing.
 */
export type AtomModel = "gsz" | "hf";

const N_MAX_DIAGRAM = 6;

/**
 * The `/api/systems` request I have in flight, or null.
 *
 * I keep it at module level rather than in the store on purpose: no view
 * renders it or should re-render on it, it is bookkeeping for one fetch. See
 * `loadSystems` for who races for it.
 */
let inFlightSystems: Promise<{ systems: SystemInfo[] }> | null = null;

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
  /** Whether I show real line strengths in the Spectrum view. On by default,
   *  because uniform bars silently assert that every line is equally strong,
   *  which is false. */
  intensities: boolean;
  /** Whether I weight the Spectrum view by LTE populations. Off by default,
   *  because it is a model with a temperature in it and you should switch it
   *  on deliberately rather than getting it without asking. */
  thermal: boolean;
  temperatureK: number;
  /** log10(n_e / cm^-3). My control is logarithmic, so my state is too. */
  logNe: number;
  /** Whether I synthesize a line-profile curve instead of leaving the lines as
   *  bars. Off by default, because it costs a wider request and only means
   *  something once a width mechanism is switched on. */
  profile: boolean;
  /** log10 of the spectrograph resolving power R = lambda/dlambda, or null when
   *  there is no instrument at all. A model of a machine, never of the atom. */
  logResolvingPower: number | null;
  /** The wavelength window [nm] I synthesize the profile over, or null for the
   *  full across-n range. You set it by clicking a line. */
  profileZoom: [number, number] | null;
  /** Whether I show the curve of growth for the zoomed line. Off by default,
   *  because it answers a different question from the profile (how the line
   *  responds to more gas, not what it looks like) and deserves to be asked
   *  for. */
  showCurveOfGrowth: boolean;
  curveOfGrowth: CurveOfGrowthInfo | null;
  /** Whether I put the whole line list in front of a continuum instead of
   *  watching it emit. A different question from either panel above: not what
   *  the gas gives off, but what it takes out of light passing through. Off by
   *  default, and I need populations for it, so I need LTE on. */
  absorption: boolean;
  /** log10(column density of the element / m^-2). The knob my curve of growth
   *  sweeps, held here at one value for every line at once. */
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
  /** The fraction of the electron I need the surface to enclose. The level follows. */
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
  /** Hydrogenic keys give me a LevelsResponse; screened atoms give me ScreenedLevels. */
  levels: LevelsResponse | ScreenedLevels | null;
  spectrum: SpectrumResponse | null;
  /** null means the Aufbau ground config, which my server fills in; otherwise an explicit config string. */
  config: string | null;
  model: AtomModel;
  /**
   * The finished Hartree-Fock solve, or null.
   *
   * I derive it from (system, config) and from nothing else: an HF solve knows
   * about occupied subshells, not about the (n, l, m) state I am drawing. So I
   * reset it explicitly in setSystem and setConfig rather than putting it in
   * INVALIDATED, exactly like classicalGhost, and a change of n does not throw
   * away seconds of solve.
   */
  hf: HFLevels | null;
  hfStatus: SampleStatus;
  /**
   * Whether I keep the exchange term in my Hartree-Fock solve.
   *
   * False asks me for the Hartree model: electrons that repel but are
   * distinguishable, which I return COUNTERFACTUAL. It is a physics input, so
   * it clears `hf` exactly the way setConfig does: the two models are
   * different atoms and I must never leave a stale one under a flipped switch.
   *
   * True by default, and I deliberately do not remember it across a system
   * change: altered physics should be something you asked for on the atom in
   * front of you, not something I inherited from the last one.
   */
  exchange: boolean;
  /**
   * Whether I keep the occupancy cap in my Hartree-Fock solve.
   *
   * False is the stronger counterfactual: no cap, so every electron falls into
   * the 1s and the atom has no shells left to have. It contains the exchange
   * one, since antisymmetry is what the exclusion principle IS, so I couple
   * the two flags here rather than leaving them free, and `pauli: false,
   * exchange: true` never leaves this store. My server would answer 422 for
   * it, and I will not pass through a state my API defines as meaningless.
   *
   * True by default, and I reset it in setSystem exactly like `exchange`.
   */
  pauli: boolean;
  /**
   * Whether I draw both models at once in the density plot.
   *
   * I keep it out of INVALIDATED and do not spread it with them. It names an
   * extra curve on one payload rather than a different atom, so my cloud,
   * plane and surface are all still exactly as true as they were; throwing
   * them away would be seconds of solve spent telling you nothing. Only the
   * radial response goes, because only the radial response is missing a field.
   */
  compare: boolean;
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
  setCompare: (compare: boolean) => void;
  /**
   * The tour you are taking, or null.
   *
   * `savedState` is your own state, which I snapshot on entry and restore on
   * exit. If you are three minutes into a chlorine Hartree-Fock solve I should
   * not lose it because you clicked a tour out of curiosity.
   */
  tourId: string | null;
  stepIndex: number;
  savedState: UrlState | null;
  startTour: (id: string, step?: number) => void;
  exitTour: () => void;
  /** Leave the tour, and I record that you read it to the end. */
  finishTour: () => void;
  goToStep: (i: number) => void;
  /**
   * Whether I still have the first-visit invitation on screen.
   *
   * I seed it from the browser's memory, so if you have already answered it,
   * by taking a tour or by skipping past it, you never meet it again. I stay
   * usable underneath it either way: my invitation is a bar, not a gate.
   */
  inviteOpen: boolean;
  /** The tours you have read to the last step, for my menu's done marks. */
  completedTours: string[];
  dismissInvite: () => void;
  loadHF: () => Promise<void>;
  /**
   * I solve the atom before drawing it, under Hartree-Fock only.
   *
   * The solve is what tells me which subshells exist, so a picture I fire
   * before it lands may be about to be refused. This resolves to whether I
   * have a solve available; false means I have already put the reason on
   * `error`.
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

/** Everything I derive from (n, l, m, system, basis), which I clear when any of them changes. */
export const INVALIDATED = {
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
  // A mesh is a contour of one particular |psi|^2, so it goes as stale as my
  // cloud does the moment the state changes. I leave the requested fraction
  // out of here: that is a question you asked, and it survives to be asked
  // again of the next orbital.
  iso: null,
  isoStatus: "idle" as SampleStatus,
  isoProgress: 0,
  radial: null,
  levels: null,
  spectrum: null,
  curveOfGrowth: null,
  absorptionData: null,
  // A zoom window names a wavelength, and a wavelength names a line of one
  // particular system. If I carried it across a system change I would point
  // the profile at empty spectrum and quietly return nothing.
  profileZoom: null as [number, number] | null,
};

/**
 * What the browser remembers about tours, which I read once at startup.
 *
 * I read it here rather than in the components that need it because the record
 * only ever changes through my own actions, so one read at load with my store
 * as the live copy cannot disagree with itself. If a second tab finishes a
 * tour I will not notice until reload, which costs a returning reader nothing.
 */
const startingMemory = readMemory();

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
  // there. I chose this default so I open on the contrast the view exists to
  // show rather than on a flat line or an all-black one.
  logColumn: 20,
  // I keep profileZoom's default in INVALIDATED, which I spread below.
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
  // I default to gsz, so that every deep link written before I could do
  // Hartree-Fock keeps resolving to the physics it was written against.
  model: "gsz",
  hf: null,
  hfStatus: "idle",
  exchange: true,
  pauli: true,
  compare: false,
  tourId: null,
  stepIndex: 0,
  savedState: null,
  inviteOpen: shouldInvite(startingMemory),
  completedTours: startingMemory.completed,
  ...INVALIDATED,
  // My classical ghost data depends on (n, system) but not (l, m, basis), so I
  // reset it explicitly here rather than keeping it in INVALIDATED (a basis
  // change keeps it).
  setQuantumNumbers: (n, l, m) =>
    set({ ...clampState(n, l, m), ...INVALIDATED, classicalGhost: null, classicalStatus: "idle" }),
  setSystem: (system) =>
    set((state) => ({
      system,
      // Sulfur and chlorine have no GSZ parameters, so I cannot draw them with
      // the screened model at all. When you select one I move to Hartree-Fock
      // rather than leaving a model selected under which I refuse every
      // request.
      model: resolveModel(state.systems, system, state.model),
      ...INVALIDATED,
      // selecting a system resets me to the Aufbau ground config, which my
      // server fills in
      config: null,
      classicalGhost: null,
      classicalStatus: "idle",
      forceLaw: null,
      forceStatus: "idle",
      // a solve belongs to one atom; see my comment on `hf`
      hf: null,
      hfStatus: "idle" as SampleStatus,
      // and I do not carry altered physics to the next atom
      exchange: true,
      pauli: true,
      // nor a comparison: it is something you asked for about the atom in
      // front of you, and the next one may not have both models.
      compare: false,
    })),
  // config is its own physics input: I clear everything derived from it but
  // keep the system you selected.
  //
  // I spread the full INVALIDATED rather than the four level payloads I used
  // to clear. Since Phase 26 the configuration reaches my cloud, plane,
  // surface and radial curve as well, because a Hartree-Fock picture is an
  // orbital of one particular configuration. A 2p I drew under 1s2 2s2 2p6
  // sitting beneath a picker that now reads 1s2 2s2 2p5 3s1 is exactly the
  // stale-physics render this block exists to make impossible.
  setConfig: (config) => set({ config, ...INVALIDATED, hf: null, hfStatus: "idle" }),
  // Switching model changes what my numbers mean, so everything I derived
  // under the old one goes. The HF solve itself survives: I key it on the
  // atom, not on which model I am displaying, so switching away and back is
  // free.
  setModel: (model) => set({ model, ...INVALIDATED }),
  // Same shape as setConfig: the solve under the old setting is a different
  // atom, so it goes rather than sitting stale beneath a flipped switch. I put
  // the status back to idle rather than to sampling, because nothing is
  // running yet and a view that reported "solving" before a request existed
  // would be describing work nobody started.
  // Turning exchange back ON also restores the cap, and that is physics rather
  // than tidiness: an exchange term exists because the wavefunction is
  // antisymmetric, and an antisymmetric wavefunction is what the exclusion
  // principle is. There is no state with one and not the other, so I cannot
  // hold one.
  //
  // INVALIDATED goes with it since Phase 26: the switch reaches my cloud,
  // plane, surface and radial curve now, and a Hartree orbital is a different
  // curve rather than the same curve at a different accuracy.
  setExchange: (exchange) =>
    set((s) => ({
      exchange,
      pauli: exchange ? true : s.pauli,
      ...INVALIDATED,
      hf: null,
      hfStatus: "idle",
    })),
  // Off takes exchange with it, for the same reason in the other direction.
  // Back on restores real physics rather than leaving you in the weaker
  // counterfactual you never asked for.
  //
  // I clear `config` in both directions, so my solve uses the ground
  // configuration of whichever rule is now in force: 1s^N with the cap off,
  // Aufbau with it on. A configuration carried across the switch would be a
  // different atom on one side of it, and my server withholds the comparison
  // for exactly that reason, so leaving it set would silently cost you the
  // comparison.
  setPauli: (pauli) =>
    set({
      pauli,
      exchange: pauli,
      config: null,
      ...INVALIDATED,
      hf: null,
      hfStatus: "idle",
    }),
  // My one many-electron control that is not a physics input. Nothing about
  // the atom changed, so my solve, cloud, plane and surface all stand; only
  // the radial response goes, because only it is missing a field.
  setCompare: (compare) => set({ compare, radial: null }),
  // A tour step is a whole state, not a patch, so I send it through tourReset
  // rather than a raw setState: see my comment on tourReset for what would
  // otherwise render under the new labels.
  startTour: (id, step = 0) => {
    const tour = tourById(id);
    // For an id I do not know I leave everything alone, my invitation
    // included: a stale deep link should not cost you the offer of a tour.
    if (!tour) return;
    // Starting one answers my invitation whichever door it came through: the
    // invitation's own button, the menu, or a link someone shared.
    rememberDismissed();
    set((s) => {
      const i = clampStep(tour, step);
      return {
        ...tourReset(stepState(tour.steps[i]), s.systems),
        tourId: id,
        stepIndex: i,
        inviteOpen: false,
        // Only on entry. If you re-enter mid-tour I must not overwrite your
        // own state with a tour step's.
        savedState: s.savedState ?? currentUrlState(s),
      };
    });
  },
  goToStep: (i) =>
    set((s) => {
      const tour = s.tourId ? tourById(s.tourId) : null;
      if (!tour) return {};
      const next = clampStep(tour, i);
      return { ...tourReset(stepState(tour.steps[next]), s.systems), stepIndex: next };
    }),
  exitTour: () =>
    set((s) => ({
      ...(s.savedState ? tourReset(s.savedState, s.systems) : {}),
      tourId: null,
      stepIndex: 0,
      savedState: null,
    })),
  // Leaving and finishing differ by one fact worth keeping: that you reached
  // the end. The exit itself is exitTour's, not a second copy of it.
  finishTour: () => {
    const id = get().tourId;
    if (id) set({ completedTours: rememberCompleted(id).completed });
    get().exitTour();
  },
  dismissInvite: () => {
    rememberDismissed();
    set({ inviteOpen: false });
  },
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
  // Each of these changes what I ask my engine for, so my cached spectrum goes
  // stale the instant they move. Same rule as setIntensities.
  setThermal: (thermal) => set({ thermal, spectrum: null }),
  setTemperatureK: (temperatureK) => set({ temperatureK, spectrum: null }),
  setLogNe: (logNe) => set({ logNe, spectrum: null }),
  // I synthesize the profile server-side, so every knob that changes its shape
  // invalidates my cached response exactly like the thermal ones.
  setProfile: (profile) => set({ profile, spectrum: null }),
  setLogResolvingPower: (logResolvingPower) =>
    set({ logResolvingPower, spectrum: null }),
  // The curve belongs to one line, so when you move the window I discard it too.
  setProfileZoom: (profileZoom) =>
    set({ profileZoom, spectrum: null, curveOfGrowth: null, absorptionData: null }),
  setShowCurveOfGrowth: (showCurveOfGrowth) => set({ showCurveOfGrowth }),
  // I compute the absorption curve for one column against one gas, so I
  // discard it on either knob. Same rule as the thermal ones above.
  setAbsorption: (absorption) => set({ absorption, absorptionData: null }),
  setLogColumn: (logColumn) => set({ logColumn, absorptionData: null }),
  // a pure render choice: I have nothing physical to invalidate
  setNucleusMode: (nucleusMode) => set({ nucleusMode }),
  setCount: (count) => set({ count }),
  setPlaneQuantity: (planeQuantity) =>
    set({ planeQuantity, plane: null, planeStatus: "idle", planeProgress: 0 }),
  setFps: (fps) => set({ fps }),
  // my lab slice: independent of the main (n,l,m,system) physics, so I never put it in INVALIDATED
  setLabConst: (partial) =>
    set((s) => ({
      labConst: { ...s.labConst, ...partial },
      whatif: null,
      whatifStatus: "idle",
    })),
  setLabZ: (labZ) => set({ labZ, whatif: null, whatifStatus: "idle" }),
  // overlay visibility is presentational; the data I draw carries its own provenance
  setGhost: (on) => {
    set({ ghost: on });
    if (on && get().classicalStatus === "idle") void get().loadClassical();
  },
  // my force-law slice: its own axis (preset, params, l), independent of the
  // main (n,l,m,system) physics, so I never put it in INVALIDATED. Changing
  // the preset, a param, or l clears only my force-law data. forceViz is
  // presentational and clears nothing, which is a store invariant. A system
  // change clears it too, since Z and mu change with it.
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
      // In What-If I only use hydrogenic z{N} systems, so I narrow the union defensively.
      const real = await client.getLevels(sys, N_MAX_DIAGRAM, true);
      if (client.isScreenedLevels(real)) throw new Error("what-if expects hydrogenic levels");
      // I draw the altered diagram only when the derived alpha stays in the perturbative range
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
    // Three of my callers want this table on first paint: the Controls panel,
    // the ensureHF gate, and loadHF. Each guards on `systems.length === 0`,
    // which is still zero for all three until the first response lands, so if
    // I did not share the request in flight I would open by fetching the same
    // 16 kB three times. I clear it in `finally` so I retry a failed load
    // rather than remembering it.
    if (inFlightSystems === null) inFlightSystems = client.getSystems();
    let systems: SystemInfo[];
    try {
      systems = (await inFlightSystems).systems;
    } finally {
      inFlightSystems = null;
    }
    // A deep link can name ?system=s&model=gsz, which I can only know is wrong
    // once the table saying sulfur has no GSZ parameters has arrived. This is
    // the moment it arrives, so this is where I correct that link.
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
  // Which of my two representations I draw is a viewing choice over the same
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
    const { n, l, system, systems, compare } = get();
    set({
      radial: await client.getRadial(
        n,
        l,
        system,
        undefined,
        manyElectronParams(get()),
        resolveCompare(systems, system, compare),
      ),
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
    // Z and the electron count live on the system table, so I cannot request a
    // solve before it has loaded. I ask for it rather than guessing them from
    // the key: the key is a label, the table is the authority. I go through
    // loadSystems rather than the client directly, so I share the request any
    // other caller already has in flight.
    if (systems.length === 0) await get().loadSystems();
    const table = get().systems;
    const info = table.find((s) => s.key === system);
    if (info === undefined || info.n_electrons === null) {
      set({
        hfStatus: "error",
        error: `I need an atom with a known electron count to run Hartree-Fock; ${system} has none`,
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
      // The solve reports me no intermediate progress (my server says why), so
      // I have nothing to show between 0 and 1 and I pretend nothing.
      await client.watchJob(job.id, () => {});
      const meta = await client.getJobMeta(job.id);
      if (!client.isHFLevels(meta)) throw new Error("expected hartree-fock job meta");
      set({ hf: meta, hfStatus: "ready" });
    } catch (err) {
      set({ hfStatus: "error", error: err instanceof Error ? err.message : String(err) });
    }
  },
  ensureHF: async () => {
    // I take the table first, because it is what decides the model. A deep
    // link can say ?system=s&model=gsz, and firing that request before the
    // table lands spends a 400 on a question I am one fetch away from
    // answering myself. It is cheap: after the first load `systems` is
    // populated and this is a length check.
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
        // I use the same window as my profile panel. Across the full 90 to
        // 7500 nm range a line is narrower than a pixel, so the only place I
        // can show an absorption line's actual shape is a zoom, exactly as for
        // emission.
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
