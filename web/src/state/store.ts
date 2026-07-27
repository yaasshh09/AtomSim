import { create } from "zustand";
import * as client from "../api/client";
import type { Basis, ConstMultipliers, PlaneQuantity } from "../api/client";
import type {
  ClassicalGhost,
  ConstantsReport,
  ForceLawResult,
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
  radial: RadialResponse | null;
  /** Hydrogenic keys return LevelsResponse; screened atoms return ScreenedLevels. */
  levels: LevelsResponse | ScreenedLevels | null;
  spectrum: SpectrumResponse | null;
  /** null = Aufbau ground config (server fills it); else an explicit config string. */
  config: string | null;
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
  ...INVALIDATED,
  // classical ghost data depends on (n, system) but not (l, m, basis), so it is
  // reset explicitly here rather than living in INVALIDATED (basis changes keep it).
  setQuantumNumbers: (n, l, m) =>
    set({ ...clampState(n, l, m), ...INVALIDATED, classicalGhost: null, classicalStatus: "idle" }),
  setSystem: (system) =>
    set({
      system,
      ...INVALIDATED,
      // selecting a system resets to the Aufbau ground config (server fills it)
      config: null,
      classicalGhost: null,
      classicalStatus: "idle",
      forceLaw: null,
      forceStatus: "idle",
    }),
  // config is its own physics input (screened atoms only): it clears the derived
  // level/spectrum/state payloads but keeps the selected system.
  setConfig: (config) => set({ config, levels: null, spectrum: null, stateInfo: null }),
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
    set({ systems: (await client.getSystems()).systems });
  },
  loadStateInfo: async () => {
    const { n, l, m, system, fineStructure } = get();
    set({ stateInfo: await client.getState(n, l, m, system, fineStructure) });
  },
  sample: async () => {
    const { n, l, m, count, basis, system } = get();
    set({ status: "sampling", progress: 0, error: null });
    try {
      const job = await client.createSampleJob({ n, l, m, count, basis, system });
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
  loadPlane: async () => {
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
    const { n, l, system } = get();
    set({ radial: await client.getRadial(n, l, system) });
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
