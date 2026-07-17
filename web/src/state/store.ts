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
  SpectrumResponse,
  StateResponse,
  SystemInfo,
} from "../api/types";
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
  levels: LevelsResponse | null;
  spectrum: SpectrumResponse | null;
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
  forceP: number;
  forceL: number;
  forceLaw: ForceLawResult | null;
  forceStatus: SampleStatus;
  setForceP: (p: number) => void;
  setForceL: (l: number) => void;
  loadForceLaw: () => Promise<void>;
  setGhost: (on: boolean) => void;
  loadClassical: () => Promise<void>;
  setLabConst: (partial: Partial<ConstMultipliers>) => void;
  setLabZ: (labZ: number) => void;
  loadWhatIf: () => Promise<void>;
  setQuantumNumbers: (n: number, l: number, m: number) => void;
  setSystem: (system: string) => void;
  setBasis: (basis: Basis) => void;
  setView: (view: ViewMode) => void;
  setColorMode: (colorMode: ColorMode) => void;
  setFineStructure: (fineStructure: boolean) => void;
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
  forceP: 1.0,
  forceL: 0,
  forceLaw: null,
  forceStatus: "idle",
  ...INVALIDATED,
  // classical ghost data depends on (n, system) but not (l, m, basis), so it is
  // reset explicitly here rather than living in INVALIDATED (basis changes keep it).
  setQuantumNumbers: (n, l, m) =>
    set({ ...clampState(n, l, m), ...INVALIDATED, classicalGhost: null, classicalStatus: "idle" }),
  setSystem: (system) =>
    set({
      system,
      ...INVALIDATED,
      classicalGhost: null,
      classicalStatus: "idle",
      forceLaw: null,
      forceStatus: "idle",
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
  // force-law slice: its own axis (forceP, forceL) — independent of the main
  // (n,l,m,system) physics, so it is never in INVALIDATED; changing p or l
  // clears only the force-law data. System changes clear it too (Z/mu change).
  setForceP: (p) =>
    set({
      forceP: Math.min(Math.max(p, 0.5), 1.5),
      forceLaw: null,
      forceStatus: "idle",
    }),
  setForceL: (l) =>
    set({ forceL: Math.max(0, Math.round(l)), forceLaw: null, forceStatus: "idle" }),
  loadForceLaw: async () => {
    const { forceP, forceL, system } = get();
    set({ forceStatus: "sampling", error: null });
    try {
      const forceLaw = await client.getForceLaw(system, forceP, forceL);
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
      const real = await client.getLevels(sys, N_MAX_DIAGRAM, true);
      // altered diagram only when the derived alpha stays in the perturbative range
      const altered =
        report.altered && isAlphaValid(alpha)
          ? await client.getLevels(sys, N_MAX_DIAGRAM, true, alpha)
          : null;
      set({ whatif: { report, real, altered }, whatifStatus: "ready" });
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
    const { system, fineStructure } = get();
    set({ levels: await client.getLevels(system, N_MAX_DIAGRAM, fineStructure) });
  },
  loadSpectrum: async () => {
    const { system, fineStructure } = get();
    set({ spectrum: await client.getSpectrum(system, N_MAX_DIAGRAM, fineStructure) });
  },
}));
