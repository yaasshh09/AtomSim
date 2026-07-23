/* Deep links = the demo-script hook surface (spec M4): every app state the
 * Phase 2 guided tour needs is addressable by URL alone. Parsing validates
 * hard — a junk parameter is dropped, never propagated into the store. */
import type { Basis, ConstMultipliers, PlaneQuantity } from "../api/client";
import type { ColorMode, ViewMode } from "../state/store";
import type { NucleusMode } from "./nucleus";
import {
  DEFAULT_EXPR,
  PRESET_PARAMS,
  clampParam,
  defaultParams,
  validateExprClient,
  type ForcePreset,
} from "./forceLaw";
import { clampState } from "./quantum";
import { CONST_MAX, CONST_MIN, CONSTANT_KEYS, type ConstantKey } from "./whatif";

export interface UrlState {
  n: number;
  l: number;
  m: number;
  system: string;
  basis: Basis;
  view: ViewMode;
  colorMode: ColorMode;
  fineStructure: boolean;
  dirac: boolean;
  ghost: boolean;
  nucleusMode: NucleusMode;
  planeQuantity: PlaneQuantity;
  labConst: ConstMultipliers;
  labZ: number;
  forcePreset: ForcePreset;
  forceParams: Record<string, number>;
  forceL: number;
  /** custom force-law expression V(r); only meaningful when forcePreset==="custom" */
  forceExpr: string;
  /** screened-atom electron configuration; null = Aufbau ground (default) */
  config: string | null;
}

export const URL_DEFAULTS: UrlState = {
  n: 1,
  l: 0,
  m: 0,
  system: "h",
  basis: "complex",
  view: "cloud",
  colorMode: "solid",
  fineStructure: false,
  dirac: false,
  ghost: false,
  nucleusMode: "marker",
  planeQuantity: "density",
  labConst: { hbar: 1, e: 1, m_e: 1, eps0: 1, c: 1 },
  labZ: 1,
  forcePreset: "powerlaw",
  forceParams: defaultParams("powerlaw"),
  forceL: 0,
  forceExpr: DEFAULT_EXPR,
  config: null,
};

// a config string is compact subshell tokens: "1s2 2s2 2p6 3p1"
const CONFIG_RE = /^(\d[spdfgh]\d+)( \d[spdfgh]\d+)*$/;

// mirrors the n select in Controls (N_CHOICES max)
const N_MAX_UI = 6;

const VIEWS: ViewMode[] = ["cloud", "plane", "radial", "levels", "spectrum", "whatif", "forcelaw"];
const COLORS: ColorMode[] = ["solid", "density", "phase"];
const BASES: Basis[] = ["complex", "real"];
const NUCLEUS: NucleusMode[] = ["hidden", "true-scale", "marker"];
const PLANES: PlaneQuantity[] = ["density", "psi"];
const FORCE_PRESETS: ForcePreset[] = [
  "powerlaw",
  "yukawa",
  "harmonic",
  "finitewell",
  "coulombcore",
  "custom",
];
const SYSTEM_KEY = /^[a-z0-9+-]{1,16}$/;

// short URL names for the five constant multipliers (m_e -> "me")
const CONST_PARAMS: Record<ConstantKey, string> = {
  hbar: "hbar",
  e: "e",
  m_e: "me",
  eps0: "eps0",
  c: "c",
};

function pickEnum<T extends string>(raw: string | null, allowed: T[]): T | undefined {
  return allowed.includes(raw as T) ? (raw as T) : undefined;
}

function pickInt(raw: string | null): number | undefined {
  if (raw === null || !/^-?\d+$/.test(raw)) return undefined;
  return Number(raw);
}

function pickFloat(raw: string | null): number | undefined {
  if (raw === null || !/^-?\d*\.?\d+(e-?\d+)?$/i.test(raw)) return undefined;
  const v = Number(raw);
  return Number.isFinite(v) ? v : undefined;
}

/** Validated partial state from a query string; invalid params are dropped. */
export function parseAppUrl(search: string): Partial<UrlState> {
  const q = new URLSearchParams(search);
  const out: Partial<UrlState> = {};

  const n = pickInt(q.get("n"));
  const l = pickInt(q.get("l"));
  const m = pickInt(q.get("m"));
  if (n !== undefined || l !== undefined || m !== undefined) {
    const clamped = clampState(
      Math.min(n ?? URL_DEFAULTS.n, N_MAX_UI),
      l ?? URL_DEFAULTS.l,
      m ?? URL_DEFAULTS.m,
    );
    out.n = clamped.n;
    out.l = clamped.l;
    out.m = clamped.m;
  }

  const system = q.get("system");
  if (system !== null && SYSTEM_KEY.test(system)) out.system = system;

  const basis = pickEnum(q.get("basis"), BASES);
  if (basis) out.basis = basis;
  const view = pickEnum(q.get("view"), VIEWS);
  if (view) out.view = view;
  let color = pickEnum(q.get("color"), COLORS);
  // mirror of the store guard: phase needs the complex basis
  if (color === "phase" && (basis ?? URL_DEFAULTS.basis) === "real") color = "density";
  if (color) out.colorMode = color;
  const nucleus = pickEnum(q.get("nucleus"), NUCLEUS);
  if (nucleus) out.nucleusMode = nucleus;
  const plane = pickEnum(q.get("plane"), PLANES);
  if (plane) out.planeQuantity = plane;

  const fs = q.get("fs");
  if (fs === "1" || fs === "true") out.fineStructure = true;
  else if (fs === "0" || fs === "false") out.fineStructure = false;

  if (q.get("dirac") === "1") out.dirac = true;

  const ghost = q.get("ghost");
  if (ghost === "1" || ghost === "true") out.ghost = true;
  else if (ghost === "0" || ghost === "false") out.ghost = false;

  const lc: Partial<ConstMultipliers> = {};
  for (const k of CONSTANT_KEYS) {
    const v = pickFloat(q.get(CONST_PARAMS[k]));
    if (v !== undefined && v > 0) lc[k] = Math.min(Math.max(v, CONST_MIN), CONST_MAX);
  }
  if (Object.keys(lc).length > 0) out.labConst = { ...URL_DEFAULTS.labConst, ...lc };

  const z = pickInt(q.get("z"));
  if (z !== undefined) out.labZ = Math.min(Math.max(z, 1), 10);

  // Force-law preset + params: one axis independent of the main physics. The
  // preset selects which param names are live; each is validated and clamped to
  // its own spec range, missing params fall back to the preset defaults. Fields
  // are only written when the URL actually mentions the force law (a preset or
  // one of its params), so an empty query stays an empty override set.
  const presetRaw = q.get("preset");
  const preset = pickEnum(presetRaw, FORCE_PRESETS) ?? "powerlaw";
  const params = defaultParams(preset);
  let sawForceParam = false;
  for (const spec of PRESET_PARAMS[preset]) {
    const v = pickFloat(q.get(spec.name));
    if (v !== undefined) {
      params[spec.name] = clampParam(spec, v);
      sawForceParam = true;
    }
  }
  if ((presetRaw !== null && pickEnum(presetRaw, FORCE_PRESETS) !== undefined) || sawForceParam) {
    out.forcePreset = preset;
    out.forceParams = params;
  }

  const fl = pickInt(q.get("fl"));
  if (fl !== undefined && fl >= 0) out.forceL = fl;

  // Custom force-law expression: only read for the custom preset, and only when
  // the lightweight client check passes (the server AST parser is the authority).
  if (out.forcePreset === "custom") {
    const rawExpr = q.get("expr");
    if (rawExpr !== null && validateExprClient(rawExpr) === null) out.forceExpr = rawExpr;
  }

  const config = q.get("config");
  if (config !== null && CONFIG_RE.test(config)) out.config = config;

  return out;
}

/** Canonical query string for a state; default values are omitted ("" if all default). */
export function serializeAppUrl(state: UrlState): string {
  const q = new URLSearchParams();
  if (state.n !== URL_DEFAULTS.n) q.set("n", String(state.n));
  if (state.l !== URL_DEFAULTS.l) q.set("l", String(state.l));
  if (state.m !== URL_DEFAULTS.m) q.set("m", String(state.m));
  if (state.system !== URL_DEFAULTS.system) q.set("system", state.system);
  if (state.basis !== URL_DEFAULTS.basis) q.set("basis", state.basis);
  if (state.view !== URL_DEFAULTS.view) q.set("view", state.view);
  if (state.colorMode !== URL_DEFAULTS.colorMode) q.set("color", state.colorMode);
  if (state.fineStructure !== URL_DEFAULTS.fineStructure) q.set("fs", "1");
  if (state.dirac && state.fineStructure) q.set("dirac", "1");
  if (state.ghost !== URL_DEFAULTS.ghost) q.set("ghost", "1");
  if (state.nucleusMode !== URL_DEFAULTS.nucleusMode) q.set("nucleus", state.nucleusMode);
  if (state.planeQuantity !== URL_DEFAULTS.planeQuantity) q.set("plane", state.planeQuantity);
  for (const k of CONSTANT_KEYS) {
    if (Math.abs(state.labConst[k] - URL_DEFAULTS.labConst[k]) > 1e-9) {
      q.set(CONST_PARAMS[k], String(state.labConst[k]));
    }
  }
  if (state.labZ !== URL_DEFAULTS.labZ) q.set("z", String(state.labZ));
  if (state.forcePreset !== URL_DEFAULTS.forcePreset) q.set("preset", state.forcePreset);
  for (const spec of PRESET_PARAMS[state.forcePreset]) {
    const v = state.forceParams[spec.name];
    if (v !== undefined && Math.abs(v - spec.default) > 1e-9) q.set(spec.name, String(v));
  }
  if (state.forceL !== URL_DEFAULTS.forceL) q.set("fl", String(state.forceL));
  if (state.forcePreset === "custom" && state.forceExpr !== URL_DEFAULTS.forceExpr) {
    q.set("expr", state.forceExpr);
  }
  if (state.config) q.set("config", state.config);
  // note: '+' stays percent-encoded (%2B) — a literal '+' in a query string
  // reads back as a space, which would break the he+ round-trip
  const s = q.toString();
  return s ? `?${s}` : "";
}
