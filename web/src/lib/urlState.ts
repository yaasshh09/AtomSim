/* Deep links = the demo-script hook surface (spec M4): every app state the
 * Phase 2 guided tour needs is addressable by URL alone. Parsing validates
 * hard — a junk parameter is dropped, never propagated into the store. */
import type { Basis, ConstMultipliers, PlaneQuantity } from "../api/client";
import type { AtomModel, ColorMode, SurfaceMode, ViewMode } from "../state/store";
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
  /** Both models' total density on one axis in the Radial view; defaults off. */
  compare: boolean;
  bField: number;
  eField: number;
  hyperfine: boolean;
  /** Line strengths in the Spectrum view; defaults on, so the URL marks it off. */
  intensities: boolean;
  /** LTE weighting in the Spectrum view; defaults off, so the URL marks it on. */
  thermal: boolean;
  temperatureK: number;
  /** log10 of the electron density in cm^-3. Held as the log because the
   *  control is a log slider and round-tripping 1e13 as a decimal string is
   *  needless precision loss. */
  logNe: number;
  /** Voigt line-profile synthesis in the Spectrum view; defaults off. */
  profile: boolean;
  /** log10 of the spectrograph resolving power, or null for no instrument.
   *  Log for the same reason as logNe: the control is a log slider. */
  logResolvingPower: number | null;
  /** Wavelength window [nm] a profile is zoomed to, or null for full range.
   *  Carried so a link can point at one line's shape, which is the whole
   *  reason the zoom exists. */
  profileZoom: [number, number] | null;
  /** Absorption spectrum in the Spectrum view; defaults off. */
  absorption: boolean;
  /** log10 of the column density in m^-2. Carried because it is a physical
   *  knob and not a disclosure toggle: moving it is what walks the gas from
   *  a faithful census to a saturated one, and a link to "hydrogen at 10^22,
   *  Lyman black and Balmer invisible" has to survive being shared. */
  logColumn: number;
  ghost: boolean;
  nucleusMode: NucleusMode;
  planeQuantity: PlaneQuantity;
  /** What the 3-D view draws: the cloud, the enclosing surface, or both. */
  surfaceMode: SurfaceMode;
  /**
   * The fraction of the electron the surface encloses.
   *
   * Carried because it is the question the picture answers, not a display
   * preference: "the 90% contour" and "the 50% contour" of one orbital are
   * different claims about the same atom, and a link to a lesson about the
   * difference has to survive being shared. Free-valued rather than restricted
   * to the offered presets, so a hand-written ?iso=0.6827 works.
   */
  isoFraction: number;
  labConst: ConstMultipliers;
  labZ: number;
  forcePreset: ForcePreset;
  forceParams: Record<string, number>;
  forceL: number;
  /** custom force-law expression V(r); only meaningful when forcePreset==="custom" */
  forceExpr: string;
  /** screened-atom electron configuration; null = Aufbau ground (default) */
  config: string | null;
  /**
   * Which many-electron model an atom's levels come from.
   *
   * "gsz" is the fitted screened central field, "hf" the self-consistent
   * Hartree-Fock solve. Both are APPROXIMATION, but of different things, and
   * they disagree - so this is physics input, not a display toggle, and the
   * store invalidates on it.
   *
   * Defaults to "gsz" so every deep link written before Hartree-Fock existed
   * keeps resolving to the physics it was written against.
   *
   * The key is `model`; note `hf` is separately in use as the *key* for the
   * hyperfine toggle. Different keys, no collision, easy to misread.
   */
  model: AtomModel;
  /**
   * Whether the Hartree-Fock solve keeps its exchange term.
   *
   * False is the Hartree model: electrons that repel but are distinguishable,
   * returned COUNTERFACTUAL. Serialized as `nox=1` rather than `exchange=0`,
   * with the default polarity chosen so that ABSENCE means real physics — a
   * link that predates this toggle, or one a user hand-trims, cannot land
   * anyone in altered physics by omission.
   *
   * `nox` and not `x`: `hf` is already the hyperfine key and this file has one
   * near-collision too many already.
   */
  exchange: boolean;
  /**
   * Whether the Hartree-Fock solve keeps the occupancy cap.
   *
   * False collapses the configuration to 1s^N. Serialized as `nopauli=1`, same
   * polarity and same reason as `nox`: absence means real physics, so no link
   * written before this existed can put a reader inside a counterfactual.
   *
   * `nopauli=1` implies exchange off, and the parser enforces that rather than
   * trusting the query string. A hand-edited `?nopauli=1` with no `nox` is not
   * a state to honour literally — it names a model the API rejects — so it is
   * read as the collapse it obviously means.
   */
  pauli: boolean;
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
  compare: false,
  bField: 0,
  eField: 0,
  hyperfine: false,
  intensities: true,
  thermal: false,
  profile: false,
  logResolvingPower: null,
  profileZoom: null,
  absorption: false,
  logColumn: 20,
  // A stellar photosphere: warm enough that the excited levels are populated
  // at all, dense enough that hydrogen is not yet mostly ionized.
  temperatureK: 10000,
  logNe: 13,
  ghost: false,
  nucleusMode: "marker",
  planeQuantity: "density",
  surfaceMode: "cloud",
  isoFraction: 0.9,
  labConst: { hbar: 1, e: 1, m_e: 1, eps0: 1, c: 1 },
  labZ: 1,
  forcePreset: "powerlaw",
  forceParams: defaultParams("powerlaw"),
  forceL: 0,
  forceExpr: DEFAULT_EXPR,
  config: null,
  model: "gsz",
  exchange: true,
  pauli: true,
};

// a config string is compact subshell tokens: "1s2 2s2 2p6 3p1"
const CONFIG_RE = /^(\d[spdfgh]\d+)( \d[spdfgh]\d+)*$/;

// mirrors the n select in Controls (N_CHOICES max)
const N_MAX_UI = 6;

const VIEWS: ViewMode[] = ["cloud", "plane", "radial", "levels", "spectrum", "whatif", "forcelaw"];
const COLORS: ColorMode[] = ["solid", "density", "phase"];
const BASES: Basis[] = ["complex", "real"];
/** Which many-electron model an atom's levels come from. See UrlState.model. */
const MODELS: AtomModel[] = ["gsz", "hf"];
const NUCLEUS: NucleusMode[] = ["hidden", "true-scale", "marker"];
const PLANES: PlaneQuantity[] = ["density", "psi"];
const SURFACES: SurfaceMode[] = ["cloud", "surface", "both"];
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

  // Dropped rather than thrown on, like every other parameter here: this
  // module's contract is that junk never reaches the store, and a link with a
  // typo should still open the app rather than fail to render.
  const model = pickEnum(q.get("model"), MODELS);
  if (model) out.model = model;

  if (q.get("nox") === "1") out.exchange = false;
  // The cap and antisymmetry are one rule, so `nopauli=1` carries `nox` with
  // it whether or not the link says so. Honouring a bare `?nopauli=1`
  // literally would build a request the API answers 422 to, which is a worse
  // reading of the user's intent than the obvious one.
  if (q.get("nopauli") === "1") {
    out.pauli = false;
    out.exchange = false;
  }

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
  const surf = pickEnum(q.get("surf"), SURFACES);
  if (surf) out.surfaceMode = surf;
  const isoFraction = pickFloat(q.get("iso"));
  // Open interval, hard: 0 and 1 are not contours, and the server would answer
  // 422 for either. A junk value drops back to the default rather than
  // travelling into a request.
  if (isoFraction !== undefined && isoFraction > 0 && isoFraction < 1) {
    out.isoFraction = isoFraction;
  }

  const fs = q.get("fs");
  if (fs === "1" || fs === "true") out.fineStructure = true;
  else if (fs === "0" || fs === "false") out.fineStructure = false;

  if (q.get("dirac") === "1") out.dirac = true;

  if (q.get("compare") === "1") out.compare = true;

  const b = Number(q.get("b"));
  if (Number.isFinite(b) && b > 0) out.bField = b;

  // param is "ef" (electric field): "e" is already the charge multiplier (CONST_PARAMS)
  const ef = Number(q.get("ef"));
  if (Number.isFinite(ef) && ef > 0) out.eField = ef;

  if (q.get("hf") === "1") out.hyperfine = true;

  // Defaults on, so only the off state is carried: "int=0".
  if (q.get("int") === "0") out.intensities = false;

  if (q.get("lte") === "1") out.thermal = true;
  const tk = pickFloat(q.get("tk"));
  if (tk !== undefined && tk >= 1e2 && tk <= 1e6) out.temperatureK = tk;
  const ne = pickFloat(q.get("ne"));
  if (ne !== undefined && ne >= 4 && ne <= 22) out.logNe = ne;

  if (q.get("prof") === "1") out.profile = true;
  const rp = pickFloat(q.get("rp"));
  if (rp !== undefined && rp >= 2 && rp <= 7) out.logResolvingPower = rp;
  const zoom = q.get("zoom");
  if (zoom) {
    const [lo, hi] = zoom.split(",").map(Number);
    // Both ends, both real light, and in order: a malformed window would ask
    // the engine for a spectrum that cannot exist.
    if (Number.isFinite(lo) && Number.isFinite(hi) && lo > 0 && hi > lo) {
      out.profileZoom = [lo, hi];
    }
  }

  if (q.get("abs") === "1") out.absorption = true;
  const col = pickFloat(q.get("col"));
  // Same bounds as the slider. Outside them the engine still answers, but the
  // spectrum is either a flat line or entirely black and the view has nothing
  // to show.
  if (col !== undefined && col >= 14 && col <= 26) out.logColumn = col;

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
  if (state.compare) q.set("compare", "1");
  if (state.bField > 0 && state.fineStructure) q.set("b", String(state.bField));
  if (state.eField > 0) q.set("ef", String(state.eField));
  if (state.hyperfine) q.set("hf", "1");
  if (!state.intensities) q.set("int", "0");
  if (state.thermal) {
    q.set("lte", "1");
    if (state.temperatureK !== URL_DEFAULTS.temperatureK) {
      q.set("tk", String(state.temperatureK));
    }
    if (state.logNe !== URL_DEFAULTS.logNe) q.set("ne", String(state.logNe));
  }
  if (state.profile) {
    q.set("prof", "1");
    if (state.logResolvingPower !== null) {
      q.set("rp", String(state.logResolvingPower));
    }
    if (state.profileZoom) {
      q.set("zoom", `${state.profileZoom[0]},${state.profileZoom[1]}`);
    }
  }
  if (state.absorption) {
    q.set("abs", "1");
    if (state.logColumn !== URL_DEFAULTS.logColumn) {
      q.set("col", String(state.logColumn));
    }
  }
  if (state.ghost !== URL_DEFAULTS.ghost) q.set("ghost", "1");
  if (state.nucleusMode !== URL_DEFAULTS.nucleusMode) q.set("nucleus", state.nucleusMode);
  if (state.planeQuantity !== URL_DEFAULTS.planeQuantity) q.set("plane", state.planeQuantity);
  if (state.surfaceMode !== URL_DEFAULTS.surfaceMode) q.set("surf", state.surfaceMode);
  // Only when a surface is being drawn: a fraction in a link that shows a point
  // cloud names a contour nobody is looking at.
  if (state.surfaceMode !== "cloud" && state.isoFraction !== URL_DEFAULTS.isoFraction) {
    q.set("iso", String(state.isoFraction));
  }
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
  // Written unconditionally when non-default, including for hydrogenic
  // systems where it currently selects nothing. Dropping it there would make
  // the round trip depend on which system is loaded, and a link is supposed to
  // carry the state it was written from.
  if (state.model !== URL_DEFAULTS.model) q.set("model", state.model);
  if (!state.exchange) q.set("nox", "1");
  // Written alongside `nox` rather than instead of it. The pair round-trips
  // through a parser that would infer the missing one anyway, but a link a
  // user reads or edits should say both things it means.
  if (!state.pauli) q.set("nopauli", "1");
  // note: '+' stays percent-encoded (%2B) — a literal '+' in a query string
  // reads back as a space, which would break the he+ round-trip
  const s = q.toString();
  return s ? `?${s}` : "";
}
