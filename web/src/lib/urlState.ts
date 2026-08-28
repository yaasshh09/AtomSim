/* My deep links are the demo-script hook surface (spec M4): every state the
 * Phase 2 guided tour needs, I make addressable by URL alone. I validate hard
 * when I parse, and I drop a junk parameter rather than letting it into my
 * store. */
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
  /** Whether I put both models' total density on one axis in the Radial view; I default this off. */
  compare: boolean;
  bField: number;
  eField: number;
  hyperfine: boolean;
  /** Whether I draw line strengths in the Spectrum view; I default this on, so the URL marks it off. */
  intensities: boolean;
  /** Whether I apply LTE weighting in the Spectrum view; I default this off, so the URL marks it on. */
  thermal: boolean;
  temperatureK: number;
  /** log10 of the electron density in cm^-3. I hold the log because the
   *  control is a log slider and round-tripping 1e13 as a decimal string
   *  would lose precision for nothing. */
  logNe: number;
  /** Whether I synthesize Voigt line profiles in the Spectrum view; I default this off. */
  profile: boolean;
  /** log10 of the spectrograph resolving power, or null when there is no
   *  instrument. I hold the log for the same reason as logNe: the control is
   *  a log slider. */
  logResolvingPower: number | null;
  /** The wavelength window [nm] I have zoomed a profile to, or null for the
   *  full range. I carry it so a link can point at one line's shape, which is
   *  the whole reason I offer the zoom. */
  profileZoom: [number, number] | null;
  /** Whether I draw the absorption spectrum in the Spectrum view; I default this off. */
  absorption: boolean;
  /** log10 of the column density in m^-2. I carry it because it is a physical
   *  knob and not a disclosure toggle: moving it is what walks the gas from a
   *  faithful census to a saturated one, and a link to "hydrogen at 10^22,
   *  Lyman black and Balmer invisible" has to survive being shared. */
  logColumn: number;
  ghost: boolean;
  nucleusMode: NucleusMode;
  planeQuantity: PlaneQuantity;
  /** What I draw in the 3-D view: the cloud, the enclosing surface, or both. */
  surfaceMode: SurfaceMode;
  /**
   * The fraction of the electron my surface encloses.
   *
   * I carry it because it is the question the picture answers, not a display
   * preference: "the 90% contour" and "the 50% contour" of one orbital are
   * different claims about the same atom, and a link to a lesson about the
   * difference has to survive being shared. I leave it free-valued rather than
   * restricting it to the presets I offer, so a hand-written ?iso=0.6827 works.
   */
  isoFraction: number;
  labConst: ConstMultipliers;
  labZ: number;
  forcePreset: ForcePreset;
  forceParams: Record<string, number>;
  forceL: number;
  /** The custom force-law expression V(r) I solve; only meaningful when forcePreset==="custom" */
  forceExpr: string;
  /** The screened-atom electron configuration I use; null means the Aufbau ground state (my default) */
  config: string | null;
  /**
   * Which many-electron model I take an atom's levels from.
   *
   * "gsz" is the fitted screened central field, "hf" the self-consistent
   * Hartree-Fock solve. Both are APPROXIMATION, but of different things, and
   * they disagree, so this is physics input rather than a display toggle and
   * my store invalidates on it.
   *
   * I default to "gsz" so every deep link written before I could do
   * Hartree-Fock keeps resolving to the physics it was written against.
   *
   * The key is `model`; note that I use `hf` separately as the *key* for the
   * hyperfine toggle. Different keys, no collision, easy to misread.
   */
  model: AtomModel;
  /**
   * Whether I keep the exchange term in my Hartree-Fock solve.
   *
   * False is the Hartree model: electrons that repel but are distinguishable,
   * which I return COUNTERFACTUAL. I serialize it as `nox=1` rather than
   * `exchange=0`, and I chose that polarity so ABSENCE means real physics: a
   * link that predates this toggle, or one you hand-trim, cannot land anyone
   * in altered physics by omission.
   *
   * `nox` and not `x`: I already use `hf` for hyperfine, and this file has one
   * near-collision too many already.
   */
  exchange: boolean;
  /**
   * Whether I keep the occupancy cap in my Hartree-Fock solve.
   *
   * False collapses the configuration to 1s^N. I serialize it as `nopauli=1`,
   * same polarity and same reason as `nox`: absence means real physics, so no
   * link written before this existed can put you inside a counterfactual.
   *
   * `nopauli=1` implies exchange off, and I enforce that when I parse rather
   * than trusting the query string. A hand-edited `?nopauli=1` with no `nox`
   * is not a state I should honour literally, since it names a model my API
   * rejects, so I read it as the collapse it obviously means.
   */
  pauli: boolean;
  /**
   * The tour you are taking, and how far into it you are.
   *
   * I carry it so the demo script is a URL: "?tour=hydrogen-honestly&step=4"
   * is a thing you can send someone. I write `step` only alongside a tour,
   * because a bare step number describes nothing.
   */
  tour: string | null;
  step: number;
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
  // I start in a stellar photosphere: warm enough that the excited levels are
  // populated at all, dense enough that hydrogen is not yet mostly ionized.
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
  tour: null,
  step: 0,
};

// I read a config string as compact subshell tokens: "1s2 2s2 2p6 3p1"
const CONFIG_RE = /^(\d[spdfgh]\d+)( \d[spdfgh]\d+)*$/;

// I mirror the n select in Controls here (N_CHOICES max)
const N_MAX_UI = 6;

const VIEWS: ViewMode[] = ["cloud", "plane", "radial", "levels", "spectrum", "whatif", "forcelaw"];
const COLORS: ColorMode[] = ["solid", "density", "phase"];
const BASES: Basis[] = ["complex", "real"];
/** Which many-electron model I take an atom's levels from. See UrlState.model. */
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

// the short URL names I give the five constant multipliers (m_e -> "me")
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

/**
 * The addressable slice of one of my store snapshots.
 *
 * I keep it here rather than in main.tsx because two callers now need the same
 * list: the subscriber that keeps my URL describing the live state, and my
 * tour, which snapshots your state on entry and restores it on exit. A second
 * hand-maintained copy of thirty-seven field names is how those two drift.
 */
export function currentUrlState(s: Omit<UrlState, "tour" | "step">): UrlState {
  return {
    n: s.n,
    l: s.l,
    m: s.m,
    system: s.system,
    basis: s.basis,
    view: s.view,
    colorMode: s.colorMode,
    fineStructure: s.fineStructure,
    dirac: s.dirac,
    compare: s.compare,
    bField: s.bField,
    eField: s.eField,
    hyperfine: s.hyperfine,
    intensities: s.intensities,
    thermal: s.thermal,
    temperatureK: s.temperatureK,
    logNe: s.logNe,
    profile: s.profile,
    logResolvingPower: s.logResolvingPower,
    profileZoom: s.profileZoom,
    absorption: s.absorption,
    logColumn: s.logColumn,
    ghost: s.ghost,
    nucleusMode: s.nucleusMode,
    planeQuantity: s.planeQuantity,
    surfaceMode: s.surfaceMode,
    isoFraction: s.isoFraction,
    labConst: s.labConst,
    labZ: s.labZ,
    forcePreset: s.forcePreset,
    forceParams: s.forceParams,
    forceL: s.forceL,
    forceExpr: s.forceExpr,
    config: s.config,
    model: s.model,
    exchange: s.exchange,
    pauli: s.pauli,
    // I do not read these off the snapshot. My store holds them as
    // tourId/stepIndex, and my two callers want opposite things: the URL
    // subscriber overrides both with the live tour, while the tour's own entry
    // snapshot wants your state without a tour in it, since that is what I
    // restore when you exit.
    tour: null,
    step: 0,
  };
}

/** The validated partial state I read out of a query string; I drop invalid params. */
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

  // I drop it rather than throwing, like every other parameter here: my
  // contract is that junk never reaches the store, and a link with a typo
  // should still open me rather than fail to render.
  const model = pickEnum(q.get("model"), MODELS);
  if (model) out.model = model;

  if (q.get("nox") === "1") out.exchange = false;
  // The cap and antisymmetry are one rule, so I carry `nox` along with
  // `nopauli=1` whether or not the link says so. If I honoured a bare
  // `?nopauli=1` literally I would build a request my API answers 422 to,
  // which is a worse reading of your intent than the obvious one.
  if (q.get("nopauli") === "1") {
    out.pauli = false;
    out.exchange = false;
  }

  const basis = pickEnum(q.get("basis"), BASES);
  if (basis) out.basis = basis;
  const view = pickEnum(q.get("view"), VIEWS);
  if (view) out.view = view;
  let color = pickEnum(q.get("color"), COLORS);
  // I mirror my store guard here: phase needs the complex basis
  if (color === "phase" && (basis ?? URL_DEFAULTS.basis) === "real") color = "density";
  if (color) out.colorMode = color;
  const nucleus = pickEnum(q.get("nucleus"), NUCLEUS);
  if (nucleus) out.nucleusMode = nucleus;
  const plane = pickEnum(q.get("plane"), PLANES);
  if (plane) out.planeQuantity = plane;
  const surf = pickEnum(q.get("surf"), SURFACES);
  if (surf) out.surfaceMode = surf;
  const isoFraction = pickFloat(q.get("iso"));
  // I hold the interval open, hard: 0 and 1 are not contours, and my server
  // would answer 422 for either. I drop a junk value back to my default rather
  // than letting it travel into a request.
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

  // I name the param "ef" (electric field): I already use "e" for the charge multiplier (CONST_PARAMS)
  const ef = Number(q.get("ef"));
  if (Number.isFinite(ef) && ef > 0) out.eField = ef;

  if (q.get("hf") === "1") out.hyperfine = true;

  // I default this on, so I only carry the off state: "int=0".
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
    // I want both ends, both real light, and in order: with a malformed window
    // I would be asking my engine for a spectrum that cannot exist.
    if (Number.isFinite(lo) && Number.isFinite(hi) && lo > 0 && hi > lo) {
      out.profileZoom = [lo, hi];
    }
  }

  if (q.get("abs") === "1") out.absorption = true;
  const col = pickFloat(q.get("col"));
  // I use the same bounds as my slider. Outside them my engine still answers,
  // but the spectrum is either a flat line or entirely black and I have
  // nothing to show you.
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

  // The force-law preset and its params are one axis, independent of my main
  // physics. The preset selects which param names are live; I validate each
  // one and clamp it to its own spec range, and I fall back to the preset
  // defaults for any that are missing. I only write these fields when the URL
  // actually mentions the force law (a preset or one of its params), so an
  // empty query stays an empty override set.
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

  // I read the custom force-law expression only for the custom preset, and
  // only when my lightweight browser check passes (my server's AST parser is
  // the authority).
  if (out.forcePreset === "custom") {
    const rawExpr = q.get("expr");
    if (rawExpr !== null && validateExprClient(rawExpr) === null) out.forceExpr = rawExpr;
  }

  const config = q.get("config");
  if (config !== null && CONFIG_RE.test(config)) out.config = config;

  const tour = q.get("tour");
  if (tour) {
    out.tour = tour;
    const step = pickInt(q.get("step"));
    // I clamp this to the tour's real length in clampStep at apply time; here
    // it only has to be a non-negative integer, since I do not know how long
    // any tour is from inside this module.
    out.step = step !== undefined && step >= 0 ? step : 0;
  }

  return out;
}

/** The canonical query string I write for a state; I omit default values ("" if everything is default). */
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
  // I write this only when I am drawing a surface: a fraction in a link that
  // shows a point cloud names a contour nobody is looking at.
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
  // I write this whenever it is non-default, including for hydrogenic systems
  // where it currently selects nothing. If I dropped it there my round trip
  // would depend on which system is loaded, and a link is supposed to carry
  // the state it was written from.
  if (state.model !== URL_DEFAULTS.model) q.set("model", state.model);
  if (!state.exchange) q.set("nox", "1");
  // I write this alongside `nox` rather than instead of it. The pair
  // round-trips through a parser that would infer the missing one anyway, but
  // a link you read or edit should say both things it means.
  if (!state.pauli) q.set("nopauli", "1");
  if (state.tour) {
    q.set("tour", state.tour);
    if (state.step !== URL_DEFAULTS.step) q.set("step", String(state.step));
  }
  // note: I leave '+' percent-encoded (%2B), because a literal '+' in a query
  // string reads back as a space, which would break my he+ round-trip
  const s = q.toString();
  return s ? `?${s}` : "";
}
