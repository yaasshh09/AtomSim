import type {
  AbsorptionInfo,
  ClassicalGhost,
  ConstantsReport,
  ForceLawResult,
  HFLevels,
  JobInfo,
  JobMeta,
  LevelsResponse,
  RadialResponse,
  ScreenedLevels,
  CurveOfGrowthInfo,
  SpectrumResponse,
  StateResponse,
  SystemsResponse,
} from "./types";

/** The /api/levels payload is hydrogenic or screened; discriminate on `orbitals`. */
export function isScreenedLevels(
  body: LevelsResponse | ScreenedLevels,
): body is ScreenedLevels {
  return "orbitals" in body;
}

export type Basis = "complex" | "real";
export type PlaneQuantity = "density" | "psi";

/**
 * A number, safe to drop into a query string.
 *
 * JavaScript stringifies anything from 1e21 up in exponential form, so
 * `String(1e21)` is "1e+21" and the raw `+` decodes server-side as a space:
 * the API sees "1e 21" and rejects it. Both of this app's big physical knobs
 * cross that threshold — electron density goes to 1e22 cm^-3 and column
 * density to 1e26 m^-2 — so the failure is at the top of a slider's travel,
 * not in some unreachable corner.
 */
export function num(v: number): string {
  return encodeURIComponent(String(v));
}

/**
 * A string, safe to drop into a query string.
 *
 * The same failure as `num`, arrived at from the other side. `he+` is a real
 * preset key, a bare `+` in a query string decodes to a space, and so the
 * server was handed the system "he " and refused it. That is not one broken
 * request: selecting He+ 422'd its state, levels, spectrum, radial curve and
 * every thumbnail in the strip at once, because all of them spell the system
 * into a URL. The cloud and cross-section kept working, since jobs POST JSON,
 * which is what made it look like a display problem rather than an encoding
 * one.
 *
 * Anything that reaches a query string as text goes through here, not just the
 * keys that happen to need it today: `+` is legal in a preset key and the next
 * one to contain it should not have to rediscover this.
 */
export function key(v: string): string {
  return encodeURIComponent(v);
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

/**
 * The server's own words, when it bothered to write any.
 *
 * FastAPI puts a refusal's reason in `detail`, and several of them are the
 * whole point of the response: the Hartree-Fock endpoint explains that a
 * neutral potassium atom is declined for its 4s shell rather than for its Z,
 * which is exactly the misreading a bare "HTTP 400" would leave in place.
 * Falls back to the status code when the body is not JSON or carries no
 * detail, so a proxy returning HTML still produces something legible.
 */
async function errorFrom(url: string, res: Response): Promise<Error> {
  let detail: string | null = null;
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    detail = null;
  }
  return new Error(detail ?? `${url}: HTTP ${res.status}`);
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await errorFrom(url, res);
  return res.json() as Promise<T>;
}

export function getSystems(): Promise<SystemsResponse> {
  return getJson("/api/systems");
}

export function getState(
  n: number,
  l: number,
  m: number,
  system: string,
  fineStructure: boolean,
): Promise<StateResponse> {
  return getJson(
    `/api/state/${n}/${l}/${m}?system=${key(system)}&fine_structure=${fineStructure}`,
  );
}

export function getRadial(
  n: number,
  l: number,
  system: string,
  points?: number,
): Promise<RadialResponse> {
  const p = points === undefined ? "" : `&points=${points}`;
  return getJson(`/api/radial/${n}/${l}?system=${key(system)}${p}`);
}

export function getLevels(
  system: string,
  nMax: number,
  fineStructure: boolean,
  alpha?: number,
  config?: string | null,
  dirac = false,
  bField = 0,
  eField = 0,
  hyperfine = false,
): Promise<LevelsResponse | ScreenedLevels> {
  const a = alpha === undefined ? "" : `&alpha=${alpha}`;
  const c = config ? `&config=${encodeURIComponent(config)}` : "";
  const d = dirac ? "&dirac=true" : "";
  const b = bField > 0 ? `&b_field=${bField}` : "";
  const e = eField > 0 ? `&e_field=${eField}` : "";
  const h = hyperfine ? "&hyperfine=true" : "";
  return getJson(
    `/api/levels?system=${key(system)}&n_max=${nMax}&fine_structure=${fineStructure}${a}${c}${d}${b}${e}${h}`,
  );
}

export interface ConstMultipliers {
  hbar: number;
  e: number;
  m_e: number;
  eps0: number;
  c: number;
}

export function getConstants(m: ConstMultipliers): Promise<ConstantsReport> {
  return getJson(
    `/api/constants?hbar=${m.hbar}&e=${m.e}&m_e=${m.m_e}&eps0=${m.eps0}&c=${m.c}`,
  );
}

export function getClassical(system: string, n: number): Promise<ClassicalGhost> {
  return getJson(`/api/classical?system=${key(system)}&n=${n}`);
}

export function getForceLaw(
  system: string,
  preset: string,
  params: Record<string, number>,
  l: number,
  nStates = 4,
  expr?: string,
): Promise<ForceLawResult> {
  const q = new URLSearchParams({
    system,
    preset,
    l: String(l),
    n_states: String(nStates),
  });
  for (const [k, v] of Object.entries(params)) q.set(k, String(v));
  if (expr !== undefined) q.set("expr", expr);
  return getJson(`/api/forcelaw?${q.toString()}`);
}

/** LTE conditions. Both or neither: ionization depends on the pair, and the
 *  API rejects half of them rather than inventing the other. */
export interface ThermalParams {
  temperatureK: number;
  electronDensityCm3: number;
}

/** Line-profile synthesis: off unless `on`, and zoomed only if a window is set. */
export interface ProfileParams {
  on: boolean;
  /** Gaussian slit function R = lambda/dlambda; null means no instrument. */
  resolvingPower?: number | null;
  /** Wavelength window [nm]; null uses the across-n range the bars use. */
  window?: [number, number] | null;
}

export function getSpectrum(
  system: string,
  nMax: number,
  fineStructure: boolean,
  config?: string | null,
  intensities = false,
  thermal?: ThermalParams | null,
  profile?: ProfileParams | null,
): Promise<SpectrumResponse> {
  const c = config ? `&config=${encodeURIComponent(config)}` : "";
  const t = thermal
    ? `&temperature_k=${num(thermal.temperatureK)}` +
      `&electron_density_cm3=${num(thermal.electronDensityCm3)}`
    : "";
  let p = "";
  if (profile?.on) {
    p = "&profile=true";
    if (profile.resolvingPower != null) {
      p += `&resolving_power=${num(profile.resolvingPower)}`;
    }
    if (profile.window) {
      p += `&lambda_min=${num(profile.window[0])}&lambda_max=${num(profile.window[1])}`;
    }
  }
  return getJson(
    `/api/spectrum?system=${key(system)}&n_max=${nMax}&fine_structure=${fineStructure}` +
      `&intensities=${intensities}${c}${t}${p}`,
  );
}

export interface CurveOfGrowthParams {
  system: string;
  nMax: number;
  fineStructure: boolean;
  lambdaNm: number;
  thermal: ThermalParams;
  resolvingPower?: number | null;
  config?: string | null;
}

export function getCurveOfGrowth(p: CurveOfGrowthParams): Promise<CurveOfGrowthInfo> {
  const c = p.config ? `&config=${encodeURIComponent(p.config)}` : "";
  const r = p.resolvingPower != null ? `&resolving_power=${num(p.resolvingPower)}` : "";
  return getJson(
    `/api/curve-of-growth?system=${key(p.system)}&n_max=${p.nMax}` +
      `&fine_structure=${p.fineStructure}&lambda_nm=${num(p.lambdaNm)}` +
      `&temperature_k=${num(p.thermal.temperatureK)}` +
      `&electron_density_cm3=${num(p.thermal.electronDensityCm3)}${r}${c}`,
  );
}

export interface AbsorptionParams {
  system: string;
  nMax: number;
  fineStructure: boolean;
  /** For the element, not per line: each line's own lower-level fraction
   *  turns this into that line's absorbers. */
  columnDensityM2: number;
  thermal: ThermalParams;
  resolvingPower?: number | null;
  window?: [number, number] | null;
  config?: string | null;
}

export function getAbsorption(p: AbsorptionParams): Promise<AbsorptionInfo> {
  const c = p.config ? `&config=${encodeURIComponent(p.config)}` : "";
  const r = p.resolvingPower != null ? `&resolving_power=${num(p.resolvingPower)}` : "";
  // No window means the engine sizes its own, which is the safe default: a
  // saturated line is far wider than its FWHM, and a window guessed from the
  // line's width silently returns a short equivalent width.
  const w = p.window
    ? `&lambda_min=${num(p.window[0])}&lambda_max=${num(p.window[1])}`
    : "";
  return getJson(
    `/api/absorption?system=${key(p.system)}&n_max=${p.nMax}` +
      `&fine_structure=${p.fineStructure}` +
      `&column_density_m2=${num(p.columnDensityM2)}` +
      `&temperature_k=${num(p.thermal.temperatureK)}` +
      `&electron_density_cm3=${num(p.thermal.electronDensityCm3)}${r}${w}${c}`,
  );
}

export interface SampleParams {
  n: number;
  l: number;
  m: number;
  count: number;
  seed?: number;
  basis: Basis;
  system: string;
}

export function createSampleJob(params: SampleParams): Promise<JobInfo> {
  return postJson("/api/jobs/sample", { seed: 0, ...params });
}

export interface PlaneParams {
  n: number;
  l: number;
  m: number;
  quantity: PlaneQuantity;
  basis: Basis;
  system: string;
  resolution?: number;
}

export function createPlaneJob(params: PlaneParams): Promise<JobInfo> {
  return postJson("/api/jobs/plane", { resolution: 512, ...params });
}

export interface HFParams {
  z: number;
  /** Defaults to neutral on the server; set it for an ion. */
  n_electrons?: number;
  /** Defaults to the Aufbau ground configuration. */
  config?: string | null;
  /**
   * False solves the Hartree model instead: distinguishable electrons, no
   * exchange. Omit for real physics — the server defaults to true, so a caller
   * cannot ask for the counterfactual by forgetting a field.
   */
  exchange?: boolean;
}

/**
 * Start a Hartree-Fock solve.
 *
 * A job and not a plain GET because the solve is seconds, not milliseconds.
 * Rejections are synchronous and carry their reason - see errorFrom - so a
 * configuration this solver cannot handle fails here rather than eight
 * seconds later inside the worker.
 */
export function createHFJob(params: HFParams): Promise<JobInfo> {
  return postJson("/api/jobs/hf", params);
}

/** The job `meta` payload is a sample, a plane, or a Hartree-Fock solve. */
export function isHFLevels(meta: JobMeta): meta is HFLevels {
  return meta.kind === "hf";
}

export function watchJob(jobId: string, onProgress: (p: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/jobs/${jobId}`);
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data as string) as {
        status: string;
        progress: number;
        error: string | null;
      };
      onProgress(msg.progress);
      if (msg.status === "done") {
        ws.close();
        resolve();
      } else if (msg.status === "error") {
        ws.close();
        reject(new Error(msg.error ?? "job failed"));
      }
    };
    ws.onerror = () => reject(new Error("websocket error"));
  });
}

export function getJobMeta(jobId: string): Promise<JobMeta> {
  return getJson(`/api/jobs/${jobId}/meta`);
}

export async function getChannel(jobId: string, channel?: string): Promise<Float32Array> {
  const url = channel
    ? `/api/jobs/${jobId}/data?channel=${channel}`
    : `/api/jobs/${jobId}/data`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return decodeFloats(await res.arrayBuffer());
}

export function decodeFloats(buffer: ArrayBuffer): Float32Array {
  if (buffer.byteLength % 4 !== 0) {
    throw new Error(`byte length ${buffer.byteLength} is not a multiple of 4 (float32)`);
  }
  return new Float32Array(buffer);
}

export function decodePositions(buffer: ArrayBuffer): Float32Array {
  if (buffer.byteLength % 12 !== 0) {
    throw new Error(
      `positions byte length ${buffer.byteLength} is not a multiple of 12 (xyz float32)`,
    );
  }
  return new Float32Array(buffer);
}

export function thumbnailUrl(
  n: number,
  l: number,
  m: number,
  system: string,
  basis: Basis,
  size: number,
): string {
  return `/api/thumbnail/${n}/${l}/${m}?system=${key(system)}&basis=${basis}&size=${size}`;
}
