import type {
  ClassicalGhost,
  ConstantsReport,
  ForceLawResult,
  JobInfo,
  JobMeta,
  LevelsResponse,
  RadialResponse,
  ScreenedLevels,
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

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
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
    `/api/state/${n}/${l}/${m}?system=${system}&fine_structure=${fineStructure}`,
  );
}

export function getRadial(
  n: number,
  l: number,
  system: string,
  points?: number,
): Promise<RadialResponse> {
  const p = points === undefined ? "" : `&points=${points}`;
  return getJson(`/api/radial/${n}/${l}?system=${system}${p}`);
}

export function getLevels(
  system: string,
  nMax: number,
  fineStructure: boolean,
  alpha?: number,
  config?: string | null,
  dirac = false,
): Promise<LevelsResponse | ScreenedLevels> {
  const a = alpha === undefined ? "" : `&alpha=${alpha}`;
  const c = config ? `&config=${encodeURIComponent(config)}` : "";
  const d = dirac ? "&dirac=true" : "";
  return getJson(
    `/api/levels?system=${system}&n_max=${nMax}&fine_structure=${fineStructure}${a}${c}${d}`,
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
  return getJson(`/api/classical?system=${encodeURIComponent(system)}&n=${n}`);
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

export function getSpectrum(
  system: string,
  nMax: number,
  fineStructure: boolean,
  config?: string | null,
): Promise<SpectrumResponse> {
  const c = config ? `&config=${encodeURIComponent(config)}` : "";
  return getJson(
    `/api/spectrum?system=${system}&n_max=${nMax}&fine_structure=${fineStructure}${c}`,
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
  return `/api/thumbnail/${n}/${l}/${m}?system=${system}&basis=${basis}&size=${size}`;
}
