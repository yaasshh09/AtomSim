import type { SystemInfo } from "../api/types";

/**
 * The nuclear-mass line: an electron around a +1 nucleus of mass M m_e, from
 * positronium (M = 1) through hydrogen (M = 1836) to past tritium.
 *
 * Nothing physical is computed here. The reduced mass, its provenance and the
 * system's name all come from the server (`/api/system/m<M>`); this file only
 * maps a slider position to a key and back. The real presets on the line are
 * snapped to by key, so standing on hydrogen means the CODATA preset with its
 * measured radius and hyperfine moment, not a mass key that merely lands close.
 */

/** The slider's range, matching MASS_RATIO_MIN/MAX in systems.py. */
export const MASS_MIN = 1;
export const MASS_MAX = 10000;

/** The presets that are an electron orbiting a +1 nucleus, so sit on the line. */
const LINE_PRESETS = ["ps", "h", "d", "t"];

/**
 * Muonium (an electron around an antimuon) is real but has no preset, so its
 * landmark is a mass key. The server marks it COUNTERFACTUAL like every other
 * mass key and names muonium in the assumption, which is the honest reading:
 * the mass is right, the antimuon's own properties are not modelled.
 */
const MUONIUM: Landmark = { key: "m206.768", mass: 206.768, label: "muonium" };

const MASS_KEY = /^m(\d+(?:\.\d+)?)$/;

export interface Landmark {
  key: string;
  mass: number;
  label: string;
}

/** M for a mass key, or null for anything else (including "mu-h"). */
export function parseMassKey(key: string): number | null {
  const found = MASS_KEY.exec(key);
  return found ? Number(found[1]) : null;
}

/**
 * The key for a slider mass: four significant figures in plain decimals, the
 * one spelling the server's pattern accepts, so URLs and caches agree.
 */
export function massKey(mass: number): string {
  const clamped = Math.min(Math.max(mass, MASS_MIN), MASS_MAX);
  return `m${Number(clamped.toPrecision(4))}`;
}

/** Whether the slider applies: a mass key, or a preset that sits on the line. */
export function onMassLine(key: string): boolean {
  return parseMassKey(key) !== null || LINE_PRESETS.includes(key);
}

/** Short names for the presets, so the slider's tick labels stay short. */
const SHORT: Record<string, string> = {
  ps: "positronium", h: "hydrogen", d: "deuterium", t: "tritium",
};

/**
 * The real particles on the line, lightest first. A preset's mass is read from
 * the server's m/M (the nucleus is 1/(m/M) electron masses when the orbiter is
 * an electron), so the client carries no mass constants of its own.
 */
export function landmarks(systems: SystemInfo[]): Landmark[] {
  const presets = systems
    .filter((s) => LINE_PRESETS.includes(s.key) && s.m_over_m_nucleus > 0)
    .map((s) => ({ key: s.key, mass: 1 / s.m_over_m_nucleus, label: SHORT[s.key] }));
  return [...presets, MUONIUM].sort((a, b) => a.mass - b.mass);
}

/** The nuclear mass of the selected system in m_e, or null off the line. */
export function nuclearMass(systems: SystemInfo[], key: string): number | null {
  const parsed = parseMassKey(key);
  if (parsed !== null) return parsed;
  return landmarks(systems).find((l) => l.key === key)?.mass ?? null;
}

/**
 * The key for a slider position. Within `tol` decades of a landmark it snaps
 * to that landmark's key, because a hand on a slider cannot hit 1836.15267 and
 * should not have to in order to reach real hydrogen.
 */
export function snapKey(mass: number, marks: Landmark[], tol = 0.01): string {
  const near = marks.find((l) => Math.abs(Math.log10(mass) - Math.log10(l.mass)) < tol);
  return near ? near.key : massKey(mass);
}

/** Replace any mass-key entry in the table with this one, presets untouched. */
export function withMassSystem(systems: SystemInfo[], info: SystemInfo): SystemInfo[] {
  return [...systems.filter((s) => parseMassKey(s.key) === null), info];
}
