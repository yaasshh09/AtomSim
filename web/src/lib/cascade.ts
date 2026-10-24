import type { Cascade } from "../api/types";

const L_LETTERS = "spdfghik";

export function levelName(n: number, l: number): string {
  return `${n}${L_LETTERS[l] ?? `(l=${l})`}`;
}

const TIME_UNITS: [number, string][] = [
  [1, "s"],
  [1e-3, "ms"],
  [1e-6, "µs"],
  [1e-9, "ns"],
  [1e-12, "ps"],
  [1e-15, "fs"],
];

/** A duration in the largest unit that keeps it at or above 1: 5.27e-9 -> "5.27 ns". */
export function formatSeconds(s: number, digits = 3): string {
  if (!Number.isFinite(s)) return "∞";
  if (s === 0) return "0 s";
  const [scale, unit] = TIME_UNITS.find(([f]) => Math.abs(s) >= f) ?? TIME_UNITS[TIME_UNITS.length - 1];
  return `${Number((s / scale).toPrecision(digits))} ${unit}`;
}

export interface GrotrianNode {
  n: number;
  l: number;
  /** 0..1 across the columns (by l) and 0 (top, highest n) .. 1 (bottom, n = 1). */
  x: number;
  y: number;
  visit: number;
  terminal: boolean;
}

export interface GrotrianArrow {
  from: GrotrianNode;
  to: GrotrianNode;
  photons: number;
  wavelengthNm: number;
}

/**
 * A Grotrian diagram: one column per l, one row per n, arrows for every decay.
 *
 * Rows are evenly spaced by n, not by energy (see CASCADE_GROTRIAN_LIBERTY):
 * on the true -1/n^2 scale the top rows would merge.
 */
export function grotrian(c: Cascade): { nodes: GrotrianNode[]; arrows: GrotrianArrow[] } {
  const nTop = c.start[0];
  const lMax = Math.max(1, ...c.levels.map((lv) => lv.l));
  const nodes = c.levels.map((lv) => ({
    n: lv.n,
    l: lv.l,
    x: lv.l / lMax,
    y: nTop === 1 ? 1 : (nTop - lv.n) / (nTop - 1),
    visit: lv.visit_probability.value,
    terminal: lv.terminal,
  }));
  const at = new Map(nodes.map((nd) => [`${nd.n},${nd.l}`, nd]));
  const arrows = c.branches.map((b) => ({
    from: at.get(`${b.upper[0]},${b.upper[1]}`)!,
    to: at.get(`${b.lower[0]},${b.lower[1]}`)!,
    photons: b.photons.value,
    wavelengthNm: b.wavelength.value,
  }));
  return { nodes, arrows };
}

/** Where the atom ends up: the visit probability of each terminal level. */
export function endpoints(c: Cascade): { name: string; probability: number }[] {
  return c.levels
    .filter((lv) => lv.terminal)
    .map((lv) => ({ name: levelName(lv.n, lv.l), probability: lv.visit_probability.value }))
    .sort((a, b) => b.probability - a.probability);
}

/** The brightest lines of the cascade, by expected photons per excited atom. */
export function topBranches(c: Cascade, count: number) {
  return [...c.branches].sort((a, b) => b.photons.value - a.photons.value).slice(0, count);
}
