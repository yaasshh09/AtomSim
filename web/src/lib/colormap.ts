import type { Lut } from "./luts";

/** My clamped LUT lookup: t in [0, 1] -> [r, g, b] bytes. */
export function lutColor(lut: Lut, t: number): readonly [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  return lut[Math.min(lut.length - 1, Math.floor(clamped * lut.length))];
}

/**
 * VISUAL LIBERTY (which I disclose wherever I use it): I gamma-compress
 * density brightness, t = (rho / rho_max)^DENSITY_GAMMA, so faint outer lobes
 * stay visible. I mirror GAMMA in src/atomsim/server/thumbnails.py.
 */
export const DENSITY_GAMMA = 0.5;

export function densityT(value: number, vmax: number): number {
  if (vmax <= 0) return 0;
  const clamped = Math.min(Math.max(value, 0), vmax);
  return (clamped / vmax) ** DENSITY_GAMMA;
}

/** I map a signed value to [0, 1] with zero at 0.5, for my diverging colormaps. */
export function signedT(value: number, vabs: number): number {
  if (vabs <= 0) return 0.5;
  return (Math.min(Math.max(value / vabs, -1), 1) + 1) / 2;
}

export function maxOf(values: Float32Array): number {
  let m = 0;
  for (let i = 0; i < values.length; i++) m = Math.max(m, values[i]);
  return m;
}

export function maxAbs(values: Float32Array): number {
  let m = 0;
  for (let i = 0; i < values.length; i++) m = Math.max(m, Math.abs(values[i]));
  return m;
}

/** The cyclic colour I give arg(psi) in [-pi, pi]: the full HSL hue wheel. */
export function phaseColor(phase: number): readonly [number, number, number] {
  const h = (phase + Math.PI) / (2 * Math.PI);
  return hslToRgb(h - Math.floor(h), 1, 0.55);
}

export function hslToRgb(
  h: number,
  s: number,
  l: number,
): readonly [number, number, number] {
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const f = (t: number): number => {
    let x = t;
    if (x < 0) x += 1;
    if (x > 1) x -= 1;
    if (x < 1 / 6) return p + (q - p) * 6 * x;
    if (x < 1 / 2) return q;
    if (x < 2 / 3) return p + (q - p) * (2 / 3 - x) * 6;
    return p;
  };
  return [
    Math.round(f(h + 1 / 3) * 255),
    Math.round(f(h) * 255),
    Math.round(f(h - 1 / 3) * 255),
  ];
}
