import { PHASE, type Lut } from "./luts";

/** Clamped LUT lookup: t in [0, 1] -> [r, g, b] bytes. */
export function lutColor(lut: Lut, t: number): readonly [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  return lut[Math.min(lut.length - 1, Math.floor(clamped * lut.length))];
}

/**
 * VISUAL LIBERTY, disclosed everywhere it is used: density brightness is
 * gamma-compressed, t = (rho / rho_max)^DENSITY_GAMMA, so faint outer lobes
 * stay visible. Mirrors GAMMA in src/atomsim/server/thumbnails.py.
 */
export const DENSITY_GAMMA = 0.5;

export function densityT(value: number, vmax: number): number {
  if (vmax <= 0) return 0;
  const clamped = Math.min(Math.max(value, 0), vmax);
  return (clamped / vmax) ** DENSITY_GAMMA;
}

/** Maps a signed value to [0, 1] with zero at 0.5, for the diverging colormaps. */
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

/**
 * The cyclic colour for arg(psi), any real phase (wrapped to [0, 2 pi)).
 *
 * The PHASE table rather than an HSL hue wheel: for red-green colour-blind
 * readers the wheel collapsed phases a third of a turn apart into one colour.
 * See phase_rows() in scripts/gen_luts.py and tests/test_lut_cvd.py.
 */
export function phaseColor(phase: number): readonly [number, number, number] {
  const turn = phase / (2 * Math.PI);
  const t = turn - Math.floor(turn);
  return PHASE[Math.min(Math.floor(t * PHASE.length), PHASE.length - 1)];
}
