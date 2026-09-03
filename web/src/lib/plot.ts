import type { ScaleLinear } from "d3-scale";

/** The SVG path traced through (xs[i], ys[i]) under the given scales. */
export function linePath(
  xs: readonly number[],
  ys: readonly number[],
  x: ScaleLinear<number, number>,
  y: ScaleLinear<number, number>,
): string {
  return xs
    .map((v, i) => `${i === 0 ? "M" : "L"}${x(v).toFixed(2)},${y(ys[i]).toFixed(2)}`)
    .join("");
}

/**
 * The grid indices where a sampled curve changes sign.
 *
 * This reads the drawn curve and claims nothing beyond it: the positions are
 * linear interpolations between two adjacent samples that straddle zero, which
 * is where the plotted polyline crosses the axis, and therefore exactly what a
 * marker drawn there points at. It is not the engine's node count. That number
 * comes off /api/state with its own provenance and appears in the left rail. If
 * the two ever disagree, the engine is right, and the difference means the grid
 * is too coarse to show it.
 *
 * Exact zeros are skipped rather than counted. A sample that lands on zero
 * without the curve changing sign is a touch, not a crossing, and P(r) = r²R²
 * is made of those.
 */
export function zeroCrossings(
  grid: readonly number[],
  values: readonly number[],
): number[] {
  const out: number[] = [];
  for (let i = 1; i < values.length; i++) {
    const a = values[i - 1];
    const b = values[i];
    if (a === 0 || b === 0) continue;
    if (a < 0 === b < 0) continue;
    // Linear interpolation to the crossing: t is the fraction of the interval
    // at which the straight segment between two samples reaches zero.
    const t = a / (a - b);
    out.push(grid[i - 1] + t * (grid[i] - grid[i - 1]));
  }
  return out;
}

/**
 * The index one past the last sample worth drawing, for a curve that may go
 * negative.
 *
 * `informativeEnd` in lib/spark.ts does this for the left rail's P(r) card, but
 * it measures against a peak found with `v > peak`, so on R(r) the whole
 * negative lobe is invisible to it. This is the same idea on |v|.
 *
 * The window is a presentational choice, kept honest exactly one way: the axis
 * is labelled with the r this returns, never with the end of the grid, and the
 * caption says a tail was trimmed and where. The radial grid is sized for the
 * solver rather than for a picture (hydrogen's 3s runs to 180 bohr while R(r)
 * is flat past about 30), so over the whole grid the plot would be one spike
 * and 150 bohr of nothing.
 */
export function informativeEndSigned(
  values: readonly number[],
  floor = 1e-3,
): number {
  if (values.length === 0) return 0;
  let peak = 0;
  for (const v of values) {
    const a = Math.abs(v);
    if (Number.isFinite(a) && a > peak) peak = a;
  }
  if (peak <= 0) return values.length;
  const cutoff = peak * floor;
  for (let i = values.length - 1; i >= 0; i--) {
    const a = Math.abs(values[i]);
    if (Number.isFinite(a) && a > cutoff) return Math.max(i + 1, 2);
  }
  return values.length;
}
