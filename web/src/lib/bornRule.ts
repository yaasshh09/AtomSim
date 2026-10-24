/**
 * The Born rule, one measurement at a time.
 *
 * Each revealed point is one draw from |psi|^2 out of a sample job, and the
 * engine's KS tests hold those draws to the exact distribution (sampling.py).
 * What this file adds is only bookkeeping: radii of the first k draws, their
 * histogram, the exact radial CDF to compare them against, and the
 * Kolmogorov-Smirnov distance between the two.
 */

/** r of the first `k` points of an interleaved (x, y, z) Float32Array. */
export function radii(positions: Float32Array, k: number): Float64Array {
  const count = Math.min(k, Math.floor(positions.length / 3));
  const out = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const x = positions[3 * i];
    const y = positions[3 * i + 1];
    const z = positions[3 * i + 2];
    out[i] = Math.hypot(x, y, z);
  }
  return out;
}

/** The exact CDF of r on P(r)'s own grid, by the trapezoid rule, scaled to end at 1. */
export function exactCdf(grid: number[], p: number[]): Float64Array {
  const out = new Float64Array(grid.length);
  for (let i = 1; i < grid.length; i++) {
    out[i] = out[i - 1] + 0.5 * (p[i] + p[i - 1]) * (grid[i] - grid[i - 1]);
  }
  const total = out[out.length - 1] || 1;
  for (let i = 0; i < out.length; i++) out[i] /= total;
  return out;
}

/** F(r) by linear interpolation in the tabulated CDF; 0 below the grid, 1 above. */
function cdfAt(r: number, grid: number[], cdf: Float64Array): number {
  if (r <= grid[0]) return 0;
  if (r >= grid[grid.length - 1]) return 1;
  let lo = 0;
  let hi = grid.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (grid[mid] <= r) lo = mid;
    else hi = mid;
  }
  const t = (r - grid[lo]) / (grid[hi] - grid[lo]);
  return cdf[lo] + t * (cdf[hi] - cdf[lo]);
}

/** sup |F_N(r) - F(r)|, the Kolmogorov-Smirnov distance of the draws so far. */
export function ksDistance(rs: Float64Array, grid: number[], cdf: Float64Array): number {
  const sorted = Float64Array.from(rs).sort();
  const n = sorted.length;
  let d = 0;
  for (let i = 0; i < n; i++) {
    const f = cdfAt(sorted[i], grid, cdf);
    d = Math.max(d, (i + 1) / n - f, f - i / n);
  }
  return d;
}

/**
 * The typical KS distance for N honest draws: 0.83/sqrt(N), the median of the
 * limiting Kolmogorov distribution (its mean is sqrt(pi/2) ln 2 = 0.87). Draws
 * from the wrong distribution stop shrinking; these keep going.
 */
export function typicalKs(n: number): number {
  return n > 0 ? 0.8276 / Math.sqrt(n) : 1;
}

/** Counts per bin over [0, rMax], as a probability density in 1/bohr. */
export function histogram(rs: Float64Array, rMax: number, bins: number): number[] {
  const counts = new Array<number>(bins).fill(0);
  const w = rMax / bins;
  for (const r of rs) {
    const b = Math.floor(r / w);
    if (b >= 0 && b < bins) counts[b] += 1;
  }
  const n = rs.length || 1;
  return counts.map((c) => c / (n * w));
}
