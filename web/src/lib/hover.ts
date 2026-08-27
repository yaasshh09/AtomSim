/**
 * Reading a value off a curve.
 *
 * Every plot in this app draws a sampled curve and then leaves the reader to
 * estimate values against a tick. That estimate is the one number on screen
 * the app has not disclosed the precision of, so these helpers exist to hand
 * back the sample itself: the returned index always names a real grid point,
 * never an interpolation, and the caller prints the engine's own value at it.
 *
 * Interpolating between samples would be inventing data at a resolution the
 * solve does not have, which is exactly the thing this codebase does not do
 * quietly. If a curve looks coarse under the crosshair, it is coarse.
 */

/**
 * Index of the entry in an ascending array nearest `target`.
 *
 * Returns -1 for an empty array; clamps to the ends outside the domain, so a
 * pointer dragged past the axis reads the last real sample instead of nothing.
 */
export function nearestIndex(xs: readonly number[], target: number): number {
  const n = xs.length;
  if (n === 0) return -1;
  if (n === 1) return 0;
  let lo = 0;
  let hi = n - 1;
  // Binary search for the insertion point. The grids here run to thousands of
  // points and this is called on every pointer move.
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= target) lo = mid;
    else hi = mid;
  }
  return target - xs[lo] <= xs[hi] - target ? lo : hi;
}

/**
 * Where a pointer landed, in the SVG's own viewBox coordinates.
 *
 * Every plot here is `width: 100%; height: auto` with the default
 * `preserveAspectRatio`, so the box is scaled uniformly and one ratio converts
 * both axes. A zero-width rect (an element measured before layout) returns 0
 * rather than NaN, which would otherwise propagate into the crosshair's x.
 */
export function viewBoxX(clientX: number, rect: DOMRect, viewBoxWidth: number): number {
  if (rect.width <= 0) return 0;
  return ((clientX - rect.left) / rect.width) * viewBoxWidth;
}

/** Is `x` (viewBox units) inside the plotting area between the margins? */
export function withinPlot(x: number, left: number, right: number): boolean {
  return x >= left && x <= right;
}

/**
 * A number for a hover readout: enough figures to be useful, never more.
 *
 * Values in these plots span radial probabilities near 1e-4 and densities near
 * 30, so a fixed decimal count prints either noise or zeros. Three significant
 * figures is the readout's own precision and is not a claim about the solve's.
 */
export function formatHover(value: number): string {
  if (value === 0) return "0";
  const mag = Math.abs(value);
  if (mag >= 1e4 || mag < 1e-3) return value.toExponential(2);
  return String(Number(value.toPrecision(3)));
}
