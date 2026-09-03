/**
 * How a value gets read off a curve.
 *
 * Every plot here is a sampled curve, and without this the reader is left
 * estimating values against a tick. That estimate would be the one number on
 * screen whose precision nobody had disclosed, so these helpers hand back the
 * sample itself: the returned index always names a real grid point, never an
 * interpolation, and the engine's own value at it gets printed.
 *
 * Interpolating between samples would invent data at a resolution the solve
 * does not have, which is exactly the thing this app never does quietly. If a
 * curve looks coarse under the crosshair, it is coarse.
 */

/**
 * The index of the entry in an ascending array nearest `target`.
 *
 * An empty array returns -1, and values outside the domain clamp to the ends,
 * so a pointer dragged past the axis reads the last real sample instead of
 * nothing.
 */
export function nearestIndex(xs: readonly number[], target: number): number {
  const n = xs.length;
  if (n === 0) return -1;
  if (n === 1) return 0;
  let lo = 0;
  let hi = n - 1;
  // Binary search for the insertion point. The grids run to thousands of
  // points, and this runs on every pointer move.
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
 * Every plot is drawn as `width: 100%; height: auto` with the default
 * `preserveAspectRatio`, so the box scales uniformly and one ratio converts
 * both axes. A zero-width rect (an element measured before layout) returns 0
 * rather than NaN, which would otherwise propagate into the crosshair's x.
 */
export function viewBoxX(clientX: number, rect: DOMRect, viewBoxWidth: number): number {
  if (rect.width <= 0) return 0;
  return ((clientX - rect.left) / rect.width) * viewBoxWidth;
}

/** Is `x` (viewBox units) inside the plotting area, between the margins? */
export function withinPlot(x: number, left: number, right: number): boolean {
  return x >= left && x <= right;
}

/**
 * A number for a hover readout: enough figures to be useful, never more.
 *
 * These plots span radial probabilities near 1e-4 and densities near 30, so a
 * fixed decimal count would print either noise or zeros. Three significant
 * figures is the readout's own precision, and it does not claim to be the
 * solve's.
 */
export function formatHover(value: number): string {
  if (value === 0) return "0";
  const mag = Math.abs(value);
  if (mag >= 1e4 || mag < 1e-3) return value.toExponential(2);
  return String(Number(value.toPrecision(3)));
}
