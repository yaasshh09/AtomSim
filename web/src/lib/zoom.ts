/**
 * Zooming, done on the domain rather than on the picture.
 *
 * The obvious way to zoom an SVG is a transform on the drawn group, and it is
 * the wrong way here. A transform scales the strokes, the tick text and the
 * marker labels with the data, so a zoomed plot ends up with 4 px curves and
 * axis numbers that no longer line up with anything. Worse, the axis would go
 * on saying what it said before, which is the one thing this app never does:
 * the numbers written on a plot always describe what is drawn on it.
 *
 * So a zoom here narrows the domain and everything downstream is recomputed
 * from it. Ticks re-space, labels re-spread, the curve is re-sampled onto the
 * new scale. Nothing is stretched and no number goes stale.
 *
 * A window never leaves the full domain, so scrolling cannot wander off into
 * empty space, and it cannot go below MIN_SPAN either: past about four decades
 * of magnification the float grid under the curve runs out before the picture
 * does, and a plot that keeps zooming into an interval containing two samples
 * is drawing a straight line and calling it data.
 */

export type Domain = readonly [number, number];

/** Deepest magnification: the window can shrink to this fraction of the full range. */
const MIN_SPAN = 1e-4;

/** A log axis needs a positive domain; anything else is drawn linearly. */
function usableLog(full: Domain, log: boolean): boolean {
  return log && full[0] > 0 && full[1] > 0;
}

const fwd = (v: number, log: boolean) => (log ? Math.log10(v) : v);
const inv = (t: number, log: boolean) => (log ? 10 ** t : t);

/** [lo, hi] with lo <= hi, whichever order it arrived in. */
function ordered(d: Domain): [number, number] {
  return d[0] <= d[1] ? [d[0], d[1]] : [d[1], d[0]];
}

/**
 * A window pushed back inside its full domain, keeping its span.
 *
 * Panning is expressed as "move the window and then fix it up", so this is
 * where a pan against the edge stops rather than overshooting into blank axis.
 */
export function clampView(view: Domain, full: Domain, log = false): [number, number] {
  const lg = usableLog(full, log);
  const [fLo, fHi] = ordered(full).map((v) => fwd(v, lg)) as [number, number];
  const fSpan = fHi - fLo;
  if (!(fSpan > 0)) return ordered(full);
  const [vLo0, vHi0] = ordered(view).map((v) => fwd(v, lg)) as [number, number];
  const span = Math.min(Math.max(vHi0 - vLo0, fSpan * MIN_SPAN), fSpan);
  let lo = vLo0;
  if (lo + span > fHi) lo = fHi - span;
  if (lo < fLo) lo = fLo;
  return [inv(lo, lg), inv(lo + span, lg)];
}

/**
 * The window after a zoom of `factor` about `anchor`.
 *
 * `anchor` is where the pointer is, as a fraction from the near end of the
 * window to the far end, so the value under the cursor stays under the cursor.
 * A factor below 1 zooms in. Anchoring at the pointer rather than at the
 * centre is what makes wheel-zoom feel like moving a magnifier over the plot
 * instead of driving a pair of sliders.
 */
export function zoomView(
  view: Domain,
  full: Domain,
  factor: number,
  anchor = 0.5,
  log = false,
): [number, number] {
  const lg = usableLog(full, log);
  const [vLo, vHi] = ordered(view).map((v) => fwd(v, lg)) as [number, number];
  const span = vHi - vLo;
  if (!(span > 0) || !(factor > 0)) return ordered(view);
  const a = Math.min(Math.max(anchor, 0), 1);
  const at = vLo + a * span;
  const next = span * factor;
  return clampView([inv(at - a * next, lg), inv(at + (1 - a) * next, lg)], full, log);
}

/**
 * The window slid by `fraction` of its own span. Positive moves it up-domain.
 *
 * Panning by a fraction of the *window* rather than of the full range is what
 * keeps a drag feeling the same at every magnification: one plot width of
 * pointer travel always moves the view by one plot width of data.
 */
export function panView(
  view: Domain,
  full: Domain,
  fraction: number,
  log = false,
): [number, number] {
  const lg = usableLog(full, log);
  const [vLo, vHi] = ordered(view).map((v) => fwd(v, lg)) as [number, number];
  const shift = (vHi - vLo) * fraction;
  return clampView([inv(vLo + shift, lg), inv(vHi + shift, lg)], full, log);
}

/** How many times magnified this window is. 1 is the whole range. */
export function zoomFactor(view: Domain, full: Domain, log = false): number {
  const lg = usableLog(full, log);
  const [vLo, vHi] = ordered(view).map((v) => fwd(v, lg)) as [number, number];
  const [fLo, fHi] = ordered(full).map((v) => fwd(v, lg)) as [number, number];
  const span = vHi - vLo;
  if (!(span > 0) || !(fHi - fLo > 0)) return 1;
  return (fHi - fLo) / span;
}

/** Is this window narrower than the full range, by more than rounding? */
export function isZoomed(view: Domain, full: Domain, log = false): boolean {
  return zoomFactor(view, full, log) > 1.001;
}

/**
 * The values inside a window, as index bounds one past the last.
 *
 * Curves are still drawn from the full array and clipped, but the label
 * spreaders on the ladders need to know which rows are actually on screen:
 * spreading a label for a rung three screens above the frame pushes every
 * visible label out of the way to make room for one nobody can see.
 */
export function withinView<T>(rows: readonly T[], value: (row: T) => number, view: Domain): T[] {
  const [lo, hi] = ordered(view);
  return rows.filter((r) => {
    const v = value(r);
    return v >= lo && v <= hi;
  });
}
