/**
 * Where the bottom sheet comes to rest.
 *
 * Kept out of the component on purpose. Snapping is the only part of a drag
 * that involves a decision, and it is a decision about three numbers: how far
 * the sheet was moved, how fast it was moving when it was let go, and how tall
 * the screen is. None of that needs a DOM, so none of it is tested through
 * one; the component is left with pointer plumbing that has nothing to get
 * wrong.
 */

export const SNAPS = ["collapsed", "half", "full"] as const;
export type Snap = (typeof SNAPS)[number];

/** The peek bar: the system name and the live n, l, m, and nothing else.
 *  Mirrored as `--peek-h` in index.css, where the stage's bottom padding and
 *  the tour card's dock both measure against it. */
export const PEEK_HEIGHT = 56;

/**
 * Under this, the snap fractions shrink.
 *
 * Every landscape phone is here (844x390 is the common one), and a sheet at
 * 90% of 390px is a sheet that has eaten the instrument. The half snap exists
 * precisely so the control under a thumb and the plot it changes are on screen
 * together, and it stops meaning that the moment it covers the plot.
 */
export const SHORT_VIEWPORT = 500;

export function snapHeight(snap: Snap, viewportHeight: number): number {
  if (snap === "collapsed") return PEEK_HEIGHT;
  const short = viewportHeight < SHORT_VIEWPORT;
  const fraction = snap === "half" ? (short ? 0.35 : 0.45) : short ? 0.6 : 0.9;
  return Math.round(viewportHeight * fraction);
}

/** Past this, in pixels per millisecond, a drag is a flick and goes one step. */
const FLING = 0.5;

/**
 * The snap a release lands on.
 *
 * `offset` is how far the drag moved, in pixels, negative upwards, so the
 * height the sheet is being held at is its resting height minus the offset. A
 * slow drag lands on whichever snap that height is nearest; a flick goes one
 * step in the direction it was thrown, however short it was, because a flick
 * is a statement of intent rather than a measurement.
 */
export function nextSnap(
  current: Snap,
  offset: number,
  velocity: number,
  viewportHeight: number,
): Snap {
  const i = SNAPS.indexOf(current);
  if (velocity <= -FLING) return SNAPS[Math.min(i + 1, SNAPS.length - 1)];
  if (velocity >= FLING) return SNAPS[Math.max(i - 1, 0)];
  const held = snapHeight(current, viewportHeight) - offset;
  const distance = (s: Snap) => Math.abs(snapHeight(s, viewportHeight) - held);
  return SNAPS.reduce((best, s) => (distance(s) < distance(best) ? s : best));
}
