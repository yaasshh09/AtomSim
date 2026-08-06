/**
 * Downsample a curve to a fixed number of bar heights, normalized to its peak.
 *
 * Used by the left rail's P(r) card, which is a shape summary in about 130
 * pixels, not a plot: it has no axes and carries no scale, and the Radial view
 * is where the same curve is drawn to be measured.
 *
 * Each bar is the **maximum** over its bin, not the mean. On a radial
 * probability the peak is the feature the card exists to show, where the
 * electron most likely is, and averaging a sharp 1s peak against the tail
 * beside it moves the maximum of the drawn shape away from the maximum of the
 * function. Taking the max keeps the peak where the physics put it, at the
 * cost of drawing the curve slightly fatter than it is. That is the trade a
 * summary can make; a plot could not.
 *
 * Returns an empty array when there is nothing honest to draw, no samples, or
 * a curve whose peak is not a positive finite number. Callers render the card
 * only for a non-empty result, so an absent curve shows no card rather than a
 * flat one, which would read as "P(r) = 0 everywhere".
 */
/**
 * Index one past the last sample worth drawing, for a curve on a grid that
 * runs much further out than the curve does.
 *
 * The radial grid is sized for the solver, not for a thumbnail: hydrogen's 3d
 * runs to 180 bohr while P(r) is back under a thousandth of its peak by about
 * 40. Drawn over the whole grid the card is four bars and a flat line, which
 * looks like a broken chart rather than a wavefunction.
 *
 * Windowing to where the curve actually lives is a presentational choice, and
 * the caller is required to keep it honest the only way that works: by
 * labelling the axis with the r this returns rather than with the end of the
 * grid. A window is fine; a window drawn under the full grid's label is not.
 */
export function informativeEnd(values: number[], floor = 1e-3): number {
  if (values.length === 0) return 0;
  let peak = 0;
  for (const v of values) {
    if (Number.isFinite(v) && v > peak) peak = v;
  }
  if (peak <= 0) return values.length;
  const cutoff = peak * floor;
  for (let i = values.length - 1; i >= 0; i--) {
    const v = values[i];
    if (Number.isFinite(v) && v > cutoff) {
      // One past the last sample above the floor, and never so short that
      // there is nothing left to draw.
      return Math.max(i + 1, 2);
    }
  }
  return values.length;
}

export function sparkBars(values: number[], bars: number): number[] {
  if (bars <= 0 || values.length === 0) return [];
  let peak = 0;
  for (const v of values) {
    if (Number.isFinite(v) && v > peak) peak = v;
  }
  if (peak <= 0) return [];
  const out: number[] = [];
  for (let i = 0; i < bars; i++) {
    const start = Math.floor((i * values.length) / bars);
    // At least one sample per bar, so a curve shorter than the bar count still
    // fills the card instead of leaving holes in it.
    const end = Math.max(start + 1, Math.floor(((i + 1) * values.length) / bars));
    let hi = 0;
    for (let j = start; j < end && j < values.length; j++) {
      const v = values[j];
      if (Number.isFinite(v) && v > hi) hi = v;
    }
    out.push(hi / peak);
  }
  return out;
}
