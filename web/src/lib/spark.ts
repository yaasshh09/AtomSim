/**
 * Downsample a curve to a fixed number of bar heights, normalized to its peak.
 *
 * This feeds the left rail's P(r) card, which is a shape summary in about 130
 * pixels, not a plot: it has no axes and carries no scale. The Radial view is
 * where the same curve gets drawn to be measured.
 *
 * Each bar is the **maximum** over its bin, not the mean. On a radial
 * probability the peak is the feature the card exists to show, where the
 * electron most likely is, and averaging a sharp 1s peak against the tail
 * beside it would move the maximum of the drawn shape away from the maximum of
 * the function. Taking the max keeps the peak where the physics put it, at the
 * cost of a curve drawn slightly fatter than it is. That is a trade a summary
 * can make and a plot could not.
 *
 * An empty array comes back when there is nothing honest to draw: no samples,
 * or a curve whose peak is not a positive finite number. Callers render the
 * card only for a non-empty result, so an absent curve gets no card rather
 * than a flat one, which would read as "P(r) = 0 everywhere".
 */
/**
 * The index one past the last sample worth drawing, for a curve on a grid that
 * runs much further out than the curve does.
 *
 * The radial grid is sized for the solver, not for a thumbnail: hydrogen's 3d
 * runs to 180 bohr while P(r) is back under a thousandth of its peak by about
 * 40. Over the whole grid the card would be four bars and a flat line, which
 * looks like a broken chart rather than a wavefunction.
 *
 * Windowing to where the curve actually lives is a presentational choice, kept
 * honest the only way that works: the axis is labelled with the r this returns
 * rather than with the end of the grid. A window is fine. A window drawn under
 * the full grid's label is not.
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
      // Stop one past the last sample above the floor, and never so short that
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
