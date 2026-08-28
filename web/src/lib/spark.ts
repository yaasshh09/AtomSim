/**
 * I downsample a curve to a fixed number of bar heights, normalized to its
 * peak.
 *
 * I use this for my left rail's P(r) card, which is a shape summary in about
 * 130 pixels, not a plot: it has no axes and carries no scale, and my Radial
 * view is where I draw the same curve to be measured.
 *
 * I make each bar the **maximum** over its bin, not the mean. On a radial
 * probability the peak is the feature the card exists to show, where the
 * electron most likely is, and if I averaged a sharp 1s peak against the tail
 * beside it I would move the maximum of the drawn shape away from the maximum
 * of the function. Taking the max keeps the peak where the physics put it, and
 * costs me a curve drawn slightly fatter than it is. That is the trade a
 * summary can make; a plot could not.
 *
 * I return an empty array when I have nothing honest to draw, meaning no
 * samples, or a curve whose peak is not a positive finite number. My callers
 * render the card only for a non-empty result, so for an absent curve I show
 * no card rather than a flat one, which would read as "P(r) = 0 everywhere".
 */
/**
 * The index one past the last sample I think is worth drawing, for a curve on
 * a grid that runs much further out than the curve does.
 *
 * My radial grid is sized for the solver, not for a thumbnail: hydrogen's 3d
 * runs to 180 bohr while P(r) is back under a thousandth of its peak by about
 * 40. Over the whole grid my card would be four bars and a flat line, which
 * looks like a broken chart rather than a wavefunction.
 *
 * Windowing to where the curve actually lives is a presentational choice, and
 * I keep it honest the only way that works: I label the axis with the r this
 * returns rather than with the end of the grid. A window is fine; a window
 * drawn under the full grid's label is not.
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
      // I stop one past the last sample above the floor, and never so short
      // that I have nothing left to draw.
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
    // I take at least one sample per bar, so a curve shorter than the bar
    // count still fills the card instead of leaving holes in it.
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
