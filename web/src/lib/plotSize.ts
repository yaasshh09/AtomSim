import { useCallback, useState } from "react";

/**
 * Plot geometry in real pixels.
 *
 * Every plot in the app draws into a fixed viewBox at `width: 100%`, so the
 * SVG scaled to whatever its container was and took every font size in it
 * along: a 12px axis label rendered at about 5px in a 300px column and at 21px
 * on a 1600px one. The same picture, and only one of those is readable.
 *
 * The fix is to stop treating the viewBox as a design grid. Measure the
 * container, use the measurement as the viewBox width, and one SVG unit is one
 * CSS pixel: a 12px label is 12px wherever it is drawn. Every x coordinate in
 * these views is already written relative to the width (`W - M.right`,
 * `W / 4`), so the change is a measurement replacing a constant rather than a
 * re-layout.
 */

/**
 * The narrowest width a plot is built for, unless it asks for more.
 *
 * One job, and only one: a `ResizeObserver` reports 0 for an element that has
 * not been laid out yet, and a zero width propagates into a d3 scale as a
 * zero-span range and out of it as `NaN` in every coordinate, which draws
 * nothing and says nothing about why. This is the value that stands in until
 * the first real measurement arrives.
 *
 * It used to be 320, back when it was also a layout constant, and at 320 it
 * was the last place a plot could still be wider than the screen. A 320px
 * phone gives its view about 305px of container; the floor rounded that back
 * up to 320 and the SVG's `min-width` pushed 15px of the picture off the right
 * edge, which is the exact failure the measured geometry was written to end.
 * The floor sits below every real container now, so a real measurement always
 * wins, and a container narrower than this is a viewport no browser ships.
 *
 * The wider floors the ladder views name are a separate question, and it is
 * not a width: see `plotFit`.
 */
export const MIN_PLOT_WIDTH = 240;

export function plotWidth(measured: number, min: number = MIN_PLOT_WIDTH): number {
  // Rounded because a fractional viewBox width puts every gridline on a half
  // pixel, and a 1px axis rule drawn across two device pixels reads as grey.
  return Math.max(min, Math.round(measured));
}

/**
 * The width a plot is drawn at, and whether that is under the width its
 * furniture was designed for.
 *
 * A view's floor used to be enforced by widening the viewBox past the
 * container, which pushed the difference into a horizontal scrollbar. On a
 * desktop that is a scrollbar nobody meets, because the centre column is wider
 * than every floor. On a 390px phone the ladder views were 200px wider than
 * the screen, and a plot you have to drag sideways to finish reading is a plot
 * with half of itself hidden.
 *
 * So the floor stops being a width and becomes a question: is there room for
 * the furniture this view carries side by side? Under it the view is told
 * `compact` and lays its panels out one above the other instead, on the width
 * that is actually there. The absolute floor stays, because a zero-width
 * measurement is still NaN in every coordinate.
 */
export function plotFit(
  measured: number,
  min: number = MIN_PLOT_WIDTH,
): { width: number; compact: boolean } {
  const width = plotWidth(measured);
  return { width, compact: width < min };
}

/**
 * A plot's height, following its width between two limits.
 *
 * With a fixed viewBox the rendered height was always `width * ratio`, so a
 * constant height here would have visibly reshaped every desktop plot the
 * moment the width became real. Keeping the ratio holds desktop where it is;
 * the floor is what lets a phone have a plot that is tall and narrow rather
 * than a 90px letterbox, and the ceiling stops a wide monitor turning a
 * two-curve plot into a full page of empty axis.
 */
export function plotHeight(
  width: number,
  ratio: number,
  min: number,
  max: number,
): number {
  return Math.round(Math.min(max, Math.max(min, width * ratio)));
}

/**
 * The measured width of a plot's container, as a callback ref to put on it.
 *
 * A callback ref rather than a `RefObject`, because several of these views
 * return a different wrapper element while their data loads. A ref object
 * would still point at the unmounted one and the observer would go on
 * reporting its last width forever; a callback ref is told about the swap.
 */
export function usePlotWidth(min: number = MIN_PLOT_WIDTH): {
  width: number;
  compact: boolean;
  ref: (el: HTMLElement | null) => void;
} {
  const [measured, setMeasured] = useState(min);
  const ref = useCallback((el: HTMLElement | null) => {
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      // The content box, so the wrapper's own padding is not counted as room
      // to draw in.
      setMeasured(entry.contentRect.width);
    });
    observer.observe(el);
    setMeasured(el.clientWidth - paddingX(el));
    return () => observer.disconnect();
  }, []);
  return { ...plotFit(measured, min), ref };
}

function paddingX(el: HTMLElement): number {
  const s = getComputedStyle(el);
  return parseFloat(s.paddingLeft) + parseFloat(s.paddingRight);
}
