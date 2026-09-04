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
 * A floor rather than a target, and it does two jobs. A `ResizeObserver`
 * reports 0 for an element that has not been laid out yet, and a zero width
 * propagates into a d3 scale as a zero-span range and out of it as `NaN` in
 * every coordinate, which draws nothing and says nothing about why.
 *
 * The second job is the reason `min` is a parameter. A curve against an axis
 * holds together at 320: the only furniture is a margin wide enough for a tick
 * label. An energy ladder does not, because it carries a column of rung labels
 * on the left, a column of energies on the right and, on hydrogen, a fine
 * structure fan beside both, and no fraction of 320 fits three columns of
 * text. Those views name a wider floor. Under it the SVG scales down as it
 * always did, which is the bounded fallback: the picture gets smaller rather
 * than getting its labels sliced off.
 */
export const MIN_PLOT_WIDTH = 320;

export function plotWidth(measured: number, min: number = MIN_PLOT_WIDTH): number {
  // Rounded because a fractional viewBox width puts every gridline on a half
  // pixel, and a 1px axis rule drawn across two device pixels reads as grey.
  return Math.max(min, Math.round(measured));
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
  ref: (el: HTMLElement | null) => void;
} {
  const [width, setWidth] = useState(min);
  const ref = useCallback(
    (el: HTMLElement | null) => {
      if (!el) return;
      const observer = new ResizeObserver(([entry]) => {
        // The content box, so the wrapper's own padding is not counted as room
        // to draw in.
        setWidth(plotWidth(entry.contentRect.width, min));
      });
      observer.observe(el);
      setWidth(plotWidth(el.clientWidth - paddingX(el), min));
      return () => observer.disconnect();
    },
    [min],
  );
  return { width, ref };
}

function paddingX(el: HTMLElement): number {
  const s = getComputedStyle(el);
  return parseFloat(s.paddingLeft) + parseFloat(s.paddingRight);
}
