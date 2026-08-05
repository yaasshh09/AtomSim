/** The rectangle a `getBoundingClientRect` gives, narrowed to what is used. */
export interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * The ring's rectangle, padded out from the control it surrounds.
 *
 * Null for a zero-sized box. An anchor that is unmounted, display:none, or
 * inside a collapsed <details> measures 0 by 0, and ringing it would draw a
 * dot in the top-left corner pointing at nothing.
 */
export function spotlightBox(box: Box, pad: number) {
  if (box.width <= 0 || box.height <= 0) return null;
  return {
    x: box.left - pad,
    y: box.top - pad,
    w: box.width + 2 * pad,
    h: box.height + 2 * pad,
  };
}
