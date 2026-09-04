import { useEffect, useState } from "react";

/**
 * The narrowest viewport the three-column instrument fits in.
 *
 * Not a round number chosen for looks. The two rails are 300px each because
 * the left one carries an energy in hartree beside its tier badge and wraps
 * below 268 (see `.app-grid` in index.css). That leaves 300px of centre column
 * at exactly 900.
 *
 * It used to be a refusal: below it the app rendered a notice asking for a
 * wider screen, on the argument that a squeezed plot is a dishonest plot. The
 * argument was right and the threshold was not, because a 300px centre column
 * squeezes a plot exactly as hard as a phone does, and that is the width the
 * notice stopped applying at. Measured plot geometry fixed the squeeze on both
 * sides of the line (lib/plotSize.ts), and what is left here is a layout
 * choice: which of the two shells to build.
 */
export const MIN_WIDTH = 900;

/** Whether to build the stacked shell rather than the three-column one. */
export function isNarrow(width: number): boolean {
  return width < MIN_WIDTH;
}

/** Tracks the viewport, which the shell picks on and the sheet snaps to. */
export function useViewport(): { width: number; height: number } {
  const [size, setSize] = useState(() => ({
    width: window.innerWidth,
    height: window.innerHeight,
  }));
  useEffect(() => {
    const onResize = () =>
      setSize({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return size;
}
