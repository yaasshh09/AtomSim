import { useEffect, useState } from "react";

/** The narrowest viewport the three-column instrument fits in.
 *
 * Not a round number picked for looks. The two rails are 300px each because
 * the left one carries an energy in hartree beside its tier badge and wraps
 * below 268 (see `.app-grid` in index.css). That leaves 300px of centre column
 * at exactly 900, which is already the floor for a plot whose axis labels stay
 * readable. Below it the layout does not get cramped, it gets wrong. */
export const MIN_WIDTH = 900;

export function needsWiderScreen(width: number): boolean {
  return width < MIN_WIDTH;
}

/** Track viewport width, so the notice can say how far off this screen is. */
export function useViewportWidth(): number {
  const [width, setWidth] = useState(() => window.innerWidth);
  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return width;
}

/** Shown instead of the app when the screen cannot hold it.
 *
 * The alternative was a stacked mobile layout, and the reason against it is
 * the same reason the rest of this project exists: a squeezed plot still looks
 * like a plot. Someone would read a number off an axis too small to label
 * correctly and believe it. Saying "not here" is the honest version of not
 * being able to show it. */
export function NarrowNotice({ width }: { width: number }) {
  return (
    <div className="narrow-notice">
      <div className="narrow-notice-inner">
        <p className="brand">atomsim</p>
        <h1>Needs a wider screen.</h1>
        <p>
          atomsim is a three-column instrument: a 3D viewport with a column of
          readouts on one side and about twenty controls on the other. It could
          be stacked to fit a phone, but only by shrinking the plots past the
          point where you can read a number off an axis, and a plot you cannot
          read is not an honest plot. So it asks for room instead.
        </p>
        <p className="narrow-notice-measure">
          this screen <strong>{width}px</strong> · needs{" "}
          <strong>{MIN_WIDTH}px</strong>
        </p>
        <p className="narrow-notice-what">
          It models the quantum mechanics of atoms without ever quietly lying
          about them. Every number it shows carries a tier that says how it was
          got: <span className="tier">EXACT</span>,{" "}
          <span className="tier">NUMERICAL</span>,{" "}
          <span className="tier">APPROXIMATION</span>,{" "}
          <span className="tier">COUNTERFACTUAL</span>, or{" "}
          <span className="tier">VISUAL LIBERTY</span>.
        </p>
      </div>
    </div>
  );
}
