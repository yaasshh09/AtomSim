import { useCallback, useEffect, useRef, useState } from "react";
import { useAppStore } from "../state/store";
import { tourById } from "../tours/registry";
import { spotlightBox } from "../tours/spotlight";

const PAD = 6;

type Ring = ReturnType<typeof spotlightBox>;

function same(a: Ring, b: Ring): boolean {
  if (a === null || b === null) return a === b;
  return a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;
}

/**
 * The ring drawn around the control a step is talking about.
 *
 * Fixed-position and pointer-events:none, so it never intercepts a click: the
 * controls stay live during a tour, and ringing one must not stop it working.
 * If the anchor is missing or measures zero, no ring renders at all and the
 * card downstairs carries the step on its own.
 *
 * The anchor gets re-found and re-measured after every render rather than
 * looked up once. Walking the flagship tour in a browser is what forced that.
 * Two failures, one cause: measuring once assumes the layout has finished
 * moving, and it has not.
 *
 * The Dirac toggle only exists after the levels payload arrives, so a one-shot
 * querySelector found nothing and gave up for good, leaving no ring at all on
 * that step. The surface controls kept their exact size and slid 247 px up
 * once the cloud view finished laying out, so a ResizeObserver never fired and
 * the ring sat a third of a screen below the control it pointed at.
 *
 * Re-syncing after every render fixes both, because every layout change here
 * comes from a store change and this component re-renders on all of them. The
 * observers below cover the rest: a window resize, and a settle that arrives
 * without a render. `same` is what stops that looping, since setting state
 * from an effect with no dependency array would otherwise re-render forever on
 * a fresh object.
 */
export function TourSpotlight() {
  const { tourId, stepIndex, sheet, setSheet } = useAppStore();
  const [box, setBox] = useState<Ring>(null);
  const boxRef = useRef<Ring>(null);
  const tour = tourId ? tourById(tourId) : null;
  const anchor = tour?.steps[stepIndex]?.spotlight ?? null;
  /** The last anchor the sheet was opened for. See the block that sets it. */
  const openedFor = useRef<string | null>(null);

  const sync = useCallback(() => {
    const el = anchor ? document.querySelector(`[data-tour="${anchor}"]`) : null;
    /* Thirteen of the sixteen anchors live in Controls or InfoPanel, and on the
       stacked shell both of those are inside the sheet, so most steps of most
       tours point at something the sheet is sitting on. Landing on one of them
       opens the sheet far enough to show it and scrolls the panel to it.

       Once per step, which is what the ref is for, and not once per render. A
       rule that reopened the sheet whenever it found it closed would take it
       back off anyone who closed it mid-step to look at the picture, which is
       a reasonable thing to want during a tour about the picture. The step is
       only counted as handled once the element has actually been found, since
       some anchors (the Dirac toggle) do not exist until their payload has
       arrived. */
    if (el && openedFor.current !== anchor) {
      openedFor.current = anchor;
      if (el.closest(".mobile-sheet")) {
        if (useAppStore.getState().sheet === "collapsed") setSheet("half");
        // Instant, not smooth: the sheet runs a height transition at the same
        // time, and a smooth scroll inside a box that is still growing lands
        // short of where it was aimed.
        el.scrollIntoView({ block: "center" });
      }
    }
    /* A collapsed sheet clips its body rather than unmounting it, so an anchor
       inside it still reports a real size, positioned under the bottom edge of
       the screen. Structural rather than measured for exactly that reason: a
       size check would pass and the ring would be drawn off the bottom of the
       page. No ring, and the card downstairs carries the step, which is what
       already happens for any anchor that is not on screen. */
    const buried =
      el !== null && sheet === "collapsed" && el.closest(".mobile-sheet") !== null;
    const next = el && !buried ? spotlightBox(el.getBoundingClientRect(), PAD) : null;
    if (same(boxRef.current, next)) return;
    boxRef.current = next;
    setBox(next);
  }, [anchor, sheet, setSheet]);

  // No dependency array: sync after every render, which is after every store
  // change.
  useEffect(sync);

  useEffect(() => {
    if (!anchor) return;
    // Two frames, because the frame right after a step lands is often one
    // layout pass short of settled.
    let second = 0;
    const first = requestAnimationFrame(() => {
      sync();
      second = requestAnimationFrame(sync);
    });
    const observer = new ResizeObserver(sync);
    if (document.body) observer.observe(document.body);
    const el = document.querySelector(`[data-tour="${anchor}"]`);
    if (el) observer.observe(el);
    window.addEventListener("resize", sync);
    return () => {
      cancelAnimationFrame(first);
      cancelAnimationFrame(second);
      observer.disconnect();
      window.removeEventListener("resize", sync);
    };
  }, [anchor, sync]);

  if (!box) return null;
  return (
    <div
      className="tour-ring"
      style={{ left: box.x, top: box.y, width: box.w, height: box.h }}
      aria-hidden="true"
    />
  );
}
