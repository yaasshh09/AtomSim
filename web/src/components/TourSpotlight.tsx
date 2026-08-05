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
 * A ring around the control a step is talking about.
 *
 * Fixed-position, pointer-events:none, so it never intercepts a click: the
 * controls stay live during a tour and ringing one must not stop it working.
 * If the anchor is missing or measures zero the ring simply does not render,
 * and the card downstairs carries the step on its own.
 *
 * The anchor is re-found and re-measured after every render rather than looked
 * up once, and walking the flagship in a browser is what forced that. Two
 * failures, one cause: measuring once assumes the app has finished moving, and
 * it has not.
 *
 * The Dirac toggle only exists after the levels payload arrives, so a one-shot
 * querySelector found nothing and gave up for good; that step drew no ring at
 * all. The surface controls kept their exact size and slid 247 px up once the
 * cloud view finished laying out, so a ResizeObserver never fired and the ring
 * sat a third of a screen below the control it was pointing at.
 *
 * Re-syncing after every render fixes both, because every layout change in
 * this app comes from a store change and this component re-renders on all of
 * them. The observers below cover the rest: a window resize, and a settle that
 * arrives without a render. `same` is what stops that from looping, since
 * setting state from an effect with no dependency array would otherwise
 * re-render forever on a fresh object.
 */
export function TourSpotlight() {
  const { tourId, stepIndex } = useAppStore();
  const [box, setBox] = useState<Ring>(null);
  const boxRef = useRef<Ring>(null);
  const tour = tourId ? tourById(tourId) : null;
  const anchor = tour?.steps[stepIndex]?.spotlight ?? null;

  const sync = useCallback(() => {
    const el = anchor ? document.querySelector(`[data-tour="${anchor}"]`) : null;
    const next = el ? spotlightBox(el.getBoundingClientRect(), PAD) : null;
    if (same(boxRef.current, next)) return;
    boxRef.current = next;
    setBox(next);
  }, [anchor]);

  // No dependency array: after every render, which is after every store change.
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
