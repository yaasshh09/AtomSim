import { useEffect, useState } from "react";
import { useAppStore } from "../state/store";
import { tourById } from "../tours/registry";
import { spotlightBox } from "../tours/spotlight";

const PAD = 6;

/**
 * A ring around the control a step is talking about.
 *
 * Fixed-position, pointer-events:none, so it never intercepts a click: the
 * controls stay live during a tour and ringing one must not stop it working.
 * If the anchor is missing or measures zero the ring simply does not render,
 * and the card downstairs carries the step on its own.
 */
export function TourSpotlight() {
  const { tourId, stepIndex } = useAppStore();
  const [box, setBox] = useState<ReturnType<typeof spotlightBox>>(null);
  const tour = tourId ? tourById(tourId) : null;
  const anchor = tour?.steps[stepIndex]?.spotlight ?? null;

  useEffect(() => {
    if (!anchor) {
      setBox(null);
      return;
    }
    const el = document.querySelector(`[data-tour="${anchor}"]`);
    if (!el) {
      setBox(null);
      return;
    }
    const measure = () => setBox(spotlightBox(el.getBoundingClientRect(), PAD));

    // One requestAnimationFrame is not enough, and the browser check is what
    // proved it: entering a tour mounts the card downstairs, that resizes the
    // grid, and the control this ring points at was still moving when the
    // first frame measured. The ring settled 5 px high and 7 px short and
    // stayed there until a window resize happened to re-measure it.
    //
    // So the anchor is watched rather than sampled. The observer on the
    // element catches it changing size, and the one on the shell catches a
    // sibling panel appearing and pushing it, which is a move the element's
    // own observer never sees.
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    if (document.body) observer.observe(document.body);
    const id = requestAnimationFrame(measure);
    window.addEventListener("resize", measure);
    return () => {
      cancelAnimationFrame(id);
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [anchor, stepIndex, tourId]);

  if (!box) return null;
  return (
    <div
      className="tour-ring"
      style={{ left: box.x, top: box.y, width: box.w, height: box.h }}
      aria-hidden="true"
    />
  );
}
