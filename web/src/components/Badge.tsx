import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Provenance } from "../api/types";
import { formatErrorScale } from "../lib/liberties";

/* I tune these to my instrument palette: EXACT gets the shell's own mint, so
   the tier I am proudest of is the colour I built the whole UI around. I leave
   counterfactual pink alone on purpose, because index.css uses that exact hue
   for the ghost HUD, the counterfactual banner and the counterfactual rungs,
   and a badge that drifted off it would stop matching the thing it labels. */
const COLORS: Record<string, string> = {
  exact: "#34e0a1",
  numerical: "#6cb7ff",
  approximation: "#fbbf24",
  counterfactual: "#f472b6",
  visual_liberty: "#b48bd9",
};

/** The clear space I keep between the panel and every window edge, in px. */
const MARGIN = 8;
/** The gap I leave between a badge and the panel it opens, in px. */
const GAP = 6;

type Point = { left: number; top: number };

/* I put the panel below the badge when it fits there, above it when it does
   not, and pinned to the window when neither side has room: on a short
   viewport I then show the top of the panel and scroll the rest, rather than
   running half of it off-screen. Horizontally I start the panel at the badge
   and slide it back in whenever that would overhang the right edge. */
export function placeInspector(
  anchor: { left: number; top: number; bottom: number },
  panel: { width: number; height: number },
  view: { width: number; height: number },
): Point {
  let top = anchor.bottom + GAP;
  if (top + panel.height > view.height - MARGIN) {
    const above = anchor.top - GAP - panel.height;
    top = above >= MARGIN ? above : Math.max(MARGIN, view.height - MARGIN - panel.height);
  }
  const rightmost = Math.max(MARGIN, view.width - MARGIN - panel.width);
  return { left: Math.min(Math.max(anchor.left, MARGIN), rightmost), top };
}

export function Badge({ provenance }: { provenance: Provenance }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<Point | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const color = COLORS[provenance.fidelity] ?? "#ffffff";

  const reposition = useCallback(() => {
    const button = buttonRef.current;
    const panel = panelRef.current;
    if (!button || !panel) return;
    setPos(
      placeInspector(
        button.getBoundingClientRect(),
        { width: panel.offsetWidth, height: panel.offsetHeight },
        { width: window.innerWidth, height: window.innerHeight },
      ),
    );
  }, []);

  useLayoutEffect(() => {
    if (!open) {
      setPos(null);
      return;
    }
    reposition();
  }, [open, reposition]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    const onDown = (e: Event) => {
      const target = e.target as Node;
      if (buttonRef.current?.contains(target)) return;
      if (panelRef.current?.contains(target)) return;
      setOpen(false);
    };
    // When the badge scrolls out of sight I have to close the panel, or it
    // outlives the thing it points at and floats over unrelated readouts. An
    // observer is what tells me: its intersection rect is already clipped by
    // every scrolling ancestor, so it sees a badge hidden inside a rail, which
    // a plain viewport comparison would call visible.
    const seen = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) setOpen(false);
      },
      { threshold: 0 },
    );
    if (buttonRef.current) seen.observe(buttonRef.current);
    // I listen in the capture phase, because the thing that scrolls is a rail
    // or a view wrapper and not the window: a bubbling listener never hears it
    // and my panel would drift away from the badge it belongs to.
    window.addEventListener("scroll", reposition, true);
    window.addEventListener("resize", reposition);
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown, true);
    return () => {
      seen.disconnect();
      window.removeEventListener("scroll", reposition, true);
      window.removeEventListener("resize", reposition);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown, true);
    };
  }, [open, reposition]);

  return (
    <span className="badge-wrap">
      <button
        ref={buttonRef}
        type="button"
        className="badge"
        style={{ borderColor: color, color }}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        {provenance.fidelity.replace("_", " ").toUpperCase()}
      </button>
      {open &&
        createPortal(
          /* I portal into the body, not into the badge. My badges live inside
             the scrolling rails and on top of the canvas, and both of those
             clip: a rail is `overflow-y: auto`, which makes its cross axis
             `auto` as well, so an absolutely positioned panel wider than the
             rail was cut off and opened a strip of empty background beside it.
             Positioned in viewport coordinates from here, the panel has no
             clipping ancestor left and every word of a provenance stays inside
             one rectangle, wherever the badge that opened it happens to sit. */
          <div
            ref={panelRef}
            className="badge-inspector"
            // I hide it rather than unmounting it for the first frame: I have
            // to lay the panel out before I can measure it, and `visibility`
            // keeps it in the layout while keeping the unplaced frame unseen.
            style={{
              left: pos?.left ?? 0,
              top: pos?.top ?? 0,
              visibility: pos ? "visible" : "hidden",
            }}
          >
            <p>
              <strong>Method:</strong> {provenance.method}
            </p>
            {provenance.assumptions.length > 0 && (
              <ul>
                {provenance.assumptions.map((a) => (
                  <li key={a}>{a}</li>
                ))}
              </ul>
            )}
            {provenance.error_estimate !== null && (
              <p>
                <strong>Error scale:</strong> {formatErrorScale(provenance.error_estimate)}
              </p>
            )}
            {provenance.refinement && (
              <p>
                <strong>To improve:</strong> {provenance.refinement}
              </p>
            )}
          </div>,
          document.body,
        )}
    </span>
  );
}
