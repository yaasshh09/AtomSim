import { useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { stateLabel } from "../lib/quantum";
import { nextSnap, PEEK_HEIGHT, snapHeight } from "../lib/sheet";
import { useAppStore } from "../state/store";
import { Controls } from "./Controls";
import { InfoPanel } from "./InfoPanel";

/**
 * The controls and the readouts, on a phone, in a sheet that drags.
 *
 * It wraps `InfoPanel` and `Controls` exactly as they are. A second set of
 * controls written for a small screen would be two panels to keep in step and
 * the first place the two shells would drift apart on what the app can do:
 * every knob would have to be added twice, and the one that got forgotten
 * would be silently missing rather than visibly broken. Both panels are
 * already vertical stacks of labelled fields, which is what a sheet body wants
 * anyway.
 *
 * Half is the state this is designed for. An instrument is used by changing
 * something and watching the picture answer, and half is the only snap where
 * the control under a thumb and the plot it changes are both on screen.
 * Collapsed gives the picture the whole viewport; full is for reading down the
 * whole control set.
 */
export function MobileSheet({ viewportHeight }: { viewportHeight: number }) {
  const { n, l, m, system, systems, sheet, setSheet } = useAppStore();
  const sys = systems.find((s) => s.key === system);
  const [drag, setDrag] = useState<number | null>(null);
  /* The pointer's own history, kept in a ref rather than in state: velocity is
     read once, at the release, and re-rendering the sheet on every pointer
     sample to store a number nothing draws would be a frame's work per move. */
  const track = useRef<{ id: number; y0: number; y: number; t: number; v: number } | null>(
    null,
  );

  const resting = snapHeight(sheet, viewportHeight);
  const height = Math.min(
    Math.max(resting - (drag ?? 0), PEEK_HEIGHT),
    snapHeight("full", viewportHeight),
  );

  const onPointerDown = (e: ReactPointerEvent<HTMLElement>) => {
    track.current = { id: e.pointerId, y0: e.clientY, y: e.clientY, t: e.timeStamp, v: 0 };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: ReactPointerEvent<HTMLElement>) => {
    const t = track.current;
    if (!t || t.id !== e.pointerId) return;
    const dt = e.timeStamp - t.t;
    if (dt > 0) t.v = (e.clientY - t.y) / dt;
    t.y = e.clientY;
    t.t = e.timeStamp;
    setDrag(e.clientY - t.y0);
  };
  const onPointerUp = (e: ReactPointerEvent<HTMLElement>) => {
    const t = track.current;
    if (!t || t.id !== e.pointerId) return;
    track.current = null;
    setDrag(null);
    // A press and a release in the same place is a tap, not a drag, and a tap
    // on the handle is the one-handed way in and out.
    const moved = Math.abs(e.clientY - t.y0) > 4;
    if (!moved) {
      setSheet(sheet === "collapsed" ? "half" : "collapsed");
      return;
    }
    setSheet(nextSnap(sheet, e.clientY - t.y0, t.v, viewportHeight));
  };

  return (
    <section
      className={`mobile-sheet mobile-sheet-${sheet}${drag === null ? "" : " mobile-sheet-dragging"}`}
      style={{ height }}
      aria-label="controls and readouts"
    >
      <header
        className="mobile-sheet-handle"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <span className="mobile-sheet-grip" aria-hidden="true" />
        <span className="mobile-sheet-peek">
          {sys ? sys.name : system} · {stateLabel(n, l, m)}
        </span>
        <button
          type="button"
          className="mobile-sheet-toggle"
          aria-expanded={sheet !== "collapsed"}
          onClick={() => setSheet(sheet === "collapsed" ? "half" : "collapsed")}
        >
          {sheet === "collapsed" ? "controls" : "hide"}
        </button>
      </header>
      {/* Mounted at every snap, including collapsed. Unmounting it would throw
          away the panels' own scroll positions and the configuration field's
          uncommitted draft every time the sheet was closed, and would make the
          tour's anchors vanish from the document exactly when it needs to
          measure them. */}
      <div className="mobile-sheet-body" inert={sheet === "collapsed"}>
        {/* Controls first, readouts under them. On the desktop shell the
            readouts are a rail you glance at without moving anything, but in a
            sheet opened to half there is one screenful, and the half snap
            exists so that a control and the plot it changes are visible
            together. Putting the state card first would spend that screenful
            on a summary of what is already drawn behind it. */}
        <Controls />
        <InfoPanel />
      </div>
    </section>
  );
}
