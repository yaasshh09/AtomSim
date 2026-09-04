import { useCallback, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent, RefObject } from "react";
import { isZoomed, panView, zoomFactor, zoomView, type Domain } from "../lib/zoom";

/** One axis of a zoomable plot: what it covers, where it is drawn, how it maps. */
export type ZoomAxis = {
  /** The whole of the data, which is also the furthest out a zoom can go. */
  domain: Domain;
  /** The pixel span it occupies, in viewBox units, in scale order. */
  range: Domain;
  log?: boolean;
};

export type PlotZoom = {
  /** The window to build this render's scale from. Equals `domain` unzoomed. */
  x: [number, number];
  y: [number, number];
  zoomed: boolean;
  /** Magnification, for the readout: 1 is the whole range. */
  factor: number;
  dragging: boolean;
  reset: () => void;
  /** Zoom about the centre, for the buttons. Below 1 zooms in. */
  by: (factor: number) => void;
  /** Ref for the plot this zoom belongs to. */
  ref: (el: SVGSVGElement | null) => void;
  /** Ref for a second plot sharing the same window, e.g. a residual panel. */
  follower: (el: SVGSVGElement | null) => void;
  /** The primary element, for anything else that needs to measure it. */
  element: RefObject<SVGSVGElement | null>;
  handlers: {
    onPointerDown: (e: ReactPointerEvent<SVGSVGElement>) => void;
    onPointerMove: (e: ReactPointerEvent<SVGSVGElement>) => void;
    onPointerUp: (e: ReactPointerEvent<SVGSVGElement>) => void;
    onPointerCancel: (e: ReactPointerEvent<SVGSVGElement>) => void;
    onDoubleClick: () => void;
  };
};

const WHEEL_IN = 1 / 1.18;
const WHEEL_OUT = 1.18;
/** Pointer travel, in viewBox units, before a press counts as a drag. */
const DRAG_SLOP = 3;

const FULL: [number, number] = [0, 1];

/**
 * Wheel-to-zoom and drag-to-pan over one or two axes of an SVG plot.
 *
 * The hook owns a window per axis and nothing else. It does not build scales,
 * because the caller's scale may be log, may be nice()d, and may be shared
 * with a second plot; and it does not draw, because clipping the drawn layer
 * is a decision about which parts of a plot are data (the curve) and which are
 * furniture that has to stay put (the axis, the labels).
 *
 * Zooming the domain rather than transforming the picture is what keeps the
 * axis honest: ticks re-space, labels re-spread and the curve is re-sampled,
 * so the numbers printed on a magnified plot are the numbers it is showing.
 *
 * The window resets whenever the full domain changes. Selecting a new atom or
 * a new orbital replaces the data underneath, and a window held over from the
 * previous one would be a magnification of somewhere that no longer means
 * anything.
 */
export function usePlotZoom(spec: {
  width: number;
  height: number;
  x?: ZoomAxis;
  y?: ZoomAxis;
}): PlotZoom {
  const { width, height, x: ax, y: ay } = spec;
  const element = useRef<SVGSVGElement | null>(null);
  const [xView, setXView] = useState<[number, number] | null>(null);
  const [yView, setYView] = useState<[number, number] | null>(null);
  const [dragging, setDragging] = useState(false);

  // The full domains, as one string. Comparing the arrays themselves is no
  // use: they are rebuilt every render from the payload, so they are never the
  // same object even when they hold the same numbers.
  const key =
    `${ax?.domain[0]},${ax?.domain[1]},${ax?.log}|` +
    `${ay?.domain[0]},${ay?.domain[1]},${ay?.log}`;
  const [prevKey, setPrevKey] = useState(key);
  if (key !== prevKey) {
    setPrevKey(key);
    setXView(null);
    setYView(null);
  }

  const x = xView ?? (ax ? ([ax.domain[0], ax.domain[1]] as [number, number]) : FULL);
  const y = yView ?? (ay ? ([ay.domain[0], ay.domain[1]] as [number, number]) : FULL);

  // Live values for the wheel listener, which is registered once per element
  // and would otherwise close over the first render's windows forever.
  const live = useRef({ ax, ay, x, y, width, height });
  live.current = { ax, ay, x, y, width, height };

  /* Updated from the previous window rather than from the one this render was
     built with. A trackpad delivers a flick as a burst of wheel events inside
     one frame, and reading the rendered window would make all of them zoom
     from the same starting point: six notches would land as one. */
  const zoomAbout = useCallback((factor: number, fx: number, fy: number) => {
    const { ax: cx, ay: cy } = live.current;
    if (cx) {
      setXView((prev) =>
        zoomView(prev ?? [cx.domain[0], cx.domain[1]], cx.domain, factor, fx, cx.log ?? false),
      );
    }
    if (cy) {
      setYView((prev) =>
        zoomView(prev ?? [cy.domain[0], cy.domain[1]], cy.domain, factor, fy, cy.log ?? false),
      );
    }
  }, []);

  /* A ref callback rather than an effect, because a second plot can share this
     window (the spectrum and its residuals are one wavelength axis drawn
     twice) and each of them needs its own listener. React 19 runs the returned
     cleanup when the element goes away. */
  const wheelRef = useCallback(
    (el: SVGSVGElement | null) => {
      if (!el) return;
      const onWheel = (e: WheelEvent) => {
        const s = live.current;
        if (!s.ax && !s.ay) return;
        // Non-passive on purpose: without preventDefault the page scrolls out
        // from under the plot the moment someone tries to zoom it.
        e.preventDefault();
        const rect = el.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;
        const vx = ((e.clientX - rect.left) / rect.width) * s.width;
        const vy = ((e.clientY - rect.top) / rect.height) * s.height;
        const frac = (v: number, r: Domain) => (v - r[0]) / (r[1] - r[0]);
        zoomAbout(
          e.deltaY > 0 ? WHEEL_OUT : WHEEL_IN,
          s.ax ? frac(vx, s.ax.range) : 0.5,
          s.ay ? frac(vy, s.ay.range) : 0.5,
        );
      };
      el.addEventListener("wheel", onWheel, { passive: false });
      return () => el.removeEventListener("wheel", onWheel);
    },
    [zoomAbout],
  );

  const ref = useCallback(
    (el: SVGSVGElement | null) => {
      element.current = el;
      const off = wheelRef(el);
      return () => {
        element.current = null;
        off?.();
      };
    },
    [wheelRef],
  );

  const drag = useRef<{ id: number; cx: number; cy: number; moved: boolean } | null>(null);

  const onPointerDown = useCallback((e: ReactPointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    drag.current = { id: e.pointerId, cx: e.clientX, cy: e.clientY, moved: false };
  }, []);

  /* Geometry comes from the element under the pointer, not from the primary
     one: a follower plot is dragged with the same handlers, and capturing the
     pointer on the wrong element would drop the drag on its first move. */
  const onPointerMove = useCallback((e: ReactPointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    const el = e.currentTarget;
    if (!d || d.id !== e.pointerId) return;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const s = live.current;
    const dx = ((e.clientX - d.cx) / rect.width) * s.width;
    const dy = ((e.clientY - d.cy) / rect.height) * s.height;
    if (!d.moved) {
      // A click on a rung or a bar is a press and a release in the same place,
      // so panning only starts past the slop and those clicks still land.
      if (Math.abs(dx) < DRAG_SLOP && Math.abs(dy) < DRAG_SLOP) return;
      d.moved = true;
      setDragging(true);
      el.setPointerCapture(e.pointerId);
    }
    d.cx = e.clientX;
    d.cy = e.clientY;
    const { ax: cx, ay: cy } = s;
    if (cx) {
      const span = cx.range[1] - cx.range[0];
      if (span !== 0) {
        setXView((prev) =>
          panView(prev ?? [cx.domain[0], cx.domain[1]], cx.domain, -dx / span, cx.log ?? false),
        );
      }
    }
    if (cy) {
      const span = cy.range[1] - cy.range[0];
      if (span !== 0) {
        setYView((prev) =>
          panView(prev ?? [cy.domain[0], cy.domain[1]], cy.domain, -dy / span, cy.log ?? false),
        );
      }
    }
  }, []);

  const endDrag = useCallback((e: ReactPointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    if (!d || d.id !== e.pointerId) return;
    if (d.moved && e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    drag.current = null;
    setDragging(false);
  }, []);

  const reset = useCallback(() => {
    setXView(null);
    setYView(null);
  }, []);

  const factor = Math.max(
    ax ? zoomFactor(x, ax.domain, ax.log ?? false) : 1,
    ay ? zoomFactor(y, ay.domain, ay.log ?? false) : 1,
  );

  return {
    x,
    y,
    zoomed:
      (ax !== undefined && isZoomed(x, ax.domain, ax.log ?? false)) ||
      (ay !== undefined && isZoomed(y, ay.domain, ay.log ?? false)),
    factor,
    dragging,
    reset,
    by: (f: number) => zoomAbout(f, 0.5, 0.5),
    ref,
    follower: wheelRef,
    element,
    handlers: {
      onPointerDown,
      onPointerMove,
      onPointerUp: endDrag,
      onPointerCancel: endDrag,
      onDoubleClick: reset,
    },
  };
}

/**
 * The zoom's controls, and its state in words.
 *
 * The magnification is printed rather than implied. A plot that has been
 * scrolled into is showing a slice of its data, and someone arriving at it
 * mid-session has no way to tell that from a plot whose data happens to be
 * narrow. "3.2x" plus a way back out is the whole disclosure.
 */
export function ZoomControls({
  zoom,
  what = "the plot",
}: {
  zoom: PlotZoom;
  /** What zooming acts on, for the hint line: "the wavelength axis". */
  what?: string;
}) {
  return (
    <div className="zoom-bar">
      <div className="zoom-buttons">
        <button
          type="button"
          onClick={() => zoom.by(WHEEL_IN * WHEEL_IN)}
          aria-label="zoom in"
          title="zoom in"
        >
          +
        </button>
        <button
          type="button"
          onClick={() => zoom.by(WHEEL_OUT * WHEEL_OUT)}
          aria-label="zoom out"
          disabled={!zoom.zoomed}
          title="zoom out"
        >
          −
        </button>
        <button
          type="button"
          onClick={zoom.reset}
          disabled={!zoom.zoomed}
          title="back to the full range"
        >
          reset
        </button>
      </div>
      <span className="zoom-state">
        {zoom.zoomed ? (
          <>
            <strong>{zoom.factor.toFixed(1)}×</strong> on {what}
          </>
        ) : (
          <>scroll to zoom {what}, drag to pan, double-click to reset</>
        )}
      </span>
    </div>
  );
}

/**
 * What the window cut off, marked at the edge it went out of.
 *
 * A ladder zoomed into a gap between two shells is a correct picture of an
 * empty stretch of energy, and it looks exactly like a plot that failed to
 * load. These two lines are the difference: they say which way the missing
 * rungs went, and how many there are, so the frame reads as a window on
 * something rather than as nothing.
 *
 * The counts are the caller's, because which direction is "above" depends on
 * the axis: a log binding-energy ladder puts its smallest value at the top.
 */
export function OffWindowMarks({
  above,
  below,
  x,
  top,
  bottom,
  noun,
}: {
  above: number;
  below: number;
  x: number;
  top: number;
  bottom: number;
  noun: string;
}) {
  const plural = (n: number) => (n === 1 ? noun : `${noun}s`);
  return (
    <>
      {above > 0 && (
        <text x={x} y={top} className="tick off-window">
          {`\u2191 ${above} more ${plural(above)} above this window`}
        </text>
      )}
      {below > 0 && (
        <text x={x} y={bottom} className="tick off-window">
          {`\u2193 ${below} more ${plural(below)} below this window`}
        </text>
      )}
    </>
  );
}
