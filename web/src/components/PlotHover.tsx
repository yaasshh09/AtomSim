import { useCallback, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent, RefObject } from "react";
import { viewBoxX } from "../lib/hover";
import { mathTspans } from "../lib/mathText";

/**
 * Pointer tracking on an SVG plot, in the plot's own coordinates.
 *
 * This hook stores a viewBox x and nothing else. Which sample that lands on,
 * and what gets printed for it, is the caller's decision, because only the
 * caller knows which of its arrays is the grid and what the values mean.
 * Keeping the lookup here would quietly pick a nearest-point rule for curves
 * this file has never seen.
 */
export function usePlotHover(
  viewBoxWidth: number,
  /** The plot's element, when something else already owns the ref (the zoom). */
  shared?: RefObject<SVGSVGElement | null>,
) {
  const own = useRef<SVGSVGElement | null>(null);
  const ref = shared ?? own;
  const [x, setX] = useState<number | null>(null);
  const onPointerMove = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>) => {
      const el = ref.current;
      if (!el) return;
      setX(viewBoxX(e.clientX, el.getBoundingClientRect(), viewBoxWidth));
    },
    [viewBoxWidth],
  );
  const onPointerLeave = useCallback(() => setX(null), []);
  return { ref, x, onPointerMove, onPointerLeave };
}

/**
 * The crosshair and its readout.
 *
 * Drawn at the sample, not at the pointer: `px` is where the nearest grid
 * point actually is, so the line sits on a number the engine computed rather
 * than between two of them. On a coarse grid the crosshair therefore snaps,
 * which is the honest behaviour. It is showing where the data is.
 */
export function HoverReadout({
  px,
  py,
  top,
  bottom,
  lines,
  width,
  rightMargin,
}: {
  px: number;
  py: number | null;
  top: number;
  bottom: number;
  /** The readout rows, longest first: the box is sized off the first one. */
  lines: string[];
  width: number;
  rightMargin: number;
}) {
  const boxW = Math.max(...lines.map((s) => s.length)) * 5.6 + 12;
  const boxH = lines.length * 12 + 8;
  // Flips to the left of the crosshair when the box would overrun the frame,
  // so a readout near the right edge stays inside the plot instead of being
  // clipped by the viewBox.
  const flip = px + boxW + 10 > width - rightMargin;
  const bx = flip ? px - boxW - 8 : px + 8;
  const by = Math.min(Math.max(py ?? top, top), bottom - boxH);
  return (
    <g className="hover-layer" pointerEvents="none">
      <line x1={px} x2={px} y1={top} y2={bottom} className="hover-line" />
      {py !== null && <circle cx={px} cy={py} r={3} className="hover-dot" />}
      <rect x={bx} y={by} width={boxW} height={boxH} rx={2} className="hover-box" />
      {lines.map((text, i) => (
        <text key={text} x={bx + 6} y={by + 13 + i * 12} className="tick hover-text">
          {mathTspans(text)}
        </text>
      ))}
    </g>
  );
}
