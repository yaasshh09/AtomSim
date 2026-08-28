import { useCallback, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { viewBoxX } from "../lib/hover";

/**
 * How I track the pointer on an SVG plot, in the plot's own coordinates.
 *
 * This hook stores a viewBox x and nothing else. Which sample that lands on,
 * and what I print for it, is the caller's decision, because only the caller
 * knows which of its arrays is the grid and what the values mean. If I kept
 * the lookup here I would be quietly choosing a nearest-point rule for curves
 * I have never seen.
 */
export function usePlotHover(viewBoxWidth: number) {
  const ref = useRef<SVGSVGElement | null>(null);
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
 * My crosshair and its readout.
 *
 * I draw it at the sample, not at the pointer: `px` is where the nearest grid
 * point actually is, so my line sits on a number my engine computed rather
 * than between two of them. On a coarse grid my crosshair therefore snaps,
 * which is the honest behaviour; it is showing you where the data is.
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
  /** My readout rows, longest first: I size the box off the first one. */
  lines: string[];
  width: number;
  rightMargin: number;
}) {
  const boxW = Math.max(...lines.map((s) => s.length)) * 5.6 + 12;
  const boxH = lines.length * 12 + 8;
  // I flip to the left of the crosshair when the box would overrun the frame,
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
          {text}
        </text>
      ))}
    </g>
  );
}
