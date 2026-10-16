import { scaleLinear } from "d3-scale";
import { useRef } from "react";
import type { ReactNode, RefObject } from "react";
import { minorTicks } from "../lib/axis";
import { GridLines, PlotTools } from "./PlotZoom";

/** Overlay viewBox size. Strokes are non-scaling, so this only sets the text scale. */
const S = 1000;

/**
 * A y = 0 plane raster with a bohr grid over it and the plot toolbar under it,
 * shared by the 2-D cross-section and the time evolution view.
 *
 * The grid is placed by pixel centres, not by the canvas edges: the engine
 * samples `resolution` points from -h to +h inclusive, so each edge pixel is
 * centred on ±h and the image runs half a pixel past it. Drawn edge to edge
 * instead, every line would sit up to half a pixel off the value it names.
 */
export function PlaneStack({
  canvasRef,
  halfExtent: h,
  resolution,
  csv,
  name,
  children,
}: {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  /** Half the plane's width in bohr, or 0 before the first grid arrives. */
  halfExtent: number;
  resolution: number;
  csv?: () => string;
  name: string;
  /** Status lines drawn over the frame (computing..., errors). */
  children?: ReactNode;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const half = resolution > 1 ? h / (resolution - 1) : 0;
  const x = scaleLinear([-h - half, h + half], [0, S]);
  const z = scaleLinear([-h - half, h + half], [S, 0]);
  const major = h > 0 ? scaleLinear([-h, h], [0, 1]).ticks(6) : [];
  const minor = minorTicks(major, [-h, h]);
  // Labels stay off the frame edge, where they would print over the axis
  // names and over each other in the corner. The caption states the extent.
  const labelled = major.filter((t) => Math.abs(t) < 0.9 * h);
  return (
    <>
      <div className="plane-frame" ref={frameRef}>
        <div className="plane-canvas plane-stack">
          <canvas ref={canvasRef} className="plane-raster" />
          {h > 0 && (
            <svg
              ref={svgRef}
              viewBox={`0 0 ${S} ${S}`}
              className="plane-grid"
              preserveAspectRatio="none"
              aria-hidden="true"
            >
              <GridLines
                x1={0} x2={S} y1={0} y2={S}
                xs={major.map(x)} xMinor={minor.map(x)}
                ys={major.map(z)} yMinor={minor.map(z)}
              />
              {labelled.map((t) => (
                <text key={`x${t}`} x={x(t)} y={S - 14} textAnchor="middle" className="plane-grid-label">
                  {t}
                </text>
              ))}
              {labelled.map((t) => (
                <text key={`z${t}`} x={14} y={z(t)} dy="0.32em" className="plane-grid-label">
                  {t}
                </text>
              ))}
              <text x={S - 14} y={S - 52} textAnchor="end" className="plane-grid-label">
                x [bohr]
              </text>
              <text x={14} y={44} className="plane-grid-label">
                z [bohr]
              </text>
            </svg>
          )}
        </div>
        {children}
      </div>
      <div className="zoom-bar">
        <PlotTools
          svg={() => svgRef.current}
          canvas={() => canvasRef.current}
          frame={() => frameRef.current}
          csv={csv}
          name={name}
        />
      </div>
    </>
  );
}
