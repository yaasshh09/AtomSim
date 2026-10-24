import { useEffect, useRef, useState } from "react";
import { getNodes } from "../api/client";
import type { NodeSet } from "../api/types";
import { currentCaption, nodeCaption } from "../lib/explain";
import { HF_ORBITAL_CAPTION } from "../lib/hfModel";
import { Notation } from "../lib/mathText";
import { toCsv } from "../lib/plotExport";
import { rasterize } from "../lib/rasterize";
import { sizeAnchor } from "../lib/scaleBar";
import { isHydrogenic } from "../lib/systemKind";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { PlaneStack } from "./PlaneStack";

/**
 * The exact nodes where they cut this slice. A sphere is a circle, a cone about
 * z is two rays at ±theta from +z, and a nodal plane through z meets y = 0 only
 * along the z axis, unless it is y = 0 itself (the sin-type real orbitals), which
 * the caption says instead of drawing.
 */
export function NodeMarks({
  nodes,
  x,
  z,
  reach,
}: {
  nodes: NodeSet;
  x: (b: number) => number;
  z: (b: number) => number;
  reach: number;
}) {
  const axis = nodes.z_axis || nodes.planes.some((p) => Math.abs(Math.sin(p.value)) > 1e-9);
  return (
    <g className="node-marks" aria-hidden="true">
      {nodes.radial.map((q) => (
        <circle key={q.label} cx={x(0)} cy={z(0)} r={x(q.value) - x(0)} />
      ))}
      {nodes.cones.flatMap((q) =>
        [1, -1].map((side) => (
          <line
            key={`${q.label}${side}`}
            x1={x(0)}
            y1={z(0)}
            x2={x(side * reach * Math.sin(q.value))}
            y2={z(reach * Math.cos(q.value))}
          />
        )),
      )}
      {axis && <line x1={x(0)} y1={z(-reach)} x2={x(0)} y2={z(reach)} />}
    </g>
  );
}

export function PlaneView() {
  const {
    n, l, m, system, basis, planeQuantity, plane, planeStatus, planeProgress,
    error, loadPlane, setPlaneQuantity,
  } = useAppStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const systems = useAppStore((st) => st.systems);
  const hydrogenic = isHydrogenic(systems, system);
  const [showNodes, setShowNodes] = useState(false);
  const [nodes, setNodes] = useState<NodeSet | null>(null);

  useEffect(() => {
    setNodes(null);
    if (!showNodes || !hydrogenic) return;
    let live = true;
    getNodes(n, l, m, system, basis).then(
      (ns) => live && setNodes(ns),
      () => {},
    );
    return () => {
      live = false;
    };
  }, [showNodes, hydrogenic, n, l, m, system, basis]);

  useEffect(() => {
    if (!plane && planeStatus === "idle") void loadPlane();
  }, [plane, planeStatus, loadPlane, n, l, m, system, basis, planeQuantity]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !plane) return;
    const { resolution, quantity } = plane.meta;
    canvas.width = resolution;
    canvas.height = resolution;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(
      new ImageData(rasterize(plane.values, resolution, quantity), resolution, resolution),
      0,
      0,
    );
  }, [plane]);

  return (
    <div className="view-wrap">
      <div className="view-header">
        <span className="plot-title">
          {plane ? (
            <Notation>{`${plane.meta.label} [${plane.meta.unit}]`}</Notation>
          ) : (
            "2D cross-section (y = 0 plane)"
          )}
        </span>
        {plane && <Badge provenance={plane.meta.provenance} />}
        <div className="seg">
          <button
            type="button"
            className={planeQuantity === "density" ? "seg-on" : ""}
            onClick={() => setPlaneQuantity("density")}
          >
            |ψ|²
          </button>
          <button
            type="button"
            className={planeQuantity === "psi" ? "seg-on" : ""}
            onClick={() => setPlaneQuantity("psi")}
          >
            ψ (signed)
          </button>
          <button
            type="button"
            className={planeQuantity === "current" ? "seg-on" : ""}
            onClick={() => setPlaneQuantity("current")}
            title="The probability current through this plane"
          >
            current j
          </button>
        </div>
        <button
          type="button"
          className={showNodes ? "ghost seg-on" : "ghost"}
          aria-pressed={showNodes}
          onClick={() => setShowNodes((v) => !v)}
          title="Mark where psi is exactly zero"
        >
          nodes
        </button>
        {showNodes && nodes && <Badge provenance={nodes.provenance} />}
      </div>
      <PlaneStack
        canvasRef={canvasRef}
        halfExtent={plane?.meta.half_extent ?? 0}
        resolution={plane?.meta.resolution ?? 0}
        name={`atomsim-plane-${planeQuantity}`}
        light={plane ? plane.meta.quantity !== "density" : false}
        bohrPm={plane?.meta.bohr_pm}
        overlay={
          showNodes && nodes && plane
            ? (x, z) => (
                <NodeMarks nodes={nodes} x={x} z={z} reach={2 * plane.meta.half_extent} />
              )
            : undefined
        }
        csv={
          plane
            ? () => {
                // Row i is z ascending, column j is x ascending, both on
                // linspace(-h, h, resolution): the engine's own samples.
                const { resolution: res, half_extent: h, unit, quantity } = plane.meta;
                const at = (k: number) => -h + (2 * h * k) / (res - 1);
                const rows: number[][] = [];
                for (let i = 0; i < res; i++) {
                  for (let j = 0; j < res; j++) rows.push([at(j), at(i), plane.values[i * res + j]]);
                }
                return toCsv(["x_bohr", "z_bohr", `${quantity} [${unit}]`], rows);
              }
            : undefined
        }
      >
        {planeStatus === "sampling" && (
          <p className="hint">computing… {(planeProgress * 100).toFixed(0)}%</p>
        )}
        {planeStatus === "error" && error && <p className="error">{error}</p>}
      </PlaneStack>
      {plane && (
        <p className="caption">
          x, z ∈ [−{plane.meta.half_extent.toFixed(1)}, +
          {plane.meta.half_extent.toFixed(1)}] bohr; z vertical (quantization axis).{" "}
          {sizeAnchor(2 * plane.meta.half_extent, plane.meta.bohr_pm)}{" "}
          {plane.meta.quantity === "density" ? (
            "The inferno brightness is γ-compressed at exponent 0.5 (VISUAL LIBERTY, so faint lobes stay visible)."
          ) : plane.meta.quantity === "psi" ? (
            <Notation>{"A diverging RdBu, linear in ψ: blue < 0 < red. ψ is real on this plane (e^{imφ} = ±1)."}</Notation>
          ) : (
            <Notation>{currentCaption(basis, plane.meta.m)}</Notation>
          )}
        </p>
      )}
      {showNodes && nodes && <p className="caption">{nodeCaption(nodes)}</p>}
      {showNodes && !hydrogenic && (
        <p className="caption">
          Exact nodes are drawn for the hydrogen-like systems only. A many-electron
          orbital has nodes too, but they are zeros of a numerical function, and
          drawing them with the same dashed line would give them a tier they have not
          earned.
        </p>
      )}
      {plane?.meta.model === "hf" && <p className="caption">{HF_ORBITAL_CAPTION}</p>}
    </div>
  );
}
