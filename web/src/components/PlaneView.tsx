import { useEffect, useRef, useState } from "react";
import { getNodes } from "../api/client";
import type { NodeSet } from "../api/types";
import { currentCaption, nodeCaption } from "../lib/explain";
import { HF_ORBITAL_CAPTION } from "../lib/hfModel";
import { Notation } from "../lib/mathText";
import { toCsv } from "../lib/plotExport";
import { maxAbs, maxOf } from "../lib/colormap";
import { rasterize } from "../lib/rasterize";
import { sizeAnchor } from "../lib/scaleBar";
import { isHydrogenic } from "../lib/systemKind";
import { isY0, planeQuantityOn, Y0_SLICE, type PlaneSlice } from "../lib/urlState";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { ComparePicker } from "./ComparePicker";
import { Disclosure } from "./Disclosure";
import { Slider } from "./Field";
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
    error, loadPlane, setPlaneQuantity, planeSlice, setPlaneSlice,
    planeCompare, setPlaneCompare, comparePlane, comparePlaneStatus, loadComparePlane,
  } = useAppStore();
  const compareRef = useRef<HTMLCanvasElement>(null);
  const tilted = !isY0(planeSlice);
  // The sliders move a local copy; the store, and so a new grid, follows 300 ms
  // after the hand stops, so a drag is one solve rather than thirty.
  const [draft, setDraft] = useState<PlaneSlice>(planeSlice);
  useEffect(() => setDraft(planeSlice), [planeSlice]);
  useEffect(() => {
    if (draft === planeSlice) return;
    const t = setTimeout(() => setPlaneSlice(draft), 300);
    return () => clearTimeout(t);
  }, [draft, planeSlice, setPlaneSlice]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const systems = useAppStore((st) => st.systems);
  const hydrogenic = isHydrogenic(systems, system);
  const [showNodes, setShowNodes] = useState(false);
  const [nodes, setNodes] = useState<NodeSet | null>(null);

  useEffect(() => {
    setNodes(null);
    if (!showNodes || !hydrogenic || tilted) return;
    let live = true;
    getNodes(n, l, m, system, basis).then(
      (ns) => live && setNodes(ns),
      () => {},
    );
    return () => {
      live = false;
    };
  }, [showNodes, hydrogenic, tilted, n, l, m, system, basis]);

  useEffect(() => {
    if (!plane && planeStatus === "idle") void loadPlane();
  }, [plane, planeStatus, loadPlane, n, l, m, system, basis, planeQuantity, planeSlice]);

  // The comparison is drawn on the main plane's extent, so it can only be
  // asked for once that extent is known.
  useEffect(() => {
    if (planeCompare && plane && !comparePlane && comparePlaneStatus === "idle") {
      void loadComparePlane(plane.meta.half_extent);
    }
  }, [planeCompare, plane, comparePlane, comparePlaneStatus, loadComparePlane]);

  // Side by side, both frames share one colour scale, so equal brightness is
  // equal density; alone, a frame keeps its own maximum as it always has.
  const shared =
    plane && comparePlane
      ? plane.meta.quantity === "density"
        ? Math.max(maxOf(plane.values), maxOf(comparePlane.values))
        : Math.max(maxAbs(plane.values), maxAbs(comparePlane.values))
      : undefined;

  useEffect(() => {
    const paint = (canvas: HTMLCanvasElement | null, grid: typeof plane) => {
      if (!canvas || !grid) return;
      const { resolution, quantity } = grid.meta;
      canvas.width = resolution;
      canvas.height = resolution;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.putImageData(
        new ImageData(rasterize(grid.values, resolution, quantity, shared), resolution, resolution),
        0,
        0,
      );
    };
    paint(canvasRef.current, plane);
    paint(compareRef.current, comparePlane);
  }, [plane, comparePlane, shared]);

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
            disabled={planeQuantityOn(planeSlice, "psi", basis, m) !== "psi"}
            title={
              planeQuantityOn(planeSlice, "psi", basis, m) !== "psi"
                ? "psi is complex on this plane; use the real basis or the y = 0 plane"
                : undefined
            }
          >
            ψ (signed)
          </button>
          <button
            type="button"
            className={planeQuantity === "current" ? "seg-on" : ""}
            onClick={() => setPlaneQuantity("current")}
            disabled={tilted}
            title={
              tilted
                ? "The current is drawn through the y = 0 plane only"
                : "The probability current through this plane"
            }
          >
            current j
          </button>
        </div>
        <button
          type="button"
          className={showNodes ? "ghost seg-on" : "ghost"}
          aria-pressed={showNodes}
          onClick={() => setShowNodes((v) => !v)}
          disabled={tilted}
          title={
            tilted
              ? "Exact node marks are drawn on the y = 0 plane only"
              : "Mark where psi is exactly zero"
          }
        >
          nodes
        </button>
        {showNodes && nodes && !tilted && <Badge provenance={nodes.provenance} />}
      </div>
      <Disclosure summary={`Slice plane: ${tilted ? "tilted" : "y = 0, z up"}`}>
        <Slider
          label="Normal's angle from z"
          readout={`${draft.theta}°`}
          min={0}
          max={180}
          step={5}
          value={draft.theta}
          onChange={(theta) => setDraft({ ...draft, theta })}
        />
        <Slider
          label="Normal's azimuth about z"
          readout={`${draft.phi}°`}
          min={0}
          max={355}
          step={5}
          value={draft.phi}
          onChange={(phi) => setDraft({ ...draft, phi })}
        />
        <Slider
          label="Distance from the nucleus"
          readout={`${draft.offset} bohr`}
          min={-20}
          max={20}
          step={0.25}
          value={draft.offset}
          onChange={(offset) => setDraft({ ...draft, offset })}
        />
        <button
          type="button"
          className="ghost"
          disabled={isY0(draft)}
          onClick={() => setDraft(Y0_SLICE)}
        >
          back to y = 0
        </button>
      </Disclosure>
      <Disclosure
        summary={
          planeCompare
            ? `Compare: beside ${planeCompare.n},${planeCompare.l},${planeCompare.m} (${planeCompare.system})`
            : "Compare with another state"
        }
      >
        {planeQuantity === "current" ? (
          <p className="caption">The comparison draws densities and signed ψ, not the current.</p>
        ) : planeCompare ? (
          <>
            <ComparePicker value={planeCompare} systems={systems} onChange={setPlaneCompare} />
            <button type="button" className="ghost" onClick={() => setPlaneCompare(null)}>
              stop comparing
            </button>
          </>
        ) : (
          <button
            type="button"
            className="ghost"
            onClick={() => setPlaneCompare({ n, l, m, system: system === "h" ? "mu-h" : "h" })}
          >
            draw a second state beside this one
          </button>
        )}
      </Disclosure>
      <div className={comparePlane && planeCompare ? "plane-compare" : undefined}>
      <div className="plane-cell">
      <PlaneStack
        canvasRef={canvasRef}
        halfExtent={plane?.meta.half_extent ?? 0}
        resolution={plane?.meta.resolution ?? 0}
        name={`atomsim-plane-${planeQuantity}`}
        light={plane ? plane.meta.quantity !== "density" : false}
        bohrPm={plane?.meta.bohr_pm}
        axisNames={tilted ? ["across", "up"] : ["x", "z"]}
        overlay={
          showNodes && nodes && plane && !tilted
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
                const [across, up] = tilted ? ["across_bohr", "up_bohr"] : ["x_bohr", "z_bohr"];
                return toCsv([across, up, `${quantity} [${unit}]`], rows);
              }
            : undefined
        }
      >
        {planeStatus === "sampling" && (
          <p className="hint">computing… {(planeProgress * 100).toFixed(0)}%</p>
        )}
        {planeStatus === "error" && error && <p className="error">{error}</p>}
      </PlaneStack>
      </div>
      {planeCompare && planeQuantity !== "current" && (
        <div className="plane-cell">
        <PlaneStack
          canvasRef={compareRef}
          halfExtent={comparePlane?.meta.half_extent ?? 0}
          resolution={comparePlane?.meta.resolution ?? 0}
          name={`atomsim-plane-compare-${planeQuantity}`}
          light={comparePlane ? comparePlane.meta.quantity !== "density" : false}
          bohrPm={comparePlane?.meta.bohr_pm}
          axisNames={tilted ? ["across", "up"] : ["x", "z"]}
        >
          {comparePlaneStatus === "sampling" && <p className="hint">computing the comparison…</p>}
          {comparePlaneStatus === "error" && error && <p className="error">{error}</p>}
        </PlaneStack>
        </div>
      )}
      </div>
      {plane && comparePlane && planeCompare && (
        <p className="caption">
          Left: <Notation>{plane.meta.label}</Notation> <Badge provenance={plane.meta.provenance} />.
          Right: <Notation>{comparePlane.meta.label}</Notation>{" "}
          <Badge provenance={comparePlane.meta.provenance} /> ({planeCompare.system}). Same plane,
          same frame and one colour scale for both, so sizes and brightnesses compare directly; a
          state much smaller than the frame is drawn small rather than rescaled to fill it.
        </p>
      )}
      {plane && (
        <p className="caption">
          {tilted ? (
            <>
              Both axes span [−{plane.meta.half_extent.toFixed(1)}, +
              {plane.meta.half_extent.toFixed(1)}] bohr; across runs along (
              {plane.meta.axis_across.map((v) => v.toFixed(2)).join(", ")}) and up along (
              {plane.meta.axis_up.map((v) => v.toFixed(2)).join(", ")}) in (x, y, z).{" "}
            </>
          ) : (
            <>
              x, z ∈ [−{plane.meta.half_extent.toFixed(1)}, +
              {plane.meta.half_extent.toFixed(1)}] bohr; z vertical (quantization axis).{" "}
            </>
          )}
          {sizeAnchor(2 * plane.meta.half_extent, plane.meta.bohr_pm)}{" "}
          {plane.meta.quantity === "density" ? (
            "The inferno brightness is γ-compressed at exponent 0.5 (VISUAL LIBERTY, so faint lobes stay visible)."
          ) : plane.meta.quantity === "psi" ? (
            <Notation>
              {tilted
                ? "A diverging RdBu, linear in ψ: blue < 0 < red. ψ is real everywhere in this basis, so the sign is honest on any plane."
                : "A diverging RdBu, linear in ψ: blue < 0 < red. ψ is real on this plane (e^{imφ} = ±1)."}
            </Notation>
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
