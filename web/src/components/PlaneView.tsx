import { useEffect, useRef } from "react";
import { currentCaption } from "../lib/explain";
import { HF_ORBITAL_CAPTION } from "../lib/hfModel";
import { Notation } from "../lib/mathText";
import { toCsv } from "../lib/plotExport";
import { rasterize } from "../lib/rasterize";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { PlaneStack } from "./PlaneStack";

export function PlaneView() {
  const {
    n, l, m, system, basis, planeQuantity, plane, planeStatus, planeProgress,
    error, loadPlane, setPlaneQuantity,
  } = useAppStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);

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
      </div>
      <PlaneStack
        canvasRef={canvasRef}
        halfExtent={plane?.meta.half_extent ?? 0}
        resolution={plane?.meta.resolution ?? 0}
        name={`atomsim-plane-${planeQuantity}`}
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
          {plane.meta.quantity === "density" ? (
            "The inferno brightness is γ-compressed at exponent 0.5 (VISUAL LIBERTY, so faint lobes stay visible)."
          ) : plane.meta.quantity === "psi" ? (
            <Notation>{"A diverging RdBu, linear in ψ: blue < 0 < red. ψ is real on this plane (e^{imφ} = ±1)."}</Notation>
          ) : (
            <Notation>{currentCaption(basis, plane.meta.m)}</Notation>
          )}
        </p>
      )}
      {plane?.meta.model === "hf" && <p className="caption">{HF_ORBITAL_CAPTION}</p>}
    </div>
  );
}
