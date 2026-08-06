import { useEffect, useRef } from "react";
import { HF_ORBITAL_CAPTION } from "../lib/hfModel";
import { rasterize } from "../lib/rasterize";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

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
          {plane ? `${plane.meta.label} [${plane.meta.unit}]` : "2D cross-section (y = 0 plane)"}
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
        </div>
      </div>
      <div className="plane-frame">
        <canvas ref={canvasRef} className="plane-canvas" />
        {planeStatus === "sampling" && (
          <p className="hint">computing… {(planeProgress * 100).toFixed(0)}%</p>
        )}
        {planeStatus === "error" && error && <p className="error">{error}</p>}
      </div>
      {plane && (
        <p className="caption">
          x, z ∈ [−{plane.meta.half_extent.toFixed(1)}, +
          {plane.meta.half_extent.toFixed(1)}] bohr; z vertical (quantization axis).{" "}
          {plane.meta.quantity === "density"
            ? "Inferno brightness is γ-compressed, exponent 0.5 (VISUAL LIBERTY, reveals faint lobes)."
            : "Diverging RdBu, linear in ψ: blue < 0 < red. ψ is real on this plane (e^{imφ} = ±1)."}
        </p>
      )}
      {plane?.meta.model === "hf" && <p className="caption">{HF_ORBITAL_CAPTION}</p>}
    </div>
  );
}
