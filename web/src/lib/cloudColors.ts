import type { ColorMode } from "../state/store";
import { densityT, lutColor, maxOf, phaseColor } from "./colormap";
import { INFERNO } from "./luts";

/** The per-vertex RGB floats (0-1) for the cloud, or null when it is drawn in a solid colour. */
export function buildCloudColors(
  mode: ColorMode,
  density: Float32Array | null,
  phase: Float32Array | null,
): Float32Array | null {
  if (mode === "density" && density) {
    const vmax = maxOf(density);
    const out = new Float32Array(density.length * 3);
    for (let i = 0; i < density.length; i++) {
      const [r, g, b] = lutColor(INFERNO, densityT(density[i], vmax));
      out[3 * i] = r / 255;
      out[3 * i + 1] = g / 255;
      out[3 * i + 2] = b / 255;
    }
    return out;
  }
  if (mode === "phase" && phase) {
    const out = new Float32Array(phase.length * 3);
    for (let i = 0; i < phase.length; i++) {
      const [r, g, b] = phaseColor(phase[i]);
      out[3 * i] = r / 255;
      out[3 * i + 1] = g / 255;
      out[3 * i + 2] = b / 255;
    }
    return out;
  }
  return null;
}
