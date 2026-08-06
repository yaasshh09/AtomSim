import { densityT, lutColor, maxAbs, maxOf, signedT } from "./colormap";
import { INFERNO, RDBU_R } from "./luts";

/**
 * PlaneGrid float32 values (row i = z ascending) -> RGBA pixels (row 0 = top).
 * density: inferno with the disclosed gamma compression (VISUAL LIBERTY).
 * psi: diverging RdBu_r, LINEAR in psi, zero at the midpoint, signed structure
 * is the honest point of that mode, so no gamma is applied.
 */
export function rasterize(
  values: Float32Array,
  resolution: number,
  quantity: "density" | "psi",
): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(resolution * resolution * 4);
  const vmax = quantity === "density" ? maxOf(values) : maxAbs(values);
  for (let row = 0; row < resolution; row++) {
    const src = resolution - 1 - row; // canvas row 0 is the top (+z)
    for (let col = 0; col < resolution; col++) {
      const v = values[src * resolution + col];
      const [r, g, b] =
        quantity === "density"
          ? lutColor(INFERNO, densityT(v, vmax))
          : lutColor(RDBU_R, signedT(v, vmax));
      const o = 4 * (row * resolution + col);
      out[o] = r;
      out[o + 1] = g;
      out[o + 2] = b;
      out[o + 3] = 255;
    }
  }
  return out;
}
