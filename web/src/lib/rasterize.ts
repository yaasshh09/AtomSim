import { densityT, lutColor, maxAbs, maxOf, phaseColor, signedT } from "./colormap";
import { INFERNO, RDBU_R } from "./luts";

/**
 * Turns PlaneGrid float32 values (row i = z ascending) into RGBA pixels
 * (row 0 = top).
 * Density gets inferno with the disclosed gamma compression (VISUAL LIBERTY).
 * psi and the current get diverging RdBu_r, LINEAR with zero at the midpoint.
 * The signed structure is the honest point of those modes, so no gamma is
 * applied to them.
 */
export function rasterize(
  values: Float32Array,
  resolution: number,
  quantity: "density" | "psi" | "current",
  /** A fixed colour scale. Omitted, the frame's own maximum. */
  scale?: number,
): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(resolution * resolution * 4);
  const vmax = scale ?? (quantity === "density" ? maxOf(values) : maxAbs(values));
  for (let row = 0; row < resolution; row++) {
    const src = resolution - 1 - row; // +z goes at canvas row 0, the top
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

/**
 * A complex field as hue and brightness: hue is arg Psi on the same wheel as
 * the point cloud's phase colouring, brightness is |Psi|^2 through the same
 * gamma-compressed scale the density mode uses, so a lobe is as bright in
 * either mode and only its colour says more.
 */
export function rasterizePhase(
  re: Float32Array,
  im: Float32Array,
  resolution: number,
  scale: number,
): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(resolution * resolution * 4);
  for (let row = 0; row < resolution; row++) {
    const src = resolution - 1 - row;
    for (let col = 0; col < resolution; col++) {
      const k = src * resolution + col;
      const v = densityT(re[k] * re[k] + im[k] * im[k], scale);
      const [r, g, b] = phaseColor(Math.atan2(im[k], re[k]));
      const o = 4 * (row * resolution + col);
      out[o] = r * v;
      out[o + 1] = g * v;
      out[o + 2] = b * v;
      out[o + 3] = 255;
    }
  }
  return out;
}
