/**
 * The engine's float32 grid as 8-bit levels of the gamma-compressed density,
 * reordered for a WebGL 3-D texture.
 *
 * The engine stores [i, j, k] at (x_i, y_j, z_k) with k fastest. A Data3DTexture
 * wants its width axis fastest, so the texture's (u, v, w) here are (z, y, x),
 * and the shader samples with the coordinates swapped back. The level is
 * round(255 sqrt(rho / rho_max)): the same t the cloud's colours use.
 */
export function volumeTexels(values: Float32Array, n: number): Uint8Array {
  if (values.length !== n * n * n) {
    throw new Error(`expected ${n}^3 values, got ${values.length}`);
  }
  let max = 0;
  for (let i = 0; i < values.length; i++) if (values[i] > max) max = values[i];
  const out = new Uint8Array(values.length);
  if (!(max > 0)) return out;
  for (let i = 0; i < values.length; i++) {
    out[i] = Math.round(255 * Math.sqrt(Math.max(values[i], 0) / max));
  }
  return out;
}
