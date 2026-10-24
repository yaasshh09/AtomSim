/**
 * 3-D exports: the isosurface as binary STL, the point cloud as binary PLY.
 *
 * Both are written straight from the arrays the engine sent, so nothing about
 * them is resampled or smoothed on the way out. Coordinates stay in bohr: STL
 * has no unit field at all, and PLY's is free text, so the file names say
 * "bohr" and the PLY header carries it as a comment. A printer or CAD tool that
 * assumes millimetres will make a 1-bohr feature 1 mm, which is a fine scale
 * for a model and the wrong one for a measurement.
 */

/** Binary STL: 80-byte header, triangle count, then 50 bytes per triangle. */
export function stlBinary(
  vertices: Float32Array,
  triangles: Uint32Array,
  header = "AtomSim isosurface, units: bohr",
): ArrayBuffer {
  const count = Math.floor(triangles.length / 3);
  const buf = new ArrayBuffer(84 + 50 * count);
  const view = new DataView(buf);
  const head = new TextEncoder().encode(header.slice(0, 80));
  new Uint8Array(buf, 0, 80).set(head);
  view.setUint32(80, count, true);
  let o = 84;
  for (let t = 0; t < count; t++) {
    const ia = 3 * triangles[3 * t];
    const ib = 3 * triangles[3 * t + 1];
    const ic = 3 * triangles[3 * t + 2];
    const ux = vertices[ib] - vertices[ia];
    const uy = vertices[ib + 1] - vertices[ia + 1];
    const uz = vertices[ib + 2] - vertices[ia + 2];
    const vx = vertices[ic] - vertices[ia];
    const vy = vertices[ic + 1] - vertices[ia + 1];
    const vz = vertices[ic + 2] - vertices[ia + 2];
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len;
    ny /= len;
    nz /= len;
    for (const v of [nx, ny, nz]) {
      view.setFloat32(o, v, true);
      o += 4;
    }
    for (const i of [ia, ib, ic]) {
      for (let k = 0; k < 3; k++) {
        view.setFloat32(o, vertices[i + k], true);
        o += 4;
      }
    }
    view.setUint16(o, 0, true); // attribute byte count
    o += 2;
  }
  return buf;
}

/** Binary little-endian PLY of the sampled points, with |psi|^2 per point when given. */
export function plyPoints(positions: Float32Array, density?: Float32Array | null): ArrayBuffer {
  const count = Math.floor(positions.length / 3);
  const withDensity = !!density && density.length >= count;
  const header =
    "ply\nformat binary_little_endian 1.0\ncomment AtomSim point cloud, units: bohr\n" +
    (withDensity ? "comment density is |psi|^2 at the point, bohr^-3\n" : "") +
    `element vertex ${count}\nproperty float x\nproperty float y\nproperty float z\n` +
    (withDensity ? "property float density\n" : "") +
    "end_header\n";
  const head = new TextEncoder().encode(header);
  const stride = withDensity ? 16 : 12;
  const buf = new ArrayBuffer(head.length + stride * count);
  new Uint8Array(buf, 0, head.length).set(head);
  const view = new DataView(buf, head.length);
  for (let i = 0; i < count; i++) {
    view.setFloat32(stride * i, positions[3 * i], true);
    view.setFloat32(stride * i + 4, positions[3 * i + 1], true);
    view.setFloat32(stride * i + 8, positions[3 * i + 2], true);
    if (withDensity) view.setFloat32(stride * i + 12, (density as Float32Array)[i], true);
  }
  return buf;
}
