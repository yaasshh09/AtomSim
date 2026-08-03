import type { IsoMeta } from "../api/types";
import { phaseColor } from "./colormap";

/**
 * Per-vertex RGB floats (0-1) for the surface, from arg(psi) at each vertex.
 *
 * The same `phaseColor` the point cloud uses, on purpose: a p orbital's two
 * lobes must be the same two colours whether they are drawn as points or as a
 * shell, or the picture would say the representation changed the physics.
 *
 * This is also the step that makes the surface honest about what it is drawn
 * through. |psi|^2 is blind to sign, so a mesh coloured by nothing is a
 * single-coloured dumbbell, and a single-coloured dumbbell teaches the wrong
 * thing about bonding.
 */
export function buildSurfaceColors(phase: Float32Array): Float32Array {
  const out = new Float32Array(phase.length * 3);
  for (let i = 0; i < phase.length; i++) {
    const [r, g, b] = phaseColor(phase[i]);
    out[3 * i] = r / 255;
    out[3 * i + 1] = g / 255;
    out[3 * i + 2] = b / 255;
  }
  return out;
}

/**
 * The sentence a textbook lobe is missing, in the words of the measurement.
 *
 * Built from `enclosed_fraction` rather than from the requested fraction: the
 * grid delivers what it delivers, and the caption states that, not the ask.
 */
export function enclosedCaption(meta: IsoMeta): string {
  const inside = (meta.enclosed_fraction.value * 100).toFixed(1);
  const outside = (meta.outside_fraction * 100).toFixed(1);
  return `encloses ${inside}% of the electron — it is outside this surface ${outside}% of the time`;
}

/**
 * How far the camera should sit to frame the whole surface.
 *
 * From the vertices rather than from the box: the box is fitted to hold 99.9%
 * of the electron and is several times larger than the contour drawn in it, so
 * framing on the box would leave a 90% surface as a dot in the middle.
 */
export function surfaceExtent(vertices: Float32Array): number {
  let max = 0;
  for (let i = 0; i < vertices.length; i += 3) {
    const r = Math.hypot(vertices[i], vertices[i + 1], vertices[i + 2]);
    if (r > max) max = r;
  }
  return max;
}

/**
 * Whether the mesh's component count can be believed as a shape claim.
 *
 * The engine reports the pieces it actually cut, and near a node that number is
 * a property of the grid as much as of the orbital: lobes separated by less
 * than a cell come out fused. So the count is shown with the caveat attached
 * whenever the state has a node that could be doing this — any l > 0 — and
 * shown plainly for an s state, which has no angular node to fuse across.
 */
export function componentsCaption(meta: IsoMeta): string {
  const pieces = `${meta.components} piece${meta.components === 1 ? "" : "s"}`;
  if (meta.l === 0) return pieces;
  return `${pieces} at this resolution (lobes closer than one cell come out joined)`;
}
