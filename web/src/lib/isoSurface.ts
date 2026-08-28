import type { IsoMeta } from "../api/types";
import { phaseColor } from "./colormap";

/**
 * The per-vertex RGB floats (0-1) I give the surface, from arg(psi) at each
 * vertex.
 *
 * I use the same `phaseColor` as my point cloud on purpose: a p orbital's two
 * lobes must be the same two colours whether I draw them as points or as a
 * shell, or my picture would say the representation changed the physics.
 *
 * This is also the step that makes my surface honest about what it is drawn
 * through. |psi|^2 is blind to sign, so if I coloured the mesh by nothing I
 * would hand you a single-coloured dumbbell, and a single-coloured dumbbell
 * teaches the wrong thing about bonding.
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
 * The sentence a textbook lobe is missing, said in the words of the
 * measurement.
 *
 * I build it from `enclosed_fraction` rather than from the fraction you asked
 * for: the grid delivers what it delivers, and I state that, not the ask.
 */
export function enclosedCaption(meta: IsoMeta): string {
  const inside = (meta.enclosed_fraction.value * 100).toFixed(1);
  const outside = (meta.outside_fraction * 100).toFixed(1);
  return `I enclose ${inside}% of the electron here, and it is outside this surface ${outside}% of the time`;
}

/**
 * How far back I put the camera to frame the whole surface.
 *
 * I measure it from the vertices rather than from the box: I fit the box to
 * hold 99.9% of the electron, which makes it several times larger than the
 * contour I draw inside it, so framing on the box would leave a 90% surface as
 * a dot in the middle.
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
 * Whether you can believe my mesh's component count as a claim about shape.
 *
 * My engine reports the pieces it actually cut, and near a node that number is
 * a property of the grid as much as of the orbital: lobes separated by less
 * than a cell come out fused. So I attach the caveat whenever the state has a
 * node that could be doing this, meaning any l > 0, and I state the count
 * plainly for an s state, which has no angular node to fuse across.
 */
export function componentsCaption(meta: IsoMeta): string {
  const pieces = `${meta.components} piece${meta.components === 1 ? "" : "s"}`;
  if (meta.l === 0) return pieces;
  return `${pieces} at the resolution I used (lobes closer than one cell come out joined)`;
}
