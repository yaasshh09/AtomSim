/**
 * The one rotation I use to take engine coordinates onto the screen.
 *
 * My engine emits (x, y, z) in bohr with **z** as the quantization axis.
 * three.js puts +y up, so if I drew the data unrotated I would stand the
 * quantization axis out of the screen and every |m|-dependent shape would read
 * end-on. Rotating -90 degrees about x maps engine z onto three.js +y, which is
 * what I mean in `RENDER_LIBERTIES` by "z quantization axis drawn
 * screen-vertical".
 *
 * I export it rather than writing it out at each use site because it is a
 * shared convention, and a renderer that disagrees with its neighbour about
 * which way is up mislabels the physics rather than merely looking wrong. My
 * axis triad is the case that proves it: drawn in three.js coordinates it
 * labelled the screen-vertical axis "y", directly contradicting the cloud
 * beside it and the liberty note under it.
 */
export const PHYSICS_TO_SCREEN: [number, number, number] = [-Math.PI / 2, 0, 0];
