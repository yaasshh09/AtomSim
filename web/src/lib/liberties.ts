import type { Provenance } from "../api/types";
import { MARKER_DIVISOR } from "./nucleus";

/**
 * I render a provenance `error_estimate` at a precision it can actually
 * support.
 *
 * These reach me as raw doubles, and if I printed one verbatim I would write
 * things like "0.00034049718827628214", an error bar quoted to twenty
 * significant figures, claiming a precision the error bar is the admission of
 * not having. Two figures is what an error scale carries; anything past that
 * is noise wearing a digit's clothes, and overstating precision is the one bug
 * I refuse to ship.
 *
 * A known gap I will not paper over: `Provenance` carries no unit, while my
 * engine's contract (provenance.py) is that `error_estimate` is in the unit of
 * the quantity it describes. So I show the magnitude without one. Fixing that
 * properly means giving Provenance a unit field in the engine and the schema;
 * if I invented a unit here I would be printing a guess as a fact.
 */
export function formatErrorScale(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return "0";
  // toPrecision(2) already switches to exponential on its own above 1e2 and
  // below 1e-6, and it is right to: "4300" for 4321 would assert two zeros it
  // does not have. The one place it leaves me wanting is 1e-6 to 1e-3, where
  // it gives "0.00034" and you have to count the leading zeros, so I send that
  // band to exponential too.
  return Math.abs(x) < 1e-3 ? x.toExponential(1) : x.toPrecision(2);
}

/** I am the authority on my own rendering choices, and I disclose them rather than hide them. */
export const RENDER_LIBERTIES: Provenance = {
  fidelity: "visual_liberty",
  method: "I render engine-sampled positions as three.js point sprites",
  assumptions: [
    "I draw the z quantization axis screen-vertical (my data stays xyz in bohr)",
    "point size, opacity and additive glow are my presentation, not physics",
    "I gamma-compress density colour brightness: t = (rho/rho_max)^0.5",
  ],
  error_estimate: null,
  refinement: "I take the positions, density and phase channels from the engine unmodified",
};

export const NUCLEUS_MARKER_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "I draw the nucleus as a fixed-size marker sphere at the origin",
  assumptions: [
    `I set the marker radius to camera distance / ${MARKER_DIVISOR}, which is presentation, not physics`,
    "the true position (the origin) and the r_rms I read out are exact",
    "switch to 'true scale' and I will show you the honest, subpixel size",
  ],
  error_estimate: null,
  refinement: "I state the magnification factor live in the canvas caption",
};

/**
 * The most spiral windings I draw in the classical ghost overlay. Neither a
 * sampled line nor a 60 fps point can resolve the real revolution count
 * (~1e5), and if I drew it I would alias it into noise, so I cap the azimuthal
 * winding count for display and disclose it here. My radius law, clock and
 * readouts stay exact.
 */
export const GHOST_DISPLAY_WINDINGS = 16;

export const CLASSICAL_SLOWMO: Provenance = {
  fidelity: "visual_liberty",
  method: "I show the classical collapse in slow motion; my live clock shows real simulated time",
  assumptions: [
    "the playback speed is a viewing choice I made, not physics",
    `I draw the spiral with at most ${GHOST_DISPLAY_WINDINGS} windings; the honest revolution count is my orbits readout`,
  ],
  error_estimate: null,
  refinement: "I state the slow-motion factor live in the ghost HUD",
};

/**
 * The trap I have to stay out of here: my bar chart of spectral lines looks
 * exactly like an observed spectrum, so shading it by A invites you to read it
 * as "this is how bright the line is". It is not. Observed brightness is
 * N_upper * A * h nu and depends on level populations, which I do not model
 * unless you turn the LTE control on. Without it, what my bars show is the
 * spontaneous emission rate, log-compressed because A spans about four decades
 * across a hydrogen line list.
 */
/**
 * The curve I synthesize is engine physics; the *y axis* I draw it on is not.
 * A spectral emissivity spans many decades across a hydrogen line list, so a
 * linear trace would show you Lyman-alpha and a flat floor. So I log-compress
 * the full-range curve and clip it a fixed number of decades below its own
 * peak, which is a reading aid and has to say so: my zoomed panel plots the
 * same data linearly, where a line's true shape is the whole point.
 */
export const PROFILE_DECADES = 6;

export const SPECTRUM_PROFILE_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: `I draw the full-range profile on log10 intensity, clipped ${PROFILE_DECADES} decades below the peak`,
  assumptions: [
    "I synthesize the curve itself in the engine: Voigt profiles at engine widths",
    `I draw anything fainter than 1e-${PROFILE_DECADES} of the peak at the floor, not at zero`,
    "my log compression is a reading aid; zoom a line and I plot it linearly",
    "my wavelength axis is logarithmic, so a line's drawn width is not its shape",
  ],
  error_estimate: null,
  refinement: "in my zoomed panel I plot intensity linearly on a linear wavelength axis",
};

export const SPECTRUM_INTENSITY_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "I scale bar height and opacity by log10 of the Einstein A coefficient",
  assumptions: [
    "I compute A (the spontaneous emission rate, s^-1) in the engine, exact to quadrature roundoff",
    "my bar height and opacity are a log-compressed rate, NOT a predicted observed intensity",
    "I model no level populations here: turn on LTE weighting and I will",
    "the log compression is my presentation; I print the decade range in the caption",
  ],
  error_estimate: null,
  refinement: "LTE weighting (my temperature and density controls) turns these rates into emissivities",
};

/**
 * With LTE weighting on my bars are a modelled emissivity rather than a bare
 * rate, so my old "no populations here" disclosure no longer applies. What is
 * still a presentational choice is the log compression, and what is still not
 * an observed brightness is the number itself: my model is optically thin, so
 * a strong line in a thick medium would not really look like this. The physics
 * assumptions ride on each line's own emissivity provenance; here I cover only
 * what my drawing adds on top.
 */
export const SPECTRUM_EMISSIVITY_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "I scale bar height and opacity by log10 of the LTE emissivity",
  assumptions: [
    "I compute the emissivity (eV/s per atom) in the engine from Boltzmann populations and Saha ionization",
    "an emissivity is still not an observed brightness: my model is optically thin",
    "the log compression is my presentation; I print the decade range in the caption",
    "I draw lines too faint to reach the floor of the scale at the floor rather than dropping them",
  ],
  error_estimate: null,
  refinement: "radiative transfer through a finite optical depth would give you a predicted brightness",
};

/**
 * A Hartree-Fock ladder spans more than two decades of binding energy: argon's
 * 1s sits at -3227.5 eV and its 3p at -16.1 eV, a ratio of 200. If I drew that
 * linearly, argon's entire valence shell (3s at -34.8 and 3p at -16.1, which
 * is all of its chemistry) would fall inside the top 1.1% of the frame, about
 * two pixels apart, and my picture would say "argon has a 1s and a smudge". So
 * I make the axis logarithmic in binding energy.
 *
 * That buys readability and costs me the zero: the ionization limit is
 * infinitely far up a log axis and I cannot draw it as a rung. I mark it at
 * the top edge and name it as off-scale rather than quietly moving it to a
 * finite height, because a line drawn at the top would read as a level that is
 * there.
 */
export const HF_LADDER_AXIS_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "I draw the orbital ladder on log10 of binding energy |ε| [eV], deepest at the bottom",
  assumptions: [
    "the energies themselves are engine values I leave unmodified; I compress only the axis",
    "spacing on my axis is a ratio, not a difference: two rungs one decade apart differ 10x",
    "0 eV (the ionization limit) is off the top of a logarithmic axis, not at my top rung",
    "every occupied orbital is bound, so |ε| never crosses zero and my log is always defined",
  ],
  error_estimate: null,
  refinement: "I print the eV value of every rung beside it, on the linear scale I computed it on",
};

/**
 * What I add on top of the engine's triangles.
 *
 * The mesh itself is measured: its vertices sit where linear interpolation
 * puts the level on a cell edge, which is my engine's business and carries the
 * engine's provenance. Three things here are mine, and each of them makes the
 * surface look more certain than it is. My smooth shading averages normals
 * across facets, which hides the faceting that is the honest signature of a
 * finite grid. A lit opaque shell says "edge" to the eye, which is the exact
 * misreading my enclosed-fraction readout exists to prevent. And I draw the
 * surface slightly transparent so the cloud shows through it, which is a
 * choice about legibility rather than about the atom.
 */
export const ISOSURFACE_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "I light a three.js mesh of engine-extracted triangles, smooth-shaded and translucent",
  assumptions: [
    "I average vertex normals across facets, so I smooth away the faceting a finite grid really has",
    "my lighting and translucency are presentation; the geometry is the engine's, unmodified",
    "a solid-looking shell is not an edge; the enclosed fraction I print beside it is the whole claim",
    "I colour each vertex by arg(psi) through the same map I use for the point cloud",
  ],
  error_estimate: null,
  refinement: "I print the enclosed fraction, its grid-halving error and the escaped mass live",
};

export const THUMBNAIL_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "I render this inferno PNG of |psi|^2 on the y=0 plane server-side, as a navigation aid",
  assumptions: [
    "I gamma-compress the brightness: t = (rho/rho_max)^0.5",
    "this is not a measurement surface: I give it no axes and no scale",
  ],
  error_estimate: null,
  refinement: "open my 2D cross-section view and I will label and scale it",
};
