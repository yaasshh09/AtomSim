import type { Provenance } from "../api/types";
import { MARKER_DIVISOR } from "./nucleus";

/**
 * Render a provenance `error_estimate` at a precision it can actually support.
 *
 * These arrive as raw doubles, and printing one verbatim would give
 * "0.00034049718827628214": an error bar quoted to twenty significant figures,
 * claiming exactly the precision it exists to admit not having. Two figures is
 * what an error scale carries; anything past that is noise wearing a digit's
 * clothes, and overstating precision is the one bug this app must never ship.
 *
 * A known gap, left visible rather than papered over: `Provenance` carries no
 * unit, while the engine's contract (provenance.py) is that `error_estimate` is
 * in the unit of the quantity it describes. So the magnitude shows without one.
 * Fixing it properly means giving Provenance a unit field in the engine and the
 * schema; inventing a unit here would print a guess as a fact.
 */
export function formatErrorScale(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return "0";
  // toPrecision(2) already switches to exponential on its own above 1e2 and
  // below 1e-6, and it is right to: "4300" for 4321 would assert two zeros it
  // does not have. The one place it falls short is 1e-6 to 1e-3, where it gives
  // "0.00034" and the reader has to count leading zeros, so that band goes to
  // exponential too.
  return Math.abs(x) < 1e-3 ? x.toExponential(1) : x.toPrecision(2);
}

/** This module is the authority on the app's rendering choices, and it discloses them rather than hiding them. */
export const RENDER_LIBERTIES: Provenance = {
  fidelity: "visual_liberty",
  method: "engine-sampled positions rendered as three.js point sprites",
  assumptions: [
    "the z quantization axis is drawn screen-vertical (the data stays xyz in bohr)",
    "point size, opacity and additive glow are presentation, not physics",
    "density colour brightness is gamma-compressed: t = (rho/rho_max)^0.5",
  ],
  error_estimate: null,
  refinement: "the positions, density and phase channels come from the engine unmodified",
};

export const NUCLEUS_MARKER_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "the nucleus is drawn as a fixed-size marker sphere at the origin",
  assumptions: [
    `the marker radius is camera distance / ${MARKER_DIVISOR}, which is presentation, not physics`,
    "the true position (the origin) and the r_rms in the readout are exact",
    "switch to 'true scale' for the honest, subpixel size",
  ],
  error_estimate: null,
  refinement: "the magnification factor is stated live in the canvas caption",
};

/**
 * The most spiral windings the classical ghost overlay draws. Neither a sampled
 * line nor a 60 fps point can resolve the real revolution count (~1e5), and
 * drawing it would alias into noise, so the azimuthal winding count is capped
 * for display and disclosed here. The radius law, clock and readouts stay
 * exact.
 */
export const GHOST_DISPLAY_WINDINGS = 16;

export const CLASSICAL_SLOWMO: Provenance = {
  fidelity: "visual_liberty",
  method: "the classical collapse plays in slow motion; the live clock shows real simulated time",
  assumptions: [
    "the playback speed is a viewing choice, not physics",
    `the spiral is drawn with at most ${GHOST_DISPLAY_WINDINGS} windings; the honest revolution count is in the orbits readout`,
  ],
  error_estimate: null,
  refinement: "the slow-motion factor is stated live in the ghost HUD",
};

/**
 * The trap to stay out of here: a bar chart of spectral lines looks exactly
 * like an observed spectrum, so shading it by A invites the reading "this is
 * how bright the line is". It is not. Observed brightness is N_upper * A * h nu
 * and depends on level populations, which are modelled only when the LTE
 * control is on. Without it, the bars show the spontaneous emission rate,
 * log-compressed because A spans about four decades across a hydrogen line
 * list.
 */
/**
 * The synthesized curve is engine physics; the *y axis* it is drawn on is not.
 * A spectral emissivity spans many decades across a hydrogen line list, so a
 * linear trace would show Lyman-alpha and a flat floor. Hence the full-range
 * curve is log-compressed and clipped a fixed number of decades below its own
 * peak, which is a reading aid and has to say so: the zoomed panel plots the
 * same data linearly, where a line's true shape is the whole point.
 */
export const PROFILE_DECADES = 6;

export const SPECTRUM_PROFILE_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: `the full-range profile is drawn on log10 intensity, clipped ${PROFILE_DECADES} decades below the peak`,
  assumptions: [
    "the curve itself is synthesized in the engine: Voigt profiles at engine widths",
    `anything fainter than 1e-${PROFILE_DECADES} of the peak is drawn at the floor, not at zero`,
    "the log compression is a reading aid; zoom a line and it is plotted linearly",
    "the wavelength axis is logarithmic, so a line's drawn width is not its shape",
  ],
  error_estimate: null,
  refinement: "the zoomed panel plots intensity linearly on a linear wavelength axis",
};

export const SPECTRUM_INTENSITY_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "bar height and opacity scale with log10 of the Einstein A coefficient",
  assumptions: [
    "A (the spontaneous emission rate, s^-1) is computed in the engine, exact to quadrature roundoff",
    "bar height and opacity are a log-compressed rate, NOT a predicted observed intensity",
    "no level populations are modelled here: turn on LTE weighting for those",
    "the log compression is presentation; the decade range is printed in the caption",
  ],
  error_estimate: null,
  refinement: "LTE weighting (the temperature and density controls) turns these rates into emissivities",
};

/**
 * With LTE weighting on, the bars are a modelled emissivity rather than a bare
 * rate, so the "no populations here" disclosure no longer applies. What is
 * still a presentational choice is the log compression, and what is still not
 * an observed brightness is the number itself: the model is optically thin, so
 * a strong line in a thick medium would not really look like this. The physics
 * assumptions ride on each line's own emissivity provenance; this covers only
 * what the drawing adds on top.
 */
export const SPECTRUM_EMISSIVITY_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "bar height and opacity scale with log10 of the LTE emissivity",
  assumptions: [
    "the emissivity (eV/s per atom) is computed in the engine from Boltzmann populations and Saha ionization",
    "an emissivity is still not an observed brightness: the model is optically thin",
    "the log compression is presentation; the decade range is printed in the caption",
    "lines too faint to reach the floor of the scale are drawn at the floor rather than dropped",
  ],
  error_estimate: null,
  refinement: "radiative transfer through a finite optical depth would give a predicted brightness",
};

/**
 * A Hartree-Fock ladder spans more than two decades of binding energy: argon's
 * 1s sits at -3227.5 eV and its 3p at -16.1 eV, a ratio of 200. Drawn linearly,
 * argon's entire valence shell (3s at -34.8 and 3p at -16.1, which is all of
 * its chemistry) would fall inside the top 1.1% of the frame, about two pixels
 * apart, and the picture would say "argon has a 1s and a smudge". Hence a
 * logarithmic axis in binding energy.
 *
 * That buys readability and costs the zero: the ionization limit is infinitely
 * far up a log axis and cannot be drawn as a rung. It is marked at the top edge
 * and named as off-scale rather than quietly moved to a finite height, because
 * a line drawn at the top would read as a level that is there.
 */
export const HF_LADDER_AXIS_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "the orbital ladder is drawn on log10 of binding energy |ε| [eV], deepest at the bottom",
  assumptions: [
    "the energies themselves are engine values, left unmodified; only the axis is compressed",
    "spacing on this axis is a ratio, not a difference: two rungs one decade apart differ 10x",
    "0 eV (the ionization limit) is off the top of a logarithmic axis, not at the top rung",
    "every occupied orbital is bound, so |ε| never crosses zero and the log is always defined",
  ],
  error_estimate: null,
  refinement: "every rung prints its eV value beside it, on the linear scale it was computed on",
};

/**
 * What this view adds on top of the engine's triangles.
 *
 * The mesh itself is measured: its vertices sit where linear interpolation puts
 * the level on a cell edge, which is the engine's business and carries the
 * engine's provenance. Three things here belong to the renderer, and each makes
 * the surface look more certain than it is. Smooth shading averages normals
 * across facets, which hides the faceting that is the honest signature of a
 * finite grid. A lit opaque shell says "edge" to the eye, which is the exact
 * misreading the enclosed-fraction readout exists to prevent. And the surface
 * is drawn slightly transparent so the cloud shows through it, which is a
 * choice about legibility rather than about the atom.
 */
export const ISOSURFACE_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "a three.js mesh of engine-extracted triangles, lit, smooth-shaded and translucent",
  assumptions: [
    "vertex normals are averaged across facets, smoothing away the faceting a finite grid really has",
    "the lighting and translucency are presentation; the geometry is the engine's, unmodified",
    "a solid-looking shell is not an edge; the enclosed fraction printed beside it is the whole claim",
    "each vertex is coloured by arg(psi) through the same map the point cloud uses",
  ],
  error_estimate: null,
  refinement: "the enclosed fraction, its grid-halving error and the escaped mass are printed live",
};

export const THUMBNAIL_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "an inferno PNG of |psi|^2 on the y=0 plane, rendered server-side as a navigation aid",
  assumptions: [
    "the brightness is gamma-compressed: t = (rho/rho_max)^0.5",
    "this is not a measurement surface: it carries no axes and no scale",
  ],
  error_estimate: null,
  refinement: "the 2D cross-section view shows the same plane labelled and to scale",
};
