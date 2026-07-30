import type { Provenance } from "../api/types";
import { MARKER_DIVISOR } from "./nucleus";

/**
 * Render a provenance `error_estimate` at a precision it can actually support.
 *
 * These arrive as raw doubles, and printing one verbatim gives things like
 * "0.00034049718827628214" — an error bar quoted to twenty significant figures,
 * which claims a precision the error bar is the admission of not having. Two
 * figures is what an error scale carries; anything past that is noise wearing a
 * digit's clothes, and in this codebase overstating precision is the bug.
 *
 * Known gap, deliberately not papered over: `Provenance` carries no unit, while
 * the engine's contract (provenance.py) is that `error_estimate` is in the unit
 * of the quantity it describes. So the magnitude is shown without one. Fixing
 * that properly means giving Provenance a unit field in the engine and the
 * schema; inventing a unit here would be a guess printed as a fact.
 */
export function formatErrorScale(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return "0";
  // toPrecision(2) already switches to exponential on its own above 1e2 and
  // below 1e-6, and it is right to: "4300" for 4321 would assert two zeros it
  // does not have. The one place it is left wanting is 1e-6 to 1e-3, where it
  // gives "0.00034" and the leading zeros have to be counted, so that band is
  // sent to exponential too.
  return Math.abs(x) < 1e-3 ? x.toExponential(1) : x.toPrecision(2);
}

/** The frontend is the authority on its own rendering choices — disclosed, never hidden. */
export const RENDER_LIBERTIES: Provenance = {
  fidelity: "visual_liberty",
  method: "three.js point-sprite rendering of engine-sampled positions",
  assumptions: [
    "z quantization axis drawn screen-vertical (data stays xyz in bohr)",
    "point size, opacity and additive glow are presentation, not physics",
    "density colour brightness gamma-compressed: t = (rho/rho_max)^0.5",
  ],
  error_estimate: null,
  refinement: "positions, density and phase channels come from the engine unmodified",
};

export const NUCLEUS_MARKER_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "nucleus drawn as a fixed-size marker sphere at the origin",
  assumptions: [
    `marker radius = camera distance / ${MARKER_DIVISOR} — presentation, not physics`,
    "true position (the origin) and the r_rms readout are exact",
    "switch to 'true scale' to see the honest, subpixel size",
  ],
  error_estimate: null,
  refinement: "the magnification factor is stated live in the canvas caption",
};

/**
 * Max spiral windings drawn by the classical ghost overlay. The real revolution
 * count (~1e5) cannot be resolved by a sampled line or a 60 fps point — drawing
 * it would alias into noise — so the azimuthal winding count is capped for
 * display and disclosed here. Radius law, clock, and readouts stay exact.
 */
export const GHOST_DISPLAY_WINDINGS = 16;

export const CLASSICAL_SLOWMO: Provenance = {
  fidelity: "visual_liberty",
  method: "classical collapse shown in slow motion; the live clock shows real simulated time",
  assumptions: [
    "playback speed is a viewing choice, not physics",
    `spiral drawn with at most ${GHOST_DISPLAY_WINDINGS} windings — the honest revolution count is the orbits readout`,
  ],
  error_estimate: null,
  refinement: "the slow-motion factor is stated live in the ghost HUD",
};

/**
 * The trap here: a bar chart of spectral lines looks exactly like an observed
 * spectrum, so shading it by A invites the reading "this is how bright the line
 * is". It is not. Observed brightness is N_upper * A * h nu and depends on level
 * populations, which are not modelled unless the LTE control is on. What the
 * bars show without it is the spontaneous emission rate, log-compressed because
 * A spans about four decades across a hydrogen line list.
 */
/**
 * The synthesized curve is engine physics; the *y axis* it is drawn on is not.
 * A spectral emissivity spans many decades across a hydrogen line list, so a
 * linear trace shows Lyman-alpha and a flat floor. The full-range curve is
 * therefore log-compressed and clipped a fixed number of decades below its own
 * peak, which is a reading aid and has to say so: the zoomed panel plots the
 * same data linearly, where a line's true shape is the whole point.
 */
export const PROFILE_DECADES = 6;

export const SPECTRUM_PROFILE_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: `full-range profile drawn on log10 intensity, clipped ${PROFILE_DECADES} decades below the peak`,
  assumptions: [
    "the curve itself is engine-synthesized: Voigt profiles at engine widths",
    `anything fainter than 1e-${PROFILE_DECADES} of the peak is drawn at the floor, not at zero`,
    "log compression is a reading aid; zoom a line to see it plotted linearly",
    "wavelength axis is logarithmic, so a line's drawn width is not its shape",
  ],
  error_estimate: null,
  refinement: "the zoomed panel plots intensity linearly on a linear wavelength axis",
};

export const SPECTRUM_INTENSITY_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "bar height and opacity scaled by log10 of the Einstein A coefficient",
  assumptions: [
    "A (spontaneous emission rate, s^-1) is engine-computed and exact to quadrature roundoff",
    "bar height/opacity is a log-compressed rate, NOT a predicted observed intensity",
    "no level populations are modelled: turn on LTE weighting for those",
    "log compression is presentation; the decade range is printed in the caption",
  ],
  error_estimate: null,
  refinement: "LTE weighting (the temperature and density controls) turns rates into emissivities",
};

/**
 * With LTE weighting on, the bars are a modelled emissivity rather than a bare
 * rate, so the old "no populations here" disclosure no longer applies. What is
 * still a presentational choice is the log compression, and what is still not
 * an observed brightness is the number itself: the model is optically thin, so
 * a strong line in a thick medium would not really look like this. The physics
 * assumptions ride on each line's own emissivity provenance; this covers only
 * what the drawing adds on top.
 */
export const SPECTRUM_EMISSIVITY_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "bar height and opacity scaled by log10 of the LTE emissivity",
  assumptions: [
    "emissivity (eV/s per atom) is engine-computed from Boltzmann populations and Saha ionization",
    "an emissivity is still not an observed brightness: the model is optically thin",
    "log compression is presentation; the decade range is printed in the caption",
    "lines too faint to reach the floor of the scale are drawn at the floor, not dropped",
  ],
  error_estimate: null,
  refinement: "radiative transfer through a finite optical depth would give a predicted brightness",
};

/**
 * A Hartree-Fock ladder spans more than two decades of binding energy: argon's
 * 1s sits at -3227.5 eV and its 3p at -16.1 eV, a ratio of 200. Drawn linearly,
 * argon's entire valence shell (3s at -34.8 and 3p at -16.1, which is all of
 * its chemistry) falls inside the top 1.1% of the frame, about two pixels
 * apart, and the picture says "argon has a 1s and a smudge". So the axis is
 * logarithmic in binding energy.
 *
 * That buys readability and costs the zero: the ionization limit is infinitely
 * far up a log axis and cannot be drawn as a rung. It is marked at the top
 * edge and named as off-scale rather than quietly moved to a finite height,
 * because a line drawn at the top would read as a level that is there.
 */
export const HF_LADDER_AXIS_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "orbital ladder drawn on log10 of binding energy |ε| [eV], deepest at the bottom",
  assumptions: [
    "the energies themselves are engine values, unmodified — only the axis is compressed",
    "spacing on this axis is a ratio, not a difference: two rungs one decade apart differ 10x",
    "0 eV (the ionization limit) is off the top of a logarithmic axis, not at the top rung",
    "every occupied orbital is bound, so |ε| never crosses zero and the log is always defined",
  ],
  error_estimate: null,
  refinement: "the eV value of every rung is printed beside it, on the linear scale it was computed on",
};

export const THUMBNAIL_LIBERTY: Provenance = {
  fidelity: "visual_liberty",
  method: "server-rendered inferno PNG of |psi|^2 on the y=0 plane (navigation aid)",
  assumptions: [
    "brightness gamma-compressed: t = (rho/rho_max)^0.5",
    "not a measurement surface: no axes, no scale",
  ],
  error_estimate: null,
  refinement: "open the 2D cross-section view for the labeled, scaled version",
};
