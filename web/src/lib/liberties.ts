import type { Provenance } from "../api/types";
import { MARKER_DIVISOR } from "./nucleus";

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
