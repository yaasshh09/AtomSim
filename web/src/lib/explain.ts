/**
 * Plain-English scaffolding for my instrument views.
 *
 * Everything here is presentation copy, never physics: nothing in this file
 * computes a quantity, and I never let it. What these strings do is answer the
 * two questions you have before you can read a plot at all, "what is this" and
 * "is that number big", and I answer the second one by naming a thing you have
 * already met rather than by rescaling anything.
 *
 * I choose the anchors so you can check them. I say "about an MRI scanner" for
 * 1.5-3 T because clinical scanners are 1.5-3 T; I draw the field-ionization
 * line at 25 MV/m because F_ion ~ 1/(16 n^4) in atomic units puts n=6 hydrogen
 * there, and n=6 is the top of my shell picker. An anchor that flattered a
 * number instead of locating it would be the same class of bug as a silent
 * unit conversion, so I keep a reason next to each one.
 */

/** One view's opening card: what I plot there, and what to look at first. */
export interface ViewLead {
  /** Plain-English name, not the axis label. */
  title: string;
  /** One sentence. What question I answer in this view. */
  lead: string;
  /** What I think you should actually look at first. Optional. */
  notice?: string;
}

export const VIEW_LEADS: Record<string, ViewLead> = {
  radial: {
    title: "Radial functions",
    lead:
      "How far the electron sits from the nucleus, asked two different ways: " +
      "I draw the wavefunction's radial part, and the probability of finding " +
      "the electron at each distance.",
    notice:
      "I let the two plots disagree at r = 0 on purpose. Hover any curve and " +
      "I will read out its value; watch the number of wiggles too, because I " +
      "always give you n − ℓ − 1 of them.",
  },
  levels: {
    title: "Energy levels",
    lead:
      "The ladder of energies I allow this atom to have. Rungs are levels, " +
      "and the gaps between them are the colours of light it emits.",
    notice:
      "Pick a detail below the ladder and I will magnify what splits a rung: " +
      "relativity, a magnetic field, an electric field, or the nucleus itself.",
  },
  spectrum: {
    title: "Spectrum, checked against NIST",
    lead:
      "Every colour of light this atom can emit. I compute them from the " +
      "level ladder and draw them beside the measured wavelengths in NIST's " +
      "atomic spectra database.",
    notice:
      "The dots on the axis are the measurements, not mine. My agreement " +
      "readout says how many of my lines land inside the stated tolerance.",
  },
  whatif: {
    title: "What if the constants were different?",
    lead:
      "Change ℏ, e, mₑ, ε₀ or c and watch which observable quantities move. " +
      "For most combinations I have nothing to show you, because nothing you " +
      "could ever measure changes.",
    notice:
      "Try a scenario below. The lesson is in what stays put: only " +
      "dimensionless ratios and fixed-ruler scales are observable at all.",
  },
  forcelaw: {
    title: "What if the force law were different?",
    lead:
      "Replace the −Z/r Coulomb attraction with something else and I solve " +
      "the same Schrödinger equation against it. The level ladder is what " +
      "changes.",
    notice:
      "I draw each potential beside the honest reference it should be " +
      "compared with, so you can see what the swap actually cost.",
  },
};

/* ------------------------------------------------------------------ */
/* Magnitude anchors                                                    */
/* ------------------------------------------------------------------ */

/**
 * I describe a magnetic field in terms of a magnet you have met.
 *
 * Clinical MRI scanners run 1.5-3 T, which is the anchor most readers own;
 * steady laboratory fields top out around 45 T, so I call 20 T (my slider's
 * max) a research magnet and not the strongest in the world.
 */
export function describeField(tesla: number): string {
  if (tesla <= 0) return "no field";
  if (tesla < 0.5) return "weaker than a lab electromagnet";
  if (tesla < 1.5) return "a strong lab electromagnet";
  if (tesla <= 3.5) return "about an MRI scanner";
  if (tesla <= 12) return "a research superconducting magnet";
  return "among the strongest steady fields built";
}

/**
 * An electric field, which I anchor at the two thresholds that matter here.
 *
 * Dry air breaks down near 3 MV/m, so below that I am describing a field you
 * could set up between electrodes in a room. My upper anchor is field
 * ionization: hydrogen's classical threshold is F ~ 1/(16 n^4) in atomic
 * units, and one atomic unit of field is 5.14e5 MV/m, which puts n=6 (my
 * largest shell) at roughly 25 MV/m.
 */
export function describeEField(mvPerM: number): string {
  if (mvPerM <= 0) return "no field";
  if (mvPerM < 3) return "below air's breakdown strength";
  if (mvPerM < 25) return "past air's breakdown; a vacuum or solid-state field";
  return "strong enough to strip hydrogen's outer shells";
}

/** A temperature, which I anchor on things that emit light. */
export function describeTemperature(kelvin: number): string {
  if (kelvin <= 400) return "around room temperature";
  if (kelvin <= 2000) return "a flame";
  if (kelvin <= 7500) return "a Sun-like photosphere";
  if (kelvin <= 30000) return "a hot blue star";
  return "a stellar corona";
}

/** An electron density, which I name by the astrophysical object that has it. */
export function describeDensity(logNe: number): string {
  if (logNe <= 7) return "a nebula";
  if (logNe <= 11) return "a thin stellar atmosphere";
  if (logNe <= 14) return "a stellar photosphere";
  return "a dense laboratory plasma";
}

/** A spectrograph's resolving power, which I name by the instrument that reaches it. */
export function describeResolvingPower(logR: number | null): string {
  if (logR === null) return "no instrument: the line's own shape";
  if (logR < 3) return "a teaching spectrometer";
  if (logR < 4.5) return "a survey spectrograph";
  if (logR < 6) return "a high-resolution echelle";
  return "beyond what most spectrographs reach";
}

/* ------------------------------------------------------------------ */
/* NIST agreement                                                       */
/* ------------------------------------------------------------------ */

/** The shape my SpectrumView comparison rows have, narrowed to what I read. */
export interface ToleranceRow {
  within_tolerance: boolean;
}

export interface NistSummary {
  within: number;
  total: number;
  /** True when every line I compared is inside the stated tolerance. */
  allWithin: boolean;
  headline: string;
}

/**
 * How the line list I computed did against the vendored measurements.
 *
 * My residual plot has always carried this, one dot at a time. Saying it in a
 * sentence is not new information and not a claim beyond the dots: it is the
 * same count, read out loud, so that "vs NIST" has an answer you can see
 * without decoding a scatter.
 */
export function nistSummary(
  rows: readonly ToleranceRow[] | null | undefined,
  tolerance: number | null | undefined,
): NistSummary | null {
  if (!rows || rows.length === 0) return null;
  const total = rows.length;
  const within = rows.filter((r) => r.within_tolerance).length;
  const allWithin = within === total;
  const tol =
    typeof tolerance === "number" ? ` (tolerance ${tolerance.toExponential(0)})` : "";
  const headline = allWithin
    ? `All ${total} computed line${total === 1 ? "" : "s"} agree with NIST${tol}.`
    : `${within} of ${total} computed lines agree with NIST${tol}; ` +
      `${total - within} sit${total - within === 1 ? "s" : ""} outside it.`;
  return { within, total, allWithin, headline };
}

/* ------------------------------------------------------------------ */
/* What-If scenarios                                                    */
/* ------------------------------------------------------------------ */

/** Multipliers on the five raw constants, mirroring ConstMultipliers. */
export interface ScenarioMultipliers {
  hbar: number;
  e: number;
  m_e: number;
  eps0: number;
  c: number;
}

export interface Scenario {
  key: string;
  label: string;
  /** What I want you to watch once it is applied. */
  blurb: string;
  multipliers: ScenarioMultipliers;
}

const REAL: ScenarioMultipliers = { hbar: 1, e: 1, m_e: 1, eps0: 1, c: 1 };

/**
 * Prepared universes, each one chosen to make a single point and no other.
 *
 * I do not pick these multipliers for decoration. "Same universe in disguise"
 * is e ×2 with ε₀ ×4 because α ∝ e²/ε₀, a₀ ∝ ε₀/e² and E_h ∝ e⁴/ε₀², so every
 * exponent cancels and all three of my readouts must come back unchanged; if
 * one of them moves, my engine and this table disagree and the engine is
 * right. "Past the model's edge" multiplies α by 4·2·4·4 = 128, the one entry
 * where I land outside the perturbative range on purpose.
 */
export const SCENARIOS: Scenario[] = [
  {
    key: "real",
    label: "Our universe",
    blurb: "Every constant at its measured value. The baseline I come back to.",
    multipliers: REAL,
  },
  {
    key: "disguise",
    label: "The same universe in disguise",
    blurb:
      "Two constants doubled and quadrupled so that every exponent cancels. " +
      "I have nothing observable to move, which is the whole lesson.",
    multipliers: { ...REAL, e: 2, eps0: 4 },
  },
  {
    key: "strong-em",
    label: "Stronger electromagnetism",
    blurb:
      "Charge doubled. α goes up by four, atoms shrink, binding jumps, and " +
      "I can show you fine structure you could actually see.",
    multipliers: { ...REAL, e: 2 },
  },
  {
    key: "heavy-electron",
    label: "A heavier electron",
    blurb:
      "Mass doubled. α does not care, but the atom halves in size and " +
      "doubles in binding: this is what a muon actually does.",
    multipliers: { ...REAL, m_e: 2 },
  },
  {
    key: "broken",
    label: "Past the model's edge",
    blurb:
      "α pushed above 0.5, where my perturbative fine structure stops " +
      "meaning anything. I say so instead of drawing it.",
    multipliers: { hbar: 0.25, e: 2, m_e: 1, eps0: 0.5, c: 0.25 },
  },
];

/** The scenario whose multipliers are currently applied, or null for none. */
export function activeScenario(current: ScenarioMultipliers): Scenario | null {
  return (
    SCENARIOS.find((s) =>
      (Object.keys(s.multipliers) as (keyof ScenarioMultipliers)[]).every(
        (k) => Math.abs(s.multipliers[k] - current[k]) < 1e-9,
      ),
    ) ?? null
  );
}

/* ------------------------------------------------------------------ */
/* Force-law presets                                                    */
/* ------------------------------------------------------------------ */

/**
 * What each preset potential is, in a sentence, and why I think anyone cares.
 *
 * I key these by the same strings as PRESET_LABELS in lib/forceLaw.ts, which
 * stays my source of truth for the formulas themselves.
 */
export const PRESET_BLURBS: Record<string, string> = {
  powerlaw:
    "Coulomb with the exponent turned into a dial. At p = 1 I am solving " +
    "hydrogen exactly; move it and the accidental ℓ-degeneracy that makes 2s " +
    "and 2p share an energy breaks immediately.",
  yukawa:
    "Coulomb cut off past a range λ, the shape a screened charge really has. " +
    "Short-ranged potentials hold only finitely many bound states, so the top " +
    "of my ladder simply stops existing.",
  harmonic:
    "A spring instead of a charge. I get levels that are evenly spaced rather " +
    "than crowding toward the top, which is the clearest way to see that the " +
    "1/n² pattern belongs to 1/r and to nothing else.",
  finitewell:
    "A flat-bottomed box of depth V₀ and width a. Make it shallow enough and " +
    "I bind nothing at all, which a Coulomb well never lets me do.",
  coulombcore:
    "Coulomb plus a repulsive core, the standard stand-in for the way inner " +
    "electrons keep an outer one away from the nucleus. It lifts the " +
    "ℓ-degeneracy in the same direction real atoms do.",
  custom:
    "Your own V(r). I solve it with the same finite-difference eigensolver I " +
    "use for every preset, and I label every level it produces counterfactual.",
};

/**
 * What each raw constant is, for my panel of five sliders labelled ℏ, e, mₑ,
 * ε₀ and c.
 *
 * Those glyphs are the right names and they are not an explanation. If you do
 * not already know what ε₀ is you cannot decide whether to drag it, and the
 * whole point of my lab is that you should drag it and watch what fails to
 * happen.
 */
export const CONSTANT_BLURBS: Record<string, string> = {
  hbar: "the quantum of action: how coarse the graininess of the world is",
  e: "the electron's charge: how hard the nucleus pulls",
  m_e: "the electron's mass: how reluctant it is to be moved",
  eps0: "the vacuum's permittivity: how much the empty space between charges weakens the pull",
  c: "the speed of light: the scale relativity starts to matter at",
};
