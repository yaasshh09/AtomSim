import { scaleLinear } from "d3-scale";

/**
 * Axis helpers for plots whose window can be many decades wide or a few
 * femtometres narrow.
 *
 * Both exist because a tick label you cannot tell apart from its neighbour is
 * worse than no tick at all: it looks like a reading and carries none.
 */

/** The units a wavelength offset displays in, coarsest first. */
const UNITS = [
  { unit: "nm" as const, perNm: 1, centreDecimals: 2 },
  { unit: "pm" as const, perNm: 1e3, centreDecimals: 4 },
  { unit: "fm" as const, perNm: 1e6, centreDecimals: 6 },
];

export interface OffsetAxis {
  /** The window centre: the wavelength the offsets are measured from. */
  centreNm: number;
  /** Multiply a (lambda - centre) difference in nm by this to get display units. */
  perNm: number;
  unit: "nm" | "pm" | "fm";
  /** Decimals for `centreNm`, fine enough to resolve one display unit. */
  centreDecimals: number;
}

/**
 * Describe a zoomed wavelength window as offsets from its centre.
 *
 * A line profile is plotted over a handful of half-widths, and hydrogen's
 * Lyman-alpha natural width is about 5e-6 nm. Absolute wavelengths across that
 * window would print "121.568" at every tick: six identical labels with no way
 * to read the scale. Spectroscopy's own answer is to name the centre once and
 * label the ticks as offsets, which is what this sets up.
 *
 * It picks the finest unit that keeps tick magnitudes from collapsing to zero:
 * nm down to a picometre-scale window, then pm, then fm.
 */
export function offsetAxis(lo: number, hi: number): OffsetAxis {
  const centreNm = (lo + hi) / 2;
  const span = Math.abs(hi - lo);
  // The coarsest unit that still puts at least a few units across the window.
  // Below 10 display units the labels start rounding into each other, which is
  // the failure this function exists to prevent.
  const chosen = UNITS.find((u) => span * u.perNm >= 10) ?? UNITS[UNITS.length - 1];
  return { centreNm, ...chosen };
}

/**
 * A tick's label: its offset from the axis centre, in the axis's unit.
 *
 * Always signed, because "0" at the centre with bare numbers either side would
 * read as absolute wavelengths.
 */
export function formatOffset(axis: OffsetAxis, wavelengthNm: number): string {
  const d = (wavelengthNm - axis.centreNm) * axis.perNm;
  // Round before testing for zero: a centre tick lands on 1e-13 rather than 0
  // in floating point, and "+0.0" beside "-0.0" is noise.
  const r = Number(d.toFixed(1));
  if (r === 0) return "0";
  return `${r > 0 ? "+" : ""}${r.toFixed(1)}`;
}

/**
 * Tick wavelengths that fall on round *offsets* from the axis centre.
 *
 * d3 picks nice values on the domain it is handed, and that domain is absolute
 * wavelengths, so a window around 121.568446 nm gets ticks at nice absolute
 * values whose offsets come out as -35.6, -25.6, ... +34.4, with no tick at
 * the line centre. Choosing the ticks in offset space instead gives -40, -20,
 * 0, +20, +40, and the centre of a line profile is exactly where a tick
 * belongs.
 */
export function offsetTicks(
  axis: OffsetAxis,
  lo: number,
  hi: number,
  count = 6,
): number[] {
  const loOff = (lo - axis.centreNm) * axis.perNm;
  const hiOff = (hi - axis.centreNm) * axis.perNm;
  return scaleLinear([loOff, hiOff], [0, 1])
    .ticks(count)
    .map((o) => axis.centreNm + o / axis.perNm);
}

/**
 * Drop tick values whose labels would overlap.
 *
 * A logarithmic wavelength axis puts 5000 and 6000 nm about four pixels apart
 * while 90 and 100 get their own room, so even a decade-aware tick generator
 * hands back a row of overprinted digits at the top end. Rather than thinning
 * by value, which would assume a particular scale, this walks the ticks in
 * drawn order and keeps one only when it clears the last kept tick by
 * `minGapPx`.
 *
 * The first and last ticks are the axis bounds and always survive, so the
 * range stays readable however crowded the middle gets.
 */
export function thinTicks(
  ticks: readonly number[],
  x: (v: number) => number,
  minGapPx: number,
): number[] {
  if (ticks.length <= 2) return [...ticks];
  const last = ticks[ticks.length - 1];
  const kept: number[] = [ticks[0]];
  let lastX = x(ticks[0]);
  for (let i = 1; i < ticks.length - 1; i++) {
    const px = x(ticks[i]);
    if (Math.abs(px - lastX) < minGapPx) continue;
    // Keeping this one would crowd the final tick, which outranks it.
    if (Math.abs(x(last) - px) < minGapPx) continue;
    kept.push(ticks[i]);
    lastX = px;
  }
  kept.push(last);
  return kept;
}

/**
 * Unlabelled ticks between the labelled ones, for the minor grid.
 *
 * Linear: each major interval split into the step's natural parts (a step of
 * 1, 2 or 5 times a power of ten becomes 5, 4 or 5 parts), so minor lines land
 * on round values. Log: the 2x..9x lines inside each decade, the ruling on
 * printed log paper. Values on a major tick are left out, and so is anything
 * outside [lo, hi].
 */
export function minorTicks(
  major: readonly number[],
  [lo, hi]: readonly [number, number],
  log = false,
): number[] {
  const a = Math.min(lo, hi);
  const b = Math.max(lo, hi);
  const out: number[] = [];
  if (log) {
    if (!(a > 0)) return out;
    for (let e = Math.floor(Math.log10(a)); e <= Math.ceil(Math.log10(b)); e++) {
      for (let k = 2; k <= 9; k++) {
        const v = k * 10 ** e;
        if (v > a && v < b) out.push(v);
      }
    }
    return out;
  }
  if (major.length < 2) return out;
  const step = Math.abs(major[1] - major[0]);
  if (!(step > 0)) return out;
  const lead = Math.round(step / 10 ** Math.floor(Math.log10(step)));
  const sub = step / (lead === 2 ? 4 : 5);
  // Rounded to the step's own precision, so 0.1 + 0.2 never reads as a new tick.
  const eps = sub * 1e-6;
  for (let i = Math.ceil(a / sub); i * sub <= b + eps; i++) {
    const v = i * sub;
    if (major.some((m) => Math.abs(m - v) < eps)) continue;
    if (v >= a - eps && v <= b + eps) out.push(v);
  }
  return out;
}
