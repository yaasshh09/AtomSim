import type { FineLevel } from "../api/types";

/** My mirror of the engine's CODATA fine-structure constant
 *  (src/atomsim/constants.py ALPHA). I use it only to position the α control
 *  and its default; my COUNTERFACTUAL banner compares the α values the server
 *  echoes back, never this constant. */
export const REAL_ALPHA = 0.0072973525643;

/** The upper bound I put on the α slider and URL, matching my server's (0, 0.5] validation. */
export const ALPHA_MAX = 0.5;

/** The fine-structure fractional error past which I stop trusting the perturbative model. */
export const FINE_WARN_FRACTION = 0.1;

/** How I write α for a human: "1/137" in the reciprocal regime, a decimal once α ≥ 0.5. */
export function formatAlpha(alpha: number): string {
  if (alpha <= 0) return "0";
  if (alpha >= 0.5) return alpha.toFixed(2);
  return `1/${Math.round(1 / alpha)}`;
}

/** True when α departs from the real value my server echoed back. */
export function isAltered(alpha: number, realAlpha: number): boolean {
  return Math.abs(alpha - realAlpha) > 1e-12 * realAlpha;
}

/** The largest fractional fine-structure error (error_estimate / |shift|) I see across levels; 0 if none. */
export function fineErrorFraction(fine: FineLevel[] | null): number {
  if (!fine || fine.length === 0) return 0;
  let max = 0;
  for (const f of fine) {
    const err = f.shift.provenance.error_estimate;
    const mag = Math.abs(f.shift.value);
    if (err !== null && mag > 0) max = Math.max(max, err / mag);
  }
  return max;
}

/** Have I pushed the perturbative fine structure past the validity I stated for it? */
export function isBeyondValidity(fine: FineLevel[] | null): boolean {
  return fineErrorFraction(fine) > FINE_WARN_FRACTION;
}

/** The eV span of the fine shifts I find within shell n (0 if fewer than two sub-levels). */
export function shellSplitting(fine: FineLevel[] | null, n: number): number {
  const s = (fine ?? []).filter((f) => f.n === n).map((f) => f.shift_ev.value);
  if (s.length < 2) return 0;
  return Math.max(...s) - Math.min(...s);
}

/** The raw-constant multiplier bounds I allow, matching my server's [0.25, 4] validation. */
export const CONST_MIN = 0.25;
export const CONST_MAX = 4;

/** The five raw constants, in the order I show them in the panel. */
export const CONSTANT_KEYS = ["hbar", "e", "m_e", "eps0", "c"] as const;
export type ConstantKey = (typeof CONSTANT_KEYS)[number];

/** The glyphs I display for the five constants. */
export const CONSTANT_LABELS: Record<ConstantKey, string> = {
  hbar: "ℏ",
  e: "e",
  m_e: "mₑ",
  eps0: "ε₀",
  c: "c",
};

/** I can only draw a physical perturbative diagram when the derived α is in (0, 0.5], my server's bound. */
export function isAlphaValid(alpha: number): boolean {
  return alpha > 0 && alpha <= ALPHA_MAX;
}

/** How I write an altered/real ratio for a human: "unchanged" at 1, otherwise "×2.00" or "×0.50". */
export function formatRatio(ratio: number): string {
  if (Math.abs(ratio - 1) < 1e-9) return "unchanged";
  return `×${ratio.toFixed(2)}`;
}
