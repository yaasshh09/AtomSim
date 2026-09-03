import type { SpectralLineInfo } from "../api/types";

/** The downward transitions out of the selected (n, l). The engine has already
 *  applied the selection rules; this only picks the relevant subset. */
export function arrowsFor(
  lines: readonly SpectralLineInfo[],
  n: number,
  l: number,
): SpectralLineInfo[] {
  return lines.filter((ln) => ln.n_upper === n && ln.l_upper === l);
}

/**
 * Which magnifier the levels view is showing, and how to ask for another.
 *
 * The right-hand column of the ladder can magnify one of four things, and the
 * choice used to come out of a chain of nested ternaries buried in the render:
 * an electric field beat a hyperfine split, which beat a Zeeman fan, which
 * beat a plain fine-structure zoom. Every one of those precedences is
 * defensible on its own, and together they made a control that silently did
 * nothing. Tick "hyperfine" with the field slider up and the screen would not
 * move, with no way to find out why.
 *
 * The exclusivity is real, since four magnifications cannot share one column,
 * so it is a picker now. These functions are the single place the picker maps
 * onto the five underlying store fields, so what gets drawn and what the
 * picker says are one decision rather than two that agree by luck.
 */
export type DetailMode = "none" | "fine" | "zeeman" | "stark" | "hyperfine";

/** The store fields the magnifier reads. */
export interface DetailState {
  fineStructure: boolean;
  bField: number;
  eField: number;
  hyperfine: boolean;
}

/**
 * The magnifier the current state produces.
 *
 * The old render's precedence is kept exactly. Changing it would change which
 * plot a saved deep link comes back to, and the query schema in lib/urlState.ts
 * is a stable contract.
 */
export function detailMode(s: DetailState): DetailMode {
  if (s.eField > 0) return "stark";
  if (s.hyperfine) return "hyperfine";
  if (!s.fineStructure) return "none";
  return s.bField > 0 ? "zeeman" : "fine";
}

/** The field strengths a mode opens at when it is selected from rest. */
export const DEFAULT_B_TESLA = 2;
export const DEFAULT_E_MV_PER_M = 20;

/**
 * The state that produces `mode`, given where the sliders already are.
 *
 * A field already dialled in is kept for when the reader comes back to it, and
 * a mode selected from rest opens at a strength where something is visible,
 * because a slider that starts at zero makes its own mode look broken.
 */
export function detailState(mode: DetailMode, current: DetailState): DetailState {
  switch (mode) {
    case "none":
      return { fineStructure: false, bField: 0, eField: 0, hyperfine: false };
    case "fine":
      return { fineStructure: true, bField: 0, eField: 0, hyperfine: false };
    case "zeeman":
      return {
        fineStructure: true,
        bField: current.bField > 0 ? current.bField : DEFAULT_B_TESLA,
        eField: 0,
        hyperfine: false,
      };
    case "stark":
      return {
        fineStructure: current.fineStructure,
        bField: current.bField,
        eField: current.eField > 0 ? current.eField : DEFAULT_E_MV_PER_M,
        hyperfine: false,
      };
    case "hyperfine":
      return {
        fineStructure: current.fineStructure,
        bField: current.bField,
        eField: 0,
        hyperfine: true,
      };
  }
}

/**
 * Push overlapping labels apart, keeping them in their original order.
 *
 * The energy ladder is linear in E and the levels go as −1/n², so n=4, 5 and 6
 * land within a few pixels of each other and their labels printed straight
 * through one another: "-0.38 eV" and "-0.54 eV" took the same row of pixels
 * and neither could be read. The crowding is the physics and it stays. What
 * moves is the text about it, so each label gets nudged to where there is room
 * and joined back to its rung with a leader line.
 *
 * Order is preserved by construction, so a label never lands beside the wrong
 * rung, which would be far worse than overlapping ones. The result is clamped
 * into [lo, hi], and it can still overlap when asked to fit more labels than
 * the band has room for; the caller checks that.
 */
export function spreadLabels(
  ys: readonly number[],
  minGap: number,
  lo: number,
  hi: number,
): number[] {
  const n = ys.length;
  if (n === 0) return [];
  // Work in ascending y, remembering where each one came from.
  const order = ys.map((_, i) => i).sort((a, b) => ys[a] - ys[b]);
  const sorted = order.map((i) => ys[i]);
  const out = sorted.slice();
  // Downward pass: nothing sits closer than minGap to the one above it.
  for (let i = 1; i < n; i++) {
    if (out[i] - out[i - 1] < minGap) out[i] = out[i - 1] + minGap;
  }
  // If that pushed the stack off the bottom, shift it back and settle upward.
  const overflow = out[n - 1] - hi;
  if (overflow > 0) {
    for (let i = 0; i < n; i++) out[i] -= overflow;
    for (let i = n - 2; i >= 0; i--) {
      if (out[i + 1] - out[i] < minGap) out[i] = out[i + 1] - minGap;
    }
  }
  for (let i = 0; i < n; i++) out[i] = Math.max(out[i], lo);
  const result = new Array<number>(n);
  order.forEach((src, i) => {
    result[src] = out[i];
  });
  return result;
}
