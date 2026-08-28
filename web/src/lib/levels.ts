import type { SpectralLineInfo } from "../api/types";

/** The downward transitions out of the selected (n, l). My engine has already
 *  applied the selection rules; here I only pick the relevant subset. */
export function arrowsFor(
  lines: readonly SpectralLineInfo[],
  n: number,
  l: number,
): SpectralLineInfo[] {
  return lines.filter((ln) => ln.n_upper === n && ln.l_upper === l);
}

/**
 * Which magnifier I am showing in the levels view, and how you ask me for
 * another.
 *
 * The right-hand column of my ladder can magnify one of four things, and I
 * used to choose between them with a chain of nested ternaries buried in the
 * render: an electric field beat a hyperfine split, which beat a Zeeman fan,
 * which beat a plain fine-structure zoom. Every one of those precedences is
 * defensible on its own, and together they made a control that silently did
 * nothing. You could tick "hyperfine" with the field slider up and I would
 * move nothing on screen, with no way for you to find out why.
 *
 * The exclusivity is real, since four magnifications cannot share one column,
 * so I make it a picker now. These functions are the single place I map
 * between the picker and the five underlying store fields, so what I draw and
 * what the picker says are the same decision rather than two that agree by
 * luck.
 */
export type DetailMode = "none" | "fine" | "zeeman" | "stark" | "hyperfine";

/** The store fields I read for the magnifier. */
export interface DetailState {
  fineStructure: boolean;
  bField: number;
  eField: number;
  hyperfine: boolean;
}

/**
 * The magnifier the current state produces.
 *
 * I keep the old render's precedence exactly: if I changed it I would change
 * which plot a saved deep link comes back to, and I treat the query schema in
 * lib/urlState.ts as a stable contract.
 */
export function detailMode(s: DetailState): DetailMode {
  if (s.eField > 0) return "stark";
  if (s.hyperfine) return "hyperfine";
  if (!s.fineStructure) return "none";
  return s.bField > 0 ? "zeeman" : "fine";
}

/** The field strengths I open a mode at when you select it from rest. */
export const DEFAULT_B_TESLA = 2;
export const DEFAULT_E_MV_PER_M = 20;

/**
 * The state that produces `mode`, given where the sliders already are.
 *
 * I keep a field you have already dialled in for when you come back to it, and
 * I open a mode selected from rest at a strength where you can see something,
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
 * I push overlapping labels apart, keeping them in their original order.
 *
 * My energy ladder is linear in E and the levels go as −1/n², so n=4, 5 and 6
 * land within a few pixels of each other and I printed their labels straight
 * through one another: "-0.38 eV" and "-0.54 eV" took the same row of pixels
 * and you could read neither. The crowding is the physics and it stays; what
 * moves is my text about it, so I nudge each label to where there is room and
 * join it back to its rung with a leader line.
 *
 * I preserve the order by construction, so I never put a label beside the
 * wrong rung, which would be a far worse failure than overlapping ones. I
 * clamp the result into [lo, hi], and it can still overlap if you ask me to
 * fit more labels than the band has room for; the caller checks that.
 */
export function spreadLabels(
  ys: readonly number[],
  minGap: number,
  lo: number,
  hi: number,
): number[] {
  const n = ys.length;
  if (n === 0) return [];
  // I work in ascending y, remembering where each one came from.
  const order = ys.map((_, i) => i).sort((a, b) => ys[a] - ys[b]);
  const sorted = order.map((i) => ys[i]);
  const out = sorted.slice();
  // Downward pass: I let nothing sit closer than minGap to the one above it.
  for (let i = 1; i < n; i++) {
    if (out[i] - out[i - 1] < minGap) out[i] = out[i - 1] + minGap;
  }
  // If that pushed the stack off the bottom, I shift it back and settle upward.
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
