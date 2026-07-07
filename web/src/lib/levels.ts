import type { SpectralLineInfo } from "../api/types";

/** Downward transitions out of the selected (n, l) — already selection-rule
 *  filtered by the engine; the frontend only picks the relevant subset. */
export function arrowsFor(
  lines: readonly SpectralLineInfo[],
  n: number,
  l: number,
): SpectralLineInfo[] {
  return lines.filter((ln) => ln.n_upper === n && ln.l_upper === l);
}
