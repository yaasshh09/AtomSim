import { URL_DEFAULTS, type UrlState } from "../lib/urlState";
import type { Tour, TourStep } from "./types";

/**
 * The full app state a step asks for.
 *
 * A full reset from the defaults, never a patch onto whatever the previous step
 * left behind. Two reasons: a step's picture is then reproducible from that
 * step's own data, which is what makes its claims mean anything, and stepping
 * backward cannot leave a later step's physics switched on underneath an
 * earlier step's prose.
 *
 * `labConst` and `forceParams` are copied rather than shared, so a step can
 * never mutate the defaults out from under every step after it.
 */
export function stepState(step: TourStep): UrlState {
  return {
    ...URL_DEFAULTS,
    labConst: { ...URL_DEFAULTS.labConst },
    forceParams: { ...URL_DEFAULTS.forceParams },
    ...step.state,
  };
}

/** An index inside the tour, whatever a hand-edited `?step=` supplied. */
export function clampStep(tour: Tour, i: number): number {
  if (!Number.isFinite(i)) return 0;
  const last = tour.steps.length - 1;
  if (last < 0) return 0;
  return Math.min(Math.max(Math.floor(i), 0), last);
}
