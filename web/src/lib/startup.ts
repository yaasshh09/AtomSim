import type { SurfaceMode, ViewMode } from "../state/store";

/** Whether to draw something before anyone asks.
 *
 * The default view is a Monte-Carlo point cloud, and until a sample runs it is
 * an empty axis triad over the words "Choose a state and press Sample". That
 * is a fine first frame for whoever built the thing and a poor one for a
 * stranger following a link, who has no reason to believe the button will do
 * anything interesting.
 *
 * Cost was never the reason to wait: the default 100k-draw sample of 1s takes
 * 18 ms in the engine. So the landing view samples itself, and every later
 * recompute stays explicit, because those are the ones the reader chose.
 */
export function shouldAutoSample(opening: {
  tour: string | null | undefined;
  view: ViewMode;
  surfaceMode: SurfaceMode;
}): boolean {
  // A tour link is not a landing: startTour applies the step's own physics and
  // clears everything derived from it. Firing a sample alongside it would
  // either throw the result away or, worse, land it under a caption describing
  // different physics than the one that produced it.
  if (opening.tour) return false;
  // Only the cloud has the empty state this exists to fill. The surface view
  // fetches its own mesh, and every other view renders from the levels payload
  // that loads regardless.
  if (opening.view !== "cloud") return false;
  return opening.surfaceMode === "cloud";
}
