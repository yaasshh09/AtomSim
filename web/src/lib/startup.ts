import type { SurfaceMode, ViewMode } from "../state/store";

/** Whether I should draw something before anyone asks me to.
 *
 * My default view is a Monte-Carlo point cloud, and until a sample runs it is
 * an empty axis triad over the words "Choose a state and press Sample". That
 * is a fine first frame for the person who built me and a poor one for a
 * stranger following a link, who has no reason to believe the button will do
 * anything interesting.
 *
 * Cost is not why I used to wait: my default 100k-draw sample of 1s takes
 * 18 ms in the engine. So I sample the landing view myself, and I keep every
 * later recompute explicit, because those are the ones you chose.
 */
export function shouldAutoSample(opening: {
  tour: string | null | undefined;
  view: ViewMode;
  surfaceMode: SurfaceMode;
}): boolean {
  // A tour link is not a landing: startTour applies the step's own physics and
  // clears everything derived from it. If I fired a sample alongside it I would
  // either throw the result away or, worse, land it under a caption describing
  // different physics than the one that produced it.
  if (opening.tour) return false;
  // Only the cloud has the empty state I exist to fill. My surface view fetches
  // its own mesh, and every other view renders from the levels payload I load
  // regardless.
  if (opening.view !== "cloud") return false;
  return opening.surfaceMode === "cloud";
}
