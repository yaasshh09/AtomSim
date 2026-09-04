import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { COUNT_CHOICES } from "./components/Controls";
import { installAnalytics } from "./lib/analytics";
import { shouldAutoSample } from "./lib/startup";
import { currentUrlState, parseAppUrl, serializeAppUrl } from "./lib/urlState";
import { isNarrow } from "./lib/viewport";
import { useAppStore } from "./state/store";
// Bundled rather than fetched from a CDN: `atomsim serve` runs locally, and
// the typography is part of the instrument, so it has to survive being
// offline.
import "@fontsource-variable/space-grotesk";
import "@fontsource-variable/jetbrains-mono";
import "./index.css";

// First, so an arrival is counted even if the render that follows throws:
// "someone opened it" is true either way, and a crash is exactly when knowing
// that someone did is worth something. It stays off unless VITE_GOATCOUNTER is
// set at build time, which the dev server and a plain `npm run build` never
// do.
installAnalytics(import.meta.env.VITE_GOATCOUNTER, document);

// A phone opens on the smallest sample rather than on 100,000 points, which a
// mid-range mobile GPU will not enjoy. A starting value only: every count stays
// selectable, and Controls says so. Read once, because a default that followed
// the viewport would silently change the picture when a tablet was turned.
if (isNarrow(window.innerWidth)) useAppStore.setState({ count: COUNT_CHOICES[0] });

// Deep links (the demo-script hooks): the URL is applied before the first
// render, and then kept describing the live state, so any moment of a session
// is shareable.
const opening = parseAppUrl(window.location.search);
useAppStore.setState(opening);

// A tour link has to run the step's state through the store's tour action, not
// just land its id in the store: the step's own physics has to be applied and
// everything derived cleared. Deferred to a microtask so the store's initial
// state exists before startTour reads it.
if (opening.tour) {
  const id = opening.tour;
  const step = opening.step ?? 0;
  queueMicrotask(() => useAppStore.getState().startTour(id, step));
}

// The opening state gets drawn rather than waiting to be asked for. Same
// microtask deferral and the same reason: the store has to be settled before
// the action reads (n, l, m, system) off it.
{
  const settled = useAppStore.getState();
  if (
    shouldAutoSample({
      tour: opening.tour,
      view: settled.view,
      surfaceMode: settled.surfaceMode,
    })
  ) {
    queueMicrotask(() => void useAppStore.getState().sample());
  }
}

useAppStore.subscribe((s) => {
  const qs = serializeAppUrl({ ...currentUrlState(s), tour: s.tourId, step: s.stepIndex });
  const next = window.location.pathname + qs;
  if (next !== window.location.pathname + window.location.search) {
    window.history.replaceState(null, "", next);
  }
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
