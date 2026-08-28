import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { installAnalytics } from "./lib/analytics";
import { shouldAutoSample } from "./lib/startup";
import { currentUrlState, parseAppUrl, serializeAppUrl } from "./lib/urlState";
import { useAppStore } from "./state/store";
// I bundle these rather than fetching them from a CDN: `atomsim serve` runs me
// locally and my typography is part of the instrument, so it has to survive
// being offline.
import "@fontsource-variable/space-grotesk";
import "@fontsource-variable/jetbrains-mono";
import "./index.css";

// I do this first, so I count an arrival even if the render that follows
// throws: "someone opened it" is true either way, and a crash is exactly when
// knowing that someone did is worth something. I stay off unless
// VITE_GOATCOUNTER is set at build time, which the dev server and a plain
// `npm run build` never do.
installAnalytics(import.meta.env.VITE_GOATCOUNTER, document);

// Deep links (my demo-script hooks): I apply the URL before the first render,
// then keep the URL describing my live state so any moment of a session is
// shareable.
const opening = parseAppUrl(window.location.search);
useAppStore.setState(opening);

// For a tour link I have to run the step's state through my store's tour
// action, not just land its id in the store: I have to apply the step's own
// physics and clear everything derived. I defer to a microtask so my store's
// initial state exists before startTour reads it.
if (opening.tour) {
  const id = opening.tour;
  const step = opening.step ?? 0;
  queueMicrotask(() => useAppStore.getState().startTour(id, step));
}

// I draw the opening state rather than asking for permission to. Same
// microtask deferral and the same reason: my store has to be settled before
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
