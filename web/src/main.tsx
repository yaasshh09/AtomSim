import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { shouldAutoSample } from "./lib/startup";
import { currentUrlState, parseAppUrl, serializeAppUrl } from "./lib/urlState";
import { useAppStore } from "./state/store";
import "katex/dist/katex.min.css";
// Bundled, not fetched from a CDN: `atomsim serve` is a local app and the
// typography is part of the instrument, so it has to survive being offline.
import "@fontsource-variable/space-grotesk";
import "@fontsource-variable/jetbrains-mono";
import "./index.css";

// Deep links (demo-script hooks): apply the URL before first render, then keep
// the URL describing the live state so any moment of a session is shareable.
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

// Draw the opening state rather than asking for permission to. Same microtask
// deferral and the same reason: the store has to be settled before the action
// reads (n, l, m, system) off it.
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
