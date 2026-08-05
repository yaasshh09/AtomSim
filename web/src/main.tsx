import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
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
useAppStore.setState(parseAppUrl(window.location.search));
useAppStore.subscribe((s) => {
  const qs = serializeAppUrl(currentUrlState(s));
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
