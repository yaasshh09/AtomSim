import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { parseAppUrl, serializeAppUrl } from "./lib/urlState";
import { useAppStore } from "./state/store";
import "katex/dist/katex.min.css";
import "./index.css";

// Deep links (demo-script hooks): apply the URL before first render, then keep
// the URL describing the live state so any moment of a session is shareable.
useAppStore.setState(parseAppUrl(window.location.search));
useAppStore.subscribe((s) => {
  const qs = serializeAppUrl({
    n: s.n,
    l: s.l,
    m: s.m,
    system: s.system,
    basis: s.basis,
    view: s.view,
    colorMode: s.colorMode,
    fineStructure: s.fineStructure,
    ghost: s.ghost,
    nucleusMode: s.nucleusMode,
    planeQuantity: s.planeQuantity,
    labConst: s.labConst,
    labZ: s.labZ,
    // TODO(Task 8): urlState's force-law fields are being generalized from a
    // single power-law exponent to per-preset params; forceP is a shim until
    // that lands, mirroring the powerlaw "p" param when present.
    forceP: s.forceParams.p ?? 1.0,
    forceL: s.forceL,
  });
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
