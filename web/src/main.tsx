import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { parseAppUrl, serializeAppUrl } from "./lib/urlState";
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
  const qs = serializeAppUrl({
    n: s.n,
    l: s.l,
    m: s.m,
    system: s.system,
    basis: s.basis,
    view: s.view,
    colorMode: s.colorMode,
    fineStructure: s.fineStructure,
    dirac: s.dirac,
    compare: s.compare,
    bField: s.bField,
    eField: s.eField,
    hyperfine: s.hyperfine,
    intensities: s.intensities,
    thermal: s.thermal,
    temperatureK: s.temperatureK,
    logNe: s.logNe,
    profile: s.profile,
    logResolvingPower: s.logResolvingPower,
    profileZoom: s.profileZoom,
    absorption: s.absorption,
    logColumn: s.logColumn,
    ghost: s.ghost,
    nucleusMode: s.nucleusMode,
    planeQuantity: s.planeQuantity,
    surfaceMode: s.surfaceMode,
    isoFraction: s.isoFraction,
    labConst: s.labConst,
    labZ: s.labZ,
    forcePreset: s.forcePreset,
    forceParams: s.forceParams,
    forceL: s.forceL,
    forceExpr: s.forceExpr,
    config: s.config,
    model: s.model,
    exchange: s.exchange,
    pauli: s.pauli,
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
