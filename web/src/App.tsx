import { lazy, Suspense } from "react";
import { Controls } from "./components/Controls";
import { ForceLawView } from "./components/ForceLawView";
import { GalleryStrip } from "./components/GalleryStrip";
import { InfoPanel } from "./components/InfoPanel";
import { LevelsView } from "./components/LevelsView";
import {
  NarrowNotice,
  needsWiderScreen,
  useViewportWidth,
} from "./components/NarrowNotice";
import { PlaneView } from "./components/PlaneView";
import { RadialView } from "./components/RadialView";
import { SpectrumView } from "./components/SpectrumView";
import { TopBar } from "./components/TopBar";
import { TourInvite } from "./components/TourInvite";
import { TourPanel } from "./components/TourPanel";
import { TourSpotlight } from "./components/TourSpotlight";
import { WhatIfView } from "./components/WhatIfView";
import { useAppStore } from "./state/store";

/** three.js and @react-three/fiber are 1.1 MB, over two thirds of the bundle,
 * and they are reachable from exactly one subtree. Split out, the shell and
 * the readouts paint while it streams, and a reader who deep-links to the
 * spectrum or walks a 2-D tour step never pays for a renderer they do not
 * look at. */
const CloudView = lazy(() =>
  import("./components/CloudView").then((m) => ({ default: m.CloudView })),
);

export default function App() {
  const view = useAppStore((s) => s.view);
  const width = useViewportWidth();

  // Gated in JS rather than by a media query so there is one threshold, and so
  // a phone never pays to build a WebGL context it will not be shown.
  if (needsWiderScreen(width)) return <NarrowNotice width={width} />;

  return (
    <div className="app-shell">
      <TopBar />
      <TourInvite />
      <div className="app-grid">
        <InfoPanel />
        <main className="center-col">
          {view === "cloud" && (
            <Suspense fallback={<div className="canvas-wrap canvas-wrap-loading" />}>
              <CloudView />
            </Suspense>
          )}
          {view === "plane" && <PlaneView />}
          {view === "radial" && <RadialView />}
          {view === "levels" && <LevelsView />}
          {view === "spectrum" && <SpectrumView />}
          {view === "whatif" && <WhatIfView />}
          {view === "forcelaw" && <ForceLawView />}
          <GalleryStrip />
          <TourPanel />
        </main>
        <Controls />
      </div>
      <TourSpotlight />
    </div>
  );
}
