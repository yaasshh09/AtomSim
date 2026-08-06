import { CloudView } from "./components/CloudView";
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
          {view === "cloud" && <CloudView />}
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
