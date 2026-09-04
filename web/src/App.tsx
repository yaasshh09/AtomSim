import { lazy, Suspense } from "react";
import type { CSSProperties } from "react";
import { Controls } from "./components/Controls";
import { ForceLawView } from "./components/ForceLawView";
import { GalleryStrip } from "./components/GalleryStrip";
import { InfoPanel } from "./components/InfoPanel";
import { LevelsView } from "./components/LevelsView";
import { MobileSheet } from "./components/MobileSheet";
import { PlaneView } from "./components/PlaneView";
import { RadialView } from "./components/RadialView";
import { SpectrumView } from "./components/SpectrumView";
import { TopBar } from "./components/TopBar";
import { TourInvite } from "./components/TourInvite";
import { TourPanel } from "./components/TourPanel";
import { TourSpotlight } from "./components/TourSpotlight";
import { WhatIfView } from "./components/WhatIfView";
import { snapHeight } from "./lib/sheet";
import { isNarrow, useViewport } from "./lib/viewport";
import { useAppStore } from "./state/store";

/** three.js and @react-three/fiber are 1.1 MB, over two thirds of the bundle,
 * and they are reachable from exactly one subtree. Split out, the shell and
 * the readouts paint while it streams, and a deep link to the spectrum or a
 * walk through a 2-D tour step never pays for a renderer nobody looks at. */
const CloudView = lazy(() =>
  import("./components/CloudView").then((m) => ({ default: m.CloudView })),
);

/** Whichever view is selected. Identical in both shells: only the chrome
 *  around the picture changes, never the picture. */
function Stage() {
  const view = useAppStore((s) => s.view);
  return (
    <>
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
    </>
  );
}

function DesktopShell() {
  return (
    <div className="app-shell">
      <TopBar />
      <TourInvite />
      <div className="app-grid">
        <InfoPanel />
        <main className="center-col">
          <Stage />
          <GalleryStrip />
          <TourPanel />
        </main>
        <Controls />
      </div>
      <TourSpotlight />
    </div>
  );
}

/**
 * The same instrument, stacked: the picture takes the screen and the two rails
 * become one sheet that drags up from the bottom.
 *
 * Two shells rather than one DOM reflowed by media queries. The sheet forces
 * the issue: its open state, its snap points and its drag all need JS, so a
 * CSS-only path would be a hybrid anyway and would pay for a second copy of
 * the chrome in the DOM on both platforms. What that costs is that crossing
 * 900px mid-session remounts the tree and drops the WebGL context with it.
 * Dragging a browser window across 900px is a desktop action, and a desktop
 * user doing it is not halfway through a measurement on a phone.
 */
function MobileShell({ height }: { height: number }) {
  const sheet = useAppStore((s) => s.sheet);
  return (
    <div
      className="app-shell app-shell-mobile"
      // Published as a variable because the tour card docks above the sheet
      // and has to move when it does. CSS cannot read a sibling's height.
      style={{ "--sheet-h": `${snapHeight(sheet, height)}px` } as CSSProperties}
    >
      <TopBar />
      <TourInvite />
      <main className="mobile-stage">
        <Stage />
        <GalleryStrip />
      </main>
      {/* Above the sheet rather than under the stage: the step text is about
          the thing the ring is drawn around, and on a phone that thing is
          usually inside the sheet. */}
      <TourPanel />
      <MobileSheet viewportHeight={height} />
      <TourSpotlight />
    </div>
  );
}

export default function App() {
  const { width, height } = useViewport();
  // Picked in JS rather than by a media query, so there is one threshold and
  // one place it is written down. See lib/viewport.ts for why it is 900.
  return isNarrow(width) ? <MobileShell height={height} /> : <DesktopShell />;
}
