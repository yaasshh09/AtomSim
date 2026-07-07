import { CloudView } from "./components/CloudView";
import { Controls } from "./components/Controls";
import { InfoPanel } from "./components/InfoPanel";
import { LevelsView } from "./components/LevelsView";
import { PlaneView } from "./components/PlaneView";
import { RadialView } from "./components/RadialView";
import { useAppStore } from "./state/store";

export default function App() {
  const view = useAppStore((s) => s.view);
  return (
    <div className="app-grid">
      <InfoPanel />
      <main className="center-col">
        {view === "cloud" && <CloudView />}
        {view === "plane" && <PlaneView />}
        {view === "radial" && <RadialView />}
        {view === "levels" && <LevelsView />}
      </main>
      <Controls />
    </div>
  );
}
