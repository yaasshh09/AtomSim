import { lazy, Suspense, useState } from "react";
import { useAppStore } from "../state/store";

const PhysicsBody = lazy(() => import("./PhysicsBody"));

export function ShowPhysics() {
  const view = useAppStore((s) => s.view);
  // The summary is the affordance and has to be there from the first frame;
  // the maths under it does not exist until someone asks for it. Latching on
  // first open rather than tracking `open` keeps the chunk from being torn
  // down and refetched every time the panel is collapsed.
  const [opened, setOpened] = useState(false);

  return (
    <details
      className="physics"
      onToggle={(e) => {
        if (e.currentTarget.open) setOpened(true);
      }}
    >
      <summary>Show the physics</summary>
      {opened && (
        <Suspense fallback={<p className="physics-note">typesetting...</p>}>
          <PhysicsBody view={view} />
        </Suspense>
      )}
    </details>
  );
}
