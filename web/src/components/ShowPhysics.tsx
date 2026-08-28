import { lazy, Suspense, useState } from "react";
import { useAppStore } from "../state/store";

const PhysicsBody = lazy(() => import("./PhysicsBody"));

export function ShowPhysics() {
  const view = useAppStore((s) => s.view);
  // The summary is the affordance and I need it from the first frame; the
  // maths under it does not exist until someone asks me for it. I latch on
  // first open rather than tracking `open`, so the chunk is not torn down and
  // refetched every time you collapse the panel.
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
