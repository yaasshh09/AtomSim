import { useState } from "react";
import { useAppStore } from "../state/store";
import { TOURS } from "../tours/registry";

/** The tour picker. Opens from the top bar, closes on pick or on Escape. */
export function TourMenu() {
  const [open, setOpen] = useState(false);
  const { tourId, startTour, exitTour } = useAppStore();
  if (tourId) {
    return (
      <button className="tour-entry" type="button" onClick={exitTour}>
        leave tour
      </button>
    );
  }
  return (
    <div className="tour-entry-wrap" onKeyDown={(e) => e.key === "Escape" && setOpen(false)}>
      <button className="tour-entry" type="button" onClick={() => setOpen(!open)}>
        guided tours
      </button>
      {open && (
        <ul className="tour-menu">
          {TOURS.map((t) => (
            <li key={t.id}>
              <button
                type="button"
                onClick={() => {
                  startTour(t.id, 0);
                  setOpen(false);
                }}
              >
                <span className="tour-menu-title">{t.title}</span>
                <span className="tour-menu-blurb">{t.blurb}</span>
                <span className="tour-menu-count">{t.steps.length} steps</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
