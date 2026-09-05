import { useEffect, useRef } from "react";
import { useAppStore } from "../state/store";
import { VIEW_OPTIONS } from "./Controls";

/**
 * The seven views as a strip of tabs, for the shell that has no rails.
 *
 * On a desktop the view list lives in the right-hand rail, always on screen,
 * under the argument that a closed dropdown would hide six of seven views
 * behind a click. On a phone that rail is the bottom sheet, and the list sits
 * below the system picker and the model toggles inside it: switching from the
 * cloud to the spectrum cost a drag, a scroll and a tap, which is a worse
 * hiding than the dropdown the rail was written to avoid. The whole
 * instrument is seven pictures, and moving between them is the single most
 * common thing anyone does here, so it gets a permanent row.
 *
 * The strip scrolls sideways rather than wrapping, because two rows of tabs
 * cost as much of the screen as the labels save, and it is the one horizontal
 * scroll on this shell that is a scroll rather than an overflow: the tabs to
 * the right of the fold are meant to be reached that way.
 *
 * The labels are shortened here rather than in `VIEW_OPTIONS` because the rail
 * has room for the full ones and the tab does not. Nothing is renamed, only
 * abbreviated, and the long name stays as the accessible label.
 */
export const SHORT: Record<string, string> = {
  cloud: "3D cloud",
  plane: "2D slice",
  radial: "Radial",
  levels: "Levels",
  spectrum: "Spectrum",
  whatif: "What-If: constants",
  forcelaw: "What-If: force law",
};

export function ViewTabs() {
  const view = useAppStore((s) => s.view);
  const setView = useAppStore((s) => s.setView);
  const live = useRef<HTMLButtonElement | null>(null);
  // A deep link can open on the last tab, and a tour step can move the view
  // from somewhere that is not this strip. Either way the tab that is now on
  // is the one that has to be visible, or the row says the wrong thing about
  // where you are.
  useEffect(() => {
    live.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [view]);
  return (
    <nav className="view-tabs" data-tour="view-list" aria-label="view">
      {VIEW_OPTIONS.map((v) => (
        <button
          key={v.value}
          ref={view === v.value ? live : undefined}
          type="button"
          className={`view-tab${view === v.value ? " view-tab-on" : ""}`}
          aria-pressed={view === v.value}
          aria-label={v.label}
          onClick={() => setView(v.value)}
        >
          {SHORT[v.value] ?? v.label}
        </button>
      ))}
    </nav>
  );
}
