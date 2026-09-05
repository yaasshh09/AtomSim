import { useEffect, useState } from "react";
import { N_MAX, clampState } from "../lib/quantum";
import { isTypingTarget, matchShortcut } from "../lib/shortcuts";
import { useAppStore } from "../state/store";
import { VIEW_OPTIONS } from "./Controls";

/** What the card lists, in the order it lists them. */
const KEYS: { keys: string; does: string }[] = [
  { keys: "up / down", does: "shell n" },
  { keys: "left / right", does: "orbital l" },
  { keys: "shift + left / right", does: "orientation m" },
  { keys: "1 - 7", does: "the seven views" },
  { keys: "?", does: "this list" },
];

/**
 * The keyboard, and the card that admits it exists.
 *
 * Walking the states is the most common thing anyone does here and it cost
 * two selects and a reach for the mouse. The map itself is in lib/shortcuts;
 * this installs the one listener and spends what it returns.
 *
 * The card is not decoration. An undocumented shortcut is a feature only its
 * author has, so the key that opens the list is in the list, and the button
 * that opens it sits where the reader is already looking.
 *
 * Hidden on the phone shell by CSS rather than by a viewport check: there is
 * no keyboard to describe, and the top bar there has no room to say so.
 */
export function Shortcuts() {
  const [open, setOpen] = useState(false);

  // The store is read inside the handler rather than closed over, so the
  // listener is installed once and still sees the live state.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTypingTarget(e.target as HTMLElement | null)) return;
      const hit = matchShortcut(e);
      if (!hit) return;
      const s = useAppStore.getState();
      switch (hit.kind) {
        case "quantum": {
          // Clamped here as well as in the action, because n has a ceiling the
          // action does not know about: the rail offers six shells, and a key
          // that walked past them would ask for a state no control can show.
          const n = hit.axis === "n" ? Math.min(s.n + hit.delta, N_MAX) : s.n;
          const l = hit.axis === "l" ? s.l + hit.delta : s.l;
          const m = hit.axis === "m" ? s.m + hit.delta : s.m;
          const next = clampState(n, l, m);
          // A move that clamps back onto where it started is not a move, and
          // spending a history entry on it would fill Back with nothing.
          if (next.n === s.n && next.l === s.l && next.m === s.m) return;
          s.setQuantumNumbers(next.n, next.l, next.m);
          break;
        }
        case "view": {
          const v = VIEW_OPTIONS[hit.index];
          if (v) s.setView(v.value);
          break;
        }
        case "help":
          setOpen((o) => !o);
          break;
        case "close":
          setOpen(false);
          return; // Escape belongs to whatever else is listening for it too
      }
      // Held back until something was actually done with the key, so arrows
      // and digits this map ignores still scroll and type as usual.
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="tour-entry-wrap shortcuts-wrap">
      <button
        className="topbar-btn"
        type="button"
        aria-expanded={open}
        title="Keyboard shortcuts"
        onClick={() => setOpen(!open)}
      >
        keys
      </button>
      {open && (
        <dl className="shortcut-card">
          {KEYS.map((k) => (
            <div key={k.keys} className="shortcut-row">
              <dt>{k.keys}</dt>
              <dd>{k.does}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}
