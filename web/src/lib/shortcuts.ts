/**
 * The keyboard map.
 *
 * Pure on purpose: the decision "which key means what" is a table with edge
 * cases (a modifier held, a text field focused), and the way to check a table
 * is to read it, not to drive a browser. The component that owns the listener
 * does nothing but look the event up here and call the store.
 *
 * The bindings follow the shape of the quantum numbers rather than the shape
 * of the rail. n is the shell, so it is the vertical axis; l and m are what
 * lives inside one, so they are horizontal, with shift picking the inner of
 * the two. Every move goes through the same clamp the selects do, so a key
 * cannot reach a state a click cannot.
 */

export type Shortcut =
  | { kind: "quantum"; axis: "n" | "l" | "m"; delta: -1 | 1 }
  | { kind: "view"; index: number }
  | { kind: "help" }
  | { kind: "close" };

/** The part of a KeyboardEvent this reads. */
export interface KeyEventLike {
  key: string;
  shiftKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  altKey: boolean;
}

/** The part of an event target this reads. */
export interface TargetLike {
  tagName?: string;
  isContentEditable?: boolean;
}

/**
 * Whether the keystroke belongs to whatever is focused rather than to the app.
 *
 * A range slider answers arrow keys itself and a select answers letters and
 * digits with type-ahead. Stealing those would break controls that already
 * work, so a focused field wins every time.
 */
export function isTypingTarget(target: TargetLike | null | undefined): boolean {
  if (!target) return false;
  if (target.isContentEditable) return true;
  const tag = (target.tagName ?? "").toUpperCase();
  return tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA";
}

/** The action a keystroke asks for, or null if it asks for nothing here. */
export function matchShortcut(e: KeyEventLike): Shortcut | null {
  // Ctrl, Cmd and Alt combinations belong to the browser and the OS. Shift is
  // the one modifier this map spends, and only to reach m.
  if (e.ctrlKey || e.metaKey || e.altKey) return null;
  switch (e.key) {
    case "ArrowUp":
      return { kind: "quantum", axis: "n", delta: 1 };
    case "ArrowDown":
      return { kind: "quantum", axis: "n", delta: -1 };
    case "ArrowRight":
      return { kind: "quantum", axis: e.shiftKey ? "m" : "l", delta: 1 };
    case "ArrowLeft":
      return { kind: "quantum", axis: e.shiftKey ? "m" : "l", delta: -1 };
    case "?":
      return { kind: "help" };
    case "Escape":
      return { kind: "close" };
  }
  // The seven views in the order the rail and the tab strip draw them.
  if (/^[1-7]$/.test(e.key) && !e.shiftKey) return { kind: "view", index: Number(e.key) - 1 };
  return null;
}
