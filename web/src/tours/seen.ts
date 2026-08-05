/**
 * What the browser remembers about tours between visits.
 *
 * Two separate facts, on purpose. `dismissed` answers "has this reader already
 * been offered a tour", and it is the only thing the invitation reads.
 * `completed` answers "which tours has this reader finished", and it is what
 * the menu's done markers read. Collapsing them into one flag would mean that
 * skipping the invitation silently claimed the reader had taken the tour,
 * which is the kind of quiet lie this project exists not to tell.
 *
 * Nothing here is physics, so nothing here carries provenance. It is a record
 * of what the reader has been shown, and it never feeds a number.
 */

const KEY = "atomsim.tours.v1";

export interface TourMemory {
  /** The reader has answered the invitation, by taking a tour or by skipping. */
  dismissed: boolean;
  /** Ids of tours read to the last step. */
  completed: string[];
}

/** A reader we have never seen. A fresh object: callers push onto `completed`. */
export function noMemory(): TourMemory {
  return { dismissed: false, completed: [] };
}

/** Does this reader still get offered the tour? */
export function shouldInvite(m: TourMemory): boolean {
  return !m.dismissed;
}

/**
 * The memory after finishing `id`.
 *
 * Finishing answers the invitation too: a reader who took a tour from the menu
 * should not then be asked whether they would like one.
 */
export function withCompleted(m: TourMemory, id: string): TourMemory {
  return {
    dismissed: true,
    completed: m.completed.includes(id) ? [...m.completed] : [...m.completed, id],
  };
}

/**
 * Read a stored record, treating anything unexpected as a first visit.
 *
 * localStorage is a namespace the reader can edit by hand and that older builds
 * may have written, so every field is checked rather than trusted. The failure
 * mode being defended against is a thrown parse on load, which would take the
 * whole app down over a preference.
 */
export function parseMemory(raw: string | null): TourMemory {
  if (!raw) return noMemory();
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return noMemory();
  }
  if (typeof value !== "object" || value === null) return noMemory();
  const rec = value as Record<string, unknown>;
  return {
    dismissed: rec.dismissed === true,
    completed: Array.isArray(rec.completed)
      ? rec.completed.filter((x): x is string => typeof x === "string")
      : [],
  };
}

/**
 * The reader's storage, or null when there is not one to be had.
 *
 * Three ways this comes back null: the module is running under vitest, which
 * has no window; a privacy mode where touching the property itself throws; a
 * browser with storage switched off. All three mean the same thing to us, and
 * none of them is worth an error the reader would have to read.
 */
function storage(): Storage | null {
  try {
    if (typeof window === "undefined") return null;
    return window.localStorage ?? null;
  } catch {
    return null;
  }
}

export function readMemory(): TourMemory {
  const s = storage();
  if (!s) return noMemory();
  try {
    return parseMemory(s.getItem(KEY));
  } catch {
    return noMemory();
  }
}

/**
 * Persist a record, or fail silently.
 *
 * A refused or full storage costs the reader nothing this session: the store
 * holds the same fact in memory, so the invitation still goes away when they
 * dismiss it. It simply will not be remembered next time.
 */
export function writeMemory(m: TourMemory): void {
  const s = storage();
  if (!s) return;
  try {
    s.setItem(KEY, JSON.stringify(m));
  } catch {
    /* full, refused, or gone: this session already behaves correctly without it */
  }
}

/** Record that the invitation has been answered. */
export function rememberDismissed(): TourMemory {
  const m = { ...readMemory(), dismissed: true };
  writeMemory(m);
  return m;
}

/** Record that a tour was read to the end. */
export function rememberCompleted(id: string): TourMemory {
  const m = withCompleted(readMemory(), id);
  writeMemory(m);
  return m;
}
