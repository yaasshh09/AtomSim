/**
 * What I remember about tours in your browser between visits.
 *
 * I keep two separate facts on purpose. `dismissed` answers "have I already
 * offered this reader a tour", and it is the only thing my invitation reads.
 * `completed` answers "which tours has this reader finished", and it is what
 * my menu's done markers read. If I collapsed them into one flag, skipping the
 * invitation would silently claim you had taken the tour, which is the kind of
 * quiet lie I exist not to tell.
 *
 * None of this is physics, so none of it carries provenance. It is a record of
 * what I have shown you, and it never feeds a number.
 */

const KEY = "atomsim.tours.v1";

export interface TourMemory {
  /** You have answered my invitation, by taking a tour or by skipping it. */
  dismissed: boolean;
  /** The ids of tours you read to the last step. */
  completed: string[];
}

/** A reader I have never seen. A fresh object, since my callers push onto `completed`. */
export function noMemory(): TourMemory {
  return { dismissed: false, completed: [] };
}

/** Do I still offer this reader the tour? */
export function shouldInvite(m: TourMemory): boolean {
  return !m.dismissed;
}

/**
 * What I remember after you finish `id`.
 *
 * Finishing answers my invitation too: if you took a tour from the menu I
 * should not then ask whether you would like one.
 */
export function withCompleted(m: TourMemory, id: string): TourMemory {
  return {
    dismissed: true,
    completed: m.completed.includes(id) ? [...m.completed] : [...m.completed, id],
  };
}

/**
 * I read a stored record, and treat anything unexpected as a first visit.
 *
 * localStorage is a namespace you can edit by hand and that older builds of me
 * may have written, so I check every field rather than trusting it. What I am
 * defending against is a thrown parse on load, which would take all of me down
 * over a preference.
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
 * Your storage, or null when there is not one to be had.
 *
 * Three ways I come back null: I am running under vitest, which has no window;
 * a privacy mode where touching the property itself throws; a browser with
 * storage switched off. All three mean the same thing to me, and none of them
 * is worth an error you would have to read.
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
 * I persist a record, or fail silently.
 *
 * A refused or full storage costs you nothing this session: my store holds the
 * same fact in memory, so my invitation still goes away when you dismiss it. I
 * simply will not remember it next time.
 */
export function writeMemory(m: TourMemory): void {
  const s = storage();
  if (!s) return;
  try {
    s.setItem(KEY, JSON.stringify(m));
  } catch {
    /* full, refused, or gone: I already behave correctly this session without it */
  }
}

/** I record that you have answered my invitation. */
export function rememberDismissed(): TourMemory {
  const m = { ...readMemory(), dismissed: true };
  writeMemory(m);
  return m;
}

/** I record that you read a tour to the end. */
export function rememberCompleted(id: string): TourMemory {
  const m = withCompleted(readMemory(), id);
  writeMemory(m);
  return m;
}
