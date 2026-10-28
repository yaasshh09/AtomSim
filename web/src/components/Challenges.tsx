import { useEffect, useState } from "react";
import { CHALLENGES, solvedBy } from "../lib/challenges";
import { Notation } from "../lib/mathText";
import { currentUrlState } from "../lib/urlState";
import { useAppStore } from "../state/store";
import { Disclosure } from "./Disclosure";

const KEY = "atomsim.challenges.solved";

/** Solved ids from this browser; a convenience, so any failure means none. */
function remembered(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(KEY) ?? "[]") as string[]);
  } catch {
    return new Set();
  }
}

function remember(ids: Set<string>): void {
  try {
    localStorage.setItem(KEY, JSON.stringify([...ids]));
  } catch {
    // Private windows and blocked storage: the ticks just do not persist.
  }
}

/**
 * Puzzles solved by putting the app into the right state, ticked off live.
 * A solved challenge shows why its answer is the answer.
 */
export function Challenges() {
  const state = useAppStore();
  const [solved, setSolved] = useState<Set<string>>(remembered);
  // Keyed by content: the set is rebuilt from the live state on every render.
  const freshKey = [...solvedBy(currentUrlState(state), state.systems)]
    .filter((id) => !solved.has(id))
    .join(",");

  useEffect(() => {
    if (!freshKey) return;
    setSolved((prev) => {
      const next = new Set([...prev, ...freshKey.split(",")]);
      remember(next);
      return next;
    });
  }, [freshKey]);

  return (
    <div data-tour="challenges">
      <Disclosure summary={`Challenges: ${solved.size} of ${CHALLENGES.length} solved`}>
        <ol className="challenge-list">
          {CHALLENGES.map((c) => {
            const done = solved.has(c.id);
            return (
              <li key={c.id} className={done ? "challenge done" : "challenge"}>
                <span aria-hidden="true">{done ? "✓ " : "○ "}</span>
                <Notation>{c.prompt}</Notation>
                {done ? (
                  <p className="caption">
                    <Notation>{c.why}</Notation>
                  </p>
                ) : (
                  <details>
                    <summary>hint</summary>
                    <p className="caption">
                      <Notation>{c.hint}</Notation>
                    </p>
                  </details>
                )}
              </li>
            );
          })}
        </ol>
      </Disclosure>
    </div>
  );
}
