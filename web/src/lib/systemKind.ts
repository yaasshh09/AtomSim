import type { SystemInfo } from "../api/types";

/**
 * Which model I am standing behind for the selected system, or `null` when I
 * do not know yet.
 *
 * That third state is the whole point. Several of my endpoints are
 * hydrogenic-only and answer 422 for a screened atom, so my components gate
 * their requests on the kind. My systems table arrives over the network, which
 * means there is a first render where every lookup misses and the honest
 * answer I can give is "no idea".
 *
 * If I wrote this as a boolean meaning "screened", that render would read
 * false, I could not tell it apart from hydrogen, and the gate would fire
 * exactly the request it exists to prevent. That is not hypothetical: it is
 * the bug I cut this helper out of, and I hit it twice independently, because
 * `!isScreened` reads like the right thing every time you write it.
 *
 * So I ask for the fact I need rather than its negation: I fetch when
 * `isHydrogenic`, explain when `isScreened`, and do neither while it is null.
 */
export type SystemKind = SystemInfo["kind"] | null;

export function systemKind(systems: SystemInfo[], key: string): SystemKind {
  return systems.find((s) => s.key === key)?.kind ?? null;
}

/** True only once my table has arrived and says hydrogenic. */
export function isHydrogenic(systems: SystemInfo[], key: string): boolean {
  return systemKind(systems, key) === "hydrogenic";
}

/** True only once my table has arrived and says screened. */
export function isScreened(systems: SystemInfo[], key: string): boolean {
  return systemKind(systems, key) === "screened";
}
