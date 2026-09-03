import type { SystemInfo } from "../api/types";

/**
 * Which model stands behind the selected system, or `null` when that is not
 * known yet.
 *
 * The third state is the whole point. Several endpoints are hydrogenic-only
 * and answer 422 for a screened atom, so components gate their requests on the
 * kind. The systems table arrives over the network, which means there is a
 * first render where every lookup misses and the only honest answer is "no
 * idea".
 *
 * Written as a boolean meaning "screened", that render would read false,
 * indistinguishable from hydrogen, and the gate would fire exactly the request
 * it exists to prevent. Not hypothetical: it is the bug this helper was cut
 * out of, and it landed twice independently, because `!isScreened` reads like
 * the right thing every single time you write it.
 *
 * So ask for the fact, not its negation: fetch when `isHydrogenic`, explain
 * when `isScreened`, and do neither while it is null.
 */
export type SystemKind = SystemInfo["kind"] | null;

export function systemKind(systems: SystemInfo[], key: string): SystemKind {
  return systems.find((s) => s.key === key)?.kind ?? null;
}

/** True only once the table has arrived and says hydrogenic. */
export function isHydrogenic(systems: SystemInfo[], key: string): boolean {
  return systemKind(systems, key) === "hydrogenic";
}

/** True only once the table has arrived and says screened. */
export function isScreened(systems: SystemInfo[], key: string): boolean {
  return systemKind(systems, key) === "screened";
}
