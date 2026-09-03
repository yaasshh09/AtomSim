import aRealSpectrum from "./a-real-spectrum.json";
import breakThePhysics from "./break-the-physics.json";
import hydrogenHonestly from "./hydrogen-honestly.json";
import manyElectrons from "./many-electrons.json";
import type { Tour } from "./types";

/**
 * Every tour on offer, in menu order.
 *
 * The JSON is imported rather than fetched, so it is bundled and typechecked
 * at build time and `atomsim serve` needs no new endpoint.
 * `tests/test_tour_claims.py` reads the same files, which is the entire reason
 * the content stays as data: one file, rendered by one half of the app and
 * checked against the engine by the other.
 *
 * The cast is the seam where JSON meets the type. `registry.test.ts` is what
 * makes it safe: it walks every tour and asserts the shape the cast promises.
 */
export const TOURS: Tour[] = [
  hydrogenHonestly as Tour,
  breakThePhysics as Tour,
  manyElectrons as Tour,
  aRealSpectrum as Tour,
];

export function tourById(id: string): Tour | null {
  return TOURS.find((t) => t.id === id) ?? null;
}

/**
 * The tour offered on a first visit.
 *
 * Named rather than read off TOURS[0], so reordering the menu cannot quietly
 * change which tour the invitation opens, and a rename of the JSON id breaks a
 * test instead of an invitation button. registry.test.ts asserts that it
 * resolves.
 */
export const FLAGSHIP_TOUR_ID = "hydrogen-honestly";
