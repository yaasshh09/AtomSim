import aRealSpectrum from "./a-real-spectrum.json";
import breakThePhysics from "./break-the-physics.json";
import hydrogenHonestly from "./hydrogen-honestly.json";
import manyElectrons from "./many-electrons.json";
import type { Tour } from "./types";

/**
 * Every tour, in menu order.
 *
 * The JSON is imported rather than fetched so it is bundled and typechecked at
 * build time, and so `atomsim serve` needs no new endpoint. The same files are
 * read by `tests/test_tour_claims.py`, which is the entire reason the content
 * is data: one file, rendered by one half of the project and checked against
 * the engine by the other.
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
 * The tour a first-time reader is offered.
 *
 * Named rather than read as TOURS[0] so that reordering the menu cannot
 * quietly change which tour the invitation opens, and so a rename of the JSON
 * id breaks a test instead of an invitation button. registry.test.ts asserts
 * it resolves.
 */
export const FLAGSHIP_TOUR_ID = "hydrogen-honestly";
