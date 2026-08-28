import aRealSpectrum from "./a-real-spectrum.json";
import breakThePhysics from "./break-the-physics.json";
import hydrogenHonestly from "./hydrogen-honestly.json";
import manyElectrons from "./many-electrons.json";
import type { Tour } from "./types";

/**
 * Every tour I offer, in menu order.
 *
 * I import the JSON rather than fetching it so it is bundled and typechecked
 * at build time, and so `atomsim serve` needs no new endpoint. My
 * `tests/test_tour_claims.py` reads the same files, which is the entire reason
 * I keep the content as data: one file, rendered by one half of me and checked
 * against the engine by the other.
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
 * The tour I offer you the first time you arrive.
 *
 * I name it rather than reading TOURS[0] so that reordering the menu cannot
 * quietly change which tour my invitation opens, and so a rename of the JSON
 * id breaks a test instead of an invitation button. registry.test.ts asserts
 * that it resolves.
 */
export const FLAGSHIP_TOUR_ID = "hydrogen-honestly";
