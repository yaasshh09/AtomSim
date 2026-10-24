import { describe, expect, it } from "vitest";
import type { HFLevels } from "../api/types";
import { RACE_CASES, rankRace, raceVerdict, shortConfig } from "./orderingPuzzle";

const H = 27.211386245988;

/** Just the fields the ranking reads, shaped like a real Hartree-Fock job meta. */
function fake(total: number, err: number, orbitals: [number, number, number][]): HFLevels {
  return {
    total_energy: { value: total, unit: "hartree", label: "E", provenance: { error_estimate: err } },
    orbitals: orbitals.map(([n, l, e]) => ({ n, l, energy_ev: { value: e * H } })),
  } as unknown as HFLevels;
}

describe("the 3d/4s race", () => {
  // Scandium's two configurations, with the solver's own numbers.
  const ground = fake(-759.73267, 0.0067, [[3, 2, -0.3436], [4, 0, -0.2101]]);
  const moved = fake(-759.6602, 0.0068, [[3, 2, -0.192], [4, 0, -0.1858]]);

  it("ranks by total energy, whatever order the solves finished in", () => {
    const rows = rankRace([{ tail: "3d2 4s1", levels: moved }, { tail: "3d1 4s2", levels: ground }]);
    expect(rows.map((r) => r.tail)).toEqual(["3d1 4s2", "3d2 4s1"]);
    expect(rows[0].gapEv).toBe(0);
    expect(rows[1].gapEv).toBeCloseTo((759.73267 - 759.6602) * H, 6);
    expect(rows[1].gapErrEv).toBeCloseTo((0.0067 + 0.0068) * H, 6);
  });

  it("names the puzzle when the winner's 3d is the deeper orbital", () => {
    const v = raceVerdict(rankRace([{ tail: "3d1 4s2", levels: ground }, { tail: "3d2 4s1", levels: moved }]));
    expect(v).toContain("[Ar] 3d 4s² has the lowest total energy");
    expect(v).toContain("lies below the 4s");
  });

  it("says when there is no puzzle", () => {
    const k = rankRace([
      { tail: "4s1", levels: fake(-599.16, 0.006, [[4, 0, -0.147]]) },
      { tail: "3d1", levels: fake(-599.07, 0.006, [[3, 2, -0.058]]) },
    ]);
    expect(raceVerdict(k)).toContain("no puzzle yet");
  });

  it("writes configurations the chemistry-book way", () => {
    expect(shortConfig("3d2 4s1")).toBe("[Ar] 3d² 4s");
    expect(RACE_CASES.map((c) => c.key)).toEqual(["k", "ca", "sc", "sc+"]);
  });
});
