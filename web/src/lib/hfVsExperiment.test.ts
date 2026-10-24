import { describe, expect, it } from "vitest";
import type { HFLevels } from "../api/types";
import { koopmansRow, meanAbsError } from "./hfVsExperiment";

const H = 27.211386245988;
function solve(orbitals: [string, number, number][]): HFLevels {
  return {
    orbitals: orbitals.map(([label, occupancy, e]) => ({
      label, occupancy, energy: { value: e }, energy_ev: { value: e * H },
    })),
  } as unknown as HFLevels;
}

describe("Koopmans against NIST", () => {
  // Potassium's real numbers: the 4s at -0.14744 hartree, NIST 4.34066373 eV.
  const k = solve([["1s", 2, -133.53], ["3p", 6, -0.954], ["4s", 1, -0.14744]]);
  const nist = { z: 19, symbol: "K", ground_level: "2S<1/2>", ie_ev: 4.34066373, uncertainty_ev: 1e-7 };

  it("takes the highest occupied orbital and signs the error as Koopmans minus experiment", () => {
    const row = koopmansRow(k, nist);
    expect(row.orbital).toBe("4s");
    expect(row.koopmans).toBeCloseTo(0.14744 * H, 6);
    expect(row.error).toBeCloseTo(0.14744 * H - 4.34066373, 6);
    expect(row.error).toBeLessThan(0); // an alkali: correlation wins, Koopmans low
  });

  it("ignores empty orbitals", () => {
    const withEmpty = solve([["4s", 1, -0.14744], ["3d", 0, -0.05]]);
    expect(koopmansRow(withEmpty, nist).orbital).toBe("4s");
  });

  it("averages the absolute errors", () => {
    const rows = [koopmansRow(k, nist), { ...koopmansRow(k, nist), error: 0.5 }];
    expect(meanAbsError(rows)).toBeCloseTo((Math.abs(rows[0].error) + 0.5) / 2, 9);
  });
});
