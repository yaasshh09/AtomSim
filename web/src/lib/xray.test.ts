import { describe, expect, it } from "vitest";
import { moseleyFit, xrayRow, type Kalpha, type XrayRow } from "./xray";

const RY = 13.605693122994;

describe("K-alpha rows", () => {
  it("averages the doublet 2:1 and adds the relativity estimate to Hartree-Fock", () => {
    const k = {
      z: 26,
      symbol: "Fe",
      ka1: { experiment_ev: 6404 },
      ka2: { experiment_ev: 6391 },
      delta_scf_ev: { value: 6340 },
      relativity_ev: { value: 50 },
      moseley_ev: { value: 6378 },
    } as unknown as Kalpha;
    const row = xrayRow(k);
    expect(row.measured).toBeCloseTo((2 * 6404 + 6391) / 3, 9);
    expect(row.hfRel).toBe(6390);
    expect(row.sqrtRy).toBeCloseTo(Math.sqrt(row.measured / RY), 12);
  });

  it("recovers Bohr's slope and zero from an exact Moseley series", () => {
    const rows = [10, 20, 30].map((z) => ({ z, sqrtRy: Math.sqrt(0.75) * (z - 1) })) as XrayRow[];
    const fit = moseleyFit(rows)!;
    expect(fit.slope).toBeCloseTo(Math.sqrt(0.75), 12);
    expect(fit.zeroAt).toBeCloseTo(1, 12);
    expect(moseleyFit(rows.slice(0, 1))).toBeNull();
  });
});
