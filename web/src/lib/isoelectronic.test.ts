import { describe, expect, it } from "vitest";
import type { HFLevels } from "../api/types";
import { ionLabel, isoRow, type IsoelectronicSequence } from "./isoelectronic";

const H = 27.211386245988;
const prov = (err: number | null) => ({ error_estimate: err });

describe("isoelectronic rows", () => {
  it("labels ions with superscript charges", () => {
    expect(ionLabel("He", 0)).toBe("He");
    expect(ionLabel("Li", 1)).toBe("Li⁺");
    expect(ionLabel("Ne", 8)).toBe("Ne⁸⁺");
    expect(ionLabel("Ca", 10)).toBe("Ca¹⁰⁺");
  });

  it("subtracts the first two orders and carries both error bars", () => {
    const seq = {
      n_electrons: 2,
      e0: { value: -1, provenance: prov(null) },
      e1: { value: 0.625, provenance: prov(1e-5) },
    } as unknown as IsoelectronicSequence;
    // Li+ at its Hartree-Fock energy: -9 + 1.875 + E2.
    const levels = {
      total_energy: { value: -7.236415, provenance: prov(3e-5) },
      orbitals: [{ label: "1s", occupancy: 2, energy: { value: -2.792 }, energy_ev: { value: -2.792 * H } }],
    } as unknown as HFLevels;
    const ion = { z: 3, symbol: "Li", spectrum: "Li II", ie_ev: 75.64, uncertainty_ev: 1e-6, kind: "theoretical" as const };
    const row = isoRow(levels, seq, ion);
    expect(row.label).toBe("Li⁺");
    expect(row.residual).toBeCloseTo(-7.236415 + 9 - 1.875, 9);
    expect(row.residualBar).toBeCloseTo(3e-5 + 3e-5, 12);
    expect(row.koopmans).toBeCloseTo(2.792 * H, 6);
    expect(row.kind).toBe("theoretical");
  });
});
