import { describe, expect, it } from "vitest";
import type { SpectralLineInfo } from "../api/types";
import { arrowsFor } from "./levels";

function line(nu: number, lu: number, nl: number): SpectralLineInfo {
  return {
    n_upper: nu,
    l_upper: lu,
    j_upper: null,
    n_lower: nl,
    l_lower: 0,
    j_lower: null,
    energy_ev: {} as never,
    wavelength_nm: {} as never,
  };
}

describe("arrowsFor", () => {
  it("keeps only transitions out of the selected state", () => {
    const lines = [line(3, 1, 1), line(3, 1, 2), line(3, 2, 2), line(2, 1, 1)];
    expect(arrowsFor(lines, 3, 1)).toHaveLength(2);
    expect(arrowsFor(lines, 2, 1)).toHaveLength(1);
    expect(arrowsFor(lines, 1, 0)).toHaveLength(0);
  });
});
