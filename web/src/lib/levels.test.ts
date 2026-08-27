import { describe, expect, it } from "vitest";
import type { SpectralLineInfo } from "../api/types";
import { arrowsFor, detailMode, detailState, spreadLabels } from "./levels";

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
    einstein_a_s: null,
    oscillator_strength: null,
    emissivity: null,
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

describe("detailMode", () => {
  const base = { fineStructure: false, bField: 0, eField: 0, hyperfine: false };
  it("is the plain ladder with nothing switched on", () => {
    expect(detailMode(base)).toBe("none");
  });
  it("reproduces the render's old precedence exactly", () => {
    // An electric field beat everything, then hyperfine, then a B field.
    // Changing any of these would change which plot a saved deep link opens.
    expect(detailMode({ ...base, eField: 5, hyperfine: true, fineStructure: true, bField: 3 }))
      .toBe("stark");
    expect(detailMode({ ...base, hyperfine: true, fineStructure: true, bField: 3 }))
      .toBe("hyperfine");
    expect(detailMode({ ...base, fineStructure: true, bField: 3 })).toBe("zeeman");
    expect(detailMode({ ...base, fineStructure: true })).toBe("fine");
  });
  it("does not offer a Zeeman fan without fine structure to split", () => {
    expect(detailMode({ ...base, bField: 3 })).toBe("none");
  });
});

describe("detailState", () => {
  const base = { fineStructure: false, bField: 0, eField: 0, hyperfine: false };
  it("round-trips every mode back through detailMode", () => {
    for (const mode of ["none", "fine", "zeeman", "stark", "hyperfine"] as const) {
      expect(detailMode(detailState(mode, base)), mode).toBe(mode);
    }
  });
  it("opens a field mode somewhere visible rather than at zero", () => {
    expect(detailState("zeeman", base).bField).toBeGreaterThan(0);
    expect(detailState("stark", base).eField).toBeGreaterThan(0);
  });
  it("keeps a strength the user already dialled in", () => {
    expect(detailState("zeeman", { ...base, bField: 11 }).bField).toBe(11);
    expect(detailState("stark", { ...base, eField: 77 }).eField).toBe(77);
  });
  it("turns fine structure on for the two modes that need it", () => {
    expect(detailState("fine", base).fineStructure).toBe(true);
    expect(detailState("zeeman", base).fineStructure).toBe(true);
  });
  it("clears everything for the plain ladder", () => {
    const s = detailState("none", { fineStructure: true, bField: 9, eField: 9, hyperfine: true });
    expect(s).toEqual(base);
  });
});

describe("spreadLabels", () => {
  it("leaves labels that already have room exactly where they are", () => {
    expect(spreadLabels([10, 40, 70], 12, 0, 100)).toEqual([10, 40, 70]);
  });
  it("opens the minimum gap between crowded labels", () => {
    const out = spreadLabels([10, 12, 14], 12, 0, 100);
    for (let i = 1; i < out.length; i++) {
      expect(out[i] - out[i - 1]).toBeGreaterThanOrEqual(12 - 1e-9);
    }
  });
  it("never reorders, so a label cannot end up beside the wrong rung", () => {
    // Deliberately unsorted input: the ladder is drawn top-down and the
    // caller must be free to hand these over in rung order.
    const ys = [70, 12, 14, 10];
    const out = spreadLabels(ys, 12, 0, 100);
    const rank = (arr: number[]) =>
      arr.map((_, i) => i).sort((a, b) => arr[a] - arr[b]).join(",");
    expect(rank(out)).toBe(rank(ys));
  });
  it("pulls a stack back inside the frame when it overflows the bottom", () => {
    const out = spreadLabels([90, 92, 94, 96], 12, 0, 100);
    expect(Math.max(...out)).toBeLessThanOrEqual(100 + 1e-9);
    expect(Math.min(...out)).toBeGreaterThanOrEqual(0 - 1e-9);
    for (let i = 1; i < out.length; i++) {
      expect(out[i] - out[i - 1]).toBeGreaterThanOrEqual(12 - 1e-9);
    }
  });
  it("handles the empty and single cases", () => {
    expect(spreadLabels([], 12, 0, 100)).toEqual([]);
    expect(spreadLabels([50], 12, 0, 100)).toEqual([50]);
  });
});
