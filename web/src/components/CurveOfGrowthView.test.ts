import { describe, expect, it } from "vitest";
import type { GrowthRegime } from "../api/types";
import { decadeTicks, logLogPath, regimeSegments } from "./CurveOfGrowthView";

const R = (s: string, n: number): GrowthRegime[] =>
  Array(n).fill(s) as GrowthRegime[];

describe("regimeSegments", () => {
  it("splits a curve into one run per branch", () => {
    const segs = regimeSegments([
      ...R("linear", 3), ...R("saturated", 2), ...R("damping", 2),
    ]);
    expect(segs.map((s) => s.regime)).toEqual(["linear", "saturated", "damping"]);
  });

  it("overlaps consecutive runs by a point so the line has no gaps", () => {
    // Without the shared endpoint the polyline breaks at every branch change,
    // which reads as missing data rather than as a transition.
    const segs = regimeSegments([...R("linear", 3), ...R("saturated", 3)]);
    expect(segs[0].end).toBe(2);
    expect(segs[1].start).toBe(2);
  });

  it("handles a single branch and an empty curve", () => {
    expect(regimeSegments(R("linear", 4))).toHaveLength(1);
    expect(regimeSegments([])).toEqual([]);
  });

  it("never starts a segment before the first point", () => {
    expect(regimeSegments(R("linear", 2))[0].start).toBe(0);
  });
});

describe("logLogPath", () => {
  it("walks the requested index range in log space", () => {
    const path = logLogPath([1, 10, 100], [1, 10, 100], 0, 2, (v) => v, (v) => v);
    expect(path).toBe("M0.00 0.00 L1.00 1.00 L2.00 2.00");
  });

  it("skips non-positive values instead of emitting NaN", () => {
    // log10(0) is -Infinity and would poison the whole path attribute.
    const path = logLogPath([1, 0, 100], [1, 1, 1], 0, 2, (v) => v, (v) => v);
    expect(path).not.toContain("NaN");
    expect(path).not.toContain("Infinity");
    expect(path.split("L")).toHaveLength(2);
  });

  it("returns an empty string when nothing is drawable", () => {
    expect(logLogPath([0], [0], 0, 0, (v) => v, (v) => v)).toBe("");
  });
});

describe("decadeTicks", () => {
  it("returns whole decades inside the range", () => {
    expect(decadeTicks(12.3, 16.8)).toEqual([13, 14, 15, 16]);
  });

  it("thins out when there are too many decades to label", () => {
    const ticks = decadeTicks(0, 40, 8);
    expect(ticks.length).toBeLessThanOrEqual(8);
    expect(ticks[0]).toBe(0);
  });

  it("copes with a range containing no whole decade", () => {
    expect(decadeTicks(12.2, 12.8)).toEqual([]);
  });
});
