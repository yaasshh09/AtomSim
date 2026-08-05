import { describe, expect, it } from "vitest";
import { informativeEnd, sparkBars } from "./spark";

describe("informativeEnd", () => {
  it("cuts the long dead tail a solver grid leaves behind", () => {
    // Peak of 100 up front, then a tail far under the 1e-3 floor.
    const values = [1, 100, 50, 10, ...new Array(200).fill(0.001)];
    expect(informativeEnd(values)).toBe(4);
  });

  it("keeps the whole curve when all of it is above the floor", () => {
    const values = [1, 2, 3, 2, 1];
    expect(informativeEnd(values)).toBe(5);
  });

  it("never returns a window too short to draw", () => {
    expect(informativeEnd([100, 0, 0, 0])).toBe(2);
  });

  it("returns the full length rather than nothing when there is no peak", () => {
    expect(informativeEnd([0, 0, 0])).toBe(3);
    expect(informativeEnd([])).toBe(0);
  });
});

describe("sparkBars", () => {
  it("returns one height per bar, normalized to the peak", () => {
    const bars = sparkBars([1, 2, 3, 4], 4);
    expect(bars).toHaveLength(4);
    expect(Math.max(...bars)).toBe(1);
    expect(bars).toEqual([0.25, 0.5, 0.75, 1]);
  });

  it("keeps the peak where the function put it", () => {
    // A sharp peak early on, then a long tail. Binning by mean would drag the
    // drawn maximum toward the tail; the card would then point at the wrong r.
    const values = [0, 10, 0, 1, 1, 1, 1, 1];
    const bars = sparkBars(values, 4);
    expect(bars.indexOf(Math.max(...bars))).toBe(0);
  });

  it("draws nothing rather than a flat line when there is no curve", () => {
    expect(sparkBars([], 8)).toEqual([]);
    expect(sparkBars([0, 0, 0], 8)).toEqual([]);
    expect(sparkBars([1, 2], 0)).toEqual([]);
  });

  it("ignores non-finite samples instead of poisoning the whole card", () => {
    const bars = sparkBars([1, Number.NaN, 2, Number.POSITIVE_INFINITY], 2);
    expect(bars).toEqual([1 / 2, 1]);
  });

  it("fills every bar when the curve is shorter than the bar count", () => {
    const bars = sparkBars([1, 2], 5);
    expect(bars).toHaveLength(5);
    expect(bars.every((b) => b > 0)).toBe(true);
  });
});
