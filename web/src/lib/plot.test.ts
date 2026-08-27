import { scaleLinear } from "d3-scale";
import { describe, expect, it } from "vitest";
import { informativeEndSigned, linePath, zeroCrossings } from "./plot";

describe("linePath", () => {
  it("builds an SVG path under identity scales", () => {
    const s = scaleLinear([0, 1], [0, 1]);
    expect(linePath([0, 0.5, 1], [0, 1, 0], s, s)).toBe(
      "M0.00,0.00L0.50,1.00L1.00,0.00",
    );
  });
  it("empty input gives an empty path", () => {
    const s = scaleLinear([0, 1], [0, 1]);
    expect(linePath([], [], s, s)).toBe("");
  });
});

describe("zeroCrossings", () => {
  it("finds the crossing where the polyline actually meets the axis", () => {
    expect(zeroCrossings([0, 1, 2], [1, -1, 1])).toEqual([0.5, 1.5]);
  });
  it("interpolates an off-centre crossing", () => {
    expect(zeroCrossings([0, 1], [3, -1])).toEqual([0.75]);
  });
  it("ignores a touch that does not change sign", () => {
    // P(r) = r^2 R^2 grazes zero at every node without ever going negative;
    // marking those as crossings would put a node mark on a curve that has none.
    expect(zeroCrossings([0, 1, 2, 3], [1, 0, 1, 4])).toEqual([]);
  });
  it("counts the n - l - 1 radial nodes of a 3s-like curve", () => {
    // Two sign changes, which is 3 - 0 - 1.
    const grid = Array.from({ length: 400 }, (_, i) => i * 0.1);
    const values = grid.map((r) => (1 - (2 / 3) * r + (2 / 27) * r * r) * Math.exp(-r / 3));
    expect(zeroCrossings(grid, values)).toHaveLength(2);
  });
  it("returns nothing for a node-free curve or an empty one", () => {
    expect(zeroCrossings([0, 1, 2], [1, 2, 3])).toEqual([]);
    expect(zeroCrossings([], [])).toEqual([]);
  });
});

describe("informativeEndSigned", () => {
  it("keeps a negative lobe that a positive-only peak search would drop", () => {
    // The bug this exists to avoid: measured against max(v) rather than
    // max(|v|), a curve that is mostly negative has no samples "above" the
    // peak at all and the window collapses.
    const values = [0.001, -1, -0.5, 0.0001, 0.00001];
    expect(informativeEndSigned(values)).toBe(3);
  });
  it("trims a flat tail to where the curve stops mattering", () => {
    const values = [1, 0.5, 0.2, 1e-9, 1e-9, 1e-9];
    expect(informativeEndSigned(values)).toBe(3);
  });
  it("keeps the whole curve when nothing is negligible", () => {
    expect(informativeEndSigned([1, 2, 3])).toBe(3);
  });
  it("never returns a window too short to draw", () => {
    expect(informativeEndSigned([1, 1e-9, 1e-9])).toBe(2);
  });
  it("returns the full length rather than nothing for an all-zero curve", () => {
    expect(informativeEndSigned([0, 0, 0])).toBe(3);
    expect(informativeEndSigned([])).toBe(0);
  });
});
