import { describe, expect, it } from "vitest";
import { formatHover, nearestIndex, viewBoxX, withinPlot } from "./hover";

describe("nearestIndex", () => {
  const grid = [0, 1, 2, 3, 4];
  it("finds an exact hit", () => {
    expect(nearestIndex(grid, 3)).toBe(3);
  });
  it("rounds to the nearer neighbour on each side of the midpoint", () => {
    expect(nearestIndex(grid, 1.4)).toBe(1);
    expect(nearestIndex(grid, 1.6)).toBe(2);
  });
  it("clamps outside the domain rather than returning nothing", () => {
    expect(nearestIndex(grid, -5)).toBe(0);
    expect(nearestIndex(grid, 99)).toBe(4);
  });
  it("handles degenerate arrays", () => {
    expect(nearestIndex([], 1)).toBe(-1);
    expect(nearestIndex([7], 1)).toBe(0);
  });
  it("agrees with a linear scan on a non-uniform grid", () => {
    // The exponential mesh the many-electron solver runs on: the binary search
    // must not assume uniform spacing, which is exactly where an off-by-one
    // would hide on a uniform test grid.
    const mesh = Array.from({ length: 200 }, (_, j) => 1e-4 * Math.exp(j * 0.06));
    for (const target of [1e-4, 0.003, 0.5, 4, 17, 1e5]) {
      let best = 0;
      for (let i = 1; i < mesh.length; i++) {
        if (Math.abs(mesh[i] - target) < Math.abs(mesh[best] - target)) best = i;
      }
      expect(nearestIndex(mesh, target)).toBe(best);
    }
  });
});

describe("viewBoxX", () => {
  const rect = { left: 100, width: 400 } as DOMRect;
  it("maps a client x onto the viewBox", () => {
    expect(viewBoxX(100, rect, 640)).toBe(0);
    expect(viewBoxX(500, rect, 640)).toBe(640);
    expect(viewBoxX(300, rect, 640)).toBe(320);
  });
  it("returns 0 rather than NaN for an unlaid-out element", () => {
    expect(viewBoxX(300, { left: 0, width: 0 } as DOMRect, 640)).toBe(0);
  });
});

describe("withinPlot", () => {
  it("is inclusive of the margins", () => {
    expect(withinPlot(56, 56, 624)).toBe(true);
    expect(withinPlot(624, 56, 624)).toBe(true);
    expect(withinPlot(55, 56, 624)).toBe(false);
    expect(withinPlot(625, 56, 624)).toBe(false);
  });
});

describe("formatHover", () => {
  it("keeps three significant figures in the readable range", () => {
    expect(formatHover(0.123456)).toBe("0.123");
    expect(formatHover(12.3456)).toBe("12.3");
  });
  it("goes exponential where fixed decimals would print zeros or noise", () => {
    expect(formatHover(1.2e-6)).toBe("1.20e-6");
    expect(formatHover(4.56e7)).toBe("4.56e+7");
  });
  it("prints an exact zero as a single character", () => {
    expect(formatHover(0)).toBe("0");
  });
  it("keeps the sign on a negative wavefunction lobe", () => {
    expect(formatHover(-0.0456)).toBe("-0.0456");
  });
});
