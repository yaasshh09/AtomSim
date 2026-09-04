import { describe, expect, it } from "vitest";
import { clampView, isZoomed, panView, withinView, zoomFactor, zoomView } from "./zoom";

describe("zoomView", () => {
  it("holds the anchored value still", () => {
    // Zooming about the middle of [0, 10] has to leave 5 in the middle.
    const [lo, hi] = zoomView([0, 10], [0, 10], 0.5, 0.5);
    expect((lo + hi) / 2).toBeCloseTo(5, 10);
    expect(hi - lo).toBeCloseTo(5, 10);
  });

  it("keeps the value under the pointer under the pointer", () => {
    const view: [number, number] = [0, 10];
    const anchor = 0.8; // the value 8
    const [lo, hi] = zoomView(view, view, 0.25, anchor);
    expect(lo + anchor * (hi - lo)).toBeCloseTo(8, 10);
  });

  it("never widens past the full range", () => {
    expect(zoomView([2, 3], [0, 10], 100)).toEqual([0, 10]);
  });

  it("stops magnifying rather than collapsing to a point", () => {
    let v: [number, number] = [0, 10];
    for (let i = 0; i < 200; i++) v = zoomView(v, [0, 10], 0.5);
    expect(v[1] - v[0]).toBeGreaterThan(0);
    expect(zoomFactor(v, [0, 10])).toBeLessThanOrEqual(1e4 + 1);
  });

  it("zooms in decades on a log axis", () => {
    // Halving the window of [1, 10000] (four decades) about its centre leaves
    // two decades, centred on 100 rather than on 5000.
    const [lo, hi] = zoomView([1, 1e4], [1, 1e4], 0.5, 0.5, true);
    expect(Math.log10(hi) - Math.log10(lo)).toBeCloseTo(2, 10);
    expect(Math.sqrt(lo * hi)).toBeCloseTo(100, 6);
  });

  it("draws a log-flagged axis linearly when the domain reaches zero", () => {
    // r starts at 0 on the radial plots, and log10(0) would take the whole
    // window to -Infinity instead of refusing the transform.
    const [lo, hi] = zoomView([0, 10], [0, 10], 0.5, 0.5, true);
    expect(Number.isFinite(lo)).toBe(true);
    expect(hi - lo).toBeCloseTo(5, 10);
  });
});

describe("panView", () => {
  it("slides by a fraction of the window, not of the range", () => {
    expect(panView([4, 6], [0, 10], 0.5)).toEqual([5, 7]);
  });

  it("stops at the edge instead of running off the axis", () => {
    expect(panView([8, 10], [0, 10], 1)).toEqual([8, 10]);
    expect(panView([0, 2], [0, 10], -1)).toEqual([0, 2]);
  });

  it("keeps the span when it hits the edge", () => {
    const [lo, hi] = panView([7, 9], [0, 10], 5);
    expect(hi - lo).toBeCloseTo(2, 10);
    expect(hi).toBeCloseTo(10, 10);
  });
});

describe("clampView", () => {
  it("accepts a window given in either order", () => {
    expect(clampView([6, 4], [0, 10])).toEqual([4, 6]);
  });

  it("returns the full range when it is degenerate", () => {
    expect(clampView([1, 2], [5, 5])).toEqual([5, 5]);
  });
});

describe("isZoomed", () => {
  it("is false at the full range and true once narrowed", () => {
    expect(isZoomed([0, 10], [0, 10])).toBe(false);
    expect(isZoomed([0, 9], [0, 10])).toBe(true);
  });
});

describe("withinView", () => {
  it("keeps only the rows the window covers", () => {
    const rows = [{ e: -13.6 }, { e: -3.4 }, { e: -1.5 }];
    expect(withinView(rows, (r) => r.e, [-4, 0])).toHaveLength(2);
  });
});
