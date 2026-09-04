import { describe, expect, it } from "vitest";
import { MIN_PLOT_WIDTH, plotFit, plotHeight, plotWidth } from "./plotSize";

describe("plotFit", () => {
  it("draws at the width that is there, not at the floor", () => {
    // The phone case. 366px of container used to produce a 560px viewBox and
    // 194px of horizontal scrollbar.
    expect(plotFit(366, 560)).toEqual({ width: 366, compact: true });
  });

  it("says nothing is compact once the furniture fits", () => {
    expect(plotFit(742, 560)).toEqual({ width: 742, compact: false });
    expect(plotFit(560, 560)).toEqual({ width: 560, compact: false });
  });

  it("keeps the absolute floor against a transient zero", () => {
    expect(plotFit(0, 560).width).toBe(MIN_PLOT_WIDTH);
  });
});

describe("plotWidth", () => {
  it("passes a real measurement through", () => {
    expect(plotWidth(742)).toBe(742);
  });

  it("holds the floor against a transient zero", () => {
    // A ResizeObserver reports 0 for an element that has not been laid out.
    // Straight through, that is a zero-span d3 range and NaN in every
    // coordinate the plot draws.
    expect(plotWidth(0)).toBe(MIN_PLOT_WIDTH);
    expect(plotWidth(-5)).toBe(MIN_PLOT_WIDTH);
    expect(plotWidth(200)).toBe(MIN_PLOT_WIDTH);
  });

  it("takes a wider floor from a view that needs one", () => {
    // A ladder carries three columns of text and does not hold together at
    // 320, so it asks for more and gets it.
    expect(plotWidth(400, 560)).toBe(560);
    expect(plotWidth(700, 560)).toBe(700);
  });

  it("rounds, so gridlines land on whole pixels", () => {
    expect(plotWidth(742.4)).toBe(742);
    expect(plotWidth(742.6)).toBe(743);
  });
});

describe("plotHeight", () => {
  it("follows the width in the middle of the range", () => {
    expect(plotHeight(800, 0.375, 180, 300)).toBe(300);
    expect(plotHeight(600, 0.375, 180, 300)).toBe(225);
  });

  it("clamps at both ends", () => {
    expect(plotHeight(320, 0.375, 180, 300)).toBe(180);
    expect(plotHeight(2000, 0.375, 180, 300)).toBe(300);
  });
});
