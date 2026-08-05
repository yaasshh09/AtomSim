import { describe, expect, it } from "vitest";
import { formatOffset, offsetAxis, offsetTicks, thinTicks } from "./axis";

describe("offsetTicks", () => {
  it("puts a tick at the line centre", () => {
    const half = 4.92e-6 * 8;
    const lo = 121.568446 - half;
    const hi = 121.568446 + half;
    const a = offsetAxis(lo, hi);
    const labels = offsetTicks(a, lo, hi).map((t) => formatOffset(a, t));
    expect(labels).toContain("0");
  });

  it("lands on round offsets, not round absolute wavelengths", () => {
    const half = 4.92e-6 * 8;
    const lo = 121.568446 - half;
    const hi = 121.568446 + half;
    const a = offsetAxis(lo, hi);
    for (const t of offsetTicks(a, lo, hi)) {
      const off = (t - a.centreNm) * a.perNm;
      // Every tick is a whole number of display units.
      expect(Math.abs(off - Math.round(off))).toBeLessThan(1e-6);
    }
  });

  it("stays inside the window it was given", () => {
    const a = offsetAxis(400, 700);
    for (const t of offsetTicks(a, 400, 700)) {
      expect(t).toBeGreaterThanOrEqual(400);
      expect(t).toBeLessThanOrEqual(700);
    }
  });
});

describe("offsetAxis", () => {
  it("picks femtometres for a natural-width window", () => {
    // Lyman-alpha at 8 half-widths of its 4.92e-6 nm natural width: the window
    // that printed "121.568" at all six ticks.
    const half = 4.92e-6 * 8;
    const a = offsetAxis(121.568 - half, 121.568 + half);
    expect(a.unit).toBe("fm");
    expect(a.centreNm).toBeCloseTo(121.568, 9);
  });

  it("picks nanometres for a window wide enough to read directly", () => {
    expect(offsetAxis(400, 700).unit).toBe("nm");
  });

  it("picks picometres in between", () => {
    expect(offsetAxis(656.0, 656.05).unit).toBe("pm");
  });

  it("prints the centre fine enough to resolve one display unit", () => {
    // fm ticks against a centre rounded to 2 dp would be offsets from a
    // wavelength 500 fm away from the one they are offsets from.
    expect(offsetAxis(121.568 - 4e-5, 121.568 + 4e-5).centreDecimals).toBe(6);
    expect(offsetAxis(400, 700).centreDecimals).toBe(2);
  });
});

describe("formatOffset", () => {
  it("gives distinguishable labels where toFixed(3) gave six identical ones", () => {
    const half = 4.92e-6 * 8;
    const a = offsetAxis(121.568 - half, 121.568 + half);
    const ticks = [
      121.568 - half,
      121.568 - half / 2,
      121.568,
      121.568 + half / 2,
      121.568 + half,
    ];
    const labels = ticks.map((t) => formatOffset(a, t));
    expect(new Set(labels).size).toBe(labels.length);
    expect(labels[2]).toBe("0");
  });

  it("signs every offset so none reads as an absolute wavelength", () => {
    const a = offsetAxis(656.0, 656.05);
    expect(formatOffset(a, 656.04)).toMatch(/^\+/);
    expect(formatOffset(a, 656.01)).toMatch(/^-/);
  });

  it("prints the centre tick as a bare zero", () => {
    const a = offsetAxis(100, 200);
    expect(formatOffset(a, 150)).toBe("0");
  });
});

describe("thinTicks", () => {
  // A log axis: 5000 and 6000 land a few pixels apart while 90 and 100 do not.
  const logX = (v: number) => Math.log10(v) * 300;

  it("drops labels that would overprint their neighbour", () => {
    const ticks = [90, 100, 200, 300, 500, 1000, 2000, 5000, 8000, 10000];
    const kept = thinTicks(ticks, logX, 30);
    expect(kept.length).toBeLessThan(ticks.length);
    for (let i = 1; i < kept.length; i++) {
      expect(Math.abs(logX(kept[i]) - logX(kept[i - 1]))).toBeGreaterThanOrEqual(30);
    }
  });

  it("always keeps both bounds, so the range stays readable", () => {
    const ticks = [90, 100, 200, 300, 500, 1000, 2000, 5000, 8000, 10000];
    const kept = thinTicks(ticks, logX, 500);
    expect(kept[0]).toBe(90);
    expect(kept[kept.length - 1]).toBe(10000);
  });

  it("leaves a short tick list alone", () => {
    expect(thinTicks([1, 2], logX, 999)).toEqual([1, 2]);
    expect(thinTicks([5], logX, 999)).toEqual([5]);
  });

  it("keeps everything when there is room", () => {
    const ticks = [1, 10, 100, 1000];
    expect(thinTicks(ticks, logX, 10)).toEqual(ticks);
  });
});
