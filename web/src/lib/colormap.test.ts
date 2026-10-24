import { describe, expect, it } from "vitest";
import {
  DENSITY_GAMMA,
  densityT,
  lutColor,
  maxAbs,
  maxOf,
  phaseColor,
  signedT,
} from "./colormap";
import { PHASE } from "./luts";
import { INFERNO, RDBU_R } from "./luts";

describe("luts", () => {
  it("has 256 rgb triples in each table", () => {
    for (const lut of [INFERNO, RDBU_R]) {
      expect(lut.length).toBe(256);
      for (const [r, g, b] of lut) {
        for (const c of [r, g, b]) {
          expect(Number.isInteger(c)).toBe(true);
          expect(c).toBeGreaterThanOrEqual(0);
          expect(c).toBeLessThanOrEqual(255);
        }
      }
    }
  });
  it("inferno runs dark to light", () => {
    const sum = (i: number) => INFERNO[i][0] + INFERNO[i][1] + INFERNO[i][2];
    expect(sum(0)).toBeLessThan(30);
    expect(sum(255)).toBeGreaterThan(600);
  });
});

describe("lutColor", () => {
  it("maps endpoints and clamps", () => {
    expect(lutColor(INFERNO, 0)).toEqual(INFERNO[0]);
    expect(lutColor(INFERNO, 1)).toEqual(INFERNO[255]);
    expect(lutColor(INFERNO, -5)).toEqual(INFERNO[0]);
    expect(lutColor(INFERNO, 5)).toEqual(INFERNO[255]);
  });
});

describe("densityT", () => {
  it("gamma-compresses with the disclosed exponent", () => {
    expect(DENSITY_GAMMA).toBe(0.5);
    expect(densityT(4, 4)).toBe(1);
    expect(densityT(1, 4)).toBeCloseTo(0.5, 12); // sqrt(0.25)
    expect(densityT(0, 4)).toBe(0);
    expect(densityT(1, 0)).toBe(0); // degenerate max
  });
});

describe("signedT", () => {
  it("centres zero at 0.5 and clamps", () => {
    expect(signedT(0, 2)).toBe(0.5);
    expect(signedT(2, 2)).toBe(1);
    expect(signedT(-2, 2)).toBe(0);
    expect(signedT(99, 2)).toBe(1);
    expect(signedT(1, 0)).toBe(0.5);
  });
});

describe("max helpers", () => {
  it("maxOf and maxAbs", () => {
    expect(maxOf(new Float32Array([1, 5, 2]))).toBe(5);
    expect(maxAbs(new Float32Array([1, -7, 2]))).toBe(7);
    expect(maxOf(new Float32Array([]))).toBe(0);
  });
});

describe("phaseColor", () => {
  it("is cyclic across the -pi/+pi seam", () => {
    expect(phaseColor(-Math.PI)).toEqual(phaseColor(Math.PI));
  });
  it("reads the colour-blind-safe PHASE table, phase 0 at its start", () => {
    expect(phaseColor(0)).toEqual(PHASE[0]);
    expect(phaseColor(2 * Math.PI)).toEqual(PHASE[0]);
    expect(phaseColor(Math.PI)).toEqual(PHASE[128]);
  });
});
