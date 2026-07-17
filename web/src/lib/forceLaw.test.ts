import { describe, expect, it } from "vitest";
import { allowedSpan, clampParam, defaultParams, PRESET_PARAMS } from "./forceLaw";

describe("forceLaw preset specs", () => {
  it("every preset has at least one param with a default in range", () => {
    for (const specs of Object.values(PRESET_PARAMS)) {
      expect(specs.length).toBeGreaterThan(0);
      for (const s of specs) expect(s.default).toBeGreaterThanOrEqual(s.min);
    }
  });

  it("defaultParams returns the spec defaults", () => {
    expect(defaultParams("yukawa")).toEqual({ lambda: 3 });
    expect(defaultParams("finitewell")).toEqual({ v0: 2, a: 3 });
  });

  it("clampParam bounds to the spec range", () => {
    const spec = PRESET_PARAMS.yukawa[0];
    expect(clampParam(spec, 999)).toBe(spec.max);
    expect(clampParam(spec, -1)).toBe(spec.min);
  });

  it("allowedSpan finds the E>V window", () => {
    const r = [1, 2, 3, 4, 5];
    const v = [-10, -8, -6, -4, -2];
    expect(allowedSpan(r, v, -5)).toEqual([1, 3]); // V<-5 at r=1,2,3
    expect(allowedSpan(r, v, -20)).toBeNull(); // E below the whole well
  });
});
