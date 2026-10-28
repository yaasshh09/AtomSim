import { describe, expect, it } from "vitest";
import { volumeTexels } from "./volume";

describe("volume texels", () => {
  it("maps the peak to 255 and zero to 0 through the square-root gamma", () => {
    const v = new Float32Array(8);
    v[0] = 4;
    v[1] = 1;
    v[7] = 0;
    const t = volumeTexels(v, 2);
    expect(t[0]).toBe(255);
    expect(t[1]).toBe(Math.round(255 * 0.5));
    expect(t[7]).toBe(0);
  });

  it("returns all zeros for an empty grid rather than dividing by zero", () => {
    expect(Array.from(volumeTexels(new Float32Array(8), 2))).toEqual(new Array(8).fill(0));
  });

  it("refuses a grid that is not a cube", () => {
    expect(() => volumeTexels(new Float32Array(9), 2)).toThrow();
  });
});
