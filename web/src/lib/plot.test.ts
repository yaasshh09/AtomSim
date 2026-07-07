import { scaleLinear } from "d3-scale";
import { describe, expect, it } from "vitest";
import { linePath } from "./plot";

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
