import { describe, expect, it } from "vitest";
import { clampState, isValidState, nodeSummary, realOrbitalLabel, stateLabel } from "./quantum";

describe("isValidState", () => {
  it("accepts physical states", () => {
    expect(isValidState(1, 0, 0)).toBe(true);
    expect(isValidState(2, 1, -1)).toBe(true);
    expect(isValidState(6, 5, 5)).toBe(true);
  });
  it("rejects unphysical states", () => {
    expect(isValidState(1, 1, 0)).toBe(false); // l == n
    expect(isValidState(3, 1, 2)).toBe(false); // |m| > l
    expect(isValidState(0, 0, 0)).toBe(false); // n < 1
    expect(isValidState(2, -1, 0)).toBe(false);
    expect(isValidState(1.5, 0, 0)).toBe(false); // non-integer
  });
});

describe("clampState", () => {
  it("clamps l and m when n shrinks", () => {
    expect(clampState(1, 2, -2)).toEqual({ n: 1, l: 0, m: 0 });
  });
  it("clamps m into [-l, l]", () => {
    expect(clampState(3, 1, 5)).toEqual({ n: 3, l: 1, m: 1 });
  });
  it("keeps valid states unchanged", () => {
    expect(clampState(4, 2, -2)).toEqual({ n: 4, l: 2, m: -2 });
  });
});

describe("stateLabel", () => {
  it("uses spectroscopic letters", () => {
    expect(stateLabel(1, 0, 0)).toBe("1s (m = 0)");
    expect(stateLabel(2, 1, 0)).toBe("2p (m = 0)");
    expect(stateLabel(3, 2, -1)).toBe("3d (m = -1)");
  });
});

describe("realOrbitalLabel", () => {
  it("mirrors atomsim.analytic.angular.real_orbital_label", () => {
    expect(realOrbitalLabel(0, 0)).toBe("s");
    expect(realOrbitalLabel(1, 0)).toBe("p_z");
    expect(realOrbitalLabel(1, 1)).toBe("p_x");
    expect(realOrbitalLabel(1, -1)).toBe("p_y");
    expect(realOrbitalLabel(2, -2)).toBe("d_xy");
    expect(realOrbitalLabel(3, 3)).toBe("f_x(x2-3y2)");
    expect(realOrbitalLabel(4, 0)).toBe("g(m=0)");
    expect(realOrbitalLabel(4, 2)).toBe("g(m=+2, cos)");
    expect(realOrbitalLabel(4, -2)).toBe("g(m=-2, sin)");
  });
});

describe("nodeSummary", () => {
  it("counts l angular nodes only for real orbitals", () => {
    expect(nodeSummary(3, 2, 1, "real")).toBe("0 radial · 2 angular");
    expect(nodeSummary(3, 2, 1, "complex")).toBe("0 radial · 1 cone + the z axis");
    expect(nodeSummary(3, 2, 0, "complex")).toBe("0 radial · 2 cones");
    expect(nodeSummary(4, 1, 0, "complex")).toBe("2 radial · 1 cone");
  });
});
