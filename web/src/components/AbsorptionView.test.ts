import { describe, expect, it } from "vitest";
import {
  saturationVerdict,
  transmissionGrey,
  transmissionPath,
} from "./AbsorptionView";

describe("transmissionPath", () => {
  const id = (v: number) => v;

  it("plots wavelength on a log axis and transmission linearly", () => {
    const d = transmissionPath([10, 100, 1000], [1, 0.5, 1], id, id);
    expect(d).toBe("M1.00 1.00 L2.00 0.50 L3.00 1.00");
  });

  it("skips points a log axis cannot place", () => {
    // A zero or negative wavelength is not a point on this plot. Passing it
    // through would put a NaN in the path and blank the whole curve.
    const d = transmissionPath([0, 100, -5], [1, 0.5, 1], id, id);
    expect(d).toBe("M2.00 0.50");
    expect(d).not.toContain("NaN");
  });

  it("is empty for an empty spectrum rather than malformed", () => {
    expect(transmissionPath([], [], id, id)).toBe("");
  });
});

describe("transmissionGrey", () => {
  it("maps transmission straight to lightness with no curve", () => {
    // A line that looks half as bright is letting half the light through.
    // Any gamma here would be an undisclosed visual liberty.
    expect(transmissionGrey(1)).toBe("rgb(255 255 255)");
    expect(transmissionGrey(0)).toBe("rgb(0 0 0)");
    expect(transmissionGrey(0.5)).toBe("rgb(128 128 128)");
  });

  it("clamps rather than emitting an out-of-range colour", () => {
    expect(transmissionGrey(1.0000001)).toBe("rgb(255 255 255)");
    expect(transmissionGrey(-1e-12)).toBe("rgb(0 0 0)");
  });
});

describe("saturationVerdict", () => {
  it("calls a thin spectrum a faithful census", () => {
    expect(saturationVerdict(1.0)).toContain("faithful census");
  });

  it("warns as soon as the strongest lines start to saturate", () => {
    expect(saturationVerdict(0.8)).toContain("understates");
  });

  it("says the gas is hidden once the cores are black", () => {
    // The number this has to defend against being read as a small correction:
    // at 0.1 the gas holds ten times what the lines appear to say.
    expect(saturationVerdict(0.1)).toContain("invisible");
  });
});
