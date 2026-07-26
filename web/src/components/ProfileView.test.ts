import { describe, expect, it } from "vitest";
import type { LineWidthInfo } from "../api/types";
import {
  dominantTerm,
  profilePath,
  profileScale,
  widthAt,
  zoomWindow,
} from "./SpectrumView";

function width(over: Partial<LineWidthInfo> = {}): LineWidthInfo {
  return {
    label: "3->2",
    wavelength_nm: 656.28,
    n_upper: 3,
    n_lower: 2,
    sigma_nm: 0.02,
    gamma_nm: 1e-5,
    fwhm_nm: 0.047,
    terms: ["natural", "Doppler"],
    ...over,
  };
}

describe("profileScale", () => {
  it("puts the peak at the top and the floor at the bottom", () => {
    const s = profileScale([1e-6, 1e-3, 1], 6);
    expect(s).not.toBeNull();
    expect(s!.t(1)).toBeCloseTo(1, 12);
    expect(s!.t(1e-6)).toBeCloseTo(0, 12);
    expect(s!.t(1e-3)).toBeCloseTo(0.5, 12);
  });

  it("clamps anything below the floor instead of dropping it", () => {
    // A faint line must stay visible as a faint line; drawing it at zero
    // height would be indistinguishable from it not existing.
    const s = profileScale([1, 1e-30], 6)!;
    expect(s.t(1e-30)).toBe(0);
    expect(s.t(0)).toBe(0);
  });

  it("returns null when there is nothing to draw", () => {
    // A fully ionized gas emits nothing, and a zero curve has no log scale.
    expect(profileScale([0, 0, 0])).toBeNull();
    expect(profileScale([])).toBeNull();
  });
});

describe("profilePath", () => {
  it("walks every sample in order", () => {
    const path = profilePath(
      [1, 2, 3], [1, 1, 1], (v) => v, (t) => t, () => 5,
    );
    expect(path).toBe("M1.00 5.00 L2.00 5.00 L3.00 5.00");
  });

  it("produces no path for an empty curve", () => {
    expect(profilePath([], [], (v) => v, (t) => t, () => 0)).toBe("");
  });
});

describe("zoomWindow", () => {
  it("brackets the line symmetrically in units of its own width", () => {
    const [lo, hi] = zoomWindow(656.28, 0.05, 8);
    expect(lo).toBeCloseTo(656.28 - 0.4, 10);
    expect(hi).toBeCloseTo(656.28 + 0.4, 10);
  });

  it("never collapses to a point on a zero-width line", () => {
    // A 2s lower level with no thermal width would otherwise produce
    // lambda_min === lambda_max, which the server rejects.
    const [lo, hi] = zoomWindow(121.567, 0);
    expect(hi).toBeGreaterThan(lo);
  });
});

describe("widthAt", () => {
  it("finds the nearest line", () => {
    const a = width({ wavelength_nm: 486.1, label: "4->2" });
    const b = width({ wavelength_nm: 656.3, label: "3->2" });
    expect(widthAt([a, b], 656.0)?.label).toBe("3->2");
    expect(widthAt([a, b], 400.0)?.label).toBe("4->2");
  });

  it("returns null when the curve carries no widths", () => {
    expect(widthAt([], 656.0)).toBeNull();
  });
});

describe("dominantTerm", () => {
  it("names thermal motion when the Gaussian core wins", () => {
    expect(dominantTerm(width({ sigma_nm: 0.02, gamma_nm: 1e-6 }))).toBe(
      "thermal motion",
    );
  });

  it("names the lifetime when the Lorentzian wins", () => {
    expect(dominantTerm(width({ sigma_nm: 0, gamma_nm: 1e-3 }))).toBe(
      "the upper level's lifetime",
    );
  });

  it("names the instrument when it is the only Gaussian term", () => {
    expect(
      dominantTerm(
        width({ sigma_nm: 0.1, gamma_nm: 1e-6, terms: ["natural", "instrumental"] }),
      ),
    ).toBe("the spectrograph");
  });

  it("says nothing when there is no width at all", () => {
    expect(dominantTerm(width({ sigma_nm: 0, gamma_nm: 0, terms: [] }))).toBe(
      "nothing",
    );
  });
});
