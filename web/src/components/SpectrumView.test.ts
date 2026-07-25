import { describe, expect, it } from "vitest";
import type { Quantity, SpectralLineInfo } from "../api/types";
import { intensityScale } from "./SpectrumView";

function line(nm: number, a: number | null): SpectralLineInfo {
  const q = (value: number, unit: string): Quantity =>
    ({ value, unit, label: "", provenance: {} }) as never;
  return {
    n_upper: 2,
    l_upper: 1,
    j_upper: null,
    n_lower: 1,
    l_lower: 0,
    j_lower: null,
    energy_ev: q(1, "eV"),
    wavelength_nm: q(nm, "nm"),
    einstein_a_s: a === null ? null : q(a, "s^-1"),
    oscillator_strength: null,
  };
}

describe("intensityScale", () => {
  it("is null when no line carries a rate, so bars stay uniform", () => {
    expect(intensityScale([line(121, null), line(656, null)])).toBeNull();
  });

  it("puts the weakest line at 0 and the strongest at 1", () => {
    const s = intensityScale([line(121, 6.27e8), line(656, 4.41e7), line(486, 8.42e6)])!;
    expect(s.t(8.42e6)).toBeCloseTo(0, 10);
    expect(s.t(6.27e8)).toBeCloseTo(1, 10);
  });

  it("separates lines that differ by decades, which a linear map would not", () => {
    const s = intensityScale([line(121, 1e9), line(656, 1e5)])!;
    // Midpoint of the log range sits at t = 0.5; linearly it would be ~1e-4.
    expect(s.t(1e7)).toBeCloseTo(0.5, 10);
  });

  it("reports the decade range it compressed, for the caption", () => {
    const s = intensityScale([line(121, 1e9), line(656, 1e5)])!;
    expect(s.lo).toBeCloseTo(5, 10);
    expect(s.hi).toBeCloseTo(9, 10);
  });

  it("draws a single line at full strength instead of dividing by zero", () => {
    const s = intensityScale([line(121, 6.27e8)])!;
    expect(s.t(6.27e8)).toBe(1);
  });

  it("treats a missing or non-positive rate as full strength, never NaN", () => {
    const s = intensityScale([line(121, 1e9), line(656, 1e5)])!;
    expect(s.t(undefined)).toBe(1);
    expect(s.t(0)).toBe(1);
  });
});
