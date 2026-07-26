import { describe, expect, it } from "vitest";
import type { Quantity, SpectralLineInfo } from "../api/types";
import { intensityScale, wavelengthWindow } from "./SpectrumView";

const q = (value: number, unit: string): Quantity =>
  ({ value, unit, label: "", provenance: {} }) as never;

function line(nm: number, a: number | null): SpectralLineInfo {
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
    emissivity: null,
  };
}

/** A line carrying both a rate and an LTE emissivity. */
function warm(nm: number, a: number, eps: number): SpectralLineInfo {
  return { ...line(nm, a), emissivity: q(eps, "eV/s per atom") };
}

/** n -> n' : the optical group. */
function across(nm: number): SpectralLineInfo {
  return { ...line(nm, 1e8), n_upper: 3, n_lower: 2 };
}

/** Within one n: the fine-structure microwave group. */
function within(nm: number): SpectralLineInfo {
  return { ...line(nm, 1e-12), n_upper: 2, n_lower: 2 };
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

  it("puts a missing or zero value at the floor, never NaN and never full", () => {
    // This used to draw at full strength, which was harmless when the only
    // quantity was A and a missing A meant "not computed". With emissivity a
    // zero is a real physical value — the gas is fully ionized and the line
    // genuinely does not emit — and drawing that at full height would be the
    // most misleading thing the view could do. The bar floor keeps it visible.
    const s = intensityScale([line(121, 1e9), line(656, 1e5)])!;
    expect(s.t(undefined)).toBe(0);
    expect(s.t(0)).toBe(0);
  });
});

describe("intensityScale over emissivity", () => {
  it("reads the emissivity field instead of the rate", () => {
    const lines = [warm(121, 1e9, 5e2), warm(656, 1e5, 5e-2)];
    const s = intensityScale(lines, "emissivity")!;
    expect(s.lo).toBeCloseTo(Math.log10(5e-2), 10);
    expect(s.hi).toBeCloseTo(Math.log10(5e2), 10);
  });

  it("is null when the lines carry no emissivity, so it falls back cleanly", () => {
    expect(intensityScale([line(121, 1e9)], "emissivity")).toBeNull();
  });

  it("can rank lines differently from the rate, which is the whole point", () => {
    // A weak-but-populated line beating a strong-but-empty one is exactly what
    // the thermal weighting exists to show.
    const lines = [warm(121, 1e9, 1e-3), warm(656, 1e5, 1e3)];
    const byRate = intensityScale(lines, "rate")!;
    const byEps = intensityScale(lines, "emissivity")!;
    expect(byRate.t(byRate.value(lines[0]))).toBe(1);
    expect(byEps.t(byEps.value(lines[0]))).toBe(0);
  });
});

describe("wavelengthWindow", () => {
  const optical = [across(121), across(656), across(1875)];
  const microwave = [within(2.1e7), within(1.1e10)];

  it("keeps everything when nothing is a within-n component", () => {
    const w = wavelengthWindow(optical, false);
    expect(w.hidden).toBe(0);
    expect(w.splittable).toBe(false);
    expect(w.hi).toBeCloseTo(1875, 6);
  });

  it("excludes the within-n components by default, and counts them", () => {
    const w = wavelengthWindow([...optical, ...microwave], false);
    expect(w.splittable).toBe(true);
    expect(w.hidden).toBe(2);
    expect(w.hi).toBeCloseTo(1875, 6);
  });

  it("includes them on request, which is what stretches the axis", () => {
    const w = wavelengthWindow([...optical, ...microwave], true);
    expect(w.hidden).toBe(0);
    expect(w.hi).toBeCloseTo(1.1e10, 6);
  });

  it("shows everything when every line is a within-n component", () => {
    // No split to make; hiding all of them would leave an empty plot.
    const w = wavelengthWindow(microwave, false);
    expect(w.hidden).toBe(0);
    expect(w.splittable).toBe(false);
  });

  it("splits on the transition, not on a wavelength threshold", () => {
    // A within-n component that happens to land among the optical lines is
    // still inside the window: the rule is structural, so no arbitrary cutoff
    // decides what the user sees.
    const w = wavelengthWindow([...optical, within(500)], false);
    expect(w.hidden).toBe(0);
  });
});
