import { describe, expect, it } from "vitest";
import { formatErrorScale, HF_LADDER_AXIS_LIBERTY, RENDER_LIBERTIES } from "./liberties";

describe("formatErrorScale", () => {
  // The case that prompted this: a Hartree-Fock energy error estimate arrives
  // as a raw double and used to render every digit of it.
  it("cuts a raw double down to a precision an error bar can support", () => {
    expect(formatErrorScale(0.00034049718827628214)).toBe("3.4e-4");
  });

  it("keeps two significant figures in the range a human reads without counting zeros", () => {
    expect(formatErrorScale(0.0034)).toBe("0.0034");
    expect(formatErrorScale(1.23456)).toBe("1.2");
  });

  // Not "4300": at two significant figures those trailing zeros are not known,
  // and writing them would be the same overstatement this function exists to
  // stop, one order of magnitude up.
  it("never pads an unknown digit with a zero", () => {
    expect(formatErrorScale(4321)).toBe("4.3e+3");
    expect(formatErrorScale(123456)).toBe("1.2e+5");
  });

  it("switches to exponential where fixed notation stops being legible", () => {
    expect(formatErrorScale(1.5e-7)).toBe("1.5e-7");
  });

  // An error estimate should never be negative or non-finite. If one is, that
  // is a bug upstream, and showing it is how it gets found — swallowing it
  // into "0" or a blank would hide exactly the thing worth seeing.
  it("passes through the values that should never happen rather than hiding them", () => {
    expect(formatErrorScale(0)).toBe("0");
    expect(formatErrorScale(-0.5)).toBe("-0.50");
    expect(formatErrorScale(Number.NaN)).toBe("NaN");
    expect(formatErrorScale(Number.POSITIVE_INFINITY)).toBe("Infinity");
  });
});

describe("disclosed liberties", () => {
  // Every constant in this module exists to be shown in a Badge. A liberty
  // with no method or no assumptions discloses nothing and would render as an
  // empty box that still says VISUAL LIBERTY, which is worse than no badge.
  it("every liberty is a visual_liberty that actually says something", () => {
    for (const lib of [RENDER_LIBERTIES, HF_LADDER_AXIS_LIBERTY]) {
      expect(lib.fidelity).toBe("visual_liberty");
      expect(lib.method.length).toBeGreaterThan(0);
      expect(lib.assumptions.length).toBeGreaterThan(0);
      expect(lib.refinement).toBeTruthy();
    }
  });

  // The log axis cannot draw the ionization limit, and that consequence is the
  // one a reader most needs told. Pin that it stays disclosed.
  it("the log-axis liberty discloses that zero is off the scale", () => {
    expect(HF_LADDER_AXIS_LIBERTY.assumptions.join(" ")).toContain("ionization limit");
  });
});
