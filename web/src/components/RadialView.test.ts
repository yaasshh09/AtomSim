import { describe, expect, it } from "vitest";
import type { Quantity, ShellPeak } from "../api/types";
import { displacedChargeText, shellCells } from "./RadialView";

const q = (value: number, error: number | null): Quantity =>
  ({
    value,
    unit: "electrons",
    label: "",
    provenance: { error_estimate: error },
  }) as never;

const shell = (p: Partial<ShellPeak>): ShellPeak => ({
  label: "M",
  gsz_radius: null,
  hf_radius: null,
  gsz_depth: null,
  hf_depth: null,
  ...p,
});

describe("displacedChargeText", () => {
  it("states the number against the electron count it is a fraction of", () => {
    expect(displacedChargeText(q(0.06, 0.006), 18)).toContain("0.060");
    expect(displacedChargeText(q(0.06, 0.006), 18)).toContain("18");
  });

  it("refuses to print a figure smaller than its own error bar", () => {
    // Helium: 0.0003 displaced against a bar of about 0.0003. Printing four
    // decimals there would claim a precision the measurement does not have.
    expect(displacedChargeText(q(0.0003, 0.0004), 2)).toMatch(/agree to within/i);
  });

  it("handles an unknown electron count without inventing one", () => {
    expect(displacedChargeText(q(0.06, 0.006), null)).not.toContain("null");
  });
});

describe("shellCells", () => {
  it("says outright that a model resolves no peak, rather than leaving a blank", () => {
    const c = shellCells(shell({ hf_radius: 3.163, hf_depth: 0.003 }));
    expect(c.gsz).toMatch(/no separate peak/i);
    expect(c.hf).toContain("3.16");
  });

  it("flags a peak whose valley is too shallow to call a shell boundary", () => {
    const c = shellCells(shell({ gsz_radius: 3.1, hf_radius: 3.163, hf_depth: 0.003 }));
    expect(c.note).toMatch(/0.3%/);
  });

  it("says nothing extra about a well separated shell", () => {
    const c = shellCells(
      shell({ gsz_radius: 1.249, hf_radius: 1.241, gsz_depth: 0.4, hf_depth: 0.4 }),
    );
    expect(c.note).toBeNull();
  });
});
