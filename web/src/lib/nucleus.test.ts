import { describe, expect, it } from "vitest";
import type { SystemInfo } from "../api/types";
import { NUCLEUS_MARKER_LIBERTY } from "./liberties";
import {
  MARKER_DIVISOR,
  formatMagnification,
  nucleusCaption,
  nucleusSphere,
} from "./nucleus";

const EXACT = {
  fidelity: "exact" as const,
  method: "m",
  assumptions: [],
  error_estimate: null,
  refinement: null,
};

function sys(radiusBohr: number | null, radiusFm: number | null): SystemInfo {
  return {
    key: "h",
    name: "Hydrogen",
    z: 1,
    mu_ratio: { value: 1, unit: "m_e", label: "mu", provenance: EXACT },
    m_over_m_nucleus: 0,
    description: "",
    nuclear_radius:
      radiusBohr === null
        ? null
        : { value: radiusBohr, unit: "bohr", label: "r", provenance: EXACT },
    nuclear_radius_fm:
      radiusFm === null
        ? null
        : { value: radiusFm, unit: "fm", label: "r", provenance: EXACT },
    kind: "hydrogenic",
    n_electrons: null,
    has_gsz: true,
  };
}

const H = sys(1.5888e-5, 0.84075);
const PS = sys(null, null);

describe("nucleusSphere", () => {
  it("hidden mode and missing radius render nothing", () => {
    expect(nucleusSphere("hidden", 1.5888e-5, 9)).toBeNull();
    expect(nucleusSphere("marker", null, 9)).toBeNull();
    expect(nucleusSphere("true-scale", null, 9)).toBeNull();
  });
  it("true scale uses the physical radius unmagnified", () => {
    const s = nucleusSphere("true-scale", 1.5888e-5, 9);
    expect(s).toEqual({ kind: "true-scale", radius: 1.5888e-5, magnification: 1 });
  });
  it("marker radius is camera-relative with honest magnification", () => {
    const s = nucleusSphere("marker", 1.5888e-5, 9);
    expect(s?.kind).toBe("marker");
    expect(s?.radius).toBeCloseTo(9 / MARKER_DIVISOR, 12);
    expect(s?.magnification).toBeCloseTo(9 / MARKER_DIVISOR / 1.5888e-5, 6);
  });
});

describe("formatMagnification", () => {
  it("rounds to two significant figures with separators", () => {
    expect(formatMagnification(6294.1)).toBe("6,300×");
    expect(formatMagnification(1234)).toBe("1,200×");
    expect(formatMagnification(87)).toBe("87×");
  });
});

describe("nucleusCaption", () => {
  it("is null when hidden or system unknown", () => {
    expect(nucleusCaption("hidden", H, nucleusSphere("hidden", 1.5888e-5, 9))).toBeNull();
    expect(nucleusCaption("marker", null, null)).toBeNull();
  });
  it("point lepton is stated honestly, never drawn", () => {
    const c = nucleusCaption("marker", PS, null);
    expect(c).toMatch(/point lepton/);
    expect(c).toMatch(/no measured size/);
  });
  it("true scale states the radius and that it is subpixel", () => {
    const c = nucleusCaption("true-scale", H, nucleusSphere("true-scale", 1.5888e-5, 9));
    expect(c).toContain("0.841 fm");
    expect(c).toContain("1.6e-5");
    expect(c).toMatch(/smaller than a pixel/);
  });
  it("marker states its magnification", () => {
    const c = nucleusCaption("marker", H, nucleusSphere("marker", 1.5888e-5, 9));
    expect(c).toMatch(/6,300×|6,290×/);
    expect(c).toMatch(/true size/);
  });
});

describe("NUCLEUS_MARKER_LIBERTY", () => {
  it("disclosed as a visual liberty and states the rule", () => {
    expect(NUCLEUS_MARKER_LIBERTY.fidelity).toBe("visual_liberty");
    expect(NUCLEUS_MARKER_LIBERTY.assumptions.join(" ")).toContain(
      `camera distance / ${MARKER_DIVISOR}`,
    );
  });
});
