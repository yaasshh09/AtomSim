import { describe, expect, it } from "vitest";
import {
  absorptionAxisMode,
  bandColumns,
  bandResolutionNote,
  saturationVerdict,
  transmissionGrey,
  transmissionPath,
} from "./AbsorptionView";

describe("bandResolutionNote", () => {
  it("says the strip agrees with the curve when the columns resolve the line", () => {
    // Zoomed onto one line: a column is far finer than the profile, so both
    // readings hit the black core. Claiming a dilution gap here would describe
    // a picture that is not on the screen.
    const note = bandResolutionNote(0, 0);
    expect(note).toContain("agree");
    expect(note).not.toContain("%");
  });

  it("quantifies the dilution when a line is narrower than a column", () => {
    // The full line list: the core is black but a 1 nm pixel barely dips.
    const note = bandResolutionNote(0, 0.96);
    expect(note).toContain("96.0%");
    expect(note).toContain("0.0%");
    expect(note).toContain("almost blank");
  });

  it("treats a small disagreement as agreement rather than warning twice", () => {
    expect(bandResolutionNote(0.2, 0.24)).toContain("agree");
    expect(bandResolutionNote(0.2, 0.9)).toContain("90.0%");
  });
});

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

describe("absorptionAxisMode", () => {
  it("labels the full line list with absolute wavelengths", () => {
    // 97 nm to 1876 nm: "100" and "1000" are the natural labels.
    expect(absorptionAxisMode(97, 1876)).toBe("log");
  });

  it("switches to offsets on a zoomed line", () => {
    // Eight half-widths of Lyman-alpha. Every absolute label here rounds to
    // "121.57", and decade ticks produce none at all, which is what the panel
    // used to draw.
    expect(absorptionAxisMode(121.4990385, 121.6378527)).toBe("offset");
  });

  it("keeps absolute labels for a merely narrow window", () => {
    // The visible band. Four significant digits still separate these ticks.
    expect(absorptionAxisMode(400, 700)).toBe("log");
  });

  it("does not divide by a zero centre", () => {
    expect(absorptionAxisMode(0, 0)).toBe("offset");
  });
});

describe("bandColumns", () => {
  const id = (v: number) => v;

  it("draws one column per unit of axis, abutting exactly", () => {
    // Sub-unit rectangles were the bug: adjacent anti-aliased edges composite
    // to about 75% of full white, so a flat continuum drew as a grey barcode.
    const cols = bandColumns([0, 1, 2, 3, 4], [1, 1, 1, 1, 1], id, 0, 4);
    expect(cols).toHaveLength(4);
    for (let i = 1; i < cols.length; i++) {
      expect(cols[i].x - cols[i - 1].x).toBeCloseTo(1, 12);
    }
    expect(cols[0].x).toBe(0);
  });

  it("keeps an unabsorbed continuum at exactly 1", () => {
    // The regression this exists to prevent: past 400 nm the real transmission
    // never drops below 0.98, and the band showed lines there anyway.
    const n = 500;
    const lam = Array.from({ length: n }, (_, i) => i / n);
    const t = new Array(n).fill(1);
    const cols = bandColumns(lam, t, id, 0, 1);
    for (const c of cols) {
      expect(c.deepest).toBeCloseTo(1, 12);
      expect(c.mean).toBeCloseTo(1, 12);
    }
  });

  it("means over the wavelengths a column covers", () => {
    // Half the column black, half of it clear, so a real pixel reads half-lit.
    const cols = bandColumns([0, 0.25, 0.75, 1], [0, 0, 1, 1], id, 0, 1);
    expect(cols).toHaveLength(1);
    expect(cols[0].mean).toBeCloseTo(0.5, 6);
  });

  it("separates the drawn depth from what a pixel would record", () => {
    // One black sample inside an otherwise clear column. `deepest` keeps the
    // line visible; `mean` says a real pixel that wide barely notices it. The
    // view draws the first and prints the second.
    const lam = [0, 0.4, 0.5, 0.6, 1];
    const t = [1, 1, 0, 1, 1];
    const cols = bandColumns(lam, t, id, 0, 1);
    expect(cols[0].deepest).toBe(0);
    expect(cols[0].mean).toBeGreaterThan(0.5);
    expect(cols[0].mean).toBeLessThan(1);
  });

  it("carries the last value across a column the grid does not reach", () => {
    // A hole would render black, which reads as total absorption, the
    // opposite of "no information here".
    const cols = bandColumns([0, 0.1], [0.5, 0.5], id, 0, 4);
    expect(cols).toHaveLength(4);
    for (const c of cols) {
      expect(c.deepest).toBeCloseTo(0.5, 12);
      expect(c.mean).toBeCloseTo(0.5, 12);
    }
  });

  it("is empty for an empty grid or a collapsed axis", () => {
    expect(bandColumns([], [], id, 0, 4)).toEqual([]);
    expect(bandColumns([0, 1], [1, 1], id, 3, 3)).toEqual([]);
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
