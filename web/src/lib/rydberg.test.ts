import { describe, expect, it } from "vitest";
import { bestMeasured, computedAt, type DefectSeries } from "./rydberg";

const series = {
  l: 0,
  computed: [5, 10].map((n) => ({ n, defect: { value: 1.38 + 1 / n } })),
  measured: [
    { n: 5, defect: 1.3527, uncertainty: 1e-6 },
    { n: 9, defect: 1.349, uncertainty: 4e-5 },
    { n: 12, defect: 1.3485, uncertainty: 5e-3 },
  ],
} as unknown as DefectSeries;

describe("quantum defect helpers", () => {
  it("finds a computed defect by n, and says null when absent", () => {
    expect(computedAt(series, 10)?.value).toBeCloseTo(1.48, 9);
    expect(computedAt(series, 7)).toBeNull();
  });

  it("quotes the highest n that is measured precisely enough", () => {
    expect(bestMeasured(series)?.n).toBe(9);
  });

  it("falls back to the most precise point when none clears the bar", () => {
    expect(bestMeasured(series, 1e-9)?.n).toBe(5);
    expect(bestMeasured({ ...series, measured: [] })).toBeNull();
  });
});
