import { describe, expect, it } from "vitest";
import { URL_DEFAULTS } from "../lib/urlState";
import { clampStep, stepState } from "./step";
import type { Tour, TourStep } from "./types";

const step = (over: Partial<TourStep> = {}): TourStep => ({
  id: "s",
  title: "t",
  body: ["b"],
  state: {},
  ...over,
});

const tour = (n: number): Tour => ({
  id: "t",
  title: "T",
  blurb: "b",
  steps: Array.from({ length: n }, (_, i) => step({ id: `s${i}` })),
});

describe("stepState", () => {
  it("fills every key from the defaults", () => {
    const s = stepState(step({ state: { n: 3, l: 2 } }));
    expect(s.n).toBe(3);
    expect(s.l).toBe(2);
    expect(s.system).toBe(URL_DEFAULTS.system);
    expect(Object.keys(s).sort()).toEqual(Object.keys(URL_DEFAULTS).sort());
  });

  it("is a full reset, not a patch onto the step before", () => {
    // The bug this exists to prevent: stepping back from the fine-structure
    // step leaving fine structure switched on over the step that says the
    // levels are degenerate.
    const withFs = stepState(step({ state: { fineStructure: true } }));
    const plain = stepState(step({ state: { n: 2 } }));
    expect(withFs.fineStructure).toBe(true);
    expect(plain.fineStructure).toBe(URL_DEFAULTS.fineStructure);
  });

  it("does not share structure with URL_DEFAULTS", () => {
    // A step must never be able to mutate the defaults for every later step.
    const s = stepState(step({ state: { n: 2 } }));
    expect(s).not.toBe(URL_DEFAULTS);
    expect(s.labConst).not.toBe(URL_DEFAULTS.labConst);
  });
});

describe("clampStep", () => {
  it("keeps an index inside the tour", () => {
    expect(clampStep(tour(5), 3)).toBe(3);
    expect(clampStep(tour(5), 9)).toBe(4);
    expect(clampStep(tour(5), -2)).toBe(0);
  });

  it("survives junk from a hand-edited URL", () => {
    expect(clampStep(tour(5), Number.NaN)).toBe(0);
    expect(clampStep(tour(5), 2.7)).toBe(2);
  });

  it("returns 0 for an empty tour rather than -1", () => {
    expect(clampStep(tour(0), 3)).toBe(0);
  });
});
