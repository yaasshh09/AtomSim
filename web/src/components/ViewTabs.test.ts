import { describe, expect, it } from "vitest";
import { VIEW_OPTIONS } from "./Controls";
import { SHORT } from "./ViewTabs";

describe("the mobile view tabs", () => {
  it("has a short label for every view the rail offers", () => {
    // The strip falls back to the rail's label, which is the long one, so a
    // view added to VIEW_OPTIONS and not here would quietly widen the row
    // rather than break. This is the check that notices.
    expect(VIEW_OPTIONS.map((v) => v.value).filter((v) => !SHORT[v])).toEqual([]);
  });

  it("names no view the rail does not have", () => {
    const known = new Set<string>(VIEW_OPTIONS.map((v) => v.value));
    expect(Object.keys(SHORT).filter((k) => !known.has(k))).toEqual([]);
  });
});
