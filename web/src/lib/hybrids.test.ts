import { describe, expect, it } from "vitest";
import { HYBRID_KEYS, hybridLabel, isHybridKey } from "./hybrids";

describe("hybrid keys", () => {
  it("match the engine's nine", () => {
    expect(HYBRID_KEYS).toEqual([
      "sp-1", "sp-2", "sp2-1", "sp2-2", "sp2-3", "sp3-1", "sp3-2", "sp3-3", "sp3-4",
    ]);
  });
  it("name themselves the chemistry way", () => {
    expect(hybridLabel("sp3-2")).toBe("sp³ hybrid, lobe 2 of 4");
    expect(isHybridKey("sp4-1")).toBe(false);
    expect(isHybridKey(null)).toBe(false);
  });
});
