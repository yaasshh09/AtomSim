import { describe, expect, it } from "vitest";
import { seriesColor, seriesName } from "./spectrum";

describe("series", () => {
  it("names the classic series", () => {
    expect(seriesName(1)).toBe("Lyman");
    expect(seriesName(2)).toBe("Balmer");
    expect(seriesName(6)).toBe("Humphreys");
    expect(seriesName(7)).toBe("to n'=7");
  });
  it("colors are stable hex strings with a fallback", () => {
    expect(seriesColor(1)).toMatch(/^#[0-9a-f]{6}$/);
    expect(seriesColor(2)).not.toBe(seriesColor(1));
    expect(seriesColor(99)).toBe("#8b98a5");
  });
});
