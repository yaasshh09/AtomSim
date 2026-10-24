import { describe, expect, it } from "vitest";
import { fixed } from "./format";

describe("fixed", () => {
  it("writes negatives with a true minus and leaves positives alone", () => {
    expect(fixed(-13.6057, 2)).toBe("−13.61");
    expect(fixed(2.5, 1)).toBe("2.5");
    expect(fixed(-0.001, 2)).toBe("−0.00");
  });
});
