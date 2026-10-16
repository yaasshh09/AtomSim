import { describe, expect, it } from "vitest";
import { toCsv } from "./plotExport";

describe("toCsv", () => {
  it("quotes only the fields that need it and leaves nulls empty", () => {
    expect(toCsv(["a", "b"], [[1, 'say "hi", ok'], [null, 2.5e-7]])).toBe(
      'a,b\n1,"say ""hi"", ok"\n,2.5e-7\n',
    );
  });
});
