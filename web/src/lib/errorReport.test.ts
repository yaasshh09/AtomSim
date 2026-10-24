import { describe, expect, it } from "vitest";
import { buildReport, reportGate } from "./errorReport";

describe("error reports", () => {
  it("send the message, the stack and the path, never the query string", () => {
    const r = buildReport(new TypeError("x is undefined"), "/?n=3&l=1&system=mu-h", "0.1.0");
    expect(r.message).toBe("TypeError: x is undefined");
    expect(r.page).toBe("/");
    expect(r.version).toBe("0.1.0");
  });

  it("cap what they send at the server's limits", () => {
    const r = buildReport("y".repeat(900), "/", "v");
    expect(r.message.length).toBe(500);
  });

  it("let each message through once, and only so many in all", () => {
    const gate = reportGate(2);
    expect([gate("a"), gate("a"), gate("b"), gate("c")]).toEqual([true, false, true, false]);
  });
});
