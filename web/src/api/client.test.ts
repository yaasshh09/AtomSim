import { describe, expect, it } from "vitest";
import { num } from "./client";

describe("num", () => {
  it("leaves ordinary numbers alone", () => {
    expect(num(10000)).toBe("10000");
    expect(num(1e13)).toBe("10000000000000");
    expect(num(656.28)).toBe("656.28");
    expect(num(-1.5)).toBe("-1.5");
  });

  it("escapes the plus in exponential form", () => {
    // The whole reason this helper exists. JS stringifies from 1e21 up in
    // exponential form, and a raw "+" in a query string decodes as a space:
    // the API receives "1e 22" and rejects it as unparseable. Both big knobs
    // in this app cross that threshold, so the bug lives at the top of a
    // slider's travel rather than somewhere unreachable.
    expect(String(1e22)).toBe("1e+22");
    expect(num(1e22)).toBe("1e%2B22");
    expect(num(1e21)).toBe("1e%2B21");
    expect(decodeURIComponent(num(1e26))).toBe("1e+26");
  });

  it("survives a round trip through URL decoding for every slider extreme", () => {
    // The actual ranges the controls expose: electron density to 1e22 cm^-3,
    // column density to 1e26 m^-2, resolving power to 1e7.
    for (const v of [1e4, 1e13, 1e22, 1e14, 1e20, 1e26, 1e2, 1e7]) {
      const parsed = Number(
        new URLSearchParams(`x=${num(v)}`).get("x"),
      );
      expect(parsed).toBe(v);
    }
  });

  it("would have caught the raw interpolation it replaces", () => {
    const raw = new URLSearchParams(`x=${String(1e22)}`).get("x");
    expect(raw).toBe("1e 22");
    expect(Number(raw)).toBeNaN();
  });
});
