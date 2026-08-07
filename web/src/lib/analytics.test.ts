import { describe, expect, it } from "vitest";
import { analyticsEndpoint } from "./analytics";

describe("analyticsEndpoint", () => {
  it("is off unless configured", () => {
    // The dev server and a plain `npm run build` both leave VITE_GOATCOUNTER
    // unset, and neither should report a visit.
    expect(analyticsEndpoint(undefined)).toBeNull();
    expect(analyticsEndpoint("")).toBeNull();
    expect(analyticsEndpoint("   ")).toBeNull();
  });

  it("is off for a value that is not a string", () => {
    // import.meta.env is typed with an index signature, so this arrives as
    // `any` and the guard is the only thing standing between a stray value
    // and `new URL`.
    expect(analyticsEndpoint(null)).toBeNull();
    expect(analyticsEndpoint(42)).toBeNull();
    expect(analyticsEndpoint({ endpoint: "https://x.goatcounter.com" })).toBeNull();
  });

  it("takes the endpoint as given", () => {
    expect(analyticsEndpoint("https://atomsim.goatcounter.com/count")).toBe(
      "https://atomsim.goatcounter.com/count",
    );
  });

  it("appends /count to a bare origin", () => {
    // The origin is what the GoatCounter dashboard shows, so it is what gets
    // pasted. Repairing this one case is the difference between working and
    // silently counting nothing.
    expect(analyticsEndpoint("https://atomsim.goatcounter.com")).toBe(
      "https://atomsim.goatcounter.com/count",
    );
    expect(analyticsEndpoint("https://atomsim.goatcounter.com/")).toBe(
      "https://atomsim.goatcounter.com/count",
    );
  });

  it("trims surrounding whitespace", () => {
    expect(analyticsEndpoint("  https://atomsim.goatcounter.com/count  ")).toBe(
      "https://atomsim.goatcounter.com/count",
    );
  });

  it("refuses anything that is not https", () => {
    // Not a repair: an http endpoint is blocked as mixed content on the
    // deployed page, so accepting it would mean counting nothing while
    // looking configured.
    expect(analyticsEndpoint("http://atomsim.goatcounter.com/count")).toBeNull();
    expect(analyticsEndpoint("//atomsim.goatcounter.com/count")).toBeNull();
    expect(analyticsEndpoint("javascript:alert(1)")).toBeNull();
  });

  it("refuses a value that is not a URL at all", () => {
    expect(analyticsEndpoint("atomsim.goatcounter.com")).toBeNull();
    expect(analyticsEndpoint("not a url")).toBeNull();
  });

  it("drops a query string or fragment", () => {
    // Always a paste accident, and it would otherwise ride along on every
    // beacon the page sends.
    expect(analyticsEndpoint("https://atomsim.goatcounter.com/count?debug=1")).toBe(
      "https://atomsim.goatcounter.com/count",
    );
    expect(analyticsEndpoint("https://atomsim.goatcounter.com/count#frag")).toBe(
      "https://atomsim.goatcounter.com/count",
    );
  });

  it("keeps a self-hosted path", () => {
    // Nothing here is specific to goatcounter.com; a self-hosted instance
    // behind a path prefix stays valid.
    expect(analyticsEndpoint("https://metrics.example.org/gc/count")).toBe(
      "https://metrics.example.org/gc/count",
    );
  });
});
