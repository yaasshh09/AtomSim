import { describe, expect, it } from "vitest";
import { analyticsEndpoint, beaconUrl, type VisitFacts } from "./analytics";

const ENDPOINT = "https://atomsim.goatcounter.com/count";

const visit = (over: Partial<VisitFacts> = {}): VisitFacts => ({
  pathname: "/",
  title: "atomsim · the quantum atom, honestly",
  referrer: "",
  screenWidth: 1920,
  automated: false,
  nonce: "abc123",
  ...over,
});

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

describe("beaconUrl", () => {
  it("reports a visit with the fields the dashboard needs", () => {
    const url = new URL(beaconUrl(ENDPOINT, visit()));
    expect(url.origin + url.pathname).toBe(ENDPOINT);
    expect(url.searchParams.get("p")).toBe("/");
    expect(url.searchParams.get("t")).toBe("atomsim · the quantum atom, honestly");
    expect(url.searchParams.get("s")).toBe("1920");
    expect(url.searchParams.get("b")).toBe("0");
    expect(url.searchParams.get("rnd")).toBe("abc123");
  });

  it("never carries the app state, whatever the address bar holds", () => {
    // The regression this file exists for. GoatCounter's count.js hardcodes
    // `q: location.search` in get_data, with no setting that reaches it, so
    // overriding its `path` cleaned the recorded page and shipped the state
    // anyway: production sent q=%3Fn%3D3%26l%3D1%26m%3D-1%26view%3Dplane.
    // Building the URL here is what makes that structurally impossible, and
    // this is the assertion that keeps it so.
    const url = beaconUrl(ENDPOINT, visit({ pathname: "/" }));
    expect(url).not.toContain("q=");
    expect(url).not.toContain("n%3D3");
    expect(url).not.toContain("view");
    expect(new URL(url).searchParams.get("q")).toBeNull();
  });

  it("sends the path and nothing but the path", () => {
    // A pathname is all installAnalytics ever passes, but if some future
    // caller hands over a full URL the beacon should carry it verbatim rather
    // than silently splitting it: visible in the dashboard beats invisible.
    const url = beaconUrl(ENDPOINT, visit({ pathname: "/tour" }));
    expect(new URL(url).searchParams.get("p")).toBe("/tour");
  });

  it("omits an empty referrer instead of sending a blank one", () => {
    expect(new URL(beaconUrl(ENDPOINT, visit())).searchParams.has("r")).toBe(false);
    expect(
      new URL(beaconUrl(ENDPOINT, visit({ referrer: "https://news.ycombinator.com/" })))
        .searchParams.get("r"),
    ).toBe("https://news.ycombinator.com/");
  });

  it("omits a screen width it does not have", () => {
    expect(new URL(beaconUrl(ENDPOINT, visit({ screenWidth: 0 }))).searchParams.has("s")).toBe(
      false,
    );
  });

  it("flags an automated browser as a bot", () => {
    // Playwright checks against production would otherwise land in the count
    // as people, and this project's own verification is not an audience.
    expect(new URL(beaconUrl(ENDPOINT, visit({ automated: true }))).searchParams.get("b")).toBe(
      "1",
    );
  });

  it("varies the cache-buster so a repeat visit is not a cached image", () => {
    const a = beaconUrl(ENDPOINT, visit({ nonce: "aaa" }));
    const b = beaconUrl(ENDPOINT, visit({ nonce: "bbb" }));
    expect(a).not.toBe(b);
  });
});
