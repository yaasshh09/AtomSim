import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getAbsorption,
  getClassical,
  getCurveOfGrowth,
  getForceLaw,
  getLevels,
  getRadial,
  getSpectrum,
  getState,
  thumbnailUrl,
} from "./client";

/**
 * He+ is a real preset whose key contains a `+`, and a bare `+` in a query
 * string decodes to a space. Spelled straight into a URL it reached the server
 * as the system "he ", which is nothing, so every GET the preset made was
 * refused at once: state, levels, spectrum, radial and the whole thumbnail
 * strip. The cloud and cross-section still worked, because jobs POST JSON, and
 * that is what made it read as a rendering fault instead of an encoding one.
 *
 * So the test sweeps every request builder rather than the one that was
 * noticed. A `+` in a preset key is legal and this app has one.
 */
const THERMAL = { temperatureK: 8000, electronDensityCm3: 1e14 };

let lastUrl = "";

function stubFetch() {
  lastUrl = "";
  vi.stubGlobal("fetch", (url: string) => {
    lastUrl = url;
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
  });
}

afterEach(() => vi.unstubAllGlobals());

const CALLS: [string, () => Promise<unknown>][] = [
  ["getState", () => getState(2, 1, 0, "he+", false)],
  ["getRadial", () => getRadial(2, 1, "he+")],
  ["getLevels", () => getLevels("he+", 6, false)],
  ["getSpectrum", () => getSpectrum("he+", 6, false)],
  ["getClassical", () => getClassical("he+", 2)],
  ["getForceLaw", () => getForceLaw("he+", "powerlaw", { p: 1 }, 0)],
  [
    "getCurveOfGrowth",
    () =>
      getCurveOfGrowth({
        system: "he+",
        nMax: 6,
        fineStructure: false,
        lambdaNm: 121.5,
        thermal: THERMAL,
      }),
  ],
  [
    "getAbsorption",
    () =>
      getAbsorption({
        system: "he+",
        nMax: 6,
        fineStructure: false,
        columnDensityM2: 1e20,
        thermal: THERMAL,
      }),
  ],
];

describe("a system key containing '+' survives the trip to the server", () => {
  it.each(CALLS)("%s encodes it", async (_name, call) => {
    stubFetch();
    await call();
    // Both halves matter. The positive says the encoding happened; the
    // negative is the one that actually failed, since a bare `+` still looks
    // like a perfectly ordinary URL when you read it.
    expect(lastUrl).toContain("system=he%2B");
    expect(lastUrl).not.toContain("system=he+");
  });

  it("thumbnailUrl encodes it too", () => {
    // Not a fetch, but nine of these render at once, so it was most of the
    // visible damage.
    const url = thumbnailUrl(2, 1, -1, "he+", "real", 96);
    expect(url).toBe("/api/thumbnail/2/1/-1?system=he%2B&basis=real&size=96");
  });

  it("leaves a key that needs no encoding alone", () => {
    expect(thumbnailUrl(2, 1, -1, "mu-h", "real", 96)).toBe(
      "/api/thumbnail/2/1/-1?system=mu-h&basis=real&size=96",
    );
  });
});
