import { describe, expect, it } from "vitest";
import { defaultParams } from "./forceLaw";
import { URL_DEFAULTS, parseAppUrl, serializeAppUrl } from "./urlState";

describe("parseAppUrl", () => {
  it("empty search yields no overrides", () => {
    expect(parseAppUrl("")).toEqual({});
    expect(parseAppUrl("?")).toEqual({});
  });

  it("parses a full valid deep link", () => {
    const p = parseAppUrl(
      "?n=3&l=2&m=-1&system=mu-h&basis=real&view=spectrum&color=density&fs=1&nucleus=true-scale&plane=psi",
    );
    expect(p).toEqual({
      n: 3,
      l: 2,
      m: -1,
      system: "mu-h",
      basis: "real",
      view: "spectrum",
      colorMode: "density",
      fineStructure: true,
      nucleusMode: "true-scale",
      planeQuantity: "psi",
    });
  });

  it("clamps quantum numbers as a triple and caps n at the UI maximum", () => {
    expect(parseAppUrl("?n=99&l=50&m=-50")).toEqual({ n: 6, l: 5, m: -5 });
    expect(parseAppUrl("?n=2&l=5&m=3")).toEqual({ n: 2, l: 1, m: 1 });
    // partial triples merge with defaults before clamping
    expect(parseAppUrl("?l=1")).toEqual({ n: 1, l: 0, m: 0 });
  });

  it("rejects junk instead of propagating it", () => {
    expect(parseAppUrl("?view=poster")).toEqual({});
    expect(parseAppUrl("?n=abc")).toEqual({});
    expect(parseAppUrl("?system=<script>")).toEqual({});
    expect(parseAppUrl("?color=vibes&nucleus=huge&plane=cartoon&basis=vibes")).toEqual({});
  });

  it("demotes phase colour under the real basis (mirror of the store guard)", () => {
    expect(parseAppUrl("?basis=real&color=phase")).toEqual({
      basis: "real",
      colorMode: "density",
    });
  });

  it("parses lab constant multipliers and Z for the what-if view", () => {
    expect(parseAppUrl("?view=whatif&e=2&eps0=4&z=3")).toEqual({
      view: "whatif",
      labConst: { hbar: 1, e: 2, m_e: 1, eps0: 4, c: 1 },
      labZ: 3,
    });
  });

  it("clamps constant multipliers to [0.25, 4] and Z to [1, 10], dropping junk", () => {
    expect(parseAppUrl("?e=9")).toEqual({
      labConst: { hbar: 1, e: 4, m_e: 1, eps0: 1, c: 1 },
    });
    expect(parseAppUrl("?me=0.1")).toEqual({
      labConst: { hbar: 1, e: 1, m_e: 0.25, eps0: 1, c: 1 },
    });
    expect(parseAppUrl("?e=0")).toEqual({});
    expect(parseAppUrl("?e=nope")).toEqual({});
    expect(parseAppUrl("?z=0")).toEqual({ labZ: 1 });
    expect(parseAppUrl("?z=99")).toEqual({ labZ: 10 });
  });
});

describe("serializeAppUrl", () => {
  it("omits defaults entirely", () => {
    expect(serializeAppUrl(URL_DEFAULTS)).toBe("");
  });

  it("serializes only non-default fields", () => {
    expect(
      serializeAppUrl({ ...URL_DEFAULTS, n: 2, l: 1, view: "plane", fineStructure: true }),
    ).toBe("?n=2&l=1&view=plane&fs=1");
  });

  it("round-trips through parseAppUrl", () => {
    const state = {
      n: 4,
      l: 2,
      m: 2,
      system: "he+",
      basis: "real" as const,
      view: "whatif" as const,
      colorMode: "density" as const,
      fineStructure: true,
      ghost: false,
      nucleusMode: "hidden" as const,
      planeQuantity: "psi" as const,
      labConst: { hbar: 1, e: 2, m_e: 1, eps0: 4, c: 1 },
      labZ: 3,
      forcePreset: "powerlaw" as const,
      forceParams: { p: 1.0 },
      forceL: 0,
      forceExpr: "-1/r",
      dirac: false,
      config: null,
    };
    const parsed = parseAppUrl(serializeAppUrl(state));
    expect({ ...URL_DEFAULTS, ...parsed }).toEqual(state);
  });

  it("round-trips the ghost toggle", () => {
    const withGhost = parseAppUrl(serializeAppUrl({ ...URL_DEFAULTS, ghost: true }));
    expect(withGhost.ghost).toBe(true);
    const withoutGhost = parseAppUrl(serializeAppUrl({ ...URL_DEFAULTS, ghost: false }));
    expect({ ...URL_DEFAULTS, ...withoutGhost }.ghost).toBe(false);
  });
});

describe("force-law url state", () => {
  it("round-trips a yukawa force-law deep link", () => {
    const state = {
      ...URL_DEFAULTS,
      view: "forcelaw" as const,
      forcePreset: "yukawa" as const,
      forceParams: { lambda: 5 },
      forceL: 1,
    };
    const q = serializeAppUrl(state);
    expect(q).toContain("preset=yukawa");
    expect(q).toContain("lambda=5");
    const back = { ...URL_DEFAULTS, ...parseAppUrl(q) };
    expect(back.forcePreset).toBe("yukawa");
    expect(back.forceParams.lambda).toBe(5);
    expect(back.forceL).toBe(1);
  });

  it("round-trips a custom V(r) deep link", () => {
    const state = {
      ...URL_DEFAULTS,
      view: "forcelaw" as const,
      forcePreset: "custom" as const,
      forceExpr: "-exp(-r)/r",
    };
    const q = serializeAppUrl(state);
    expect(q).toContain("preset=custom");
    const back = { ...URL_DEFAULTS, ...parseAppUrl(q) };
    expect(back.forcePreset).toBe("custom");
    expect(back.forceExpr).toBe("-exp(-r)/r");
  });

  it("omits preset for the default power-law and reads p", () => {
    const state = {
      ...URL_DEFAULTS,
      view: "forcelaw" as const,
      forcePreset: "powerlaw" as const,
      forceParams: { p: 1.2 },
      forceL: 0,
    };
    const q = serializeAppUrl(state);
    expect(q).not.toContain("preset=");
    expect(q).toContain("p=1.2");
    expect(parseAppUrl(q).forceParams?.p).toBe(1.2);
  });

  it("clamps an out-of-range param from the URL", () => {
    const back = parseAppUrl("?view=forcelaw&preset=yukawa&lambda=999");
    expect(back.forceParams?.lambda).toBe(20); // spec max
  });

  it("falls back to preset defaults when a param is missing", () => {
    const back = parseAppUrl("?view=forcelaw&preset=finitewell&v0=1.5");
    expect(back.forceParams?.v0).toBe(1.5);
    expect(back.forceParams?.a).toBe(defaultParams("finitewell").a); // default
  });

  it("drops a negative fl and keeps forcelaw view otherwise clean", () => {
    const out = parseAppUrl("?fl=-2");
    expect(out.forceL).toBeUndefined();
  });

  it("omits preset and fl when at defaults", () => {
    const q = serializeAppUrl({ ...URL_DEFAULTS });
    expect(q).not.toContain("fl=");
    expect(q).not.toContain("preset=");
  });

  it("round-trips a screened-atom config deep link", () => {
    const state = { ...URL_DEFAULTS, system: "na", config: "1s2 2s2 2p6 3p1" };
    const q = serializeAppUrl(state);
    expect(q).toContain("config=");
    expect({ ...URL_DEFAULTS, ...parseAppUrl(q) }.config).toBe("1s2 2s2 2p6 3p1");
  });

  it("omits config when null and drops a malformed config", () => {
    expect(serializeAppUrl({ ...URL_DEFAULTS })).not.toContain("config=");
    expect(parseAppUrl("?config=not-a-config").config).toBeUndefined();
  });
});

describe("dirac level-model url state", () => {
  it("round-trips the dirac toggle with fine structure", () => {
    const s = { ...URL_DEFAULTS, view: "levels" as const, fineStructure: true, dirac: true };
    const q = serializeAppUrl(s);
    expect(q).toContain("dirac=1");
    const back = { ...URL_DEFAULTS, ...parseAppUrl(q) };
    expect(back.dirac).toBe(true);
  });

  it("omits dirac when off, or when fine structure is off", () => {
    expect(serializeAppUrl({ ...URL_DEFAULTS, dirac: false })).not.toContain("dirac");
    expect(
      serializeAppUrl({ ...URL_DEFAULTS, fineStructure: false, dirac: true }),
    ).not.toContain("dirac");
  });
});
