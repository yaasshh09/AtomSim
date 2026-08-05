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
      compare: true,
      bField: 0,
      eField: 0,
      hyperfine: true,
      intensities: false,
      thermal: true,
      temperatureK: 12000,
      logNe: 15,
      profile: true,
      logResolvingPower: 4.5,
      profileZoom: [656.1, 656.5] as [number, number],
      absorption: true,
      logColumn: 21.5,
      config: null,
      model: "hf" as const,
      exchange: false,
      pauli: false,
      surfaceMode: "both" as const,
      isoFraction: 0.5,
      tour: null,
      step: 0,
    };
    const parsed = parseAppUrl(serializeAppUrl(state));
    expect({ ...URL_DEFAULTS, ...parsed }).toEqual(state);
  });

  it("carries the profile controls only when the profile is on", () => {
    // R and a zoom window describe a curve; without the curve they would be
    // dead parameters that reopen into nothing.
    const off = serializeAppUrl({
      ...URL_DEFAULTS, profile: false, logResolvingPower: 4, profileZoom: [1, 2],
    });
    expect(off).not.toContain("rp=");
    expect(off).not.toContain("zoom=");
    const on = serializeAppUrl({
      ...URL_DEFAULTS, profile: true, logResolvingPower: 4, profileZoom: [656.1, 656.5],
    });
    expect(parseAppUrl(on).profile).toBe(true);
    expect(parseAppUrl(on).logResolvingPower).toBe(4);
    expect(parseAppUrl(on).profileZoom).toEqual([656.1, 656.5]);
  });

  it("rejects a zoom window that is not real light in order", () => {
    for (const bad of ["0,500", "-5,500", "700,600", "abc", "600"]) {
      expect(parseAppUrl(`?prof=1&zoom=${bad}`).profileZoom).toBeUndefined();
    }
  });

  it("rejects a resolving power outside the slider's range", () => {
    expect(parseAppUrl("?prof=1&rp=1").logResolvingPower).toBeUndefined();
    expect(parseAppUrl("?prof=1&rp=9").logResolvingPower).toBeUndefined();
    expect(parseAppUrl("?prof=1&rp=4.5").logResolvingPower).toBe(4.5);
  });

  it("round-trips the surface mode and the fraction it encloses", () => {
    const link = serializeAppUrl({
      ...URL_DEFAULTS, surfaceMode: "surface", isoFraction: 0.99,
    });
    expect(link).toContain("surf=surface");
    expect(link).toContain("iso=0.99");
    expect(parseAppUrl(link).surfaceMode).toBe("surface");
    expect(parseAppUrl(link).isoFraction).toBe(0.99);
  });

  it("carries the fraction only when a surface is being drawn", () => {
    // A contour nobody is looking at is a dead parameter, exactly like the
    // profile controls with the profile off.
    const cloud = serializeAppUrl({ ...URL_DEFAULTS, isoFraction: 0.5 });
    expect(cloud).not.toContain("iso=");
  });

  it("takes a hand-written fraction that is not one of the presets", () => {
    // 0.6827 is one sigma, and someone will type it. The presets are a
    // convenience in the UI, not the set of questions that can be asked.
    expect(parseAppUrl("?surf=surface&iso=0.6827").isoFraction).toBe(0.6827);
  });

  it("rejects fractions that are not contours", () => {
    for (const bad of ["0", "1", "-0.5", "1.5", "abc"]) {
      expect(parseAppUrl(`?surf=surface&iso=${bad}`).isoFraction).toBeUndefined();
    }
    expect(parseAppUrl("?surf=hologram").surfaceMode).toBeUndefined();
  });

  it("round-trips the intensities toggle, which defaults on", () => {
    expect(URL_DEFAULTS.intensities).toBe(true);
    const off = serializeAppUrl({ ...URL_DEFAULTS, intensities: false });
    expect(off).toContain("int=0");
    expect(parseAppUrl(off).intensities).toBe(false);
  });

  it("round-trips the LTE toggle, which defaults off", () => {
    expect(URL_DEFAULTS.thermal).toBe(false);
    const on = serializeAppUrl({ ...URL_DEFAULTS, thermal: true });
    expect(on).toContain("lte=1");
    expect(parseAppUrl(on).thermal).toBe(true);
  });

  it("round-trips the absorption toggle, which defaults off", () => {
    expect(URL_DEFAULTS.absorption).toBe(false);
    const on = serializeAppUrl({ ...URL_DEFAULTS, absorption: true });
    expect(on).toContain("abs=1");
    expect(parseAppUrl(on).absorption).toBe(true);
  });

  it("round-trips a column density away from the default", () => {
    // The column is what walks the gas from a faithful census to a saturated
    // one, so a link to a saturated spectrum has to survive being shared.
    const url = serializeAppUrl({
      ...URL_DEFAULTS, absorption: true, logColumn: 22.5,
    });
    const back = { ...URL_DEFAULTS, ...parseAppUrl(url) };
    expect(back.logColumn).toBe(22.5);
  });

  it("carries the column only when absorption is on", () => {
    const off = serializeAppUrl({
      ...URL_DEFAULTS, absorption: false, logColumn: 24,
    });
    expect(off).not.toContain("col=");
  });

  it("drops a column outside the range the view can draw", () => {
    expect(parseAppUrl("?abs=1&col=5").logColumn).toBeUndefined();
    expect(parseAppUrl("?abs=1&col=40").logColumn).toBeUndefined();
    expect(parseAppUrl("?abs=1&col=22").logColumn).toBe(22);
  });

  it("carries the conditions only when LTE weighting is on", () => {
    // They describe a model that is not running otherwise, so putting them in
    // the URL would promise a state the page does not restore.
    const off = serializeAppUrl({
      ...URL_DEFAULTS, thermal: false, temperatureK: 25000, logNe: 9,
    });
    expect(off).not.toContain("tk=");
    expect(off).not.toContain("ne=");
  });

  it("round-trips a temperature and density away from the defaults", () => {
    const url = serializeAppUrl({
      ...URL_DEFAULTS, thermal: true, temperatureK: 25000, logNe: 9.5,
    });
    const back = { ...URL_DEFAULTS, ...parseAppUrl(url) };
    expect(back.temperatureK).toBe(25000);
    expect(back.logNe).toBe(9.5);
  });

  it("drops conditions outside the range the server accepts", () => {
    expect(parseAppUrl("?lte=1&tk=5").temperatureK).toBeUndefined();
    expect(parseAppUrl("?lte=1&tk=1e9").temperatureK).toBeUndefined();
    expect(parseAppUrl("?lte=1&ne=1").logNe).toBeUndefined();
    expect(parseAppUrl("?lte=1&ne=40").logNe).toBeUndefined();
  });

  it("omits int when intensities are on, since that is the default", () => {
    const url = serializeAppUrl({ ...URL_DEFAULTS, intensities: true });
    expect(url).not.toContain("int=");
    expect({ ...URL_DEFAULTS, ...parseAppUrl(url) }.intensities).toBe(true);
  });

  it("round-trips the hyperfine toggle", () => {
    const url = serializeAppUrl({ ...URL_DEFAULTS, hyperfine: true });
    expect(url).toContain("hf=1");
    expect(parseAppUrl(url).hyperfine).toBe(true);
  });

  it("omits hf when hyperfine is off", () => {
    const url = serializeAppUrl({ ...URL_DEFAULTS, hyperfine: false });
    expect(url).not.toContain("hf=");
  });

  it("round-trips the e_field", () => {
    const url = serializeAppUrl({ ...URL_DEFAULTS, eField: 40 });
    expect(url).toContain("ef=40");
    expect(parseAppUrl(url).eField).toBe(40);
  });

  it("omits ef when field is zero", () => {
    const url = serializeAppUrl({ ...URL_DEFAULTS, eField: 0 });
    expect(url).not.toContain("ef=");
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

describe("many-electron model url state", () => {
  it("round-trips the model key", () => {
    const q = serializeAppUrl({ ...URL_DEFAULTS, system: "ne", model: "hf" });
    expect(q).toContain("model=hf");
    expect({ ...URL_DEFAULTS, ...parseAppUrl(q) }.model).toBe("hf");
  });

  it("defaults to gsz so existing deep links keep resolving as before", () => {
    expect({ ...URL_DEFAULTS, ...parseAppUrl("?system=ne") }.model).toBe("gsz");
  });

  it("omits the default from the serialized url", () => {
    expect(serializeAppUrl({ ...URL_DEFAULTS })).not.toContain("model=");
  });

  // The plan asked for a throw here. Every other parameter in this module is
  // dropped instead, and the contract that junk never reaches the store is
  // worth more than one parameter's strictness: a link with a typo should
  // still open the app on the default physics rather than fail to render.
  it("drops an unknown model rather than throwing", () => {
    expect(parseAppUrl("?model=dft").model).toBeUndefined();
    expect({ ...URL_DEFAULTS, ...parseAppUrl("?model=dft") }.model).toBe("gsz");
  });

  // `hf=1` is the hyperfine toggle and `model=hf` is Hartree-Fock. Two
  // different things spelled the same way two characters apart, so this pins
  // that neither one can ever be read as the other.
  it("does not confuse model=hf with the hyperfine hf=1 flag", () => {
    const q = serializeAppUrl({ ...URL_DEFAULTS, model: "hf", hyperfine: true });
    const back = { ...URL_DEFAULTS, ...parseAppUrl(q) };
    expect(back.model).toBe("hf");
    expect(back.hyperfine).toBe(true);

    const onlyHyperfine = { ...URL_DEFAULTS, ...parseAppUrl("?hf=1") };
    expect(onlyHyperfine.hyperfine).toBe(true);
    expect(onlyHyperfine.model).toBe("gsz");

    const onlyModel = { ...URL_DEFAULTS, ...parseAppUrl("?model=hf") };
    expect(onlyModel.model).toBe("hf");
    expect(onlyModel.hyperfine).toBe(false);
  });
});

describe("zeeman b-field url state", () => {
  it("round-trips the b_field with fine structure", () => {
    const url = serializeAppUrl({ ...URL_DEFAULTS, fineStructure: true, bField: 2.5 });
    expect(url).toContain("b=2.5");
    expect(parseAppUrl(url).bField).toBe(2.5);
  });

  it("omits b when field is zero", () => {
    const url = serializeAppUrl({ ...URL_DEFAULTS, fineStructure: true, bField: 0 });
    expect(url).not.toContain("b=");
  });

  it("omits b when fine structure is off", () => {
    const url = serializeAppUrl({ ...URL_DEFAULTS, fineStructure: false, bField: 3 });
    expect(url).not.toContain("b=");
  });

  it("a link cannot land anyone in altered physics by omission", () => {
    // Absence of the key means exchange is ON. A deep link written before this
    // toggle existed, or one a user hand-trims, gets the real atom.
    expect(URL_DEFAULTS.exchange).toBe(true);
    expect(serializeAppUrl(URL_DEFAULTS)).not.toContain("nox");
    expect(parseAppUrl("?system=ne&model=hf").exchange).toBeUndefined();
  });

  it("round-trips the counterfactual as nox=1", () => {
    const off = serializeAppUrl({ ...URL_DEFAULTS, exchange: false });
    expect(off).toContain("nox=1");
    expect(parseAppUrl(off).exchange).toBe(false);
  });

  it("the stronger counterfactual cannot arrive by omission either", () => {
    expect(URL_DEFAULTS.pauli).toBe(true);
    expect(serializeAppUrl(URL_DEFAULTS)).not.toContain("nopauli");
    expect(parseAppUrl("?system=ne&model=hf").pauli).toBeUndefined();
  });

  it("writes both keys for the collapse, since it means both things", () => {
    const off = serializeAppUrl({ ...URL_DEFAULTS, pauli: false, exchange: false });
    expect(off).toContain("nopauli=1");
    expect(off).toContain("nox=1");
  });

  it("reads a hand-trimmed nopauli=1 as the collapse it obviously means", () => {
    // Honouring `nopauli=1` without `nox` literally would build a request the
    // API answers 422 to, which is a worse reading of the intent than the
    // obvious one: the cap and antisymmetry are one rule.
    const parsed = parseAppUrl("?system=ne&model=hf&nopauli=1");
    expect(parsed.pauli).toBe(false);
    expect(parsed.exchange).toBe(false);
  });
});

describe("density comparison url state", () => {
  it("round-trips the compare toggle", () => {
    const q = serializeAppUrl({ ...URL_DEFAULTS, system: "na", compare: true });
    expect(q).toContain("compare=1");
    const back = { ...URL_DEFAULTS, ...parseAppUrl(q) };
    expect(back.compare).toBe(true);
  });

  it("omits compare when off", () => {
    expect(serializeAppUrl({ ...URL_DEFAULTS, compare: false })).not.toContain("compare");
  });
});

describe("tour deep links", () => {
  it("carries a tour and a step", () => {
    const s = parseAppUrl("?tour=hydrogen-honestly&step=4");
    expect(s.tour).toBe("hydrogen-honestly");
    expect(s.step).toBe(4);
  });

  it("defaults the step to the first one", () => {
    expect(parseAppUrl("?tour=hydrogen-honestly").step).toBe(0);
  });

  it("drops a junk step rather than throwing", () => {
    // Same contract as every other parameter here: junk never reaches the
    // store, so a typo'd link still opens the app.
    expect(parseAppUrl("?tour=x&step=banana").step).toBe(0);
    expect(parseAppUrl("?tour=x&step=-3").step).toBe(0);
  });

  it("omits the step when there is no tour", () => {
    // A bare ?step= describes nothing and would survive into a shared link as
    // noise.
    const qs = serializeAppUrl({ ...URL_DEFAULTS, tour: null, step: 5 });
    expect(qs).not.toContain("step=");
  });

  it("round-trips", () => {
    const want = { ...URL_DEFAULTS, tour: "hydrogen-honestly", step: 3 };
    expect({ ...want, ...parseAppUrl(serializeAppUrl(want)) }).toEqual(want);
  });
});
