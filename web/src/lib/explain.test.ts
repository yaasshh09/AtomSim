import { describe, expect, it } from "vitest";
import {
  activeScenario,
  describeDensity,
  describeEField,
  describeField,
  describeResolvingPower,
  describeTemperature,
  nistSummary,
  PRESET_BLURBS,
  SCENARIOS,
  VIEW_LEADS,
  type ScenarioMultipliers,
} from "./explain";
import { PRESET_LABELS, type ForcePreset } from "./forceLaw";

describe("magnitude anchors", () => {
  it("names no field at zero, for every quantity that has a rest position", () => {
    expect(describeField(0)).toBe("no field");
    expect(describeEField(0)).toBe("no field");
  });
  it("puts a clinical MRI in the 1.5-3 T band", () => {
    expect(describeField(1.5)).toBe("about an MRI scanner");
    expect(describeField(3)).toBe("about an MRI scanner");
    expect(describeField(1.4)).not.toBe("about an MRI scanner");
  });
  it("splits the electric field at air's breakdown and at field ionization", () => {
    expect(describeEField(1)).toContain("below air");
    expect(describeEField(10)).toContain("past air");
    // F_ion ~ 1/(16 n^4) atomic units, 1 a.u. = 5.14e5 MV/m, so n=6 lands
    // near 25 MV/m. The band boundary is that number and not a round one.
    expect(describeEField(30)).toContain("strip hydrogen");
  });
  it("returns a non-empty anchor across each slider's full range", () => {
    for (let t = 0; t <= 20; t += 0.5) expect(describeField(t).length).toBeGreaterThan(0);
    for (let f = 0; f <= 100; f += 2.5) expect(describeEField(f).length).toBeGreaterThan(0);
    for (let e = 2; e <= 6; e += 0.25) {
      expect(describeTemperature(10 ** e).length).toBeGreaterThan(0);
    }
    for (let e = 4; e <= 22; e += 0.5) expect(describeDensity(e).length).toBeGreaterThan(0);
    for (let e = 2; e <= 7; e += 0.25) {
      expect(describeResolvingPower(e).length).toBeGreaterThan(0);
    }
  });
  it("keeps the Sun's photosphere on the Sun-like rung", () => {
    expect(describeTemperature(5772)).toBe("a Sun-like photosphere");
  });
  it("says which density band the existing photosphere hint agreed on", () => {
    expect(describeDensity(7)).toBe("a nebula");
    expect(describeDensity(13)).toBe("a stellar photosphere");
  });
  it("distinguishes no instrument from a bad one", () => {
    expect(describeResolvingPower(null)).toContain("no instrument");
    expect(describeResolvingPower(2)).not.toContain("no instrument");
  });
});

describe("nistSummary", () => {
  const row = (ok: boolean) => ({ within_tolerance: ok });
  it("is null when there is nothing to compare", () => {
    expect(nistSummary(null, 1e-6)).toBeNull();
    expect(nistSummary([], 1e-6)).toBeNull();
  });
  it("counts agreements and reports the tolerance", () => {
    const s = nistSummary([row(true), row(true), row(false)], 1e-6);
    expect(s).not.toBeNull();
    expect(s!.within).toBe(2);
    expect(s!.total).toBe(3);
    expect(s!.allWithin).toBe(false);
    expect(s!.headline).toContain("2 of 3");
    expect(s!.headline).toContain("1e-6");
  });
  it("says so plainly when everything agrees", () => {
    const s = nistSummary([row(true), row(true)], 1e-6)!;
    expect(s.allWithin).toBe(true);
    expect(s.headline).toContain("All 2");
  });
  it("agrees singular and plural", () => {
    expect(nistSummary([row(true)], null)!.headline).toContain("1 computed line ");
    expect(nistSummary([row(true), row(false)], null)!.headline).toContain("sits outside");
  });
  it("omits the tolerance clause when there is no stated tolerance", () => {
    expect(nistSummary([row(true)], null)!.headline).not.toContain("tolerance");
  });
});

describe("scenarios", () => {
  /* The three observables, as the engine derives them from the raw constants:
     α ∝ e²/(ε₀ħc), a₀ ∝ ε₀ħ²/(mₑe²), E_h ∝ mₑe⁴/(ε₀²ħ²). Written out here so
     a scenario's claim about what it changes is checked rather than asserted
     in a comment; the engine remains the authority for the values on screen. */
  const alpha = (m: ScenarioMultipliers) => m.e ** 2 / (m.eps0 * m.hbar * m.c);
  const bohr = (m: ScenarioMultipliers) => (m.eps0 * m.hbar ** 2) / (m.m_e * m.e ** 2);
  const hartree = (m: ScenarioMultipliers) =>
    (m.m_e * m.e ** 4) / (m.eps0 ** 2 * m.hbar ** 2);

  const byKey = (k: string) => SCENARIOS.find((s) => s.key === k)!.multipliers;

  it("leaves every observable exactly unchanged in the disguised universe", () => {
    const m = byKey("disguise");
    expect(alpha(m)).toBeCloseTo(1, 12);
    expect(bohr(m)).toBeCloseTo(1, 12);
    expect(hartree(m)).toBeCloseTo(1, 12);
    // And it is genuinely a different universe, not the real one relabelled.
    expect(m.e).not.toBe(1);
  });
  it("changes only what each scenario claims to change", () => {
    const strong = byKey("strong-em");
    expect(alpha(strong)).toBeCloseTo(4, 12);
    const heavy = byKey("heavy-electron");
    expect(alpha(heavy)).toBeCloseTo(1, 12);
    expect(bohr(heavy)).toBeCloseTo(0.5, 12);
    expect(hartree(heavy)).toBeCloseTo(2, 12);
  });
  it("puts the broken scenario past the perturbative bound and nothing else", () => {
    const REAL_ALPHA = 0.0072973525643;
    const broken = byKey("broken");
    expect(REAL_ALPHA * alpha(broken)).toBeGreaterThan(0.5);
    for (const s of SCENARIOS) {
      if (s.key === "broken") continue;
      expect(REAL_ALPHA * alpha(s.multipliers)).toBeLessThanOrEqual(0.5);
    }
  });
  it("keeps every multiplier inside the server's [0.25, 4] validation", () => {
    for (const s of SCENARIOS) {
      for (const v of Object.values(s.multipliers)) {
        expect(v).toBeGreaterThanOrEqual(0.25);
        expect(v).toBeLessThanOrEqual(4);
      }
    }
  });
  it("recognises the applied scenario, and only that one", () => {
    expect(activeScenario({ hbar: 1, e: 1, m_e: 1, eps0: 1, c: 1 })!.key).toBe("real");
    expect(activeScenario(byKey("strong-em"))!.key).toBe("strong-em");
    expect(activeScenario({ hbar: 1, e: 1.5, m_e: 1, eps0: 1, c: 1 })).toBeNull();
  });
  it("gives every scenario a label and something to watch", () => {
    for (const s of SCENARIOS) {
      expect(s.label.length).toBeGreaterThan(0);
      expect(s.blurb.length).toBeGreaterThan(0);
    }
  });
});

describe("copy tables", () => {
  it("leads every view the app can show", () => {
    for (const key of ["radial", "levels", "spectrum", "whatif", "forcelaw"]) {
      const lead = VIEW_LEADS[key];
      expect(lead, key).toBeDefined();
      expect(lead.title.length).toBeGreaterThan(0);
      expect(lead.lead.length).toBeGreaterThan(0);
    }
  });
  it("describes every force-law preset the picker offers", () => {
    for (const preset of Object.keys(PRESET_LABELS) as ForcePreset[]) {
      expect(PRESET_BLURBS[preset], preset).toBeDefined();
      expect(PRESET_BLURBS[preset].length).toBeGreaterThan(0);
    }
  });
});
