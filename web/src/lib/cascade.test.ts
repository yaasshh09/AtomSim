import { describe, expect, it } from "vitest";
import type { Cascade, Quantity } from "../api/types";
import { endpoints, formatSeconds, grotrian, levelName, topBranches } from "./cascade";
import { CASCADE_GROTRIAN_LIBERTY } from "./liberties";

const prov = { fidelity: "numerical" as const, method: "", assumptions: [], error_estimate: null, refinement: null };
const q = (value: number): Quantity => ({ value, unit: "", label: "", provenance: prov });

// The 3p cascade as the server sends it: 88% straight to 1s, 12% to the 2s trap.
const c3p: Cascade = {
  start: [3, 1],
  levels: [
    { n: 3, l: 1, visit_probability: q(1), lifetime: q(5.27e-9), terminal: false },
    { n: 2, l: 0, visit_probability: q(0.118), lifetime: null, terminal: true },
    { n: 1, l: 0, visit_probability: q(0.882), lifetime: null, terminal: true },
  ],
  branches: [
    { upper: [3, 1], lower: [1, 0], einstein_A: q(1.67e8), branching_ratio: q(0.882), photons: q(0.882), wavelength: q(102.5) },
    { upper: [3, 1], lower: [2, 0], einstein_A: q(2.2e7), branching_ratio: q(0.118), photons: q(0.118), wavelength: q(656.1) },
  ],
  mean_settle_time: q(5.27e-9),
  populations: [],
};

describe("formatSeconds", () => {
  it("picks the unit that keeps the number at or above one", () => {
    expect(formatSeconds(5.2686e-9)).toBe("5.27 ns");
    expect(formatSeconds(1.6e-9)).toBe("1.6 ns");
    expect(formatSeconds(2.4e-7)).toBe("240 ns");
    expect(formatSeconds(0.12)).toBe("120 ms");
    expect(formatSeconds(3)).toBe("3 s");
  });

  it("says infinity for a level that cannot decay, and zero plainly", () => {
    expect(formatSeconds(Infinity)).toBe("∞");
    expect(formatSeconds(0)).toBe("0 s");
  });
});

describe("grotrian", () => {
  it("puts the start at the top and n = 1 at the bottom, columns by l", () => {
    const { nodes, arrows } = grotrian(c3p);
    const p3 = nodes.find((nd) => nd.n === 3)!;
    const s1 = nodes.find((nd) => nd.n === 1)!;
    expect(p3.y).toBe(0);
    expect(s1.y).toBe(1);
    expect(p3.x).toBe(1);
    expect(s1.x).toBe(0);
    expect(arrows).toHaveLength(2);
    expect(arrows.every((a) => a.to.y > a.from.y)).toBe(true);
  });
});

describe("summaries", () => {
  it("ranks where the atom ends up and the brightest lines", () => {
    expect(endpoints(c3p).map((e) => e.name)).toEqual(["1s", "2s"]);
    expect(topBranches(c3p, 1)[0].wavelength.value).toBe(102.5);
    expect(levelName(4, 3)).toBe("4f");
  });

  it("discloses that the rows are not an energy scale", () => {
    expect(CASCADE_GROTRIAN_LIBERTY.fidelity).toBe("visual_liberty");
    expect(CASCADE_GROTRIAN_LIBERTY.assumptions.join(" ")).toContain("not by energy");
  });
});
