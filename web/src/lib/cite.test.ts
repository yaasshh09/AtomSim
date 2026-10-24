import { describe, expect, it } from "vitest";
import type { Provenance } from "../api/types";
import { citeBibtex } from "./cite";

const prov = (method: string, assumptions: string[] = []): Provenance =>
  ({ fidelity: "approximation", method, assumptions, error_estimate: null, refinement: null }) as Provenance;

describe("citeBibtex", () => {
  it("always cites the software, with its version and access date", () => {
    const b = citeBibtex(prov("closed-form <r>"), "0.1.0", "2026-10-24");
    expect(b).toContain("@software{atomsim");
    expect(b).toContain("version = {0.1.0}");
    expect(b).toContain("urldate = {2026-10-24}");
    expect(b).not.toContain("@article");
  });

  it("adds only the primary sources the provenance names", () => {
    const gsz = citeBibtex(prov("GSZ screened central field"), "0.1.0", "2026-10-24");
    expect(gsz).toContain("@article{szydlik1974");
    expect(gsz).not.toContain("nistasd");
    const both = citeBibtex(prov("Bohr energy", ["converted to eV via CODATA", "compared with NIST"]), "0.1.0", "x");
    expect(both).toContain("@misc{codata");
    expect(both).toContain("@misc{nistasd");
  });

  it("opens with a comment naming the tier and the method", () => {
    expect(citeBibtex(prov("Koopmans"), "0.1.0", "x").split("\n")[0]).toBe("% APPROXIMATION: Koopmans");
  });
});
