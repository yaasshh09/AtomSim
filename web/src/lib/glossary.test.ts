import { describe, expect, it } from "vitest";
import { GLOSSARY, glossParts } from "./glossary";

describe("glossary", () => {
  it("defines every term in a short plain sentence, without dashes", () => {
    for (const [term, def] of Object.entries(GLOSSARY)) {
      expect(def.length, term).toBeLessThanOrEqual(160);
      expect(def, term).not.toMatch(/[—–]/);
      expect(def.endsWith("."), term).toBe(true);
    }
  });

  it("marks the first occurrence of a term and leaves the text whole", () => {
    const text = "An orbital is a wavefunction. Every orbital has nodes.";
    const parts = glossParts(text);
    expect(parts.map((p) => p.text).join("")).toBe(text);
    expect(parts.filter((p) => p.term).map((p) => p.term)).toEqual(["orbital", "wavefunction"]);
  });

  it("prefers the longest term and matches whole words only", () => {
    const parts = glossParts("Hartree-Fock energies are in hartree; nodes are not a nodeless thing.");
    expect(parts.filter((p) => p.term).map((p) => p.term)).toEqual(["Hartree-Fock", "hartree"]);
  });

  it("matches regardless of case but keeps the text as written", () => {
    const parts = glossParts("The Exchange term.");
    expect(parts[1]).toEqual({ text: "Exchange", term: "exchange" });
  });
});
