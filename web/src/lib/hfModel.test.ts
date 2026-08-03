import { describe, expect, it } from "vitest";
import type { HFLevels } from "../api/types";
import { HF_ORBITAL_CAPTION, manyElectronParams, subshellAvailable } from "./hfModel";

const NEON = {
  kind: "hf",
  z: 10,
  n_electrons: 10,
  config: "1s2 2s2 2p6",
  orbitals: [
    { n: 1, l: 0 },
    { n: 2, l: 0 },
    { n: 2, l: 1 },
  ],
} as unknown as HFLevels;

describe("manyElectronParams", () => {
  it("carries all four fields a job needs", () => {
    const p = manyElectronParams({
      model: "hf",
      config: "1s2 2s2 2p5 3s1",
      exchange: false,
      pauli: true,
    });
    expect(p).toEqual({
      model: "hf",
      config: "1s2 2s2 2p5 3s1",
      exchange: false,
      pauli: true,
    });
  });

  it("sends the real physics under the screened model", () => {
    // The counterfactual switches name Hartree-Fock's rules. Forwarding a
    // stale one under GSZ would put a flag on the wire that the picture does
    // not honour, and the server would echo it back into a badge.
    const p = manyElectronParams({
      model: "gsz",
      config: "1s2 2s2 2p6",
      exchange: false,
      pauli: false,
    });
    expect(p).toEqual({
      model: "gsz",
      config: "1s2 2s2 2p6",
      exchange: true,
      pauli: true,
    });
  });
});

describe("subshellAvailable", () => {
  it("allows everything under the screened model", () => {
    expect(subshellAvailable(null, "gsz", 3, 2)).toBe(true);
    expect(subshellAvailable(NEON, "gsz", 3, 2)).toBe(true);
  });

  it("allows everything before the solve has landed", () => {
    // Not knowing is not the same as knowing it is empty, and greying a
    // control on a guess teaches the wrong thing about the atom.
    expect(subshellAvailable(null, "hf", 3, 2)).toBe(true);
  });

  it("allows an occupied subshell and refuses an empty one", () => {
    expect(subshellAvailable(NEON, "hf", 2, 1)).toBe(true);
    expect(subshellAvailable(NEON, "hf", 1, 0)).toBe(true);
    expect(subshellAvailable(NEON, "hf", 3, 2)).toBe(false);
    expect(subshellAvailable(NEON, "hf", 3, 0)).toBe(false);
  });
});

describe("HF_ORBITAL_CAPTION", () => {
  it("says both halves of the claim", () => {
    expect(HF_ORBITAL_CAPTION).toContain("not an observable");
    expect(HF_ORBITAL_CAPTION).toContain("spherical");
  });
});
