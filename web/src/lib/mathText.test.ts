import { describe, expect, it } from "vitest";
import { mathParts } from "./mathText";

/** The parts written back as one line: `_x` is dropped, `^x` is raised. */
function shape(s: string): string {
  return mathParts(s)
    .map((p) => (p.kind === "sub" ? `_${p.text}` : p.kind === "sup" ? `^${p.text}` : p.text))
    .join("");
}

/** Everything at the baseline, which is the string as a reader sees it. */
function flat(s: string): string {
  return mathParts(s)
    .map((p) => p.text)
    .join("");
}

describe("engine labels and units", () => {
  it("sets the radial function's label", () => {
    expect(shape("R_6,0 (Z=1, mu/m_e=0.999456)")).toBe("R_6,0 (Z=1, μ/m_e=0.999456)");
  });

  it("sets the probability label and its unit", () => {
    expect(shape("P_6,1(r) = r^2 R^2 [bohr^-1]")).toBe("P_6,1(r) = r^2 R^2 [bohr^−1]");
  });

  it("sets a fractional unit exponent whole", () => {
    expect(shape("bohr^-3/2")).toBe("bohr^−3/2");
  });

  it("sets a plane label", () => {
    expect(shape("|psi_3,1,-1|^2 on the y=0 plane")).toBe("|ψ_3,1,−1|^2 on the y=0 plane");
  });

  it("reads a sum over orbitals", () => {
    expect(flat("D(r) = sum_a q_a P_a(r)^2")).toBe("D(r) = Σa qa Pa(r)2");
  });

  it("brackets an expectation value and a matrix element", () => {
    expect(flat("<r>_2,0")).toBe("⟨r⟩2,0");
    expect(flat("<3,1|r|2,0>")).toBe("⟨3,1|r|2,0⟩");
  });

  it("subscripts under a Greek base, spelled out or already a character", () => {
    expect(shape("(λ_computed − λ_NIST)/λ_NIST")).toBe("(λ_computed − λ_NIST)/λ_NIST");
    expect(shape("mu_ratio = 0.9995")).toBe("μ_ratio = 0.9995");
  });

  it("gives Schrodinger the umlaut an ASCII source could not", () => {
    expect(flat("non-relativistic Schrodinger equation")).toBe(
      "non-relativistic Schrödinger equation",
    );
  });

  it("turns a transition arrow and an inequality into their own glyphs", () => {
    expect(flat("A 3p->2s")).toBe("A 3p→2s");
    expect(flat("E_HF >= E_exact")).toBe("EHF ≥ Eexact");
  });
});

describe("numbers", () => {
  it("sets exponential notation as a power of ten", () => {
    expect(shape("1.234e-3")).toBe("1.234×10^−3");
    expect(shape("4e+22")).toBe("4×10^22");
  });

  it("drops the digit of a bare indexed symbol", () => {
    expect(shape("-V0 to V0")).toBe("-V_0 to V_0");
    expect(shape("n1=2, n2=0")).toBe("n_1=2, n_2=0");
  });

  it("leaves digits that are part of a word or a number alone", () => {
    expect(shape("He2 and log10 and 2n and 10")).toBe("He2 and log10 and 2n and 10");
  });
});

describe("what it must not touch", () => {
  // The provenance strings name real functions and real parameters. A
  // subscript through the middle of one turns a thing the reader can grep for
  // into a thing they cannot.
  it("leaves identifiers as they were typed", () => {
    for (const id of [
      "hf_atom.py",
      "sph_harm_y",
      "eigh_tridiagonal",
      "solve_radial_with_error",
    ]) {
      expect(shape(id)).toBe(id);
    }
    expect(shape("column_density_m2 must be in [0, 1e30]")).toBe(
      "column_density_m2 must be in [0, 1×10^30]",
    );
  });

  it("leaves ordinary prose alone", () => {
    const prose = "the two models place under 1.5% of the electrons differently";
    expect(shape(prose)).toBe(prose);
  });

  it("keeps angle brackets that are not a bracket", () => {
    expect(flat("a value <something longer than a symbol> here")).toBe(
      "a value <something longer than a symbol> here",
    );
  });

  it("never adds or drops characters, only heights", () => {
    // The count is the check: everything the parser recognises is the same
    // glyphs at a different height, so a label cannot quietly lose a minus
    // sign or a digit on the way to the screen.
    const src = "P_6,1(r) = r^2 R^2 [bohr^-1]";
    expect(flat(src).replace(/[−]/g, "-")).toBe(src.replace(/[_^]/g, ""));
  });
});
