import { describe, expect, it } from "vitest";
import { measurementsIn } from "./prose";
import { TOURS } from "./registry";

describe("measurementsIn", () => {
  it("finds a number carrying a unit", () => {
    expect(measurementsIn("sits at -3.40 eV, always")).toEqual(["-3.40 eV"]);
    expect(measurementsIn("121.567 nm in vacuum")).toEqual(["121.567 nm"]);
    expect(measurementsIn("about 53 pm across")).toEqual(["53 pm"]);
  });

  it("ignores numbers that are not measurements", () => {
    // Orbital labels, counts, and ordinals are not claims about a value, and
    // a lint that flagged them would be turned off within a week.
    expect(measurementsIn("the 2p has four lobes, not five")).toEqual([]);
    expect(measurementsIn("step 3 of 11")).toEqual([]);
    expect(measurementsIn("a 1s2 2s2 2p6 core")).toEqual([]);
  });

  it("does not mistake a unit inside a word", () => {
    // "K" for kelvin must not match the K that starts a capitalised word.
    expect(measurementsIn("10 Kelvins of nothing")).toEqual([]);
    expect(measurementsIn("10 K of nothing")).toEqual(["10 K"]);
  });

  it("catches every unit the tours actually use", () => {
    for (const u of ["eV", "nm", "pm", "bohr", "K", "%"]) {
      expect(measurementsIn(`7 ${u}`), u).toHaveLength(1);
    }
  });
});

describe("tour prose", () => {
  it("backs every measurement with a claim", () => {
    // The failure this guards: a step quotes a number, the engine improves,
    // and the prose keeps asserting the old value with nothing to catch it.
    for (const t of TOURS) {
      for (const s of t.steps) {
        const found = s.body.flatMap(measurementsIn);
        if (found.length > 0) {
          expect(
            s.claims?.length ?? 0,
            `${t.id}/${s.id} quotes ${found.join(", ")} with no claim behind it`,
          ).toBeGreaterThan(0);
        }
      }
    }
  });
});
