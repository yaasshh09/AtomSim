import { describe, expect, it } from "vitest";
import { URL_DEFAULTS } from "../lib/urlState";
import { tourReset } from "./apply";

const systems = [
  { key: "h", name: "hydrogen", has_gsz: true },
  { key: "s", name: "sulfur", has_gsz: false },
] as never[];

describe("tourReset", () => {
  it("carries every input the step asked for", () => {
    const out = tourReset({ ...URL_DEFAULTS, n: 3, l: 2, view: "levels" }, systems);
    expect(out.n).toBe(3);
    expect(out.l).toBe(2);
    expect(out.view).toBe("levels");
  });

  it("clears every derived field, not just the level payloads", () => {
    // The failure this prevents: a step changes n and system, and the previous
    // step's cloud, plane, surface and spectrum keep rendering under the new
    // labels because raw setState never spreads INVALIDATED.
    const out = tourReset({ ...URL_DEFAULTS, n: 2 }, systems);
    for (const k of [
      "stateInfo",
      "positions",
      "density",
      "phase",
      "meta",
      "plane",
      "iso",
      "radial",
      "levels",
      "spectrum",
      "curveOfGrowth",
      "absorptionData",
    ]) {
      expect(out[k as keyof typeof out], `${k} not cleared`).toBeNull();
    }
    expect(out.status).toBe("idle");
    expect(out.planeStatus).toBe("idle");
    expect(out.isoStatus).toBe("idle");
  });

  it("clears the payloads that live outside INVALIDATED", () => {
    // A step can change n, system, config, exchange and pauli in one move, so
    // it has to clear the union of what every individual action clears, not
    // just INVALIDATED.
    const out = tourReset({ ...URL_DEFAULTS, system: "he" }, systems);
    expect(out.classicalGhost).toBeNull();
    expect(out.classicalStatus).toBe("idle");
    expect(out.hf).toBeNull();
    expect(out.hfStatus).toBe("idle");
    expect(out.forceLaw).toBeNull();
    expect(out.forceStatus).toBe("idle");
  });

  it("moves an atom with no GSZ parameters onto Hartree-Fock", () => {
    // Same guard setSystem applies. A step naming sulfur under the default
    // model asks the server for a refusal on every request.
    const out = tourReset({ ...URL_DEFAULTS, system: "s", model: "gsz" }, systems);
    expect(out.model).toBe("hf");
  });

  it("leaves a resolvable model alone", () => {
    const out = tourReset({ ...URL_DEFAULTS, system: "h", model: "gsz" }, systems);
    expect(out.model).toBe("gsz");
  });
});
