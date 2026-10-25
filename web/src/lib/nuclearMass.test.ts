import { describe, expect, it } from "vitest";
import type { SystemInfo } from "../api/types";
import {
  landmarks,
  massKey,
  nuclearMass,
  onMassLine,
  parseMassKey,
  snapKey,
  withMassSystem,
} from "./nuclearMass";

function sys(key: string, mOverM: number): SystemInfo {
  return {
    key, name: key, z: 1, m_over_m_nucleus: mOverM, description: "",
    mu_ratio: {
      value: 0, unit: "m_e", label: "",
      provenance: { fidelity: "exact", method: "", assumptions: [], error_estimate: null, refinement: null },
    },
    nuclear_radius: null, nuclear_radius_fm: null, kind: "hydrogenic", n_electrons: null,
    has_gsz: true,
  };
}

const TABLE = [
  sys("h", 1 / 1836.15), sys("d", 1 / 3670.48), sys("t", 1 / 5496.92),
  sys("mu-h", 206.768 / 1836.15), sys("ps", 1), sys("he+", 1 / 7294.3),
];

describe("mass keys", () => {
  it("parses only plain-decimal m<M> keys", () => {
    expect(parseMassKey("m206.768")).toBe(206.768);
    expect(parseMassKey("m1")).toBe(1);
    expect(parseMassKey("mu-h")).toBeNull();
    expect(parseMassKey("m1e3")).toBeNull();
    expect(parseMassKey("h")).toBeNull();
  });

  it("writes four significant figures in plain decimals, clamped to range", () => {
    expect(massKey(1)).toBe("m1");
    expect(massKey(500.123)).toBe("m500.1");
    expect(massKey(10000)).toBe("m10000");
    expect(massKey(0.2)).toBe("m1");
    expect(massKey(1e6)).toBe("m10000");
    for (const M of [1, 3.7, 42.42, 999.9, 4321]) expect(parseMassKey(massKey(M))).not.toBeNull();
  });

  it("puts electron-orbited Z = 1 presets on the line and keeps the muon and He+ off it", () => {
    for (const k of ["ps", "h", "d", "t", "m42"]) expect(onMassLine(k)).toBe(true);
    for (const k of ["mu-h", "he+", "anti-h", "ne"]) expect(onMassLine(k)).toBe(false);
  });
});

describe("landmarks", () => {
  it("lists positronium, muonium, hydrogen, deuterium, tritium in mass order", () => {
    expect(landmarks(TABLE).map((l) => l.label)).toEqual([
      "positronium", "muonium", "hydrogen", "deuterium", "tritium",
    ]);
  });

  it("reads preset masses from the server's m/M, not a client constant", () => {
    expect(nuclearMass(TABLE, "h")).toBeCloseTo(1836.15, 6);
    expect(nuclearMass(TABLE, "ps")).toBe(1);
    expect(nuclearMass(TABLE, "m500")).toBe(500);
    expect(nuclearMass(TABLE, "mu-h")).toBeNull();
  });

  it("snaps a near miss onto the real preset, and leaves the space between alone", () => {
    const marks = landmarks(TABLE);
    expect(snapKey(1830, marks)).toBe("h");
    expect(snapKey(1.01, marks)).toBe("ps");
    expect(snapKey(207.5, marks)).toBe("m206.768");
    expect(snapKey(500, marks)).toBe("m500");
  });
});

describe("withMassSystem", () => {
  it("keeps at most one mass entry so the picker never grows a history", () => {
    const once = withMassSystem(TABLE, sys("m500", 1 / 500));
    const twice = withMassSystem(once, sys("m42", 1 / 42));
    expect(twice.map((s) => s.key)).toEqual([...TABLE.map((s) => s.key), "m42"]);
  });
});
