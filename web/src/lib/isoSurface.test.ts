import { describe, expect, it } from "vitest";
import type { IsoMeta } from "../api/types";
import { phaseColor } from "./colormap";
import {
  buildSurfaceColors,
  componentsCaption,
  enclosedCaption,
  surfaceExtent,
} from "./isoSurface";

function meta(over: Partial<IsoMeta> = {}): IsoMeta {
  const prov = {
    fidelity: "numerical" as const,
    method: "test",
    assumptions: [],
    error_estimate: null,
    refinement: null,
  };
  const q = (value: number, unit: string) => ({
    value,
    unit,
    label: "test",
    provenance: prov,
  });
  return {
    kind: "isosurface",
    vertex_count: 3,
    triangle_count: 1,
    channels: [],
    target_fraction: 0.9,
    enclosed_fraction: q(0.9004, "1"),
    outside_fraction: 0.0996,
    level: q(1.2e-3, "bohr^-3"),
    escaped_fraction: q(3e-5, "1"),
    mesh_volume: q(78.7, "bohr^3"),
    voxel_volume: q(79.1, "bohr^3"),
    area: q(89.0, "bohr^2"),
    components: 1,
    half_width: 5.6,
    resolution: 96,
    axis_unit: "bohr",
    n: 1,
    l: 0,
    m: 0,
    basis: "complex",
    system: "h",
    model: "gsz",
    label: "test",
    provenance: prov,
    ...over,
  };
}

describe("buildSurfaceColors", () => {
  it("uses the same phase map as the point cloud", () => {
    // Not a copy of the expected RGB: the assertion is that the two agree, so
    // it still holds if the shared map is ever retuned.
    const colors = buildSurfaceColors(new Float32Array([0, Math.PI]));
    const [r, g, b] = phaseColor(0);
    expect(colors.slice(0, 3)).toEqual(new Float32Array([r / 255, g / 255, b / 255]));
    expect(colors).toHaveLength(6);
  });

  it("gives opposite lobes different colours", () => {
    const colors = buildSurfaceColors(new Float32Array([0, Math.PI]));
    expect(colors.slice(0, 3)).not.toEqual(colors.slice(3, 6));
  });
});

describe("enclosedCaption", () => {
  it("states the complement, which is the part textbooks leave out", () => {
    expect(enclosedCaption(meta())).toBe(
      "encloses 90.0% of the electron — it is outside this surface 10.0% of the time",
    );
  });

  it("reports what the grid delivered, not what was asked for", () => {
    // A 90% request that landed on 87% must say 87%, or the caption is the lie
    // the whole phase exists to prevent.
    const text = enclosedCaption(
      meta({
        target_fraction: 0.9,
        enclosed_fraction: { ...meta().enclosed_fraction, value: 0.8712 },
        outside_fraction: 0.1288,
      }),
    );
    expect(text).toContain("87.1%");
    expect(text).toContain("12.9%");
  });
});

describe("componentsCaption", () => {
  it("says the count plainly for an s state, which has no angular node", () => {
    expect(componentsCaption(meta())).toBe("1 piece");
  });

  it("qualifies the count wherever a node could be fusing lobes", () => {
    const text = componentsCaption(meta({ l: 1, components: 1 }));
    expect(text).toContain("1 piece");
    expect(text).toContain("one cell");
  });

  it("pluralizes", () => {
    expect(componentsCaption(meta({ l: 1, components: 2 }))).toContain("2 pieces");
  });
});

describe("surfaceExtent", () => {
  it("is the farthest vertex, not the box the mesh was cut from", () => {
    const v = new Float32Array([1, 0, 0, 0, 3, 4, 0, 0, 2]);
    expect(surfaceExtent(v)).toBe(5);
  });

  it("is zero for an empty mesh rather than NaN", () => {
    expect(surfaceExtent(new Float32Array(0))).toBe(0);
  });
});
