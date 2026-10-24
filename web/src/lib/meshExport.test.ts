import { describe, expect, it } from "vitest";
import { plyPoints, stlBinary } from "./meshExport";

describe("STL", () => {
  // One right triangle in the xy plane, wound counter-clockwise seen from +z.
  const v = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const t = new Uint32Array([0, 1, 2]);

  it("lays out header, count, normal, vertices and attribute bytes", () => {
    const buf = stlBinary(v, t);
    expect(buf.byteLength).toBe(84 + 50);
    const view = new DataView(buf);
    expect(new TextDecoder().decode(new Uint8Array(buf, 0, 31))).toBe("AtomSim isosurface, units: bohr");
    expect(view.getUint32(80, true)).toBe(1);
    expect([0, 4, 8].map((k) => view.getFloat32(84 + k, true))).toEqual([0, 0, 1]); // +z normal
    expect(view.getFloat32(84 + 12 + 12, true)).toBe(1); // second vertex x
    expect(view.getUint16(84 + 48, true)).toBe(0);
  });
});

describe("PLY", () => {
  it("writes a header a reader can trust and the floats after it", () => {
    const pos = new Float32Array([1, 2, 3, -4, 5.5, 6]);
    const buf = plyPoints(pos, new Float32Array([0.25, 0.5]));
    const bytes = new Uint8Array(buf);
    const text = new TextDecoder().decode(bytes);
    const end = text.indexOf("end_header\n") + "end_header\n".length;
    const header = text.slice(0, end);
    expect(header).toContain("element vertex 2");
    expect(header).toContain("property float density");
    expect(header).toContain("units: bohr");
    const view = new DataView(buf, end);
    expect(view.getFloat32(16 + 4, true)).toBe(5.5);
    expect(view.getFloat32(16 + 12, true)).toBe(0.5);
    expect(buf.byteLength).toBe(end + 2 * 16);
  });

  it("leaves the density out when there is none", () => {
    const text = new TextDecoder().decode(new Uint8Array(plyPoints(new Float32Array([0, 0, 0]))));
    expect(text).not.toContain("density");
  });
});
