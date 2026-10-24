import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { NodeSet, Quantity } from "../api/types";
import { nodeCaption } from "../lib/explain";
import { NodeMarks } from "./PlaneView";

const q = (value: number, label: string) => ({ value, label }) as unknown as Quantity;
const set = (o: Partial<NodeSet>): NodeSet =>
  ({ radial: [], cones: [], planes: [], z_axis: false, basis: "complex", ...o }) as NodeSet;
// 1 bohr = 10 viewBox units, origin at (100, 100), z up.
const x = (b: number) => 100 + 10 * b;
const z = (b: number) => 100 - 10 * b;

describe("the node overlay", () => {
  it("draws a radial node as a circle of its radius about the nucleus", () => {
    const svg = renderToStaticMarkup(
      <svg><NodeMarks nodes={set({ radial: [q(2, "r1")] })} x={x} z={z} reach={5} /></svg>,
    );
    expect(svg).toContain('<circle cx="100" cy="100" r="20"');
  });

  it("draws a cone as two rays at ±theta from +z, and 2p0's xy-plane cone flat", () => {
    const svg = renderToStaticMarkup(
      <svg><NodeMarks nodes={set({ cones: [q(Math.PI / 2, "c1")] })} x={x} z={z} reach={5} /></svg>,
    );
    expect(svg).toContain('x2="150" y2="100"');
    expect(svg).toContain('x2="50" y2="100"');
  });

  it("draws the z axis for m != 0 and for a real orbital's planes through z", () => {
    const axis = '<line x1="100" y1="150" x2="100" y2="50"';
    const m1 = renderToStaticMarkup(
      <svg><NodeMarks nodes={set({ z_axis: true })} x={x} z={z} reach={5} /></svg>,
    );
    expect(m1).toContain(axis);
    const px = renderToStaticMarkup(
      <svg><NodeMarks nodes={set({ planes: [q(Math.PI / 2, "p1")] })} x={x} z={z} reach={5} /></svg>,
    );
    expect(px).toContain(axis);
  });

  it("says when the slice itself is a nodal plane, and when there are no nodes", () => {
    expect(nodeCaption(set({ basis: "real", z_axis: true, planes: [q(0, "p1")] }))).toContain(
      "y = 0 itself",
    );
    expect(nodeCaption(set({}))).toContain("no nodes at all");
    expect(nodeCaption(set({ cones: [q(Math.acos(1 / Math.sqrt(3)), "c")] }))).toContain("54.7°");
  });
});
