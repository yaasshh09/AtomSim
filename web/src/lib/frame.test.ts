import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { PHYSICS_TO_SCREEN } from "./frame";

/** Apply the shared rotation to an engine-space vector. */
function toScreen(x: number, y: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(x, y, z).applyEuler(
    new THREE.Euler(...PHYSICS_TO_SCREEN),
  );
}

describe("PHYSICS_TO_SCREEN", () => {
  it("puts the quantization axis screen-vertical, as RENDER_LIBERTIES claims", () => {
    const v = toScreen(0, 0, 1);
    expect(v.y).toBeCloseTo(1, 12);
    expect(v.x).toBeCloseTo(0, 12);
    expect(v.z).toBeCloseTo(0, 12);
  });

  it("leaves engine x on screen x, so the frame is not spun about the view", () => {
    const v = toScreen(1, 0, 0);
    expect(v.x).toBeCloseTo(1, 12);
    expect(v.y).toBeCloseTo(0, 12);
    expect(v.z).toBeCloseTo(0, 12);
  });

  it("sends engine y into the screen, not up it", () => {
    const v = toScreen(0, 1, 0);
    expect(v.z).toBeCloseTo(-1, 12);
    expect(v.y).toBeCloseTo(0, 12);
  });

  it("is a rotation: lengths and handedness survive it", () => {
    const v = toScreen(3, -4, 12);
    expect(v.length()).toBeCloseTo(13, 12);
    // x cross y must still give z in the rotated frame.
    const cross = toScreen(1, 0, 0).cross(toScreen(0, 1, 0));
    expect(cross.distanceTo(toScreen(0, 0, 1))).toBeCloseTo(0, 12);
  });
});
