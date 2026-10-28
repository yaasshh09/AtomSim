import { describe, expect, it } from "vitest";
import { evolveDensity, morphBound, morphCoefficients } from "./evolve";

const a = new Float32Array([0.5, -0.2, 0.1, 0.0]);
const b = new Float32Array([0.1, 0.4, -0.3, 0.2]);

describe("the morph between two states", () => {
  it("is a normalized state at every alpha", () => {
    for (const alpha of [0, 0.3, Math.PI / 4, 1.2, Math.PI / 2]) {
      const c = morphCoefficients(alpha, -0.5);
      expect(c.reduce((s, x) => s + x.re ** 2 + x.im ** 2, 0)).toBeCloseTo(1, 12);
    }
  });

  it("starts on the first state and ends on the second", () => {
    const out = new Float32Array(4);
    evolveDensity([a, b], morphCoefficients(0, 0), [0, 0], 0, out);
    a.forEach((v, i) => expect(out[i]).toBeCloseTo(v * v, 7));
    evolveDensity([a, b], morphCoefficients(Math.PI / 2, 0), [0, 0], 0, out);
    b.forEach((v, i) => expect(out[i]).toBeCloseTo(v * v, 7));
  });

  it("crossfades the densities on the pi-pulse path and interferes in phase", () => {
    const alpha = 0.6;
    const out = new Float32Array(4);
    evolveDensity([a, b], morphCoefficients(alpha, -0.5), [0, 0], 0, out);
    const c2 = Math.cos(alpha) ** 2;
    a.forEach((v, i) => expect(out[i]).toBeCloseTo(c2 * v * v + (1 - c2) * b[i] ** 2, 7));
    evolveDensity([a, b], morphCoefficients(alpha, 0), [0, 0], 0, out);
    const mixed = (Math.cos(alpha) * a[1] + Math.sin(alpha) * b[1]) ** 2;
    expect(out[1]).toBeCloseTo(mixed, 7);
  });

  it("bounds every frame of the path", () => {
    const bound = morphBound([a, b]);
    const out = new Float32Array(4);
    for (let k = 0; k <= 30; k++) {
      evolveDensity([a, b], morphCoefficients((k * Math.PI) / 60, 0), [0, 0], 0, out);
      expect(Math.max(...out)).toBeLessThanOrEqual(bound * (1 + 1e-6));
    }
  });
});
