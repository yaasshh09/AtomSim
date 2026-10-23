import { describe, expect, it } from "vitest";
import {
  DEFAULT_SUP,
  arrowLattice,
  currentBound,
  evolveCurrent,
  evolveField,
  planeGradients,
  beatPeriods,
  coefficients,
  sameStates,
  densityBound,
  evolveDensity,
  formatFs,
  parseSup,
  serializeSup,
  supProblem,
} from "./evolve";

/** Two fake real "states" on a 3-pixel grid, with hydrogen's 1s and 2p energies. */
const PSI = [new Float32Array([1, 0.5, 0]), new Float32Array([0, 0.5, 1])];
const E = [-0.5, -0.125];
const HALF = { re: Math.SQRT1_2, im: 0 };
const PERIOD = (2 * Math.PI) / (E[1] - E[0]);

function at(t: number, coeffs = [HALF, HALF]) {
  return evolveDensity(PSI, coeffs, E, t, new Float32Array(3));
}

describe("evolveDensity", () => {
  it("is the coherent sum at t = 0", () => {
    // pixel 1: (0.5/sqrt2 + 0.5/sqrt2)^2 = 0.5
    expect(at(0)[1]).toBeCloseTo(0.5, 6);
  });

  it("destructively interferes half a Bohr period later", () => {
    expect(at(PERIOD / 2)[1]).toBeCloseTo(0, 6);
  });

  it("comes back after one period", () => {
    const a = at(0);
    const b = at(PERIOD);
    for (let i = 0; i < 3; i++) expect(b[i]).toBeCloseTo(a[i], 5);
  });

  it("does not move where only one state lives", () => {
    expect(at(0)[0]).toBeCloseTo(0.5, 6);
    expect(at(1.234)[0]).toBeCloseTo(0.5, 6);
  });

  it("does not move at all when the energies agree", () => {
    const out = evolveDensity(PSI, [HALF, HALF], [-0.125, -0.125], 3.3, new Float32Array(3));
    expect(out[1]).toBeCloseTo(0.5, 6);
  });

  it("takes the phase of c_k: a pi phase starts the mix out of step", () => {
    expect(at(0, [HALF, { re: -Math.SQRT1_2, im: 0 }])[1]).toBeCloseTo(0, 6);
    // and pi/2 sits halfway
    expect(at(0, [HALF, { re: 0, im: Math.SQRT1_2 }])[1]).toBeCloseTo(0.25, 6);
  });

  it("never exceeds the fixed colour bound", () => {
    const bound = densityBound(PSI, [HALF, HALF]);
    for (let s = 0; s < 50; s++) {
      for (const v of at((s / 50) * PERIOD)) expect(v).toBeLessThanOrEqual(bound + 1e-6);
    }
  });
});

describe("the sup URL form", () => {
  it("round-trips, negative m and fractional values included", () => {
    const mix = [
      { n: 2, l: 1, m: -1, weight: 0.25, phasePi: 1.5 },
      { n: 3, l: 2, m: 2, weight: 0.75, phasePi: 0 },
    ];
    const raws = serializeSup(mix);
    const q = new URLSearchParams();
    for (const r of raws) q.append("sup", r);
    expect(q.toString()).toBe("sup=2_1_-1_0.25_1.5&sup=3_2_2_0.75_0");
    expect(parseSup(q.getAll("sup"))).toEqual(mix);
  });

  it("drops anything the server would refuse", () => {
    for (const bad of [
      [],
      [""],
      ["1_0_0_1"],
      ["1_0_0_x_0"],
      ["1_1_0_1_0"],
      ["1_0_0_0_0"],
      ["1_0_0_-1_0", "2_0_0_1_0"],
      ["1_0_0_1_0", "1_0_0_1_0"],
      ["7_0_0_1_0"],
      ["1_0_0_1_0", "2_0_0_1_0", "3_0_0_1_0", "4_0_0_1_0", "5_0_0_1_0"],
    ]) {
      expect(parseSup(bad)).toBeUndefined();
    }
  });

  it("accepts the default", () => {
    expect(supProblem(DEFAULT_SUP)).toBeNull();
    expect(parseSup(serializeSup(DEFAULT_SUP))).toEqual(DEFAULT_SUP);
  });
});

describe("formatFs", () => {
  it("reads the 1s-2p beat in femtoseconds and long ones in picoseconds", () => {
    expect(formatFs(0.4053)).toBe("0.405 fs");
    expect(formatFs(2468)).toBe("2.47 ps");
  });
});

describe("live coefficients and beats", () => {
  it("normalizes populations and turns phases in units of pi", () => {
    const c = coefficients([
      { n: 1, l: 0, m: 0, weight: 1, phasePi: 0 },
      { n: 2, l: 1, m: 0, weight: 3, phasePi: 0.5 },
    ]);
    expect(c[0].re ** 2 + c[0].im ** 2).toBeCloseTo(0.25, 12);
    expect(c[1].re).toBeCloseTo(0, 12);
    expect(c[1].im).toBeCloseTo(Math.sqrt(0.75), 12);
  });

  it("finds the 1s-2p beat, skips empty states, and calls a degenerate mix stationary", () => {
    const mix = [
      { n: 1, l: 0, m: 0, weight: 1, phasePi: 0 },
      { n: 2, l: 1, m: 0, weight: 1, phasePi: 0 },
      { n: 3, l: 0, m: 0, weight: 0, phasePi: 0 },
    ];
    const [t] = beatPeriods([-0.5, -0.125, -1 / 18], mix);
    expect(beatPeriods([-0.5, -0.125, -1 / 18], mix)).toHaveLength(1);
    expect(t).toBeCloseTo((2 * Math.PI) / 0.375, 9);
    expect(beatPeriods([-0.125, -0.125], mix.slice(0, 2))).toEqual([]);
  });

  it("only refetches when the states change, not their weights", () => {
    const a = [{ n: 2, l: 1, m: 1, weight: 1, phasePi: 0 }];
    expect(sameStates(a, [{ ...a[0], weight: 0.2, phasePi: 1 }])).toBe(true);
    expect(sameStates(a, [{ ...a[0], m: -1 }])).toBe(false);
  });
});

describe("the in-plane current", () => {
  // f(x, z) = x^2 + 3z on a 5x5 grid of step 0.5: central differences are exact
  // inside for a quadratic, and the edges are one-sided as in np.gradient.
  const res = 5;
  const h = 0.5;
  const ax = (k: number) => -1 + h * k;
  const f = new Float32Array(res * res);
  for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) f[i * res + j] = ax(j) ** 2 + 3 * ax(i);

  it("differences like np.gradient", () => {
    const { dx, dz } = planeGradients([f], res, h);
    expect(dx[0][2 * res + 1]).toBeCloseTo(2 * ax(1), 6); // interior: exact
    expect(dz[0][2 * res + 3]).toBeCloseTo(3, 6);
    expect(dx[0][2 * res]).toBeCloseTo((ax(1) ** 2 - ax(0) ** 2) / h, 6); // edge: one-sided
  });

  it("is exactly zero for one stationary state", () => {
    const { dx, dz } = planeGradients([f], res, h);
    const { jx, jz } = evolveCurrent([f], dx, dz, [{ re: 0.6, im: 0.8 }], [-0.5], 3.7, [6, 12]);
    expect([...jx, ...jz]).toEqual([0, 0, 0, 0]);
  });

  it("is Im(a1* a2)(psi1 grad psi2 - psi2 grad psi1) for two states", () => {
    const g = new Float32Array(res * res).fill(2); // constant psi_2: no gradient
    const { dx, dz } = planeGradients([f, g], res, h);
    const c = [{ re: Math.SQRT1_2, im: 0 }, { re: 0, im: Math.SQRT1_2 }]; // relative phase pi/2
    const idx = 2 * res + 3;
    const { jx, jz } = evolveCurrent([f, g], dx, dz, c, [0, 0], 0, [idx]);
    // w12 = Im(a1* a2) = 1/2, w21 = -1/2; only psi_1 has a gradient
    expect(jx[0]).toBeCloseTo(-0.5 * 2 * dx[0][idx], 6);
    expect(jz[0]).toBeCloseTo(-0.5 * 2 * 3, 6);
    expect(Math.hypot(jx[0], jz[0])).toBeLessThanOrEqual(currentBound([f, g], dx, dz, c, [idx]));
  });

  it("builds a field whose |Psi|^2 is the density", () => {
    const re = new Float32Array(3);
    const im = new Float32Array(3);
    const c = [HALF, { re: 0, im: Math.SQRT1_2 }];
    evolveField(PSI, c, E, 1.3, re, im);
    const rho = evolveDensity(PSI, c, E, 1.3, new Float32Array(3));
    for (let i = 0; i < 3; i++) expect(re[i] ** 2 + im[i] ** 2).toBeCloseTo(rho[i], 6);
  });

  it("centres the arrow lattice", () => {
    expect(arrowLattice(7, 3)).toEqual([0, 3, 6, 21, 24, 27, 42, 45, 48]);
    expect(arrowLattice(6, 2)).toEqual([0, 2, 4, 12, 14, 16, 24, 26, 28]);
  });
});
