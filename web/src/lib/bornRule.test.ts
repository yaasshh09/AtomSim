import { describe, expect, it } from "vitest";
import { exactCdf, histogram, ksDistance, radii, typicalKs } from "./bornRule";

// P(r) for hydrogen 1s: 4 r^2 e^{-2r}, on a fine grid out to where it is negligible.
const grid = Array.from({ length: 4001 }, (_, i) => (i * 20) / 4000);
const p1s = grid.map((r) => 4 * r * r * Math.exp(-2 * r));
const cdf = exactCdf(grid, p1s);

/** Inverse-CDF draws from the same P(r): what the sample job hands the demo. */
function draws(n: number, seed = 1): Float64Array {
  let s = seed;
  const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  return Float64Array.from({ length: n }, () => {
    const u = rand();
    let i = 0;
    while (cdf[i] < u) i++;
    return grid[i];
  });
}

describe("the Born-rule bookkeeping", () => {
  it("reads radii from interleaved positions", () => {
    const pos = new Float32Array([3, 4, 0, 0, 0, 2, 1, 1, 1]);
    expect(Array.from(radii(pos, 2))).toEqual([5, 2]);
  });

  it("builds the 1s CDF: F(1 bohr) = 1 - 5 e^-2", () => {
    const i = grid.indexOf(1);
    expect(cdf[i]).toBeCloseTo(1 - 5 * Math.exp(-2), 5);
  });

  it("honest draws converge at the 1/sqrt(N) rate, the wrong ones do not", () => {
    for (const n of [100, 2500]) {
      expect(ksDistance(draws(n), grid, cdf)).toBeLessThan(3 * typicalKs(n));
    }
    const wrong = draws(2500).map((r) => r * 1.3); // a distribution 30% too wide
    expect(ksDistance(wrong, grid, cdf)).toBeGreaterThan(10 * typicalKs(2500));
  });

  it("normalizes the histogram as a density, so it sits on P(r)", () => {
    const h = histogram(draws(20000), 10, 50);
    const area = h.reduce((s, v) => s + v * (10 / 50), 0);
    expect(area).toBeCloseTo(1, 2);
    expect(h[5]).toBeCloseTo(4 * 1.1 ** 2 * Math.exp(-2.2), 1); // bin centred at 1.1 bohr
  });
});
