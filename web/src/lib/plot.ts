import type { ScaleLinear } from "d3-scale";

/** SVG path through (xs[i], ys[i]) under the given scales. */
export function linePath(
  xs: readonly number[],
  ys: readonly number[],
  x: ScaleLinear<number, number>,
  y: ScaleLinear<number, number>,
): string {
  return xs
    .map((v, i) => `${i === 0 ? "M" : "L"}${x(v).toFixed(2)},${y(ys[i]).toFixed(2)}`)
    .join("");
}
