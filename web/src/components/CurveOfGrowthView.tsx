import { scaleLinear } from "d3-scale";
import type { CurveOfGrowthInfo, GrowthRegime } from "../api/types";
import { Badge } from "./Badge";

const W = 680;
const H = 260;
const M = { left: 62, right: 16, top: 20, bottom: 34 };

/** One colour per branch, so the regimes read off the plot without a legend
 *  lookup. Deliberately the same amber the Stark warning uses for the branch
 *  where a line's strength stops measuring anything. */
export const REGIME_COLOR: Record<GrowthRegime, string> = {
  linear: "#4ade80",
  saturated: "#fbbf24",
  damping: "#7dd3fc",
};

export const REGIME_LABEL: Record<GrowthRegime, string> = {
  linear: "linear (slope 1)",
  saturated: "saturated (slope ≈ 0)",
  damping: "damping (slope ½)",
};

/**
 * Split the curve into runs of one regime each, so every segment can be drawn
 * in its own colour.
 *
 * Each run reaches back one point so consecutive runs share an endpoint;
 * without that the polyline shows a gap at every branch change, which reads as
 * missing data rather than as a transition.
 */
export function regimeSegments(
  regime: GrowthRegime[],
): { regime: GrowthRegime; start: number; end: number }[] {
  const out: { regime: GrowthRegime; start: number; end: number }[] = [];
  for (let i = 0; i < regime.length; i++) {
    const last = out[out.length - 1];
    if (last && last.regime === regime[i]) {
      last.end = i;
    } else {
      out.push({ regime: regime[i], start: Math.max(0, i - 1), end: i });
    }
  }
  return out;
}

/** Path through a log-log curve, over an index range. */
export function logLogPath(
  xs: number[],
  ys: number[],
  start: number,
  end: number,
  x: (v: number) => number,
  y: (v: number) => number,
): string {
  const parts: string[] = [];
  for (let i = start; i <= end && i < xs.length; i++) {
    if (!(xs[i] > 0) || !(ys[i] > 0)) continue;
    const cmd = parts.length === 0 ? "M" : "L";
    parts.push(`${cmd}${x(Math.log10(xs[i])).toFixed(2)} ${y(Math.log10(ys[i])).toFixed(2)}`);
  }
  return parts.join(" ");
}

/**
 * The log₁₀ range of the values a log axis can place, as [min, max].
 *
 * Zero and negative values are not points on a log axis, and neither is a
 * degenerate range: a constant array would collapse the scale onto one pixel,
 * so it is opened out by a decade either side instead.
 */
export function logDomain(values: number[]): [number, number] {
  const logs = values.filter((v) => v > 0 && Number.isFinite(v)).map(Math.log10);
  if (logs.length === 0) return [0, 1];
  const lo = Math.min(...logs);
  const hi = Math.max(...logs);
  return hi > lo ? [lo, hi] : [lo - 1, hi + 1];
}

/** Decade tick values spanning a log range, as exponents. */
export function decadeTicks(lo: number, hi: number, max = 8): number[] {
  const first = Math.ceil(lo);
  const last = Math.floor(hi);
  const all: number[] = [];
  for (let e = first; e <= last; e++) all.push(e);
  if (all.length <= max) return all;
  const step = Math.ceil(all.length / max);
  return all.filter((_, i) => i % step === 0);
}

export function CurveOfGrowthView({ cog }: { cog: CurveOfGrowthInfo }) {
  // Only the points a log axis can actually place. `logLogPath` already skips
  // the rest, so taking the domain from the raw arrays let a single zero
  // equivalent width stretch the axis down 300 decades while the curve stayed
  // in the top percent of the panel.
  const logN = logDomain(cog.column_density_m2);
  const logW = logDomain(cog.equivalent_width_nm);
  const x = scaleLinear(logN, [M.left, W - M.right]);
  const y = scaleLinear(logW, [H - M.bottom, M.top]);
  const segments = regimeSegments(cog.regime);
  const present = [...new Set(cog.regime)];

  return (
    // A fragment, not a nested .view-wrap: that class is a flex column with
    // its own overflow, and nesting one inside another collapses the child's
    // height so the plot draws as a 10:1 sliver.
    <>
      <div className="view-header">
        <span className="plot-title">
          Curve of growth: {cog.label} at {cog.wavelength_nm.toFixed(2)} nm{" "}
          <Badge provenance={cog.provenance} />
        </span>
        <span className="legend-inline">
          {present.map((r) => (
            <span key={r} style={{ color: REGIME_COLOR[r] }}>
              ▎{REGIME_LABEL[r]}
            </span>
          ))}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" className="levels-svg">
        <line
          x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom}
          className="axis"
        />
        <line
          x1={M.left} x2={M.left} y1={M.top} y2={H - M.bottom} className="axis"
        />
        {decadeTicks(x.domain()[0], x.domain()[1]).map((e) => (
          <g key={e} transform={`translate(${x(e)},${H - M.bottom})`}>
            <line y2="5" className="axis" />
            <text y="17" textAnchor="middle" className="tick">
              10<tspan className="exponent" dy="-3.5">{exponentLabel(e)}</tspan>
            </text>
          </g>
        ))}
        {decadeTicks(y.domain()[0], y.domain()[1], 6).map((e) => (
          <g key={e} transform={`translate(${M.left},${y(e)})`}>
            <line x2="-5" className="axis" />
            <text x="-9" dy="3" textAnchor="end" className="tick">
              10<tspan className="exponent" dy="-3.5">{exponentLabel(e)}</tspan>
            </text>
          </g>
        ))}
        {segments.map((seg, i) => (
          <path
            key={i}
            d={logLogPath(
              cog.column_density_m2, cog.equivalent_width_nm,
              seg.start, seg.end, x, y,
            )}
            fill="none"
            stroke={REGIME_COLOR[seg.regime]}
            strokeWidth={2}
          />
        ))}
        <text x={W - M.right} y={H - 4} textAnchor="end" className="tick">
          column density in the lower level [m⁻²]
        </text>
        <text x={4} y={12} className="tick">
          equivalent width [nm]
        </text>
      </svg>
      <p className="caption">
        How much light the line removes, against how much gas is in the way. The
        three branches are the point: while the core is transparent every atom
        absorbs as much as the last and the width tracks the column exactly
        (slope 1). Once the core goes black it cannot absorb more, so the line
        grows only through its Doppler shoulders and{" "}
        <strong>a hundred times more gas barely widens it</strong>, which is
        why a strong line is a poor measure of how much gas there is. Far
        enough along, the Lorentzian wings from the upper level's finite
        lifetime take over and growth resumes at slope ½.
      </p>
      <p className="caption">
        Every phase before this one assumed the gas was optically thin, which is
        only the first branch. f = {cog.oscillator_strength.toExponential(3)},
        Gaussian σ = {cog.sigma_nm.toExponential(2)} nm, Lorentzian γ ={" "}
        {cog.gamma_nm.toExponential(2)} nm, damping parameter a ={" "}
        {cog.damping_parameter.toExponential(2)}. The knees sit where τ at line
        centre reaches 1 and where a·τ reaches 1, so heating the gas widens the
        line and pushes the first knee to higher column, which is exactly how a
        real curve-of-growth fit measures a temperature.
      </p>
    </>
  );
}

/**
 * The exponent of a decade tick, for a raised `tspan` rather than Unicode
 * superscript characters.
 *
 * These labels used to be built from ⁰¹²³⁴⁵⁶⁷⁸⁹. Every number in a plot is set
 * in the mono, and the mono gives a superscript glyph the same advance as a
 * full-size digit, so "10²³" came out spaced like "10 ² ³" and the exponent
 * read as detached from its mantissa. A tspan carries its own size and
 * tracking, so the exponent sits tight against the ten and stays tabular.
 *
 * U+2212 for the sign, matching the minus the zoomed axes are named with, not
 * the hyphen a keyboard gives.
 */
export function exponentLabel(e: number): string {
  return (e < 0 ? "−" : "") + Math.abs(e).toString();
}
