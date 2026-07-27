import { scaleLinear } from "d3-scale";
import type { AbsorptionInfo, AbsorbingLineInfo } from "../api/types";
import { Badge } from "./Badge";
import { REGIME_COLOR, REGIME_LABEL } from "./CurveOfGrowthView";

const W = 680;
const H = 250;
const BAND_H = 34;
const M = { left: 62, right: 16, top: 18, bottom: 34 };

/**
 * Transmission as a path over a log wavelength axis.
 *
 * Log in wavelength because the lines this has to show at once span 97 nm to
 * 1876 nm, and linear in transmission because that axis is a fraction of the
 * continuum with a real zero and a real one. A log intensity axis, which the
 * emission view needs, would make every line look equally black.
 */
export function transmissionPath(
  wavelengthNm: number[],
  transmission: number[],
  x: (v: number) => number,
  y: (v: number) => number,
): string {
  const parts: string[] = [];
  for (let i = 0; i < wavelengthNm.length; i++) {
    if (!(wavelengthNm[i] > 0)) continue;
    const cmd = parts.length === 0 ? "M" : "L";
    parts.push(
      `${cmd}${x(Math.log10(wavelengthNm[i])).toFixed(2)} ${y(transmission[i]).toFixed(2)}`,
    );
  }
  return parts.join(" ");
}

/**
 * Grey level for a transmission, as a CSS colour.
 *
 * The band beneath the plot is the spectrum as an eye would see it: a bright
 * continuum with dark lines cut into it. Transmission maps straight to
 * lightness with no curve applied, so a line that looks half as bright is
 * letting half the light through. Any gamma here would be a visual liberty
 * with nothing to gain.
 */
export function transmissionGrey(t: number): string {
  const v = Math.round(255 * Math.min(1, Math.max(0, t)));
  return `rgb(${v} ${v} ${v})`;
}

/** Decade ticks over a log range, as exponents. */
function decades(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) out.push(e);
  return out;
}

/**
 * How much of the census the spectrum is losing, in words.
 *
 * The saturation number is the payload of this whole view, and a bare ratio
 * invites being read as a small correction. It is not: at 0.1 the gas holds
 * ten times what the lines appear to say.
 */
export function saturationVerdict(saturation: number): string {
  if (saturation > 0.97) {
    return "every line is optically thin, so the spectrum is a faithful census: doubling the gas would double every depth";
  }
  if (saturation > 0.5) {
    return "the strongest lines are starting to saturate, so the spectrum already understates how much gas there is";
  }
  return "the spectrum is badly saturated: the black cores cannot absorb any more, so most of the gas is invisible to these lines";
}

export function AbsorptionView({
  abs,
  zoomed = false,
}: {
  abs: AbsorptionInfo;
  zoomed?: boolean;
}) {
  const logLambda = abs.wavelength_nm.map((v) => Math.log10(v));
  const lo = Math.min(...logLambda);
  const hi = Math.max(...logLambda);
  const x = scaleLinear([lo, hi], [M.left, W - M.right]);
  const y = scaleLinear([0, 1], [H - M.bottom, M.top]);
  const path = transmissionPath(abs.wavelength_nm, abs.transmission, x, y);

  // One rectangle per grid point for the eye's-view band. The engine's grid is
  // adaptive, so the strips are narrow exactly where the lines are.
  const strips = abs.wavelength_nm.map((lam, i) => {
    const left = i === 0 ? logLambda[i] : (logLambda[i - 1] + logLambda[i]) / 2;
    const right =
      i === logLambda.length - 1
        ? logLambda[i]
        : (logLambda[i] + logLambda[i + 1]) / 2;
    return { lam, x0: x(left), x1: x(right), t: abs.transmission[i] };
  });

  const strongest = [...abs.lines].sort((a, b) => b.tau_centre - a.tau_centre);
  const present = [...new Set(abs.lines.map((d) => d.regime))];

  return (
    <>
      <div className="view-header">
        <span className="plot-title">
          Absorption — {abs.lines.length}{" "}
          {abs.lines.length === 1 ? "line" : "lines"} against a flat continuum{" "}
          <Badge provenance={abs.provenance} />
        </span>
        <span className="legend-inline">
          {present.map((r) => (
            <span key={r} style={{ color: REGIME_COLOR[r] }}>
              ▎{REGIME_LABEL[r]}
            </span>
          ))}
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H + BAND_H}`} role="img" className="levels-svg">
        <line
          x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom}
          className="axis"
        />
        <line x1={M.left} x2={M.left} y1={M.top} y2={H - M.bottom} className="axis" />
        {decades(lo, hi).map((e) => (
          <g key={e} transform={`translate(${x(e)},${H - M.bottom})`}>
            <line y2="5" className="axis" />
            <text y="17" textAnchor="middle" className="tick">
              {(10 ** e).toString()}
            </text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t} transform={`translate(${M.left},${y(t)})`}>
            <line x2="-5" className="axis" />
            <text x="-9" dy="3" textAnchor="end" className="tick">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {/* The unabsorbed continuum, so the depth of every line is read against
            the thing it is a fraction of rather than against the axis. */}
        <line
          x1={M.left} x2={W - M.right} y1={y(1)} y2={y(1)} className="zero"
        />
        <path d={path} className="transmission-curve" />

        {/* The same numbers as an eye would see them: a bright continuum with
            dark lines cut into it. */}
        {strips.map((s, i) => (
          <rect
            key={i}
            x={s.x0}
            y={H - 6}
            width={Math.max(0.5, s.x1 - s.x0)}
            height={BAND_H - 12}
            fill={transmissionGrey(s.t)}
          />
        ))}
        <text x={W - M.right} y={H - 4} textAnchor="end" className="tick">
          wavelength [nm, log]
        </text>
        <text x={4} y={12} className="tick">
          I / I₀
        </text>
      </svg>

      <p className="caption">
        <strong>
          Equivalent width {abs.equivalent_width_nm.toExponential(3)} nm
        </strong>{" "}
        against {abs.thin_limit_width_nm.toExponential(3)} nm if nothing
        saturated or overlapped, so the spectrum is showing{" "}
        <strong>{(100 * abs.saturation).toFixed(1)}%</strong> of what a naive
        sum predicts. Read plainly: {saturationVerdict(abs.saturation)}.
      </p>

      {!zoomed && (
        <p className="caption">
          Across the whole {abs.lines.length}-line range a single line is
          narrower than one pixel, so the plot above shows{" "}
          <em>where</em> the gas absorbs and not what any line looks like. The
          numbers below are the full-resolution answer either way; to see a
          line's shape, turn on line profiles and{" "}
          <strong>click a line</strong> to zoom both panels to it.
        </p>
      )}

      <p className="caption">
        One column density, {abs.column_density_m2.toExponential(2)} m⁻², is
        given for the element; each line absorbs with only the atoms in{" "}
        <em>its own</em> lower level. That is the whole reason the Lyman lines
        are black while the Balmer lines are invisible in the same gas, and it
        is the one thing an emission spectrum cannot show you.
      </p>

      <table className="line-table">
        <thead>
          <tr>
            <th>line</th>
            <th>λ [nm]</th>
            <th>f</th>
            <th>lower column [m⁻²]</th>
            <th>τ centre</th>
            <th>branch</th>
          </tr>
        </thead>
        <tbody>
          {strongest.slice(0, 12).map((d: AbsorbingLineInfo, i) => (
            <tr key={`${d.label}-${d.wavelength_nm}-${i}`}>
              <td>{d.label}</td>
              <td>{d.wavelength_nm.toFixed(3)}</td>
              <td>{d.oscillator_strength.toExponential(2)}</td>
              <td>{d.lower_column_m2.toExponential(2)}</td>
              <td>{d.tau_centre.toExponential(2)}</td>
              <td style={{ color: REGIME_COLOR[d.regime] }}>{d.regime}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {abs.lines.length > 12 && (
        <p className="caption">
          Showing the 12 deepest of {abs.lines.length} lines, ordered by optical
          depth at line centre.
        </p>
      )}

      {abs.blends.length > 0 && (
        <p className="caption">
          <strong>
            {abs.blends.length} blended{" "}
            {abs.blends.length === 1 ? "pair" : "pairs"}
          </strong>
          :{" "}
          {abs.blends.map((b) => `${b[0]}/${b[1]}`).join(", ")}. Where lines
          overlap their transmissions multiply rather than their absorptions
          adding, so two lines each removing 60% of the light remove 84%
          together and not 120%. The whole is less than the sum of the parts by
          construction, not by approximation.
        </p>
      )}

      <p className="caption">
        Grid closure {abs.flux_closure.toFixed(4)}: the summed optical depth
        integrates to this times the analytic total on the grid actually used,
        measured rather than assumed. Absorption only — the lines darken the
        continuum but never re-emit into it, so no core ever reverses.
      </p>
    </>
  );
}
