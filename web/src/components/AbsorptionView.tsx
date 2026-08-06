import { scaleLinear, scaleLog } from "d3-scale";
import type { AbsorptionInfo, AbsorbingLineInfo } from "../api/types";
import { formatOffset, offsetAxis, offsetTicks, thinTicks } from "../lib/axis";
import { Badge } from "./Badge";
import { REGIME_COLOR, REGIME_LABEL } from "./CurveOfGrowthView";

const W = 680;
const H = 250;
/* Room under the plot for the band and its label. The band used to start six
   units above this line, which put it directly under the axis title, muted
   text on a white continuum. It now clears the title completely. */
const BAND_H = 50;
const M = { left: 62, right: 16, top: 18, bottom: 34 };
const BAND_LABEL_Y = H + 14;
const BAND_Y = H + 20;
const BAND_BAR_H = 22;

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

/**
 * Whether the wavelength axis can carry absolute labels, or needs offsets.
 *
 * This panel is drawn over two very different windows. Unzoomed it spans the
 * whole line list (97 nm to 1876 nm), where "100" and "1000" are the natural
 * labels. Zoomed to one line it spans a few half-widths, 0.14 nm for
 * Lyman-alpha, and femtometres for a natural width, where every absolute label
 * rounds to the same string. Decade ticks failed at both ends: two labels
 * across the full range, and *none at all* on a zoom, because a window narrower
 * than a decade contains no whole power of ten.
 *
 * The threshold is a span of 5% of the centre, which is about where four
 * significant digits stop separating neighbouring ticks.
 */
export function absorptionAxisMode(loNm: number, hiNm: number): "log" | "offset" {
  const centre = (loNm + hiNm) / 2;
  if (!(centre > 0)) return "offset";
  return (hiNm - loNm) / centre >= 0.05 ? "log" : "offset";
}

/** One drawn column of the band under the axis. */
export interface BandColumn {
  /** Left edge, in viewBox units. */
  x: number;
  /** Deepest transmission of any sample in the column. This is what is drawn. */
  deepest: number;
  /**
   * Flux-weighted mean over the column: what a detector pixel spanning these
   * wavelengths would actually record. Reported as a number, not drawn.
   */
  mean: number;
}

/**
 * Bin the transmission grid onto whole drawable columns.
 *
 * The band used to draw one rectangle per grid point. The engine's grid is
 * adaptive, so on the full range that is ~6000 rectangles across 602 units of
 * axis, most of them a third of a unit wide. Adjacent sub-pixel rectangles do
 * not tile: each is composited with its own edge coverage, so a shared boundary
 * lands at about 75% of full white instead of 100%. The result was a grey
 * barcode across the whole band, including past 400 nm where the transmission
 * never drops below 0.98. The band was showing lines that are not in the data,
 * which is the one thing this project does not do.
 *
 * Two readings come out of each column and they are wildly different, so the
 * band has to say which one it draws. `mean` is the honest photograph: a
 * detector pixel spanning these wavelengths integrates the flux across them, so
 * a line far narrower than a column dilutes into it. Over the whole line list
 * that is the truth and it is also useless, every column comes back above
 * 0.98, and the strip renders blank white. `deepest` is the deepest sample in
 * the column, which keeps the lines locatable at the cost of overstating how
 * dark a pixel would look. The view draws `deepest`, labels it as such, and
 * prints the worst `mean` alongside so the photographic answer is still stated.
 */
export function bandColumns(
  logLambda: number[],
  transmission: number[],
  x: (logLambda: number) => number,
  x0: number,
  x1: number,
): BandColumn[] {
  const span = x1 - x0;
  if (!(span > 0) || logLambda.length === 0) return [];
  const n = Math.max(1, Math.round(span));
  const width = span / n;
  const sum = new Float64Array(n);
  const weight = new Float64Array(n);
  const deepest = new Float64Array(n).fill(Infinity);

  for (let i = 0; i < logLambda.length; i++) {
    // The strip this sample speaks for: half-way to each neighbour.
    const left = i === 0 ? logLambda[i] : (logLambda[i - 1] + logLambda[i]) / 2;
    const right =
      i === logLambda.length - 1
        ? logLambda[i]
        : (logLambda[i] + logLambda[i + 1]) / 2;
    const xa = Math.max(x0, Math.min(x(left), x(right)));
    const xb = Math.min(x1, Math.max(x(left), x(right)));
    if (!(xb > xa)) continue;
    const first = Math.min(n - 1, Math.max(0, Math.floor((xa - x0) / width)));
    const last = Math.min(n - 1, Math.max(0, Math.floor((xb - x0) / width)));
    for (let c = first; c <= last; c++) {
      const overlap =
        Math.min(xb, x0 + (c + 1) * width) - Math.max(xa, x0 + c * width);
      if (overlap <= 0) continue;
      sum[c] += transmission[i] * overlap;
      weight[c] += overlap;
      if (transmission[i] < deepest[c]) deepest[c] = transmission[i];
    }
  }

  // A column the grid does not reach carries the last value rather than a hole:
  // a gap would read as a black line, which is the opposite of no information.
  const out: BandColumn[] = [];
  let carriedDeep = transmission[0];
  let carriedMean = transmission[0];
  for (let c = 0; c < n; c++) {
    if (weight[c] > 0) {
      carriedMean = sum[c] / weight[c];
      carriedDeep = deepest[c];
    }
    out.push({ x: x0 + c * width, deepest: carriedDeep, mean: carriedMean });
  }
  return out;
}

/**
 * What the band's column width costs the lines, in words.
 *
 * The two readings of a column diverge only when a line is narrower than the
 * column. Zoomed onto one line the columns are far finer than the profile and
 * the two agree exactly; over the whole line list a column spans about a
 * nanometre while the line is a thousandth of that, and they disagree
 * completely. A caption that announced a gap in both cases would be describing
 * a picture that is not on the screen half the time, so it has to check.
 */
export function bandResolutionNote(drawn: number, mean: number): string {
  if (mean - drawn < 0.05) {
    return (
      "Here the columns are finer than the lines are wide, so a detector pixel"
      + " of the same width would record the same depth: the strip and the"
      + " curve agree."
    );
  }
  return (
    `A detector pixel that wide integrates the flux across it and would reach`
    + ` only ${(100 * mean).toFixed(1)}% of the continuum, against the`
    + ` ${(100 * drawn).toFixed(1)}% the curve reaches at full resolution. That`
    + ` gap is not a drawing error, it is why a low-resolution spectrum of this`
    + ` same gas looks almost blank, and it is exactly the dilution the`
    + ` equivalent width above is immune to.`
  );
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

  const loNm = 10 ** lo;
  const hiNm = 10 ** hi;
  const mode = absorptionAxisMode(loNm, hiNm);
  // Offsets are placed through the same log scale as everything else. Over a
  // window this narrow log is linear to well under a pixel, so the ticks come
  // out evenly spaced; over a wide one this branch is not taken.
  const offset = offsetAxis(loNm, hiNm);
  const xTicks =
    mode === "log"
      ? thinTicks(
          scaleLog([loNm, hiNm], [M.left, W - M.right]).ticks(8),
          (v) => x(Math.log10(v)),
          34,
        )
      : offsetTicks(offset, loNm, hiNm);
  const xLabel = (v: number) =>
    mode === "log" ? `${v}` : formatOffset(offset, v);

  const band = bandColumns(logLambda, abs.transmission, x, M.left, W - M.right);
  const bandWidth = band.length > 1 ? band[1].x - band[0].x : 1;
  // The two readings of the band: what is drawn, and what a real pixel of the
  // same width would record. They agree on a zoom and diverge on the full range.
  const deepestDrawn = band.reduce((m, c) => Math.min(m, c.deepest), 1);
  const deepestMean = band.reduce((m, c) => Math.min(m, c.mean), 1);

  const strongest = [...abs.lines].sort((a, b) => b.tau_centre - a.tau_centre);
  const present = [...new Set(abs.lines.map((d) => d.regime))];

  return (
    <>
      <div className="view-header">
        <span className="plot-title">
          Absorption: {abs.lines.length}{" "}
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
        {xTicks.map((v) => (
          <g key={v} transform={`translate(${x(Math.log10(v))},${H - M.bottom})`}>
            <line y2="5" className="axis" />
            <text y="17" textAnchor="middle" className="tick">
              {xLabel(v)}
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

        {/* Where the gas absorbs, as brightness. crispEdges because these tile,
            anti-aliased edges on abutting rectangles leave a seam at every
            boundary, and a seam here reads as a line. */}
        {band.map((c, i) => (
          <rect
            key={i}
            x={c.x}
            y={BAND_Y}
            width={bandWidth}
            height={BAND_BAR_H}
            fill={transmissionGrey(c.deepest)}
            shapeRendering="crispEdges"
          />
        ))}
        <text x={M.left} y={BAND_LABEL_Y} className="tick">
          deepest absorption in each column, as brightness
        </text>
        {mode === "log" ? (
          <text x={W - M.right} y={H - 4} textAnchor="end" className="tick">
            wavelength [nm, log]
          </text>
        ) : (
          /* Same treatment as the zoomed line profile beside it: name the
             centre once and label the ticks as offsets from it. */
          <text
            x={(M.left + W - M.right) / 2}
            y={H - 4}
            textAnchor="middle"
            className="tick"
          >
            λ − {offset.centreNm.toFixed(offset.centreDecimals)} nm [{offset.unit}]
          </text>
        )}
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

      <p className="caption">
        The strip is {band.length} columns wide, and each is drawn at the{" "}
        <em>deepest</em> transmission anywhere inside it, which is what makes a
        line narrower than a column findable at all.{" "}
        {bandResolutionNote(deepestDrawn, deepestMean)}
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
        measured rather than assumed. Absorption only, the lines darken the
        continuum but never re-emit into it, so no core ever reverses.
      </p>
    </>
  );
}
