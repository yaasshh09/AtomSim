import { scaleLinear, scaleLog } from "d3-scale";
import { useEffect } from "react";
import type { SpectralLineInfo } from "../api/types";
import { SPECTRUM_INTENSITY_LIBERTY } from "../lib/liberties";
import { seriesColor, seriesName } from "../lib/spectrum";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 680;
const LINES_H = 190;
const RES_H = 150;
const M = { left: 56, right: 16 };

const TOP = 28;
const BOTTOM = LINES_H - 30;

/**
 * Map Einstein A onto a drawable [0, 1]. A spans ~4 decades across a hydrogen
 * line list, so a linear map would leave everything but the top few lines
 * invisible; the compression is logarithmic and disclosed in the caption and
 * the SPECTRUM_INTENSITY_LIBERTY badge.
 */
export function intensityScale(lines: SpectralLineInfo[]) {
  const rates = lines
    .map((ln) => ln.einstein_a_s?.value)
    .filter((a): a is number => typeof a === "number" && a > 0);
  if (rates.length === 0) return null;
  const lo = Math.log10(Math.min(...rates));
  const hi = Math.log10(Math.max(...rates));
  // A degenerate range (one line, or all equal) would divide by zero; draw those
  // at full strength rather than inventing a spread that is not there.
  const span = hi - lo;
  return {
    lo,
    hi,
    t: (a: number | undefined) =>
      typeof a !== "number" || a <= 0 || span <= 0 ? 1 : (Math.log10(a) - lo) / span,
  };
}

export function SpectrumView() {
  const { system, fineStructure, intensities, spectrum, loadSpectrum, setIntensities } =
    useAppStore();
  useEffect(() => {
    void loadSpectrum();
  }, [system, fineStructure, intensities, loadSpectrum]);
  if (!spectrum) return <p className="hint-block">loading spectrum…</p>;

  const wls = spectrum.lines.map((ln) => ln.wavelength_nm.value);
  const x = scaleLog(
    [Math.min(...wls) * 0.9, Math.max(...wls) * 1.1],
    [M.left, W - M.right],
  );
  const nLowers = [...new Set(spectrum.lines.map((ln) => ln.n_lower))].sort(
    (a, b) => a - b,
  );
  const tol = spectrum.tolerance_relative;
  const comp = spectrum.comparison;
  const yRes = tol ? scaleLinear([-3 * tol, 3 * tol], [RES_H - 30, 14]) : null;
  const clampY = (v: number) => Math.min(Math.max(v, 14), RES_H - 30);

  const strength = intensities ? intensityScale(spectrum.lines) : null;
  // Shortest bar still reaches 18% of the panel: a weak line must stay visible
  // and clickable, and hiding it would be its own kind of lie.
  const barTop = (ln: SpectralLineInfo) =>
    strength ? BOTTOM - (0.18 + 0.82 * strength.t(ln.einstein_a_s?.value)) * (BOTTOM - TOP)
             : TOP;
  const barOpacity = (ln: SpectralLineInfo) =>
    strength ? 0.3 + 0.7 * strength.t(ln.einstein_a_s?.value) : 0.9;

  return (
    <div className="view-wrap">
      <div className="view-header">
        <span className="plot-title">
          Emission lines λ [nm]{" "}
          <Badge provenance={spectrum.lines[0].wavelength_nm.provenance} />
          {strength && (
            <>
              {" "}
              <Badge provenance={SPECTRUM_INTENSITY_LIBERTY} />
            </>
          )}
        </span>
        <span className="legend-inline">
          {nLowers.map((nl) => (
            <span key={nl} style={{ color: seriesColor(nl) }}>
              ▎{seriesName(nl)}
            </span>
          ))}
        </span>
      </div>
      <label className="check">
        <input
          type="checkbox"
          checked={intensities}
          onChange={(e) => setIntensities(e.target.checked)}
        />
        scale bars by line strength (Einstein A)
      </label>
      <svg viewBox={`0 0 ${W} ${LINES_H}`} role="img" className="levels-svg">
        <line
          x1={M.left} x2={W - M.right} y1={LINES_H - 24} y2={LINES_H - 24}
          className="axis"
        />
        {x.ticks(8).map((t) => (
          <g key={t} transform={`translate(${x(t)},${LINES_H - 24})`}>
            <line y2="5" className="axis" />
            <text y="17" textAnchor="middle" className="tick">
              {t}
            </text>
          </g>
        ))}
        {spectrum.lines.map((ln, i) => (
          <line
            key={i}
            x1={x(ln.wavelength_nm.value)} x2={x(ln.wavelength_nm.value)}
            y1={barTop(ln)} y2={BOTTOM}
            stroke={seriesColor(ln.n_lower)} strokeWidth={1.5} opacity={barOpacity(ln)}
          >
            <title>
              {`${ln.n_upper}→${ln.n_lower}  λ=${ln.wavelength_nm.value.toFixed(2)} nm` +
                (ln.einstein_a_s
                  ? `  A=${ln.einstein_a_s.value.toExponential(2)} s⁻¹` +
                    (ln.oscillator_strength
                      ? `  f=${ln.oscillator_strength.value.toExponential(2)}`
                      : "")
                  : "")}
            </title>
          </line>
        ))}
        {comp?.map((c, i) => (
          <circle
            key={i} cx={x(c.reference_nm)} cy={LINES_H - 27} r={2.5}
            className={c.within_tolerance ? "ref-ok" : "ref-bad"}
          />
        ))}
        <text x={W - M.right} y={16} textAnchor="end" className="tick">
          computed lines (bars) · NIST reference (dots on axis; log-λ)
        </text>
      </svg>
      {comp && yRes && tol && (
        <svg viewBox={`0 0 ${W} ${RES_H}`} role="img" className="levels-svg">
          <rect
            x={M.left} width={W - M.left - M.right}
            y={yRes(tol)} height={yRes(-tol) - yRes(tol)} className="tol-band"
          />
          <line x1={M.left} x2={W - M.right} y1={yRes(0)} y2={yRes(0)} className="zero" />
          {comp.map((c, i) => (
            <circle
              key={i} cx={x(c.reference_nm)} cy={clampY(yRes(c.relative_error))} r={3}
              className={c.within_tolerance ? "ref-ok" : "ref-bad"}
            />
          ))}
          <text x={M.left} y={12} className="tick">
            (λ_computed − λ_NIST)/λ_NIST — shaded band = stated tolerance ±{tol.toExponential(0)}
          </text>
        </svg>
      )}
      {strength && (
        <p className="caption">
          Bar height and opacity ∝ log₁₀ A over{" "}
          {`10^${strength.lo.toFixed(1)} to 10^${strength.hi.toFixed(1)} s⁻¹`} — that is
          the spontaneous emission <em>rate</em>, not a predicted observed brightness. No
          level populations (temperature, density, optical depth) are modelled here.
        </p>
      )}
      {spectrum.intensity_note && (
        <p className="caption">{spectrum.intensity_note}</p>
      )}
      <p className="caption">
        {spectrum.reference_citation
          ? `Reference: ${spectrum.reference_citation}`
          : "No vendored NIST reference for this system — computed lines only, honestly unchecked."}
      </p>
    </div>
  );
}
