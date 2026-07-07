import { scaleLinear, scaleLog } from "d3-scale";
import { useEffect } from "react";
import { seriesColor, seriesName } from "../lib/spectrum";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 680;
const LINES_H = 190;
const RES_H = 150;
const M = { left: 56, right: 16 };

export function SpectrumView() {
  const { system, fineStructure, spectrum, loadSpectrum } = useAppStore();
  useEffect(() => {
    void loadSpectrum();
  }, [system, fineStructure, loadSpectrum]);
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

  return (
    <div className="view-wrap">
      <div className="view-header">
        <span className="plot-title">
          Emission lines λ [nm]{" "}
          <Badge provenance={spectrum.lines[0].wavelength_nm.provenance} />
        </span>
        <span className="legend-inline">
          {nLowers.map((nl) => (
            <span key={nl} style={{ color: seriesColor(nl) }}>
              ▎{seriesName(nl)}
            </span>
          ))}
        </span>
      </div>
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
            y1={28} y2={LINES_H - 30}
            stroke={seriesColor(ln.n_lower)} strokeWidth={1.5} opacity={0.9}
          />
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
      <p className="caption">
        {spectrum.reference_citation
          ? `Reference: ${spectrum.reference_citation}`
          : "No vendored NIST reference for this system — computed lines only, honestly unchecked."}
      </p>
    </div>
  );
}
