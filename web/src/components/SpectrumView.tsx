import { scaleLinear, scaleLog } from "d3-scale";
import { useEffect, useState } from "react";
import type { SpectralLineInfo } from "../api/types";
import {
  SPECTRUM_EMISSIVITY_LIBERTY,
  SPECTRUM_INTENSITY_LIBERTY,
} from "../lib/liberties";
import { seriesColor, seriesName } from "../lib/spectrum";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 680;
const LINES_H = 190;
const RES_H = 150;
const M = { left: 56, right: 16 };

const TOP = 28;
const BOTTOM = LINES_H - 30;

/** What a bar's height is being driven by. */
export type BarQuantity = "rate" | "emissivity";

const PICK: Record<BarQuantity, (ln: SpectralLineInfo) => number | undefined> = {
  rate: (ln) => ln.einstein_a_s?.value,
  emissivity: (ln) => ln.emissivity?.value,
};

/**
 * Map a per-line quantity onto a drawable [0, 1]. A spans ~4 decades across a
 * hydrogen line list and an LTE emissivity spans far more, so a linear map
 * would leave everything but the top few lines invisible; the compression is
 * logarithmic and disclosed in the caption and the badge.
 */
export function intensityScale(
  lines: SpectralLineInfo[],
  quantity: BarQuantity = "rate",
) {
  const pick = PICK[quantity];
  const values = lines
    .map(pick)
    .filter((a): a is number => typeof a === "number" && a > 0);
  if (values.length === 0) return null;
  const lo = Math.log10(Math.min(...values));
  const hi = Math.log10(Math.max(...values));
  // A degenerate range (one line, or all equal) would divide by zero; draw those
  // at full strength rather than inventing a spread that is not there.
  const span = hi - lo;
  return {
    lo,
    hi,
    quantity,
    value: pick,
    // A line can be exactly 0 once the gas is fully ionized. Clamping to the
    // floor keeps it drawn and hoverable; dropping it would quietly shorten
    // the list the user is looking at.
    t: (a: number | undefined) =>
      typeof a !== "number" || a <= 0 || span <= 0
        ? span <= 0
          ? 1
          : 0
        : (Math.log10(a) - lo) / span,
  };
}

/**
 * The wavelength window the axis covers, and what it leaves out.
 *
 * A fine-structure line list puts within-n components (2p_3/2 -> 2s_1/2 and
 * friends, out at millimetres to metres) beside ordinary n -> n' optical
 * lines. On one log axis the microwave group stretches the range so far that
 * every optical line collapses into a sliver at the left.
 *
 * The split used here is structural, not a threshold: "within n" versus
 * "across n" is a property of the transition, so no arbitrary cutoff decides
 * what you see. Hidden lines stay in the data and the count is reported.
 */
export function wavelengthWindow(lines: SpectralLineInfo[], full: boolean) {
  const all = lines.map((ln) => ln.wavelength_nm.value);
  if (full || lines.length === 0) {
    return { lo: Math.min(...all), hi: Math.max(...all), hidden: 0, splittable: false };
  }
  const across = lines
    .filter((ln) => ln.n_upper !== ln.n_lower)
    .map((ln) => ln.wavelength_nm.value);
  // Nothing to split: every line is within-n, or none is. Show them all.
  if (across.length === 0 || across.length === lines.length) {
    return { lo: Math.min(...all), hi: Math.max(...all), hidden: 0, splittable: false };
  }
  const lo = Math.min(...across);
  const hi = Math.max(...across);
  return {
    lo,
    hi,
    hidden: lines.filter(
      (ln) => ln.wavelength_nm.value < lo || ln.wavelength_nm.value > hi,
    ).length,
    splittable: true,
  };
}

export function SpectrumView() {
  const {
    system, fineStructure, intensities, spectrum, loadSpectrum, setIntensities,
    thermal, temperatureK, logNe, setThermal, setTemperatureK, setLogNe,
  } = useAppStore();
  const [fullRange, setFullRange] = useState(false);
  useEffect(() => {
    void loadSpectrum();
  }, [system, fineStructure, intensities, thermal, temperatureK, logNe, loadSpectrum]);
  if (!spectrum) return <p className="hint-block">loading spectrum…</p>;

  const window_ = wavelengthWindow(spectrum.lines, fullRange);
  const shown = spectrum.lines.filter(
    (ln) =>
      ln.wavelength_nm.value >= window_.lo && ln.wavelength_nm.value <= window_.hi,
  );
  const x = scaleLog([window_.lo * 0.9, window_.hi * 1.1], [M.left, W - M.right]);
  const nLowers = [...new Set(shown.map((ln) => ln.n_lower))].sort((a, b) => a - b);
  const tol = spectrum.tolerance_relative;
  const comp = spectrum.comparison;
  const yRes = tol ? scaleLinear([-3 * tol, 3 * tol], [RES_H - 30, 14]) : null;
  const clampY = (v: number) => Math.min(Math.max(v, 14), RES_H - 30);

  // Scale over the lines actually drawn. Letting a hidden microwave component
  // set the floor would squash every visible bar to describe something the
  // user cannot see; the caption says which range the scale covers.
  const isThermal = spectrum.thermal !== null;
  const strength = intensities
    ? intensityScale(shown, isThermal ? "emissivity" : "rate")
    : null;
  // Shortest bar still reaches 18% of the panel: a weak line must stay visible
  // and clickable, and hiding it would be its own kind of lie.
  const barTop = (ln: SpectralLineInfo) =>
    strength ? BOTTOM - (0.18 + 0.82 * strength.t(strength.value(ln))) * (BOTTOM - TOP)
             : TOP;
  const barOpacity = (ln: SpectralLineInfo) =>
    strength ? 0.3 + 0.7 * strength.t(strength.value(ln)) : 0.9;
  const ionized = spectrum.thermal?.ionized_fraction.value ?? 0;

  return (
    <div className="view-wrap">
      <div className="view-header">
        <span className="plot-title">
          Emission lines λ [nm]{" "}
          <Badge provenance={spectrum.lines[0].wavelength_nm.provenance} />
          {strength && (
            <>
              {" "}
              <Badge
                provenance={
                  isThermal ? SPECTRUM_EMISSIVITY_LIBERTY : SPECTRUM_INTENSITY_LIBERTY
                }
              />
            </>
          )}
          {spectrum.thermal && (
            <>
              {" "}
              <Badge provenance={spectrum.thermal.ionized_fraction.provenance} />
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
        scale bars by line strength{intensities && !isThermal ? " (Einstein A)" : ""}
      </label>
      {intensities && (
        <label className="check">
          <input
            type="checkbox"
            checked={thermal}
            onChange={(e) => setThermal(e.target.checked)}
          />
          weight by LTE populations (Boltzmann + Saha)
        </label>
      )}
      {intensities && thermal && (
        <>
          <label className="levels-field">
            T{" "}
            <input
              type="range" min={2} max={6} step={0.02}
              value={Math.log10(temperatureK)}
              onChange={(e) => setTemperatureK(10 ** Number(e.target.value))}
            />
            {temperatureK >= 1e4
              ? ` ${(temperatureK / 1e3).toFixed(1)}k K`
              : ` ${temperatureK.toFixed(0)} K`}
          </label>
          <label className="levels-field">
            n_e{" "}
            <input
              type="range" min={4} max={22} step={0.1} value={logNe}
              onChange={(e) => setLogNe(Number(e.target.value))}
            />
            {` 10^${logNe.toFixed(1)} cm⁻³`}
            {logNe <= 7 ? " (nebula)" : logNe >= 12 && logNe <= 14 ? " (photosphere)" : ""}
          </label>
          <p className="caption">
            Ionized fraction here: <strong>{(100 * ionized).toFixed(1)}%</strong>
            {ionized > 0.99
              ? " — almost no neutral atoms are left, so every line is faint no matter how hot it gets."
              : ionized < 0.01
                ? " — essentially all neutral, so brightness is set by excitation alone."
                : "."}
          </p>
        </>
      )}
      {window_.splittable && (
        <label className="check">
          <input
            type="checkbox"
            checked={fullRange}
            onChange={(e) => setFullRange(e.target.checked)}
          />
          show the full wavelength range, including within-n components
        </label>
      )}
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
        {shown.map((ln, i) => (
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
                  : "") +
                (ln.emissivity
                  ? `  ε=${ln.emissivity.value.toExponential(2)} eV/s per atom`
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
      {strength && !isThermal && (
        <p className="caption">
          Bar height and opacity ∝ log₁₀ A over{" "}
          {`10^${strength.lo.toFixed(1)} to 10^${strength.hi.toFixed(1)} s⁻¹`} — that is
          the spontaneous emission <em>rate</em>, not a predicted observed brightness. No
          level populations are modelled: turn on LTE weighting for those.
        </p>
      )}
      {strength && isThermal && spectrum.thermal && (
        <p className="caption">
          Bar height and opacity ∝ log₁₀ ε over{" "}
          {`10^${strength.lo.toFixed(1)} to 10^${strength.hi.toFixed(1)}`} eV/s per atom,
          at T = {spectrum.thermal.temperature_k.toFixed(0)} K and n_e ={" "}
          {spectrum.thermal.electron_density_cm3.toExponential(0)} cm⁻³. That is an LTE
          emissivity: level populations from Boltzmann, ionization from Saha, and the gas
          taken to be <em>optically thin</em>. A real medium reabsorbs its own strong
          lines, which is why Lyman-α does not dominate an observed nebula the way it
          dominates this one.
        </p>
      )}
      {window_.hidden > 0 && (
        <p className="caption">
          Axis covers the across-n lines ({window_.lo.toFixed(1)}–
          {window_.hi < 1e6
            ? `${window_.hi.toFixed(0)} nm`
            : `${(window_.hi / 1e6).toFixed(1)} mm`}
          ). {window_.hidden} within-n fine-structure component
          {window_.hidden === 1 ? " is" : "s are"} outside it, out at millimetre to metre
          wavelengths. They are still in the data and still in the engine's line list —
          tick the box above to include them, which stretches the axis far enough that
          the optical lines collapse into a sliver.
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
