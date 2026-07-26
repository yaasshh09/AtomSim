import { scaleLinear, scaleLog } from "d3-scale";
import { useEffect, useState } from "react";
import type { LineWidthInfo, ProfileInfo, SpectralLineInfo } from "../api/types";
import {
  PROFILE_DECADES,
  SPECTRUM_EMISSIVITY_LIBERTY,
  SPECTRUM_INTENSITY_LIBERTY,
  SPECTRUM_PROFILE_LIBERTY,
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
 * Map a synthesized curve onto [0, 1] for the full-range trace.
 *
 * Log-compressed for the same reason the bars are: an LTE emissivity spans
 * more decades than a panel has pixels, so a linear trace would be one spike
 * over a flat floor. Everything below `decades` under the peak is drawn at the
 * floor rather than at zero, so a faint line stays visible as a faint line.
 * The zoomed panel plots the same numbers linearly, which is where a profile's
 * actual shape lives.
 */
export function profileScale(intensity: number[], decades = PROFILE_DECADES) {
  let max = 0;
  for (const v of intensity) if (v > max) max = v;
  if (!(max > 0)) return null;
  const hi = Math.log10(max);
  const lo = hi - decades;
  return {
    lo,
    hi,
    max,
    decades,
    t: (v: number) =>
      v <= 0 ? 0 : Math.max(0, Math.min(1, (Math.log10(v) - lo) / decades)),
  };
}

/** SVG polyline through a curve, given axis mappings. */
export function profilePath(
  wavelength: number[],
  intensity: number[],
  x: (lambda: number) => number,
  y: (t: number) => number,
  t: (v: number) => number,
): string {
  const parts: string[] = [];
  for (let i = 0; i < wavelength.length; i++) {
    parts.push(`${i === 0 ? "M" : "L"}${x(wavelength[i]).toFixed(2)} ${y(t(intensity[i])).toFixed(2)}`);
  }
  return parts.join(" ");
}

/**
 * The window to synthesize a single line over when it is clicked.
 *
 * Wide enough that the wings are visibly wings (a Voigt is still 1e-3 of its
 * peak at 8 half-widths out) and narrow enough that the shape fills the panel.
 */
export function zoomWindow(
  wavelengthNm: number,
  fwhmNm: number,
  halfWidths = 8,
): [number, number] {
  // A zero-width line would collapse the window to a point and return nothing.
  const half = Math.max(fwhmNm * halfWidths, wavelengthNm * 1e-7);
  return [wavelengthNm - half, wavelengthNm + half];
}

/** The width entry nearest a wavelength, or null when the curve has none. */
export function widthAt(
  widths: LineWidthInfo[],
  wavelengthNm: number,
): LineWidthInfo | null {
  let best: LineWidthInfo | null = null;
  let bestGap = Infinity;
  for (const w of widths) {
    const gap = Math.abs(w.wavelength_nm - wavelengthNm);
    if (gap < bestGap) {
      bestGap = gap;
      best = w;
    }
  }
  return best;
}

/** Which mechanism dominates a line's width, for the caption to name. */
export function dominantTerm(w: LineWidthInfo): string {
  // Gaussian and Lorentzian are not comparable term by term, but the question
  // being answered is coarse: is the shape set by the gas or by the lifetime?
  const gaussFwhm = 2.3548 * w.sigma_nm;
  const lorentzFwhm = 2 * w.gamma_nm;
  if (gaussFwhm === 0 && lorentzFwhm === 0) return "nothing";
  if (gaussFwhm >= lorentzFwhm) {
    return w.terms.includes("instrumental") && !w.terms.includes("Doppler")
      ? "the spectrograph"
      : "thermal motion";
  }
  return "the upper level's lifetime";
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

const ZOOM_H = 210;

/**
 * One line, plotted linearly on both axes: the only place in the app where a
 * profile's actual shape is visible rather than implied. The full-range trace
 * has a log wavelength axis and a log intensity axis, so a line there is a
 * spike no matter what it really looks like.
 */
function ZoomPanel({
  prof,
  window_,
  onClear,
}: {
  prof: ProfileInfo;
  window_: [number, number];
  onClear: () => void;
}) {
  const w = prof.widths.length > 0 ? prof.widths[0] : null;
  const max = Math.max(...prof.intensity, 0);
  const x = scaleLinear(window_, [M.left, W - M.right]);
  const y = scaleLinear([0, max > 0 ? max : 1], [ZOOM_H - 30, 16]);
  const path = profilePath(
    prof.wavelength_nm, prof.intensity, x, (t) => t, (v) => y(v),
  );
  const half = w ? w.fwhm_nm / 2 : 0;
  return (
    <>
      <div className="view-header">
        <span className="plot-title">
          {w ? `${w.label} line profile` : "line profile"} — linear λ, linear
          intensity{" "}
          <Badge provenance={prof.provenance} />
        </span>
        <button className="link-button" onClick={onClear} type="button">
          back to full range
        </button>
      </div>
      <svg viewBox={`0 0 ${W} ${ZOOM_H}`} role="img" className="levels-svg">
        <line
          x1={M.left} x2={W - M.right} y1={ZOOM_H - 24} y2={ZOOM_H - 24}
          className="axis"
        />
        {x.ticks(6).map((t) => (
          <g key={t} transform={`translate(${x(t)},${ZOOM_H - 24})`}>
            <line y2="5" className="axis" />
            <text y="17" textAnchor="middle" className="tick">
              {t.toFixed(3)}
            </text>
          </g>
        ))}
        {w && max > 0 && (
          <>
            {/* FWHM drawn where it is defined: across the profile at half its
                peak. A number in a caption is not the same as seeing it. */}
            <line
              x1={x(w.wavelength_nm - half)} x2={x(w.wavelength_nm + half)}
              y1={y(max / 2)} y2={y(max / 2)} className="fwhm-bar"
            />
            <text
              x={x(w.wavelength_nm)} y={y(max / 2) - 6}
              textAnchor="middle" className="tick"
            >
              FWHM {w.fwhm_nm.toExponential(2)} nm
            </text>
          </>
        )}
        <path d={path} className="profile-curve" />
      </svg>
      {w && (
        <p className="caption">
          Width set mostly by <strong>{dominantTerm(w)}</strong>. Gaussian σ ={" "}
          {w.sigma_nm.toExponential(2)} nm ({w.terms.filter((t) => t !== "natural").join(" + ") || "none"}
          ), Lorentzian γ = {w.gamma_nm.toExponential(2)} nm (natural). They do not
          add: the shape is their convolution, a Voigt, with a Gaussian core and
          Lorentzian wings — which is why the far wings sit above where a
          Gaussian would put them.
        </p>
      )}
    </>
  );
}

export function SpectrumView() {
  const {
    system, fineStructure, intensities, spectrum, loadSpectrum, setIntensities,
    thermal, temperatureK, logNe, setThermal, setTemperatureK, setLogNe,
    profile, logResolvingPower, profileZoom,
    setProfile, setLogResolvingPower, setProfileZoom,
  } = useAppStore();
  const [fullRange, setFullRange] = useState(false);
  // Set when the user deliberately backs out of a zoom, so the auto-zoom below
  // does not immediately drag them back into it.
  const [keepFull, setKeepFull] = useState(false);
  useEffect(() => {
    void loadSpectrum();
  }, [
    system, fineStructure, intensities, thermal, temperatureK, logNe,
    profile, logResolvingPower, profileZoom, loadSpectrum,
  ]);
  const prof0 = spectrum?.profile ?? null;
  // Read off the already-subscribed `spectrum` rather than a second selector.
  // A selector returning `s.spectrum?.lines ?? []` mints a fresh array on every
  // call, so its identity never matches and the effect below re-runs forever.
  const lines0 = spectrum?.lines;
  // Turning profiles on with no window selected lands on the strongest line.
  //
  // Without this the toggle looks broken, and for a defensible reason: on a log
  // axis covering 90 to 8000 nm, a line at R = 1000 is 0.14 px wide. No
  // instrument setting makes a full-range trace show a line's shape, so the
  // shape only exists in the zoomed panel, and the view should open on it.
  useEffect(() => {
    if (
      !profile || profileZoom || keepFull || !prof0
      || prof0.widths.length === 0 || !lines0
    ) {
      return;
    }
    let best = prof0.widths[0];
    let bestWeight = -Infinity;
    for (const w of prof0.widths) {
      const ln = lines0.find(
        (l) => Math.abs(l.wavelength_nm.value - w.wavelength_nm) < 1e-9,
      );
      const weight = ln?.emissivity?.value ?? ln?.einstein_a_s?.value ?? 0;
      if (weight > bestWeight) {
        bestWeight = weight;
        best = w;
      }
    }
    setProfileZoom(zoomWindow(best.wavelength_nm, best.fwhm_nm));
  }, [profile, profileZoom, keepFull, prof0, lines0, setProfileZoom]);
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

  const prof = spectrum.profile;
  // A zoomed curve belongs in its own linear panel, not smeared across a log
  // axis covering hundreds of nm where it would be one pixel wide.
  const trace = prof && !profileZoom ? profileScale(prof.intensity) : null;
  const tracePath =
    prof && trace
      ? profilePath(
          prof.wavelength_nm, prof.intensity, x,
          (t) => BOTTOM - t * (BOTTOM - TOP), trace.t,
        )
      : null;
  const zoomLine = (ln: SpectralLineInfo) => {
    if (!prof) return;
    const w = widthAt(prof.widths, ln.wavelength_nm.value);
    if (!w) return;
    setKeepFull(false);
    setProfileZoom(zoomWindow(ln.wavelength_nm.value, w.fwhm_nm));
  };

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
      <label className="check">
        <input
          type="checkbox"
          checked={profile}
          onChange={(e) => {
            setProfile(e.target.checked);
            setKeepFull(false);
            if (!e.target.checked) setProfileZoom(null);
          }}
        />
        synthesize line profiles (Voigt: natural + Doppler)
      </label>
      {profile && (
        <label className="levels-field">
          R{" "}
          <input
            type="range" min={2} max={7} step={0.05}
            value={logResolvingPower ?? 2}
            disabled={logResolvingPower === null}
            onChange={(e) => setLogResolvingPower(Number(e.target.value))}
          />
          {logResolvingPower === null
            ? " no instrument"
            : ` ${(10 ** logResolvingPower).toExponential(1)} (λ/Δλ)`}
          <button
            className="link-button"
            type="button"
            onClick={() =>
              setLogResolvingPower(logResolvingPower === null ? 4 : null)
            }
          >
            {logResolvingPower === null ? "add a spectrograph" : "remove it"}
          </button>
        </label>
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
            className={prof ? "line-clickable" : undefined}
            onClick={prof ? () => zoomLine(ln) : undefined}
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
        {/* Over the bars, not under them: at this scale a line is far narrower
            than a pixel, so the curve lands on exactly the same columns as the
            bars and would otherwise be completely hidden by them. */}
        {tracePath && <path d={tracePath} className="profile-curve" />}
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
      {prof && profileZoom && (
        <ZoomPanel
          prof={prof}
          window_={profileZoom}
          onClear={() => {
            setKeepFull(true);
            setProfileZoom(null);
          }}
        />
      )}
      {spectrum.profile_note && (
        <p className="caption">
          No profile drawn: {spectrum.profile_note}
        </p>
      )}
      {prof && trace && (
        <p className="caption">
          Curve: engine-synthesized Voigt profiles summed onto an adaptive grid,
          drawn on log₁₀ intensity over {trace.decades} decades below the peak{" "}
          <Badge provenance={SPECTRUM_PROFILE_LIBERTY} />. It integrates to{" "}
          {prof.flux_closure.toFixed(4)}× the summed line strengths, which is the
          grid's own quadrature error, measured rather than assumed. On this log
          wavelength axis every line is a spike regardless of its real shape —{" "}
          <strong>click a line</strong> to plot it linearly and see the profile
          itself.
        </p>
      )}
      {prof?.stark_note && (
        <p className="caption warn-note">{prof.stark_note}</p>
      )}
      {prof && !prof.stark_note && prof.stark_span_nm && (
        <p className="caption">
          Collisional broadening is not in this curve. At this density its linear
          Stark span would be {prof.stark_span_nm.value.toExponential(2)} nm,
          comfortably under the widths modelled here, so the shape stands.
        </p>
      )}
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
