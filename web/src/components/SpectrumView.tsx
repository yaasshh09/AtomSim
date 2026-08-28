import { scaleLinear, scaleLog } from "d3-scale";
import { useEffect, useState } from "react";
import type { LineWidthInfo, ProfileInfo, SpectralLineInfo } from "../api/types";
import { formatOffset, offsetAxis, offsetTicks, thinTicks } from "../lib/axis";
import {
  describeDensity,
  describeResolvingPower,
  describeTemperature,
  nistSummary,
  VIEW_LEADS,
} from "../lib/explain";
import { withinPlot } from "../lib/hover";
import {
  PROFILE_DECADES,
  SPECTRUM_EMISSIVITY_LIBERTY,
  SPECTRUM_INTENSITY_LIBERTY,
  SPECTRUM_PROFILE_LIBERTY,
} from "../lib/liberties";
import { seriesColor, seriesName } from "../lib/spectrum";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { AbsorptionView } from "./AbsorptionView";
import { CurveOfGrowthView } from "./CurveOfGrowthView";
import { Disclosure } from "./Disclosure";
import { ControlGroup, Slider, Toggle } from "./Field";
import { HoverReadout, usePlotHover } from "./PlotHover";
import { ViewIntro } from "./ViewIntro";

const W = 680;
const LINES_H = 190;
const RES_H = 150;
const M = { left: 56, right: 16 };

const TOP = 28;
const BOTTOM = LINES_H - 30;

/** What I am driving a bar's height with. */
export type BarQuantity = "rate" | "emissivity";

const PICK: Record<BarQuantity, (ln: SpectralLineInfo) => number | undefined> = {
  rate: (ln) => ln.einstein_a_s?.value,
  emissivity: (ln) => ln.emissivity?.value,
};

/**
 * I map a per-line quantity onto a drawable [0, 1]. A spans ~4 decades across
 * a hydrogen line list and an LTE emissivity spans far more, so a linear map
 * would leave everything but the top few lines invisible. My compression is
 * logarithmic, and I disclose it in the caption and the badge.
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
  // A degenerate range (one line, or all equal) would have me dividing by
  // zero; I draw those at full strength rather than inventing a spread that is
  // not there.
  const span = hi - lo;
  return {
    lo,
    hi,
    quantity,
    value: pick,
    // A line can be exactly 0 once the gas is fully ionized. Clamping to the
    // floor keeps it drawn and hoverable; if I dropped it I would quietly
    // shorten the list you are looking at.
    t: (a: number | undefined) =>
      typeof a !== "number" || a <= 0 || span <= 0
        ? span <= 0
          ? 1
          : 0
        : (Math.log10(a) - lo) / span,
  };
}

/**
 * I map a synthesized curve onto [0, 1] for the full-range trace.
 *
 * I log-compress it for the same reason I compress the bars: an LTE emissivity
 * spans more decades than my panel has pixels, so a linear trace would be one
 * spike over a flat floor. I draw everything below `decades` under the peak at
 * the floor rather than at zero, so a faint line stays visible as a faint
 * line. My zoomed panel plots the same numbers linearly, which is where a
 * profile's actual shape lives.
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

/** The SVG polyline I trace through a curve, given axis mappings. */
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
 * The window I synthesize a single line over when you click it.
 *
 * Wide enough that the wings are visibly wings (a Voigt is still 1e-3 of its
 * peak at 8 half-widths out) and narrow enough that the shape fills my panel.
 */
export function zoomWindow(
  wavelengthNm: number,
  fwhmNm: number,
  halfWidths = 8,
): [number, number] {
  // A zero-width line would collapse my window to a point and return nothing.
  const half = Math.max(fwhmNm * halfWidths, wavelengthNm * 1e-7);
  return [wavelengthNm - half, wavelengthNm + half];
}

/** The width entry I find nearest a wavelength, or null when the curve has none. */
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

/** Which mechanism dominates a line's width, so I can name it in the caption. */
export function dominantTerm(w: LineWidthInfo): string {
  // Gaussian and Lorentzian are not comparable term by term, but the question
  // I am answering is coarse: is the shape set by the gas or by the lifetime?
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
 * The wavelength window my axis covers, and what I leave out of it.
 *
 * A fine-structure line list puts within-n components (2p_3/2 -> 2s_1/2 and
 * friends, out at millimetres to metres) beside ordinary n -> n' optical
 * lines. On one log axis the microwave group stretches my range so far that
 * every optical line collapses into a sliver at the left.
 *
 * My split here is structural, not a threshold: "within n" versus "across n"
 * is a property of the transition, so no arbitrary cutoff of mine decides what
 * you see. I keep the hidden lines in the data and report the count.
 */
export function wavelengthWindow(lines: SpectralLineInfo[], full: boolean) {
  const all = lines.map((ln) => ln.wavelength_nm.value);
  if (full || lines.length === 0) {
    return { lo: Math.min(...all), hi: Math.max(...all), hidden: 0, splittable: false };
  }
  const across = lines
    .filter((ln) => ln.n_upper !== ln.n_lower)
    .map((ln) => ln.wavelength_nm.value);
  // Nothing to split: every line is within-n, or none is. I show them all.
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
/* I sit the zoom's x axis high enough to leave a row for the tick labels and a
   row under them for the axis title, which names the wavelength I measure the
   offsets from. At my old baseline the title overprinted the ticks. */
const ZOOM_BASE = ZOOM_H - 40;

/**
 * One line, which I plot linearly on both axes: the only place I show you a
 * profile's actual shape rather than implying it. My full-range trace has a
 * log wavelength axis and a log intensity axis, so a line there is a spike no
 * matter what it really looks like.
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
  const y = scaleLinear([0, max > 0 ? max : 1], [ZOOM_BASE - 6, 16]);
  const path = profilePath(
    prof.wavelength_nm, prof.intensity, x, (t) => t, (v) => y(v),
  );
  const half = w ? w.fwhm_nm / 2 : 0;
  // I label offsets from the window centre, not absolute wavelengths. Across
  // eight half-widths of a natural line the absolute value is constant to
  // every decimal a label has room for, so my old axis printed "121.568" six
  // times.
  const axis = offsetAxis(window_[0], window_[1]);
  return (
    <div className="zoom-panel">
      <div className="view-header">
        <span className="plot-lead">
          {w ? `One line, up close: ${w.label}` : "One line, up close"}
        </span>
        <button className="link-button" onClick={onClear} type="button">
          ← back to the full range
        </button>
      </div>
      <p className="plot-blurb">
        I have both axes linear here, so this is the line's real shape.
        Everywhere else I draw it as a spike, because a line is far narrower
        than a pixel on an axis covering hundreds of nanometres.{" "}
        <Badge provenance={prof.provenance} />
      </p>
      <svg viewBox={`0 0 ${W} ${ZOOM_H}`} role="img" className="levels-svg">
        <line
          x1={M.left} x2={W - M.right} y1={ZOOM_BASE} y2={ZOOM_BASE}
          className="axis"
        />
        {offsetTicks(axis, window_[0], window_[1]).map((t) => (
          <g key={t} transform={`translate(${x(t)},${ZOOM_BASE})`}>
            <line y2="5" className="axis" />
            <text y="17" textAnchor="middle" className="tick">
              {formatOffset(axis, t)}
            </text>
          </g>
        ))}
        {/* My axis names what the offsets are offsets from. Without this the
            numbers are a scale with no origin. */}
        <text
          x={(M.left + W - M.right) / 2}
          y={ZOOM_H - 4}
          textAnchor="middle"
          className="tick"
        >
          λ − {axis.centreNm.toFixed(axis.centreDecimals)} nm [{axis.unit}]
        </text>
        {w && max > 0 && (
          <>
            {/* I draw the FWHM where it is defined: across the profile at half
                its peak. A number in a caption is not the same as seeing it. */}
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
        <>
          <p className="caption">
            Mostly <strong>{dominantTerm(w)}</strong> sets this width.
          </p>
          <Disclosure summary="My two widths, and why they do not add">
            <p className="caption">
              Gaussian σ = {w.sigma_nm.toExponential(2)} nm (
              {w.terms.filter((t) => t !== "natural").join(" + ") || "none"}
              ), Lorentzian γ = {w.gamma_nm.toExponential(2)} nm (natural). I do
              not add them: the shape is their convolution, a Voigt, with a
              Gaussian core and Lorentzian wings, which is why the far wings sit
              above where a Gaussian would put them.
            </p>
          </Disclosure>
        </>
      )}
    </div>
  );
}

export function SpectrumView() {
  const {
    system, fineStructure, intensities, spectrum, loadSpectrum, setIntensities,
    thermal, temperatureK, logNe, setThermal, setTemperatureK, setLogNe,
    profile, logResolvingPower, profileZoom,
    setProfile, setLogResolvingPower, setProfileZoom,
    showCurveOfGrowth, curveOfGrowth, setShowCurveOfGrowth, loadCurveOfGrowth,
    absorption, logColumn, absorptionData, setAbsorption, setLogColumn,
    loadAbsorption,
  } = useAppStore();
  const [fullRange, setFullRange] = useState(false);
  // I set this when you deliberately back out of a zoom, so my auto-zoom below
  // does not immediately drag you back into it.
  const [keepFull, setKeepFull] = useState(false);
  useEffect(() => {
    void loadSpectrum();
  }, [
    system, fineStructure, intensities, thermal, temperatureK, logNe,
    profile, logResolvingPower, profileZoom, loadSpectrum,
  ]);
  const prof0 = spectrum?.profile ?? null;
  // I read off the `spectrum` I already subscribe to rather than adding a
  // second selector. A selector returning `s.spectrum?.lines ?? []` mints a
  // fresh array on every call, so its identity never matches and my effect
  // below re-runs forever.
  const lines0 = spectrum?.lines;
  // When you turn profiles on with no window selected, I land on the strongest
  // line.
  //
  // Without that my toggle looks broken, and for a defensible reason: on a log
  // axis covering 90 to 8000 nm, a line at R = 1000 is 0.14 px wide. No
  // instrument setting lets my full-range trace show a line's shape, so the
  // shape only exists in my zoomed panel, and I should open on it.
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
  // The curve is per line and costs its own request, so I only fetch it once
  // asked and only for the line actually in the window.
  const zoomCentre = profileZoom ? (profileZoom[0] + profileZoom[1]) / 2 : null;
  useEffect(() => {
    if (!showCurveOfGrowth || zoomCentre === null) return;
    void loadCurveOfGrowth(zoomCentre);
  }, [showCurveOfGrowth, zoomCentre, temperatureK, logNe, logResolvingPower,
      loadCurveOfGrowth]);
  // Absorption is its own request against the whole line list, so I only fetch
  // it once asked. It needs populations, hence my thermal gate: which level an
  // atom is in *is* what its line absorbs with.
  useEffect(() => {
    if (!absorption || !thermal) return;
    void loadAbsorption();
  }, [absorption, thermal, logColumn, system, fineStructure, temperatureK,
      logNe, logResolvingPower, profileZoom, loadAbsorption]);

  const hover = usePlotHover(W);

  if (!spectrum) {
    return (
      <div className="view-wrap">
        <ViewIntro lead={VIEW_LEADS.spectrum} />
        <p className="hint-block">I am loading the spectrum…</p>
      </div>
    );
  }

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
  const nist = nistSummary(comp, tol);

  // I scale over the lines I actually draw. If I let a hidden microwave
  // component set the floor I would squash every visible bar to describe
  // something you cannot see; my caption says which range the scale covers.
  const isThermal = spectrum.thermal !== null;
  const strength = intensities
    ? intensityScale(shown, isThermal ? "emissivity" : "rate")
    : null;
  // My shortest bar still reaches 18% of the panel: a weak line has to stay
  // visible and clickable, and hiding it would be its own kind of lie.
  const barTop = (ln: SpectralLineInfo) =>
    strength ? BOTTOM - (0.18 + 0.82 * strength.t(strength.value(ln))) * (BOTTOM - TOP)
             : TOP;
  const barOpacity = (ln: SpectralLineInfo) =>
    strength ? 0.3 + 0.7 * strength.t(strength.value(ln)) : 0.9;
  const ionized = spectrum.thermal?.ionized_fraction.value ?? 0;

  const prof = spectrum.profile;
  // A zoomed curve belongs in its own linear panel, not smeared by me across a
  // log axis covering hundreds of nm where it would be one pixel wide.
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

  /* Which line the pointer is nearest, which I measure in pixels rather than
     in wavelength. My axis is logarithmic, so a fixed tolerance in nm is a
     different visual distance at each end of it, and you are aiming with your
     eyes. */
  let hoverLine: SpectralLineInfo | null = null;
  if (hover.x !== null && withinPlot(hover.x, M.left, W - M.right)) {
    let bestGap = 7;
    for (const ln of shown) {
      const gap = Math.abs(x(ln.wavelength_nm.value) - hover.x);
      if (gap < bestGap) {
        bestGap = gap;
        hoverLine = ln;
      }
    }
  }
  const hoverLines = hoverLine
    ? [
        `${seriesName(hoverLine.n_lower)}  ${hoverLine.n_upper}→${hoverLine.n_lower}`,
        `λ = ${hoverLine.wavelength_nm.value.toFixed(2)} nm`,
        ...(hoverLine.einstein_a_s
          ? [`A = ${hoverLine.einstein_a_s.value.toExponential(2)} s⁻¹`]
          : []),
        ...(hoverLine.oscillator_strength
          ? [`f = ${hoverLine.oscillator_strength.value.toExponential(2)}`]
          : []),
        ...(hoverLine.emissivity
          ? [`ε = ${hoverLine.emissivity.value.toExponential(2)} eV/s per atom`]
          : []),
        ...(prof ? ["click and I will draw its shape"] : []),
      ]
    : [];

  return (
    <div className="view-wrap">
      <ViewIntro
        lead={VIEW_LEADS.spectrum}
        badge={<Badge provenance={spectrum.lines[0].wavelength_nm.provenance} />}
      >
        <Disclosure summary="Where a spectral line comes from">
          <p className="caption">
            An electron dropping from one rung of the energy ladder to a lower
            one has to put the difference somewhere, and it emits a single
            photon carrying exactly that much energy. Energy fixes colour, so
            each pair of rungs gives one precise wavelength, and what I am
            showing you here is the ladder's shape, sideways.
          </p>
          <p className="caption">
            I colour the lines by where they land. Everything ending on n=1 is
            the Lyman series, in the ultraviolet; everything ending on n=2 is
            Balmer, which is the visible one, and Balmer-α at 656 nm is the red
            you see in every photograph of a nebula.
          </p>
        </Disclosure>
      </ViewIntro>

      <figure className="plot">
        <figcaption>
          <span className="plot-lead">Every line this atom can emit</span>
          <span className="plot-blurb">
            Each bar is one transition, which I place at its wavelength.
            {strength
              ? " I set the bar height from line strength, compressed logarithmically."
              : " I draw every bar the same height, because I am not modelling strength yet."}
            {prof ? " Click any bar and I will plot its shape." : ""}
          </span>
          <span className="plot-provenance">
            {strength && (
              <Badge
                provenance={
                  isThermal ? SPECTRUM_EMISSIVITY_LIBERTY : SPECTRUM_INTENSITY_LIBERTY
                }
              />
            )}
            {spectrum.thermal && (
              <Badge provenance={spectrum.thermal.ionized_fraction.provenance} />
            )}
            <span className="legend-inline">
              {nLowers.map((nl) => (
                <span key={nl} style={{ color: seriesColor(nl) }}>
                  ▎{seriesName(nl)}
                </span>
              ))}
            </span>
          </span>
        </figcaption>
        <svg
          viewBox={`0 0 ${W} ${LINES_H}`}
          role="img"
          className={`levels-svg${prof ? " plot-hoverable" : ""}`}
          ref={hover.ref}
          onPointerMove={hover.onPointerMove}
          onPointerLeave={hover.onPointerLeave}
        >
          <line
            x1={M.left} x2={W - M.right} y1={LINES_H - 24} y2={LINES_H - 24}
            className="axis"
          />
          {/* I thin these: my axis is logarithmic, so d3's tick set puts 5000
              and 6000 about four pixels apart and I printed the top two
              decades as one run of digits ("5006007008009001000"). I drop
              ticks by drawn position rather than by value, so the rule holds
              whatever range the line list spans. */}
          {thinTicks(x.ticks(8), x, 34).map((t) => (
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
              stroke={seriesColor(ln.n_lower)}
              strokeWidth={hoverLine === ln ? 3 : 1.5}
              opacity={barOpacity(ln)}
              className={prof ? "line-clickable" : undefined}
              onClick={prof ? () => zoomLine(ln) : undefined}
            />
          ))}
          {/* I draw this over the bars, not under them: at this scale a line is
              far narrower than a pixel, so my curve lands on exactly the same
              columns as the bars and would otherwise be hidden by them. */}
          {tracePath && <path d={tracePath} className="profile-curve" />}
          {comp?.map((c, i) => (
            <circle
              key={i} cx={x(c.reference_nm)} cy={LINES_H - 27} r={2.5}
              className={c.within_tolerance ? "ref-ok" : "ref-bad"}
            />
          ))}
          <text x={W - M.right} y={16} textAnchor="end" className="tick">
            bars: mine · dots on the axis: measured by NIST
          </text>
          {hoverLine && (
            <HoverReadout
              px={x(hoverLine.wavelength_nm.value)}
              py={barTop(hoverLine)}
              top={TOP}
              bottom={BOTTOM}
              lines={hoverLines}
              width={W}
              rightMargin={M.right}
            />
          )}
        </svg>
      </figure>

      {/* I call this view "vs NIST", so I put the answer to that comparison in
          a sentence near the top rather than a scatter you have to decode at
          the bottom. I keep the residual plot: the count says whether I
          passed, the plot says by how much and in which direction. */}
      {nist ? (
        <section className={`nist-panel${nist.allWithin ? " nist-ok" : " nist-off"}`}>
          <p className="nist-headline">
            <span className="nist-mark" aria-hidden="true">
              {nist.allWithin ? "✓" : "!"}
            </span>
            {nist.headline}
          </p>
          {yRes && tol && (
            <svg viewBox={`0 0 ${W} ${RES_H}`} role="img" className="levels-svg">
              <rect
                x={M.left} width={W - M.left - M.right}
                y={yRes(tol)} height={yRes(-tol) - yRes(tol)} className="tol-band"
              />
              <line x1={M.left} x2={W - M.right} y1={yRes(0)} y2={yRes(0)} className="zero" />
              {comp!.map((c, i) => (
                <circle
                  key={i} cx={x(c.reference_nm)} cy={clampY(yRes(c.relative_error))} r={3}
                  className={c.within_tolerance ? "ref-ok" : "ref-bad"}
                />
              ))}
              <text x={M.left} y={12} className="tick">
                (λ_mine − λ_NIST)/λ_NIST, shaded band = the tolerance I state, ±{tol.toExponential(0)}
              </text>
            </svg>
          )}
          <p className="caption">
            Each dot is one measured line. Inside the shaded band means my
            wavelength matches the measurement to within the tolerance I claim;
            above or below means it does not.
          </p>
          <p className="caption">
            {spectrum.reference_citation
              ? `Reference: ${spectrum.reference_citation}`
              : "I have no vendored NIST reference for this system."}
          </p>
        </section>
      ) : (
        <p className="hint-block">
          {spectrum.reference_citation
            ? `Reference: ${spectrum.reference_citation}`
            : "I have no vendored NIST reference for this system, so these are my lines only, honestly unchecked."}
        </p>
      )}

      <ControlGroup
        title="How bright is each line"
        hint="By default I draw every bar the same height, because I have not modelled brightness at all. These add it, one layer at a time."
        tone={intensities ? "active" : "plain"}
      >
        <Toggle
          label="Scale the bars by line strength"
          checked={intensities}
          onChange={setIntensities}
          why={
            intensities && !isThermal
              ? "My bars now show the Einstein A coefficient: how fast an atom in the upper level falls, not how many atoms are up there."
              : "I use the spontaneous emission rate. It says which transitions an atom prefers, before any gas is involved."
          }
          tourId="spectrum-options"
        />
        {intensities && (
          <Toggle
            label="Put it in a real gas (LTE populations)"
            checked={thermal}
            onChange={setThermal}
            why={
              thermal
                ? "I use Boltzmann for which level atoms sit in and Saha for how many are ionized, so my bars are now an emissivity."
                : "A rate is not a brightness until you know how many atoms are in the upper level. Turn this on and I work that out from a temperature and a density."
            }
          />
        )}
        {intensities && thermal && (
          <>
            <Slider
              label="temperature T"
              readout={
                temperatureK >= 1e4
                  ? `${(temperatureK / 1e3).toFixed(1)}k K`
                  : `${temperatureK.toFixed(0)} K`
              }
              anchor={describeTemperature(temperatureK)}
              min={2}
              max={6}
              step={0.02}
              value={Math.log10(temperatureK)}
              onChange={(v) => setTemperatureK(10 ** v)}
            />
            <Slider
              label="electron density n_e"
              readout={`10^${logNe.toFixed(1)} cm⁻³`}
              anchor={describeDensity(logNe)}
              min={4}
              max={22}
              step={0.1}
              value={logNe}
              onChange={setLogNe}
            />
            <p className="caption">
              The ionized fraction I get here is{" "}
              <strong>{(100 * ionized).toFixed(1)}%</strong>
              {ionized > 0.99
                ? ", so almost no neutral atoms are left and I draw every line faint no matter how hot it gets."
                : ionized < 0.01
                  ? ", essentially all neutral, so excitation alone sets the brightness I show."
                  : "."}
            </p>
          </>
        )}
      </ControlGroup>

      <ControlGroup
        title="What shape is a line"
        hint="A spectral line is not infinitely thin. Turn this on and I synthesize the shape it really has, so you can look at one up close."
        tone={profile ? "active" : "plain"}
      >
        <Toggle
          label="Synthesize line profiles"
          checked={profile}
          onChange={(v) => {
            setProfile(v);
            setKeepFull(false);
            if (!v) setProfileZoom(null);
          }}
          why={
            profile
              ? "I synthesize Voigt profiles: the upper level's finite lifetime gives Lorentzian wings, thermal motion gives a Gaussian core. Click a bar above and I will open one."
              : "Two things widen every line: the upper level cannot live forever, and the atoms are moving. Turn this on and you can click a line and I will show you its shape."
          }
        />
        {profile && (
          <Slider
            label="spectrograph resolving power R"
            readout={
              logResolvingPower === null
                ? "none"
                : `${(10 ** logResolvingPower).toExponential(1)} (λ/Δλ)`
            }
            anchor={describeResolvingPower(logResolvingPower)}
            min={2}
            max={7}
            step={0.05}
            value={logResolvingPower ?? 2}
            disabled={logResolvingPower === null}
            atRest={logResolvingPower === null}
            onChange={setLogResolvingPower}
          />
        )}
        {profile && (
          <button
            className="link-button"
            type="button"
            onClick={() => setLogResolvingPower(logResolvingPower === null ? 4 : null)}
          >
            {logResolvingPower === null
              ? "add a spectrograph and I will show the line as an instrument would"
              : "remove the spectrograph and I will show the line as it is"}
          </button>
        )}
      </ControlGroup>

      {window_.splittable && (
        <ControlGroup title="What my axis covers" tone={fullRange ? "active" : "plain"}>
          <Toggle
            label="Include the within-n fine-structure components"
            checked={fullRange}
            onChange={setFullRange}
            why={
              fullRange
                ? "My axis now stretches to millimetre and metre wavelengths, which squeezes every optical line into a sliver at the left."
                : `${window_.hidden} component${window_.hidden === 1 ? "" : "s"} start and end on the same shell, so ${window_.hidden === 1 ? "it lies" : "they lie"} out at millimetre to metre wavelengths. I keep them in the data either way.`
            }
          />
        </ControlGroup>
      )}

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

      {prof && profileZoom && (
        <ControlGroup
          title="What happens with more gas in the way"
          tone={showCurveOfGrowth ? "active" : "plain"}
        >
          <Toggle
            label="Curve of growth"
            checked={showCurveOfGrowth}
            onChange={setShowCurveOfGrowth}
            why="A line cannot keep getting stronger forever. Pile up enough atoms and the core saturates, and only the wings keep growing."
            tourId="curve-of-growth-toggle"
          />
          {showCurveOfGrowth && !curveOfGrowth && (
            <p className="hint-block">I am computing the curve of growth…</p>
          )}
        </ControlGroup>
      )}
      {prof && profileZoom && showCurveOfGrowth && curveOfGrowth && (
        <CurveOfGrowthView cog={curveOfGrowth} />
      )}

      <ControlGroup
        title="Look through the gas instead of at it"
        tone={absorption && thermal ? "active" : "plain"}
      >
        <Toggle
          label="Absorption: put this gas in front of a continuum"
          checked={absorption}
          disabled={!thermal}
          disabledReason="I need the LTE populations above: which level an atom is in is what its line absorbs with."
          onChange={setAbsorption}
          why="The same lines that glow when the gas is hot appear as dark gaps when you look through it at something brighter. That is how stellar spectra are read."
        />
        {absorption && thermal && (
          <Slider
            label="column density"
            readout={`10^${logColumn.toFixed(1)} m⁻²`}
            anchor="how much of the element is in the line of sight"
            min={14}
            max={26}
            step={0.1}
            value={logColumn}
            onChange={setLogColumn}
          />
        )}
      </ControlGroup>
      {absorption && thermal &&
        (absorptionData ? (
          <AbsorptionView abs={absorptionData} zoomed={profileZoom !== null} />
        ) : (
          <p className="hint-block">I am computing the absorption spectrum…</p>
        ))}

      {spectrum.profile_note && (
        <p className="caption">I drew no profile: {spectrum.profile_note}</p>
      )}
      {prof?.stark_note && <p className="caption warn-note">{prof.stark_note}</p>}

      {(prof && trace) || strength || window_.hidden > 0 || spectrum.intensity_note ? (
        <Disclosure summary="Exactly what my heights and my curve are claiming" tone="caveat">
          {prof && trace && (
            <p className="caption">
              The curve is Voigt profiles I synthesized in the engine and summed
              onto an adaptive grid, drawn on log₁₀ intensity over{" "}
              {trace.decades} decades below the peak{" "}
              <Badge provenance={SPECTRUM_PROFILE_LIBERTY} />. It integrates to{" "}
              {prof.flux_closure.toFixed(4)}× the summed line strengths, which is
              my grid's own quadrature error, measured rather than assumed. On
              this log wavelength axis I draw every line as a spike regardless
              of its real shape, so <strong>click a line</strong> and I will
              plot it linearly and show you the profile itself.
            </p>
          )}
          {prof && !prof.stark_note && prof.stark_span_nm && (
            <p className="caption">
              I have not put collisional broadening in this curve. At this
              density its linear Stark span would be{" "}
              {prof.stark_span_nm.value.toExponential(2)} nm, comfortably under
              the widths I do model, so the shape stands.
            </p>
          )}
          {strength && !isThermal && (
            <p className="caption">
              I set bar height and opacity ∝ log₁₀ A over{" "}
              {`10^${strength.lo.toFixed(1)} to 10^${strength.hi.toFixed(1)} s⁻¹`}. That
              is the spontaneous emission <em>rate</em>, not a brightness I am
              predicting you would observe. I model no level populations here:
              turn on LTE weighting and I will.
            </p>
          )}
          {strength && isThermal && spectrum.thermal && (
            <p className="caption">
              I set bar height and opacity ∝ log₁₀ ε over{" "}
              {`10^${strength.lo.toFixed(1)} to 10^${strength.hi.toFixed(1)}`} eV/s per atom,
              at T = {spectrum.thermal.temperature_k.toFixed(0)} K and n_e ={" "}
              {spectrum.thermal.electron_density_cm3.toExponential(0)} cm⁻³. That is an LTE
              emissivity: level populations from Boltzmann, ionization from Saha, and I take
              the gas to be <em>optically thin</em>. A real medium reabsorbs its own strong
              lines, which is why Lyman-α does not dominate an observed nebula the way it
              dominates mine.
            </p>
          )}
          {window_.hidden > 0 && (
            <p className="caption">
              My axis covers the across-n lines ({window_.lo.toFixed(1)}-
              {window_.hi < 1e6
                ? `${window_.hi.toFixed(0)} nm`
                : `${(window_.hi / 1e6).toFixed(1)} mm`}
              ). {window_.hidden} within-n fine-structure component
              {window_.hidden === 1 ? " is" : "s are"} outside it, out at millimetre to metre
              wavelengths. I keep them in the data and in my line list either way: tick the
              box above and I will include them, which stretches the axis far enough that
              the optical lines collapse into a sliver.
            </p>
          )}
          {spectrum.intensity_note && (
            <p className="caption">{spectrum.intensity_note}</p>
          )}
        </Disclosure>
      ) : null}
    </div>
  );
}
