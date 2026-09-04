import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import type { ConstMultipliers } from "../api/client";
import type { DerivedObservable } from "../api/types";
import {
  activeScenario,
  CONSTANT_BLURBS,
  SCENARIOS,
  VIEW_LEADS,
  type ScenarioMultipliers,
} from "../lib/explain";
import { spreadLabels } from "../lib/levels";
import {
  CONST_MAX,
  CONST_MIN,
  CONSTANT_KEYS,
  CONSTANT_LABELS,
  fineErrorFraction,
  formatAlpha,
  formatRatio,
  shellSplitting,
} from "../lib/whatif";
import { Notation, mathTspans } from "../lib/mathText";
import { useAppStore } from "../state/store";
import { withinView } from "../lib/zoom";
import { Badge } from "./Badge";
import { Disclosure } from "./Disclosure";
import { ControlGroup, Slider } from "./Field";
import { OffWindowMarks, usePlotZoom, ZoomControls } from "./PlotZoom";
import { ViewIntro } from "./ViewIntro";

const W = 720;
const H = 480;
/** Where the gross ladder is drawn, deepest rung to the ionization limit. */
const LADDER_RANGE: [number, number] = [H - 40, 60];
const ZOOM_N = 2; // the textbook shell: the 2p3/2 - 2p1/2 split grows with alpha

const REAL_ALL: ConstMultipliers = { hbar: 1, e: 1, m_e: 1, eps0: 1, c: 1 };

export function WhatIfView() {
  const { labConst, labZ, whatif, whatifStatus, error, setLabConst, setLabZ, loadWhatIf } =
    useAppStore();

  useEffect(() => {
    if (whatif === null && whatifStatus === "idle") void loadWhatIf();
  }, [whatif, whatifStatus, loadWhatIf]);

  /* Above the two early returns below, because it is a hook. While the lab is
     still loading there is no ladder and the window is a placeholder. */
  const zoom = usePlotZoom({
    width: W,
    height: H,
    y: {
      domain: [whatif ? whatif.real.gross[0].energy.value : -0.5, 0],
      range: LADDER_RANGE,
    },
  });

  if (whatifStatus === "error") return <p className="error">{error}</p>;
  if (!whatif) {
    return (
      <div className="view-wrap">
        <ViewIntro lead={VIEW_LEADS.whatif} />
        <p className="hint-block">Loading the What-If lab…</p>
      </div>
    );
  }

  const { report, real, altered } = whatif;
  const altOn = report.altered;
  const beyondValidity = altOn && altered === null;
  const alphaValue = report.alpha.quantity.value;

  const readouts: { key: string; label: string; obs: DerivedObservable; text: string }[] = [
    {
      key: "alpha",
      label: "α: fine-structure constant",
      obs: report.alpha,
      text: `${formatAlpha(alphaValue)} (${alphaValue.toExponential(3)})`,
    },
    {
      key: "a0",
      label: "a₀: Bohr radius (atom size)",
      obs: report.bohr_radius_pm,
      text: `${report.bohr_radius_pm.quantity.value.toFixed(2)} pm`,
    },
    {
      key: "eh",
      label: "E_h: Hartree energy (binding)",
      obs: report.hartree_ev,
      text: `${report.hartree_ev.quantity.value.toFixed(3)} eV`,
    },
  ];

  // The gross ladder shows STRUCTURE only, in units of E_h (hartree). The
  // absolute scale stays in the readouts above, which is the honest way to
  // split structure from scale.
  const y = scaleLinear(zoom.y, LADDER_RANGE);
  const rx1 = 70;
  const rx2 = 300;
  // Same crowding as the levels ladder, and the same fix: the rungs go as
  // -1/n^2, which printed the top three labels through one another. Zooming
  // narrows which rungs are on screen, and only those are spread.
  const gross = withinView(real.gross, (g) => g.energy.value, zoom.y);
  const grossY = gross.map((g) => y(g.energy.value));
  const grossLabelY = spreadLabels(grossY, 13, 46, H - 28);

  // The n=2 fine split, in µE_h (hartree * 1e6). Normalized, so not real eV.
  const realFine = (real.fine ?? []).filter((f) => f.n === ZOOM_N);
  const altFine = (altered?.fine ?? []).filter((f) => f.n === ZOOM_N);
  const shifts = [...realFine, ...altFine].map((f) => f.shift.value * 1e6);
  const lo = Math.min(0, ...shifts);
  const hi = Math.max(0, ...shifts);
  const pad = (hi - lo || 1) * 0.2;
  const yz = scaleLinear([lo - pad, hi + pad], [H - 60, 90]);
  const columns = [
    { x: 470, rows: realFine, label: "real", cf: false },
    { x: 590, rows: altFine, label: "altered", cf: true },
  ];

  const errFrac = fineErrorFraction(altered?.fine ?? null);
  const splitUeH = shellSplitting(
    (altered?.fine ?? []).map((f) => ({ ...f, shift_ev: f.shift })),
    ZOOM_N,
  ) * 1e6;

  const changed = readouts.filter((r) => r.obs.changed).map((r) => r.label.split(" ")[0]);
  const caption = (() => {
    if (beyondValidity) {
      return `The derived α = ${formatAlpha(alphaValue)} exceeds 0.5, so the perturbative fine structure is meaningless here and the altered split is not drawn. The readouts still show the true α. This is the model's honest boundary, not a glitch.`;
    }
    if (altOn && changed.length === 0) {
      return "The constants moved, but α, a₀ and E_h all came back unchanged: a different universe, observationally identical to ours. That degeneracy is the whole lesson, because only dimensionless combinations and fixed-ruler scales are observable.";
    }
    if (altOn) {
      return `Altered. Observably changed: ${changed.join(", ")}. The fine-structure fractional error is about ${(errFrac * 100).toFixed(1)}%, and it grows as (Zα)²; the n=${ZOOM_N} split comes to about ${splitUeH.toFixed(1)} µE_h. The gross ladder is α-independent structure, and absolute size and binding stay in the readouts.`;
    }
    return "Drag any raw constant. α, a₀ and E_h are derived from all five, and only those dimensionless and fixed-ruler quantities are observable. Watch which ones actually move: try e ×2 and ε₀ ×4 together.";
  })();

  /* Which prepared universe is loaded, if any. The panel is five sliders over
     a five-dimensional space, and almost every point in it is uninteresting.
     Anyone who does not already know which combinations cancel has no way to
     find one by dragging, so the scenarios are the found ones. */
  const scenario = activeScenario(labConst as ScenarioMultipliers);

  return (
    <div className="view-wrap">
      <ViewIntro
        lead={VIEW_LEADS.whatif}
        badge={<Badge provenance={report.alpha.quantity.provenance} />}
      >
        <Disclosure summary="Why most changes here change nothing">
          <p className="caption">
            A constant with units is a statement about the ruler as much as
            about the world. Double the electron's charge and you have changed
            a number, but you have also changed every measuring device made of
            charges, and the two changes can cancel exactly.
          </p>
          <p className="caption">
            What survives is the dimensionless combination. α = e²/(4πε₀ℏc) has
            no units at all, so nothing about your choice of metre or second
            can move it, and that is precisely why it is the one number in this
            panel that a different universe could genuinely disagree with us
            about. Try "the same universe in disguise" below and watch all three
            readouts hold still while two constants move by a factor of four.
          </p>
        </Disclosure>
      </ViewIntro>

      {altOn && (
        <div className="counterfactual-banner">
          COUNTERFACTUAL · derived α = {formatAlpha(alphaValue)}
        </div>
      )}

      <ControlGroup
        title="Prepared universes"
        hint="Each of these makes one point and no other. Pick one, then read the three observables underneath."
        tone={scenario && scenario.key !== "real" ? "active" : "plain"}
      >
        <div className="scenario-row">
          {SCENARIOS.map((s) => (
            <button
              key={s.key}
              type="button"
              className={`ctl-choice-btn${scenario?.key === s.key ? " ctl-choice-on" : ""}`}
              aria-pressed={scenario?.key === s.key}
              onClick={() => setLabConst(s.multipliers as Partial<ConstMultipliers>)}
            >
              {s.label}
            </button>
          ))}
        </div>
        <p className="ctl-choice-hint">
          {scenario
            ? scenario.blurb
            : "Your own combination. None of the prepared ones matches it."}
        </p>
      </ControlGroup>

      <h3 className="readouts-head">The three things you could actually measure</h3>
      <dl className="readouts">
        {readouts.map((r) => (
          <div key={r.key} className="readout-row">
            <dt>
              <Notation>{r.label}</Notation>{" "}
              <Badge provenance={r.obs.quantity.provenance} />
            </dt>
            <dd>
              <Notation>{r.text}</Notation>{" "}
              <span className={r.obs.changed ? "readout-ratio changed" : "readout-ratio"}>
                {formatRatio(r.obs.ratio)}
              </span>
            </dd>
          </div>
        ))}
      </dl>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        className={`levels-svg plot-zoomable${zoom.dragging ? " plot-panning" : ""}`}
        ref={zoom.ref}
        {...zoom.handlers}
      >
        <text x={(rx1 + rx2) / 2} y={30} textAnchor="middle" className="tick">
          {mathTspans(
            `gross levels (Z=${real.system.z}): structure in units of E_h, α-independent`,
          )}
        </text>
        <OffWindowMarks
          above={real.gross.filter((g) => g.energy.value > Math.max(...zoom.y)).length}
          below={real.gross.filter((g) => g.energy.value < Math.min(...zoom.y)).length}
          x={rx1} top={46} bottom={H - 14} noun="shell"
        />
        {gross.map((g, i) => {
          const yr = grossY[i];
          const yl = grossLabelY[i];
          const nudged = Math.abs(yl - yr) > 1;
          return (
            <g key={g.n}>
              <line x1={rx1} x2={rx2} y1={yr} y2={yr} className="rung" />
              {nudged && (
                <>
                  <line x1={rx1 - 30} x2={rx1 - 6} y1={yl} y2={yr} className="leader" />
                  <line x1={rx2 + 4} x2={rx2 + 26} y1={yr} y2={yl} className="leader" />
                </>
              )}
              <text x={rx1 - 32} y={yl} dy="0.32em" textAnchor="end" className="tick">
                n={g.n}
              </text>
              <text x={rx2 + 30} y={yl} dy="0.32em" className="tick">
                2n²={g.degeneracy}
              </text>
            </g>
          );
        })}

        <text x={530} y={54} textAnchor="middle" className="tick">
          {mathTspans(`n=${ZOOM_N} fine split [µE_h], real vs altered`)}
        </text>
        {beyondValidity ? (
          <text x={530} y={H / 2} textAnchor="middle" className="tick">
            α &gt; 0.5, past where the perturbation can be trusted
          </text>
        ) : (
          columns.map((col) => (
            <g key={col.label}>
              <text x={col.x + 20} y={78} textAnchor="middle" className="tick">
                {col.label}
              </text>
              {col.rows.map((f) => (
                <g key={`${col.label}-${f.l}-${f.j}`}>
                  <line
                    x1={col.x}
                    x2={col.x + 40}
                    y1={yz(f.shift.value * 1e6)}
                    y2={yz(f.shift.value * 1e6)}
                    className={col.cf && altOn ? "rung rung-counterfactual" : "rung"}
                  />
                  <text x={col.x + 46} y={yz(f.shift.value * 1e6)} dy="0.32em" className="tick">
                    j={f.j} · {(f.shift.value * 1e6).toFixed(1)}
                  </text>
                </g>
              ))}
            </g>
          ))
        )}
      </svg>
      <ZoomControls zoom={zoom} what="the ladder's energy axis" />

      <p className={beyondValidity ? "error" : "caption"}>{caption}</p>

      <ControlGroup
        title="Or move one constant at a time"
        hint="Each slider runs from a quarter to four times its measured value. Any slider moved off ×1.00 lights up."
        tone={altOn ? "active" : "plain"}
      >
        <div className="const-sliders" data-tour="const-sliders">
          {CONSTANT_KEYS.map((k) => (
            <Slider
              key={k}
              label={`${CONSTANT_LABELS[k]} · ${CONSTANT_BLURBS[k]}`}
              readout={`×${labConst[k].toFixed(2)}`}
              atRest={labConst[k] === 1}
              min={Math.log2(CONST_MIN)}
              max={Math.log2(CONST_MAX)}
              step={0.25}
              value={Math.log2(labConst[k])}
              onChange={(v) =>
                setLabConst({ [k]: 2 ** v } as Partial<ConstMultipliers>)
              }
            />
          ))}
        </div>
      </ControlGroup>

      <div className="whatif-controls">
        <div className="stepper">
          <span>nuclear charge Z</span>
          <div className="stepper-ctl">
            <button
              type="button"
              aria-label="decrease nuclear charge"
              onClick={() => setLabZ(Math.max(1, labZ - 1))}
              disabled={labZ <= 1}
            >
              −
            </button>
            <span className="stepper-value">{labZ}</span>
            <button
              type="button"
              aria-label="increase nuclear charge"
              onClick={() => setLabZ(Math.min(10, labZ + 1))}
              disabled={labZ >= 10}
            >
              +
            </button>
          </div>
        </div>
        <button
          type="button"
          className="primary"
          disabled={!altOn}
          onClick={() => setLabConst(REAL_ALL)}
        >
          reset to real constants
        </button>
      </div>
    </div>
  );
}
