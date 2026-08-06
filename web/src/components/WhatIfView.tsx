import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import type { ConstMultipliers } from "../api/client";
import type { DerivedObservable } from "../api/types";
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
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 720;
const H = 480;
const ZOOM_N = 2; // textbook shell: 2p3/2 - 2p1/2 split grows with alpha

const REAL_ALL: ConstMultipliers = { hbar: 1, e: 1, m_e: 1, eps0: 1, c: 1 };

export function WhatIfView() {
  const { labConst, labZ, whatif, whatifStatus, error, setLabConst, setLabZ, loadWhatIf } =
    useAppStore();

  useEffect(() => {
    if (whatif === null && whatifStatus === "idle") void loadWhatIf();
  }, [whatif, whatifStatus, loadWhatIf]);

  if (whatifStatus === "error") return <p className="error">{error}</p>;
  if (!whatif) return <p className="hint-block">loading What-If lab…</p>;

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

  // Gross ladder: STRUCTURE only, in units of E_h (hartree). Absolute scale lives
  // in the readouts above, the honest structure/scale split.
  const eMin = real.gross[0].energy.value;
  const y = scaleLinear([eMin, 0], [H - 40, 60]);
  const rx1 = 70;
  const rx2 = 300;

  // n=2 fine split, in µE_h (hartree * 1e6), normalized, not real eV.
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
      return `Derived α = ${formatAlpha(alphaValue)} exceeds 0.5, the perturbative fine structure is meaningless here, so the altered split isn't drawn. The readouts still show the true α; this is the honest model boundary, not a glitch.`;
    }
    if (altOn && changed.length === 0) {
      return "You altered the constants, but α, a₀, and E_h are all unchanged, a different universe that is observationally identical to ours. That degeneracy is the whole lesson: only dimensionless combinations and fixed-ruler scales are observable.";
    }
    if (altOn) {
      return `Altered. Observably changed: ${changed.join(", ")}. Fine-structure fractional error ≈ ${(errFrac * 100).toFixed(1)}% (grows as (Zα)²); n=${ZOOM_N} split ≈ ${splitUeH.toFixed(1)} µE_h. The gross ladder is α-independent structure; absolute size and binding are in the readouts.`;
    }
    return "Drag any raw constant. α, a₀, and E_h are derived from all five, only these dimensionless and fixed-ruler quantities are observable. Watch which actually move: try e ×2 and ε₀ ×4 together.";
  })();

  return (
    <div className="view-wrap">
      <div className="view-header">
        <span className="plot-title">
          What-If: fundamental constants{" "}
          <Badge provenance={report.alpha.quantity.provenance} />
        </span>
      </div>

      {altOn && (
        <div className="counterfactual-banner">
          COUNTERFACTUAL · derived α = {formatAlpha(alphaValue)}
        </div>
      )}

      <dl className="readouts">
        {readouts.map((r) => (
          <div key={r.key} className="readout-row">
            <dt>
              {r.label} <Badge provenance={r.obs.quantity.provenance} />
            </dt>
            <dd>
              {r.text}{" "}
              <span className={r.obs.changed ? "readout-ratio changed" : "readout-ratio"}>
                {formatRatio(r.obs.ratio)}
              </span>
            </dd>
          </div>
        ))}
      </dl>

      <svg viewBox={`0 0 ${W} ${H}`} role="img" className="levels-svg">
        <text x={(rx1 + rx2) / 2} y={30} textAnchor="middle" className="tick">
          gross levels (Z={real.system.z}), structure in units of E_h, α-independent
        </text>
        {real.gross.map((g) => (
          <g key={g.n}>
            <line x1={rx1} x2={rx2} y1={y(g.energy.value)} y2={y(g.energy.value)} className="rung" />
            <text x={rx1 - 8} y={y(g.energy.value)} dy="0.32em" textAnchor="end" className="tick">
              n={g.n}
            </text>
            <text x={rx2 + 8} y={y(g.energy.value)} dy="0.32em" className="tick">
              2n²={g.degeneracy}
            </text>
          </g>
        ))}

        <text x={530} y={54} textAnchor="middle" className="tick">
          n={ZOOM_N} fine split [µE_h], real vs altered
        </text>
        {beyondValidity ? (
          <text x={530} y={H / 2} textAnchor="middle" className="tick">
            α &gt; 0.5, beyond perturbative validity
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

      <div className="const-sliders" data-tour="const-sliders">
        {CONSTANT_KEYS.map((k) => (
          <label key={k}>
            <span>
              {CONSTANT_LABELS[k]} ×{labConst[k].toFixed(2)}
            </span>
            <input
              type="range"
              min={Math.log2(CONST_MIN)}
              max={Math.log2(CONST_MAX)}
              step={0.25}
              value={Math.log2(labConst[k])}
              onChange={(e) =>
                setLabConst({ [k]: 2 ** Number(e.target.value) } as Partial<ConstMultipliers>)
              }
            />
          </label>
        ))}
      </div>

      <div className="whatif-controls">
        <div className="stepper">
          <span>nuclear charge Z</span>
          <button type="button" onClick={() => setLabZ(Math.max(1, labZ - 1))} disabled={labZ <= 1}>
            −
          </button>
          <span>{labZ}</span>
          <button type="button" onClick={() => setLabZ(Math.min(10, labZ + 1))} disabled={labZ >= 10}>
            +
          </button>
        </div>
        <button type="button" className="primary" onClick={() => setLabConst(REAL_ALL)}>
          reset to real constants
        </button>
      </div>

      <p className={beyondValidity ? "error" : "caption"}>{caption}</p>
    </div>
  );
}
