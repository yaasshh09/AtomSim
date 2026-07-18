import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import { allowedSpan, PRESET_LABELS, PRESET_PARAMS, type ForcePreset } from "../lib/forceLaw";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 680;
const H = 460;
const PAD = { top: 32, right: 24, bottom: 44, left: 64 };
const L_CHOICES = [0, 1, 2, 3];
const PRESETS: ForcePreset[] = ["powerlaw", "yukawa", "harmonic", "finitewell", "coulombcore"];

export function ForceLawView() {
  const {
    forcePreset,
    forceParams,
    forceL,
    forceViz,
    forceLaw,
    forceStatus,
    error,
    setForcePreset,
    setForceParam,
    setForceL,
    setForceViz,
    loadForceLaw,
  } = useAppStore();

  useEffect(() => {
    if (forceLaw === null && forceStatus === "idle") void loadForceLaw();
  }, [forceLaw, forceStatus, loadForceLaw]);

  const cfProv = forceLaw?.counterfactual[0]?.energy.provenance ?? null;
  const refProv = forceLaw?.reference.items[0]?.energy.provenance ?? null;

  const levelsEv = forceLaw ? forceLaw.counterfactual.map((c) => c.energy_ev.value) : [];
  const refEv = forceLaw ? forceLaw.reference.items.map((i) => i.energy_ev.value) : [];
  const curveEv = forceLaw ? forceLaw.potential_curve.v_ev : [];
  const curveR = forceLaw ? forceLaw.potential_curve.r : [];

  const allEv = [...levelsEv, ...refEv, ...curveEv];
  const emin = allEv.length ? Math.min(...allEv) : -14;
  const emax = allEv.length ? Math.max(...allEv, 0.1) : 0;
  const y = scaleLinear([emin, emax], [H - PAD.bottom, PAD.top]);
  const rmax = curveR.length ? curveR[curveR.length - 1] : 1;
  const x = scaleLinear([0, rmax], [PAD.left, W - PAD.right]);

  const shortfall = forceLaw !== null && forceLaw.bound_count < forceLaw.requested_count;

  return (
    <div className="forcelaw">
      <div className="whatif-controls">
        <label>
          Potential
          <select
            value={forcePreset}
            onChange={(e) => setForcePreset(e.target.value as ForcePreset)}
          >
            {PRESETS.map((p) => (
              <option key={p} value={p}>
                {PRESET_LABELS[p]}
              </option>
            ))}
          </select>
        </label>
        {PRESET_PARAMS[forcePreset].map((spec) => (
          <label key={spec.name}>
            {spec.name} = {(forceParams[spec.name] ?? spec.default).toFixed(2)}
            {spec.unit ? ` ${spec.unit}` : ""}
            <input
              type="range"
              min={spec.min}
              max={spec.max}
              step={spec.step}
              value={forceParams[spec.name] ?? spec.default}
              onChange={(e) => setForceParam(spec.name, Number(e.target.value))}
            />
          </label>
        ))}
        <label>
          Orbital l
          <select value={forceL} onChange={(e) => setForceL(Number(e.target.value))}>
            {L_CHOICES.map((l) => (
              <option key={l} value={l}>
                {l} ({"spdf"[l]})
              </option>
            ))}
          </select>
        </label>
        <label>
          View
          <select
            value={forceViz}
            onChange={(e) => setForceViz(e.target.value as "well" | "ladder")}
          >
            <option value="well">Potential well</option>
            <option value="ladder">Energy ladder</option>
          </select>
        </label>
      </div>

      {forceStatus === "error" && <p className="error">{error}</p>}
      {forceStatus === "sampling" && <p className="hint-block">solving force law…</p>}
      {shortfall && (
        <p className="hint-block">
          Only {forceLaw!.bound_count} bound state
          {forceLaw!.bound_count === 1 ? "" : "s"} at these parameters
          {forceLaw!.bound_count === 0 ? " — the potential is too shallow to bind." : "."}
        </p>
      )}

      {forceLaw !== null && (
        <>
          <div className="forcelaw-legend">
            {cfProv && (
              <span>
                counterfactual {forcePreset} <Badge provenance={cfProv} />
              </span>
            )}
            {refProv && (
              <span>
                reference ({forceLaw.reference.kind}) <Badge provenance={refProv} />
              </span>
            )}
          </div>

          {forceViz === "well" ? (
            <svg
              viewBox={`0 0 ${W} ${H}`}
              className="forcelaw-svg"
              role="img"
              aria-label="potential energy curve with bound levels and reference"
            >
              <path
                className="forcelaw-curve"
                d={curveR
                  .map((r, i) => `${i === 0 ? "M" : "L"} ${x(r)} ${y(curveEv[i])}`)
                  .join(" ")}
                fill="none"
              />
              {forceLaw.reference.items.map((item, i) => (
                <line
                  key={`ref-${i}`}
                  className="forcelaw-ref"
                  x1={PAD.left}
                  x2={W - PAD.right}
                  y1={y(item.energy_ev.value)}
                  y2={y(item.energy_ev.value)}
                />
              ))}
              {forceLaw.counterfactual.map((c) => {
                const span = allowedSpan(curveR, curveEv, c.energy_ev.value);
                const x1 = span ? x(span[0]) : PAD.left;
                const x2 = span ? x(span[1]) : W - PAD.right;
                return (
                  <g key={`cf-${c.radial_index}`}>
                    <line
                      className="forcelaw-cf"
                      x1={x1}
                      x2={x2}
                      y1={y(c.energy_ev.value)}
                      y2={y(c.energy_ev.value)}
                    />
                    <text x={x2 + 4} y={y(c.energy_ev.value) - 4} className="forcelaw-label">
                      {c.energy_ev.value.toFixed(2)} eV
                    </text>
                  </g>
                );
              })}
              <text x={PAD.left} y={PAD.top - 12} className="forcelaw-col">
                V(r) and bound levels — {PRESET_LABELS[forcePreset]}
              </text>
            </svg>
          ) : (
            <svg
              viewBox={`0 0 ${W} ${H}`}
              className="forcelaw-svg"
              role="img"
              aria-label="energy levels versus reference"
            >
              {forceLaw.reference.items.map((item, i) => (
                <g key={`ref-${i}`}>
                  <line
                    className="forcelaw-ref"
                    x1={PAD.left}
                    x2={W / 2 - 8}
                    y1={y(item.energy_ev.value)}
                    y2={y(item.energy_ev.value)}
                  />
                  <text x={PAD.left} y={y(item.energy_ev.value) - 4} className="forcelaw-label">
                    {item.label}
                  </text>
                </g>
              ))}
              {forceLaw.counterfactual.map((c) => (
                <g key={`cf-${c.radial_index}`}>
                  <line
                    className="forcelaw-cf"
                    x1={W / 2 + 8}
                    x2={W - PAD.right}
                    y1={y(c.energy_ev.value)}
                    y2={y(c.energy_ev.value)}
                  />
                  <text
                    x={W - PAD.right}
                    y={y(c.energy_ev.value) - 4}
                    textAnchor="end"
                    className="forcelaw-label"
                  >
                    {c.energy_ev.value.toFixed(2)} eV
                  </text>
                </g>
              ))}
              <text x={W / 4} y={PAD.top - 12} textAnchor="middle" className="forcelaw-col">
                reference
              </text>
              <text x={(3 * W) / 4} y={PAD.top - 12} textAnchor="middle" className="forcelaw-col">
                {forcePreset}
              </text>
            </svg>
          )}

          <p className="hint-block">
            The numerical levels (NUMERICAL) are drawn against this preset's honest
            reference (EXACT). Screened and finite potentials bind only finitely many
            states; the missing upper reference rungs are the states they cannot hold.
          </p>
        </>
      )}
    </div>
  );
}
