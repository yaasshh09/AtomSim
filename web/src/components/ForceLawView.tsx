import { scaleLinear } from "d3-scale";
import { useEffect, useState } from "react";
import {
  allowedSpan,
  PRESET_LABELS,
  PRESET_PARAMS,
  validateExprClient,
  type ForcePreset,
} from "../lib/forceLaw";
import { systemKind } from "../lib/systemKind";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 680;
const H = 460;
const PAD = { top: 32, right: 24, bottom: 44, left: 64 };
const L_CHOICES = [0, 1, 2, 3];
const PRESETS: ForcePreset[] = [
  "powerlaw",
  "yukawa",
  "harmonic",
  "finitewell",
  "coulombcore",
  "custom",
];
const EXPR_HELP = "allowed: r, pi, e; + - * / **; exp log sqrt sin cos tan sinh cosh tanh abs sign where(cond, a, b)";

export function ForceLawView() {
  const {
    forcePreset,
    forceParams,
    forceL,
    forceExpr,
    forceViz,
    forceLaw,
    forceStatus,
    error,
    setForcePreset,
    setForceParam,
    setForceL,
    setForceExpr,
    setForceViz,
    loadForceLaw,
    system,
    systems,
    setSystem,
  } = useAppStore();

  // This lab swaps the potential under ONE electron, and reads the selected
  // system only for its Z and reduced mass. A screened atom is not that: it is
  // eleven electrons and a fitted field, with no one-electron reduction to
  // hand over, which is why /api/forcelaw rejects it outright. So the guard is
  // on the request, not just on the menu entry, and it waits to know rather
  // than assuming: before `systems` arrives the kind is null and nothing fires.
  const kind = systemKind(systems, system);
  useEffect(() => {
    if (kind !== "hydrogenic") return;
    if (forceLaw === null && forceStatus === "idle") void loadForceLaw();
  }, [kind, forceLaw, forceStatus, loadForceLaw]);

  // The expression input is edited locally and committed (which re-solves) only
  // on Enter or blur, so a half-typed formula never fires a doomed solve.
  const [draft, setDraft] = useState(forceExpr);
  useEffect(() => setDraft(forceExpr), [forceExpr]);
  const draftError = validateExprClient(draft);
  const submitExpr = () => {
    const trimmed = draft.trim();
    if (draftError === null && trimmed !== forceExpr) setForceExpr(trimmed);
  };

  // The view stays on the menu on purpose. Which view you are looking at is a
  // separate axis from which system you picked, so making the entry disappear
  // when a screened atom is selected would leave the user hunting for a lab
  // that was there a moment ago. Say what is missing and offer the one click
  // that fixes it instead.
  if (kind === "screened") {
    return (
      <div className="hint-block">
        <p>
          The force-law lab replaces the potential under a single electron, so
          it needs a system that is a single electron. A screened atom is a
          fitted field standing in for the other electrons, and that field is
          the thing this lab would be overwriting.
        </p>
        <button type="button" className="ghost" onClick={() => setSystem("h")}>
          Switch to hydrogen
        </button>
      </div>
    );
  }

  const potLabel =
    forcePreset === "custom"
      ? `V(r) = ${forceLaw?.expression ?? forceExpr}`
      : PRESET_LABELS[forcePreset];
  const untrusted = forceLaw ? forceLaw.counterfactual.filter((c) => !c.trusted).length : 0;

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
        <label data-tour="force-preset">
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
        {forcePreset === "custom" && (
          <label className="forcelaw-expr">
            V(r) =
            <input
              type="text"
              value={draft}
              spellCheck={false}
              aria-label="custom potential expression in r"
              onChange={(e) => setDraft(e.target.value)}
              onBlur={submitExpr}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  submitExpr();
                }
              }}
            />
          </label>
        )}
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

      {forcePreset === "custom" && (
        <>
          <p className="hint-block">
            Custom V(r) is a made-up force law — every level below is COUNTERFACTUAL. {EXPR_HELP}
          </p>
          {draftError !== null && <p className="error">{draftError}</p>}
        </>
      )}
      {forceStatus === "error" && <p className="error">{error}</p>}
      {forceStatus === "sampling" && <p className="hint-block">solving force law…</p>}
      {untrusted > 0 && (
        <p className="hint-block">
          {untrusted} level{untrusted === 1 ? "" : "s"} could not be trusted (not box/grid
          converged) and {untrusted === 1 ? "is" : "are"} drawn dashed — not real bound states.
        </p>
      )}
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
                      style={c.trusted ? undefined : { opacity: 0.5, strokeDasharray: "5 3" }}
                    />
                    <text x={x2 + 4} y={y(c.energy_ev.value) - 4} className="forcelaw-label">
                      {c.energy_ev.value.toFixed(2)} eV{c.trusted ? "" : " ⚠"}
                    </text>
                  </g>
                );
              })}
              <text x={PAD.left} y={PAD.top - 12} className="forcelaw-col">
                V(r) and bound levels — {potLabel}
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
                    style={c.trusted ? undefined : { opacity: 0.5, strokeDasharray: "5 3" }}
                  />
                  <text
                    x={W - PAD.right}
                    y={y(c.energy_ev.value) - 4}
                    textAnchor="end"
                    className="forcelaw-label"
                  >
                    {c.energy_ev.value.toFixed(2)} eV{c.trusted ? "" : " ⚠"}
                  </text>
                </g>
              ))}
              <text x={W / 4} y={PAD.top - 12} textAnchor="middle" className="forcelaw-col">
                reference
              </text>
              <text x={(3 * W) / 4} y={PAD.top - 12} textAnchor="middle" className="forcelaw-col">
                {forcePreset === "custom" ? "custom V(r)" : forcePreset}
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
