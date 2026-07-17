import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 640;
const H = 460;
const PAD = { top: 32, right: 24, bottom: 40, left: 64 };
const L_CHOICES = [0, 1, 2, 3];

export function ForceLawView() {
  const { forceP, forceL, forceLaw, forceStatus, error, setForceP, setForceL, loadForceLaw } =
    useAppStore();

  useEffect(() => {
    if (forceLaw === null && forceStatus === "idle") void loadForceLaw();
  }, [forceLaw, forceStatus, loadForceLaw]);

  const cfProv = forceLaw?.counterfactual[0]?.energy.provenance ?? null;
  const refProv = forceLaw?.reference[0]?.energy.provenance ?? null;

  // Energy axis spans both sides (eV). Guard against an empty payload.
  const evs =
    forceLaw === null
      ? []
      : [
          ...forceLaw.counterfactual.map((x) => x.energy_ev.value),
          ...forceLaw.reference.map((x) => x.energy_ev.value),
        ];
  const emin = evs.length ? Math.min(...evs) : -14;
  const emax = evs.length ? Math.max(...evs, -0.1) : 0;
  const y = scaleLinear([emin, emax], [H - PAD.bottom, PAD.top]);

  return (
    <div className="forcelaw">
      <div className="whatif-controls">
        <label>
          Force-law exponent p = {forceP.toFixed(2)}
          <input
            type="range"
            min={0.5}
            max={1.5}
            step={0.05}
            value={forceP}
            onChange={(e) => setForceP(Number(e.target.value))}
          />
          <span className="hint-block">V(r) = −Z / r^p — p = 1 is real hydrogen</span>
        </label>
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
      </div>

      {forceStatus === "error" && <p className="error">{error}</p>}
      {forceStatus === "sampling" && <p className="hint-block">solving force law…</p>}
      {forceLaw !== null && (
        <>
          <div className="forcelaw-legend">
            {cfProv && (
              <span>
                counterfactual V=−Z/r^{forceP.toFixed(2)} <Badge provenance={cfProv} />
              </span>
            )}
            {refProv && (
              <span>
                real hydrogen (reference) <Badge provenance={refProv} />
              </span>
            )}
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} className="forcelaw-svg" role="img"
               aria-label="energy levels under the counterfactual force law versus real hydrogen">
            {/* reference (exact hydrogen) — ghosted, left column */}
            {forceLaw.reference.map((r) => (
              <g key={`ref-${r.n}`}>
                <line
                  x1={PAD.left}
                  x2={W / 2 - 8}
                  y1={y(r.energy_ev.value)}
                  y2={y(r.energy_ev.value)}
                  className="forcelaw-ref"
                />
                <text x={PAD.left} y={y(r.energy_ev.value) - 4} className="forcelaw-label">
                  n={r.n}
                </text>
              </g>
            ))}
            {/* counterfactual (numerical) — solid, right column */}
            {forceLaw.counterfactual.map((c) => (
              <g key={`cf-${c.radial_index}`}>
                <line
                  x1={W / 2 + 8}
                  x2={W - PAD.right}
                  y1={y(c.energy_ev.value)}
                  y2={y(c.energy_ev.value)}
                  className="forcelaw-cf"
                />
                <text x={W - PAD.right} y={y(c.energy_ev.value) - 4} textAnchor="end"
                      className="forcelaw-label">
                  {c.energy_ev.value.toFixed(2)} eV
                </text>
              </g>
            ))}
            <text x={W / 4} y={PAD.top - 12} textAnchor="middle" className="forcelaw-col">
              real hydrogen
            </text>
            <text x={(3 * W) / 4} y={PAD.top - 12} textAnchor="middle" className="forcelaw-col">
              V = −Z / r^{forceP.toFixed(2)}
            </text>
          </svg>
          <p className="hint-block">
            At p = 1 the numerical levels land on the exact hydrogen levels (solver
            calibration). Away from 1, states of the same n but different l split — the
            accidental Coulomb degeneracy is gone.
          </p>
        </>
      )}
    </div>
  );
}
