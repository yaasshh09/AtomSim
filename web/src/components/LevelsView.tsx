import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import { isScreenedLevels } from "../api/client";
import type { ScreenedLevels } from "../api/types";
import { arrowsFor } from "../lib/levels";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 680;
const H = 460;

function ScreenedLadder({ levels }: { levels: ScreenedLevels }) {
  const orbitals = levels.orbitals;
  const es = orbitals.map((o) => o.energy_ev.value);
  const eMin = Math.min(...es);
  const eMax = Math.max(...es, 0); // include the ionization threshold at 0
  const y = scaleLinear([eMin, eMax], [H - 40, 24]);
  const rungX1 = 90;
  const rungX2 = 340;
  return (
    <div className="view-wrap">
      <div className="view-header">
        <span className="plot-title">
          Screened orbital energies ε_nl [eV]{" "}
          <Badge provenance={orbitals[0].energy.provenance} />
        </span>
        <span className="plot-title">
          · {levels.config}
          {levels.is_ground ? " (ground)" : " — excited (non-ground)"}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" className="levels-svg">
        {/* ionization threshold */}
        <line x1={rungX1} x2={rungX2} y1={y(0)} y2={y(0)} className="zero" />
        <text x={rungX2 + 8} y={y(0)} dy="0.32em" className="tick">
          0 — ionization limit
        </text>
        {orbitals.map((o) => {
          const filled = o.occupancy > 0;
          return (
            <g key={`${o.n}-${o.l}`}>
              <line
                x1={rungX1} x2={rungX2}
                y1={y(o.energy_ev.value)} y2={y(o.energy_ev.value)}
                className="rung"
                strokeWidth={filled ? 3 : 1.5}
                strokeDasharray={filled ? undefined : "4 4"}
                opacity={filled ? 1 : 0.5}
              />
              <text
                x={rungX1 - 8} y={y(o.energy_ev.value)} dy="0.32em"
                textAnchor="end" className="tick"
              >
                {o.label}
                {filled ? <tspan dy="-0.5em">{o.occupancy}</tspan> : ""}
              </text>
              <text x={rungX2 + 8} y={y(o.energy_ev.value)} dy="0.32em" className="tick">
                {o.energy_ev.value.toFixed(2)} eV
                {filled ? "" : " · virtual"}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="caption">
        Independent-particle orbital energies in the Green-Sellin-Zachor screened
        central field (APPROXIMATION). Screening lifts the hydrogenic l-degeneracy
        (s below p below d for a given n). Filled subshells are solid with their
        occupancy; virtual orbitals are dashed. Total energy{" "}
        {levels.total_energy_ev.value.toFixed(2)} eV is a sum of occupancy-weighted
        orbital energies, not a variational total — see the badge.
      </p>
    </div>
  );
}

export function LevelsView() {
  const {
    n, l, system, fineStructure, dirac, setDirac, bField, setBField,
    levels, spectrum, loadLevels, loadSpectrum,
  } = useAppStore();
  useEffect(() => {
    void loadLevels();
    void loadSpectrum();
  }, [system, fineStructure, dirac, bField, loadLevels, loadSpectrum]);
  if (!levels) return <p className="hint-block">loading levels…</p>;
  if (isScreenedLevels(levels)) return <ScreenedLadder levels={levels} />;

  const eMin = levels.gross[0].energy_ev.value;
  const y = scaleLinear([eMin, 0], [H - 40, 24]);
  const rungX1 = 70;
  const rungX2 = 320;
  const arrows = spectrum ? arrowsFor(spectrum.lines, n, l) : [];
  const grossE = new Map(levels.gross.map((g) => [g.n, g.energy_ev.value]));
  const fineForN = levels.fine?.filter((f) => f.n === n) ?? [];

  return (
    <div className="view-wrap">
      <div className="view-header">
        <span className="plot-title">
          Energy levels E_n [eV]{" "}
          <Badge provenance={levels.gross[0].energy.provenance} />
        </span>
        {fineStructure && fineForN.length > 0 && (
          <span className="plot-title">
            · fine structure of n={n}{" "}
            <Badge provenance={fineForN[0].shift.provenance} />
          </span>
        )}
        {fineStructure && (
          <label className="levels-model">
            <input type="checkbox" checked={dirac} onChange={(e) => setDirac(e.target.checked)} />
            Dirac (exact)
          </label>
        )}
        {fineStructure && (
          <label className="levels-field">
            B{" "}
            <input
              type="range" min={0} max={20} step={0.1} value={bField}
              onChange={(e) => setBField(Number(e.target.value))}
            />
            {bField > 0
              ? ` ${bField.toFixed(1)} T (µ_B·B = ${(bField * 0.5 / 2.35051757e5 * 27.211386245e6).toFixed(1)} µeV)`
              : " 0 T"}
          </label>
        )}
        {!fineStructure && (
          <span className="levels-field-hint">turn on fine structure to add a field</span>
        )}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" className="levels-svg">
        {levels.gross.map((g) => (
          <g key={g.n}>
            <line
              x1={rungX1} x2={rungX2}
              y1={y(g.energy_ev.value)} y2={y(g.energy_ev.value)}
              className={g.n === n ? "rung rung-active" : "rung"}
            />
            <text
              x={rungX1 - 8} y={y(g.energy_ev.value)} dy="0.32em"
              textAnchor="end" className="tick"
            >
              n={g.n}
            </text>
            <text x={rungX2 + 8} y={y(g.energy_ev.value)} dy="0.32em" className="tick">
              {g.energy_ev.value.toFixed(2)} eV · 2n²={g.degeneracy}
            </text>
          </g>
        ))}
        {arrows.map((a, i) => {
          if (!grossE.has(a.n_upper) || !grossE.has(a.n_lower)) return null;
          const ax = rungX1 + 30 + i * 26;
          const yTop = y(grossE.get(a.n_upper) ?? 0);
          const yBot = y(grossE.get(a.n_lower) ?? 0);
          return (
            <g key={`${a.n_lower}-${a.l_lower}-${i}`} className="arrow">
              <line x1={ax} x2={ax} y1={yTop} y2={yBot - 6} />
              <path d={`M${ax - 4},${yBot - 8} L${ax + 4},${yBot - 8} L${ax},${yBot} Z`} />
              <text x={ax + 4} y={(yTop + yBot) / 2} className="tick">
                {a.wavelength_nm.value.toFixed(0)} nm
              </text>
            </g>
          );
        })}
        {fineStructure && fineForN.length > 0 &&
          (() => {
            const bohrN = grossE.get(n) ?? 0;
            const shifts = fineForN.map((f) => f.shift_ev.value);
            const subShifts = bField > 0
              ? fineForN.flatMap((f) =>
                  (f.sublevels ?? []).map((s) => s.energy_ev.value - bohrN),
                )
              : [];
            const allValues = [...shifts, ...subShifts];
            const lo = Math.min(...allValues);
            const hi = Math.max(...allValues);
            const pad = (hi - lo || 1e-9) * 0.15;
            const yz = scaleLinear([lo - pad, hi + pad], [H - 60, 48]);
            const zx1 = 470;
            const zx2 = 590;
            const sx1 = 610;
            const sx2 = 660;
            return (
              <g>
                <text x={(zx1 + zx2) / 2} y={26} textAnchor="middle" className="tick">
                  {bField > 0
                    ? `n=${n} Zeeman split [µeV] — APPROXIMATION`
                    : `n=${n} shifts [µeV] — zoomed, ${dirac ? "EXACT" : "APPROXIMATION"}`}
                </text>
                {fineForN.map((f, idx) => {
                  const subs = bField > 0 ? f.sublevels ?? [] : [];
                  const maxAbsMj = subs.length > 0
                    ? Math.max(...subs.map((s) => Math.abs(s.m_j)))
                    : 0;
                  return (
                    <g key={`${f.l}-${f.j}`}>
                      <line
                        x1={zx1} x2={zx2}
                        y1={yz(f.shift_ev.value)} y2={yz(f.shift_ev.value)}
                        className={f.l === l ? "rung rung-active" : "rung"}
                        opacity={bField > 0 ? 0.35 : 1}
                      />
                      {bField === 0 && (
                        <text
                          x={zx2 + 6}
                          y={yz(f.shift_ev.value) + (idx % 2 ? 12 : 0)}
                          dy="0.32em" className="tick"
                        >
                          l={f.l}, j={f.j} · {(f.shift_ev.value * 1e6).toFixed(1)}
                        </text>
                      )}
                      {subs.map((s) => {
                        const yS = yz(s.energy_ev.value - bohrN);
                        const extreme = Math.abs(s.m_j) === maxAbsMj;
                        return (
                          <g key={`${f.l}-${f.j}-${s.m_j}-${s.branch}`}>
                            <line
                              x1={zx2} x2={sx1}
                              y1={yz(f.shift_ev.value)} y2={yS}
                              className="rung"
                              opacity={0.25}
                            />
                            <line
                              x1={sx1} x2={sx2}
                              y1={yS} y2={yS}
                              className={f.l === l ? "rung-active" : "rung"}
                            />
                            {extreme && (
                              <text x={sx2 + 6} y={yS} dy="0.32em" className="tick">
                                m_j={s.m_j}
                              </text>
                            )}
                          </g>
                        );
                      })}
                    </g>
                  );
                })}
              </g>
            );
          })()}
      </svg>
      <p className="caption">
        Gross levels are reduced-mass exact. The right column magnifies the{" "}
        {dirac ? "relativistic" : "α²"} shifts of the selected n — the two scales differ by ~10⁵
        and are labeled, never blended.{" "}
        {dirac
          ? "Dirac is exact for a point nucleus: the energy depends on n and j only, so 2s₁/₂ and 2p₁/₂ coincide exactly. Reality splits them by the Lamb shift (QED), which this model deliberately omits — see the badge assumptions."
          : "States with equal j coincide at this order (e.g. 2s₁/₂ and 2p₁/₂ — the Lamb shift is beyond α² and honestly absent here)."}
        {bField > 0 && (
          <>
            {" "}A magnetic field splits each j-level into 2j+1 m_j sublevels (anomalous
            Zeeman, spacing g_J·µ_B·B); as B rises they reorganize toward the Paschen-Back
            pattern where (m_l, m_s) become the good labels. Linear model — the diamagnetic
            B² term is omitted.
          </>
        )}
      </p>
    </div>
  );
}
