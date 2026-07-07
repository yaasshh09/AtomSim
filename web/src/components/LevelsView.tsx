import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import { arrowsFor } from "../lib/levels";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 680;
const H = 460;

export function LevelsView() {
  const {
    n, l, system, fineStructure, levels, spectrum, loadLevels, loadSpectrum,
  } = useAppStore();
  useEffect(() => {
    void loadLevels();
    void loadSpectrum();
  }, [system, fineStructure, loadLevels, loadSpectrum]);
  if (!levels) return <p className="hint-block">loading levels…</p>;

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
            const shifts = fineForN.map((f) => f.shift_ev.value);
            const lo = Math.min(...shifts);
            const hi = Math.max(...shifts);
            const pad = (hi - lo || 1e-9) * 0.15;
            const yz = scaleLinear([lo - pad, hi + pad], [H - 60, 48]);
            const zx1 = 470;
            const zx2 = 590;
            return (
              <g>
                <text x={(zx1 + zx2) / 2} y={26} textAnchor="middle" className="tick">
                  n={n} shifts [µeV] — zoomed, APPROXIMATION
                </text>
                {fineForN.map((f, idx) => (
                  <g key={`${f.l}-${f.j}`}>
                    <line
                      x1={zx1} x2={zx2}
                      y1={yz(f.shift_ev.value)} y2={yz(f.shift_ev.value)}
                      className={f.l === l ? "rung rung-active" : "rung"}
                    />
                    <text
                      x={zx2 + 6}
                      y={yz(f.shift_ev.value) + (idx % 2 ? 12 : 0)}
                      dy="0.32em" className="tick"
                    >
                      l={f.l}, j={f.j} · {(f.shift_ev.value * 1e6).toFixed(1)}
                    </text>
                  </g>
                ))}
              </g>
            );
          })()}
      </svg>
      <p className="caption">
        Gross levels are reduced-mass exact. The right column magnifies the α²
        fine-structure shifts of the selected n — the two scales differ by ~10⁵ and are
        labeled, never blended. States with equal j coincide at this order (e.g. 2s₁/₂ and
        2p₁/₂ — the Lamb shift is beyond α² and honestly absent here).
      </p>
    </div>
  );
}
