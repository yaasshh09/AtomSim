import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import { isScreenedLevels } from "../api/client";
import type { HFLevels, ScreenedLevels } from "../api/types";
import { arrowsFor } from "../lib/levels";
import { HF_LADDER_AXIS_LIBERTY } from "../lib/liberties";
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

/**
 * The self-consistent ladder.
 *
 * Deliberately not a variant of ScreenedLadder: it plots a different axis, and
 * it has a convergence readout that the fitted model has no analogue for. The
 * two are the same shape on screen and different claims underneath, which is
 * the point of letting you switch between them.
 */
function HFLadder({ levels }: { levels: HFLevels }) {
  const orbitals = levels.orbitals;
  // Binding energy, so the log is of a positive number. Every occupied HF
  // orbital is bound; the guard is here because a rung at exactly 0 would
  // silently become -Infinity and take the whole scale with it.
  const bind = orbitals.map((o) => Math.abs(o.energy_ev.value));
  const deepest = Math.max(...bind);
  const shallowest = Math.min(...bind.filter((b) => b > 0), deepest);
  // A quarter-decade of headroom: enough that the shallowest rung is not welded
  // to the frame edge, and deliberately not more. A generous gap above the
  // topmost level reads as "this one is far from ionization" when on a log axis
  // it is the one CLOSEST to it, and the empty band means nothing at all.
  const y = scaleLinear(
    [Math.log10(shallowest) - 0.25, Math.log10(deepest)],
    [40, H - 40],
  );
  const rungX1 = 100;
  const rungX2 = 360;
  const virial = levels.virial_ratio.value;
  return (
    <div className="view-wrap">
      <div className="view-header">
        <span className="plot-title">
          {levels.exchange ? "Hartree-Fock" : "Hartree (no exchange)"} orbital
          energies ε_nl [eV] <Badge provenance={levels.provenance} />
        </span>
        <span className="plot-title">
          · {levels.symbol ?? `Z=${levels.z}`} {levels.config}
          {levels.is_ground ? " (ground)" : " — excited (non-ground)"}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" className="levels-svg">
        {/* The ionization limit cannot be a rung here — see HF_LADDER_AXIS_LIBERTY. */}
        <text x={rungX1} y={16} className="tick" opacity={0.7}>
          ↑ 0 eV (ionization limit) — off the top of a log axis
        </text>
        {orbitals.map((o) => {
          const e = o.energy_ev.value;
          const yr = y(Math.log10(Math.abs(e)));
          return (
            <g key={`${o.n}-${o.l}`}>
              <line
                x1={rungX1} x2={rungX2} y1={yr} y2={yr}
                className="rung" strokeWidth={3}
              />
              <text x={rungX1 - 8} y={yr} dy="0.32em" textAnchor="end" className="tick">
                {o.label}
                <tspan dy="-0.5em">{o.occupancy}</tspan>
              </text>
              <text x={rungX2 + 8} y={yr} dy="0.32em" className="tick">
                {/* Three significant figures across three decades of magnitude. */}
                {e.toPrecision(4)} eV
              </text>
            </g>
          );
        })}
      </svg>
      <p className="caption">
        Self-consistent-field orbital energies (
        {levels.exchange
          ? "APPROXIMATION — see the badge for what Hartree-Fock leaves out, correlation above all"
          : "COUNTERFACTUAL — see the badge; this is not an approximation to the real atom"}
        ). Energy axis is logarithmic in binding energy{" "}
        <Badge provenance={HF_LADDER_AXIS_LIBERTY} /> because the 1s and the
        valence shell differ by more than two decades. Total energy{" "}
        {levels.total_energy_ev.value.toFixed(2)} eV
        {levels.exchange
          ? " is variational, unlike the screened model's sum of orbital energies."
          : " is stationary for this model, but it is not a variational bound on the real atom: a product wavefunction is not antisymmetric, so it is not an admissible trial function for electrons and the theorem does not apply to it."}
      </p>
      {!levels.exchange && levels.exchange_energy_ev !== null && (
        <p className="caption">
          <strong>
            Exchange is worth{" "}
            {Math.abs(levels.exchange_energy_ev.value).toFixed(2)} eV of binding
          </strong>{" "}
          to this atom — the gap between the energy above and the Hartree-Fock
          one, both solved on the same mesh. Exchange binds because same-spin
          electrons keep out of each other's way, so they repel each other less
          than distinguishable ones would.{" "}
          {levels.exchange_energy_ev.value === 0
            ? "Zero here, and that is the answer rather than a missing one: no two electrons in this configuration share a spin, so there is no pair to exchange."
            : "The Pauli occupancies are untouched — nothing has piled into the 1s."}
        </p>
      )}
      <p className="caption">
        <strong>Solve diagnostics</strong> (NUMERICAL — these describe the
        computation, not the atom): {levels.converged ? "converged" : "DID NOT CONVERGE"}{" "}
        in {levels.coarse_iterations} coarse + {levels.iterations} fine SCF
        iterations on {levels.grid_points} radial points. Virial ratio
        −〈V〉/〈T〉 = {virial.toFixed(6)}, which is exactly 2 for a converged
        solution of this Hamiltonian; the departure is a measure of the grid, not
        a property of the element.
      </p>
    </div>
  );
}

// h in eV per MHz (h = 4.135667696e-15 eV·s, times 1e6 Hz/MHz), for turning a
// hyperfine energy split into its transition frequency — the payoff number.
const EV_PER_MHZ = 4.135667696e-9;

export function LevelsView() {
  const {
    n, l, system, fineStructure, dirac, setDirac, bField, setBField,
    eField, setEField, hyperfine, setHyperfine, levels, spectrum,
    loadLevels, loadSpectrum, model, config, hf, hfStatus, loadHF, error,
    exchange,
  } = useAppStore();
  const wantHF = model === "hf";
  useEffect(() => {
    void loadLevels();
    void loadSpectrum();
  }, [system, fineStructure, dirac, bField, eField, hyperfine, loadLevels, loadSpectrum]);
  useEffect(() => {
    // Only under the HF model, and only once per (system, config): the solve
    // costs seconds, so it is not something to fire on the chance it is wanted.
    if (wantHF && hf === null && hfStatus === "idle") void loadHF();
  }, [wantHF, hf, hfStatus, system, config, exchange, loadHF]);

  if (wantHF) {
    if (hfStatus === "error") {
      return (
        <p className="hint-block">
          Hartree-Fock declined this atom: {error ?? "unknown reason"}
        </p>
      );
    }
    if (hf === null) return <p className="hint-block">solving Hartree-Fock (seconds)…</p>;
    return <HFLadder levels={hf} />;
  }
  if (!levels) return <p className="hint-block">loading levels…</p>;
  if (isScreenedLevels(levels)) return <ScreenedLadder levels={levels} />;

  const eMin = levels.gross[0].energy_ev.value;
  const y = scaleLinear([eMin, 0], [H - 40, 24]);
  const rungX1 = 70;
  const rungX2 = 320;
  const arrows = spectrum ? arrowsFor(spectrum.lines, n, l) : [];
  const grossE = new Map(levels.gross.map((g) => [g.n, g.energy_ev.value]));
  const fineForN = levels.fine?.filter((f) => f.n === n) ?? [];
  // hyperfine shell for the selected n; the unavailable case is a single
  // sentinel entry (availability does not depend on n), so fall back to it.
  const hfShells = levels.hyperfine_shells ?? [];
  const hfShell =
    hfShells.find((s) => s.n === n) ??
    (hfShells.length === 1 && !hfShells[0].available ? hfShells[0] : undefined);

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
          <span className="levels-field-hint">
            turn on fine structure to add a magnetic field
          </span>
        )}
        <label className="levels-field">
          F{" "}
          <input
            type="range" min={0} max={100} step={0.5} value={eField}
            onChange={(e) => setEField(Number(e.target.value))}
          />
          {eField > 0 ? ` ${eField.toFixed(1)} MV/m` : " 0 MV/m"}
        </label>
        <label className="levels-model">
          <input
            type="checkbox" checked={hyperfine}
            onChange={(e) => setHyperfine(e.target.checked)}
          />
          hyperfine
        </label>
        {hyperfine && hfShell?.available && hfShell.A && (
          <span className="plot-title">
            · hyperfine of {n}s ({hfShell.nucleus}){" "}
            <Badge provenance={hfShell.A.provenance} />
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
        {eField === 0 && !hyperfine && fineStructure && fineForN.length > 0 &&
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
        {eField > 0 &&
          (() => {
            const gsel = levels.gross.find((g) => g.n === n);
            const subs = gsel?.sublevels ?? [];
            if (subs.length === 0) return null;
            const bohrN = gsel!.energy_ev.value;
            const shifts = subs.map((s) => s.energy_ev.value - bohrN);
            const lo = Math.min(...shifts);
            const hi = Math.max(...shifts);
            const pad = (hi - lo || 1e-9) * 0.15;
            const yz = scaleLinear([lo - pad, hi + pad], [H - 60, 48]);
            const zx1 = 470;
            const zx2 = 610;
            const kMax = Math.max(...subs.map((s) => Math.abs(s.k)));
            return (
              <g>
                <text x={(zx1 + zx2) / 2} y={26} textAnchor="middle" className="tick">
                  n={n} Stark manifold [meV] (APPROXIMATION)
                </text>
                <line x1={zx1} x2={zx2} y1={yz(0)} y2={yz(0)} className="zero" opacity={0.5} />
                {subs.map((s) => {
                  const yS = yz(s.energy_ev.value - bohrN);
                  const label = Math.abs(s.k) === kMax && s.m === 0;
                  return (
                    <g key={`${s.n1}-${s.n2}-${s.m}`}>
                      <line
                        x1={zx1} x2={zx2} y1={yS} y2={yS}
                        className={s.k === 0 ? "rung" : "rung rung-active"}
                        opacity={0.8}
                      />
                      {label && (
                        <text x={zx2 + 6} y={yS} dy="0.32em" className="tick">
                          k={s.k}
                        </text>
                      )}
                    </g>
                  );
                })}
              </g>
            );
          })()}
        {eField === 0 && hyperfine && hfShell &&
          (() => {
            const zx1 = 470;
            const zx2 = 610;
            const titleY = 26;
            if (!hfShell.available) {
              return (
                <text x={(zx1 + zx2) / 2} y={titleY} textAnchor="middle" className="tick">
                  n={n}: no hyperfine (see caption)
                </text>
              );
            }
            const lv = hfShell.levels;
            if (lv.length < 2) {
              return (
                <text x={(zx1 + zx2) / 2} y={titleY} textAnchor="middle" className="tick">
                  {hfShell.nucleus}: I=0, no splitting
                </text>
              );
            }
            const shifts = lv.map((x) => x.shift_ev.value);
            const lo = Math.min(...shifts);
            const hi = Math.max(...shifts);
            const pad = (hi - lo || 1e-9) * 0.25;
            const yz = scaleLinear([lo - pad, hi + pad], [H - 60, 60]);
            const dnuMHz = (hi - lo) / EV_PER_MHZ;
            const is21cm = hfShell.nucleus === "proton" && n === 1;
            return (
              <g>
                <text x={(zx1 + zx2) / 2} y={titleY} textAnchor="middle" className="tick">
                  n={n}s hyperfine ({hfShell.nucleus}) — APPROXIMATION
                </text>
                <line x1={zx1} x2={zx2} y1={yz(0)} y2={yz(0)} className="zero" opacity={0.5} />
                <text x={zx1 - 6} y={yz(0)} dy="0.32em" textAnchor="end" className="tick">
                  centroid
                </text>
                {lv.map((x) => {
                  const yS = yz(x.shift_ev.value);
                  return (
                    <g key={x.F}>
                      <line
                        x1={zx1} x2={zx2} y1={yS} y2={yS}
                        className="rung rung-active"
                      />
                      <text x={zx2 + 6} y={yS} dy="0.32em" className="tick">
                        F={x.F}
                      </text>
                    </g>
                  );
                })}
                <text x={(zx1 + zx2) / 2} y={H - 40} textAnchor="middle" className="tick">
                  Δν = {dnuMHz.toFixed(1)} MHz{is21cm ? " · 21 cm line" : ""}
                </text>
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
        {eField > 0 && (
          <>
            {" "}An electric field splits each n-shell into n² parabolic (n₁,n₂,m)
            sublevels fanned by the electric quantum number k = n₁−n₂. The splitting is
            linear in F, which is hydrogen's accidental l-degeneracy showing itself: a
            first-order shift appears here that non-degenerate atoms (quadratic only) never
            get. Second-order model on the gross shells; the perturbation series is
            asymptotic and breaks down near field ionization, so read the badge.
          </>
        )}
        {hyperfine && hfShell && (
          <>
            {" "}Hyperfine: the nuclear spin I couples to the electron's angular
            momentum J, splitting the {n}s level (J=½) into F = I+J states through the
            Fermi contact interaction.{" "}
            {hfShell.available
              ? hfShell.levels.length > 1
                ? (
                  <>
                    The right column magnifies the split and gives the transition
                    frequency Δν.{" "}
                    {hfShell.nucleus === "proton" && n === 1
                      ? "This 1s F=1→0 line is 1420 MHz, the 21 cm line — radio astronomy's fingerprint of neutral hydrogen. "
                      : ""}
                    s-states only (contact term); the l&gt;0 orbital+dipolar channel is
                    deferred. Non-relativistic — bound-state QED and nuclear structure
                    (~0.01%) are omitted, see the badge.
                  </>
                )
                : hfShell.note
              : hfShell.reason}
          </>
        )}
      </p>
    </div>
  );
}
