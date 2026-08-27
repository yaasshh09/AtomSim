import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import { isScreenedLevels } from "../api/client";
import type { HFLevels, PauliCollapse, ScreenedLevels } from "../api/types";
import { describeEField, describeField, VIEW_LEADS } from "../lib/explain";
import {
  arrowsFor,
  detailMode,
  detailState,
  spreadLabels,
  type DetailMode,
} from "../lib/levels";
import { HF_LADDER_AXIS_LIBERTY } from "../lib/liberties";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { Disclosure } from "./Disclosure";
import { Choice, ControlGroup, Slider, Toggle } from "./Field";
import { ViewIntro } from "./ViewIntro";

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
      <ViewIntro
        lead={{
          title: "Energy levels of a screened atom",
          lead:
            "Each rung is one subshell's energy in the fitted central field " +
            "that stands in for all the other electrons. Solid rungs hold " +
            "electrons; dashed ones are empty and waiting.",
          notice:
            "Watch s sit below p below d at the same n. Hydrogen has them on " +
            "top of each other; screening is what pulls them apart.",
        }}
        badge={<Badge provenance={orbitals[0].energy.provenance} />}
      >
        <p className="view-intro-config">
          {levels.config}
          {levels.is_ground ? " · ground configuration" : " · excited, not the ground state"}
        </p>
      </ViewIntro>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" className="levels-svg">
        {/* ionization threshold */}
        <line x1={rungX1} x2={rungX2} y1={y(0)} y2={y(0)} className="zero" />
        <text x={rungX2 + 8} y={y(0)} dy="0.32em" className="tick">
          0: ionization limit
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
        Total energy {levels.total_energy_ev.value.toFixed(2)} eV.
      </p>
      <Disclosure summary="What this model is doing, and what it is not">
        <p className="caption">
          Independent-particle orbital energies in the Green-Sellin-Zachor screened
          central field (APPROXIMATION). Screening lifts the hydrogenic l-degeneracy
          (s below p below d for a given n). Filled subshells are solid with their
          occupancy; virtual orbitals are dashed. Total energy{" "}
          {levels.total_energy_ev.value.toFixed(2)} eV is a sum of occupancy-weighted
          orbital energies, not a variational total, see the badge.
        </p>
      </Disclosure>
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
/**
 * The real atom beside the collapsed one, as two numbers and their meaning.
 *
 * Both halves come off one server-side comparison, so this component never
 * subtracts anything: differencing two payloads here would be free to mix a
 * fine solve with a coarse one and call the remainder physics.
 */
function PauliComparison({ collapse }: { collapse: PauliCollapse }) {
  const gained = Math.abs(collapse.binding_change_ev.value);
  // Exactly zero for an atom whose ground configuration is already 1s^N, which
  // is helium and nothing else here. Not a missing number - the answer.
  const noop = collapse.binding_change_ev.value === 0;
  const shrink = collapse.radius_ratio.value;
  return (
    <>
      <p className="caption">
        <strong>
          {noop
            ? "The exclusion principle costs this atom nothing"
            : `The exclusion principle costs this atom ${gained.toPrecision(4)} eV of binding`}
        </strong>{" "}
        {noop ? (
          <>
, its ground configuration is already 1s
            <sup>{2}</sup>, so lifting the cap has nothing to lift. Helium is
            the calibration case rather than the demonstration: the two solves
            agree to the last bit because they are the same solve.
          </>
        ) : (
          <>
, the real atom ({collapse.real_config}) sits at{" "}
            {collapse.real_total_energy_ev.value.toPrecision(6)} eV and the
            collapsed one at{" "}
            {(collapse.real_total_energy_ev.value + collapse.binding_change_ev.value)
              .toPrecision(6)}{" "}
            eV, both on the same mesh. Nothing holds the electrons out of the
            deep well once the cap is gone.
          </>
        )}
      </p>
      <p className="caption">
        <strong>And it is what makes the atom big.</strong> ⟨r⟩ falls from{" "}
        {collapse.real_radius.value.toFixed(3)} to{" "}
        {collapse.collapsed_radius.value.toFixed(3)} bohr, a factor of{" "}
        {shrink.toFixed(3)}. With the cap on, atomic size rises and falls across
        a period, and that oscillation is the periodic table. With it off, ⟨r⟩
        decreases forever as Z grows, every element is a smaller version of the
        last one, and there is no chemistry to have.
      </p>
      <Disclosure summary="Checked against a closed form, not only against itself">
        <p className="caption">
          N electrons in a single 1s of exponent ζ minimize at ζ* ={" "}
          {collapse.variational_zeta.value.toFixed(4)}, giving{" "}
          {collapse.variational_energy_ev.value.toPrecision(6)} eV. The solve above
          optimizes the radial function rather than an exponent, so it searches a
          larger space and has to land at or below that number, and does. The
          formula is textbook: at Z = N = 2 it is the variational helium result.
        </p>
      </Disclosure>
    </>
  );
}

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
  const modelName = !levels.pauli
    ? "No Pauli exclusion (1s^N)"
    : levels.exchange
      ? "Hartree-Fock"
      : "Hartree (no exchange)";
  return (
    <div className="view-wrap">
      <ViewIntro
        lead={{
          title: `Energy levels: ${modelName}`,
          lead:
            "Each rung is one subshell, solved self-consistently: every " +
            "electron moves in the field of all the others, and the answer " +
            "has to reproduce itself.",
          notice:
            "The 1s sits more than two decades below the valence shell, so " +
            "the axis is logarithmic in binding energy. Deeper is further down.",
        }}
        badge={<Badge provenance={levels.provenance} />}
      >
        <p className="view-intro-config">
          {levels.symbol ?? `Z=${levels.z}`} {levels.config}
          {levels.is_ground
            ? levels.pauli
              ? " · ground configuration"
              : " · ground, with no cap left to obey"
            : " · excited, not the ground state"}
        </p>
      </ViewIntro>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" className="levels-svg">
        {/* The ionization limit cannot be a rung here, see HF_LADDER_AXIS_LIBERTY. */}
        <text x={rungX1} y={16} className="tick" opacity={0.7}>
          ↑ 0 eV (ionization limit), off the top of a log axis
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
        Total energy {levels.total_energy_ev.value.toFixed(2)} eV
        {levels.exchange
          ? ", and it is variational: the real atom is at or below it."
          : ", stationary for this model but not a bound on the real atom."}{" "}
        <Badge provenance={HF_LADDER_AXIS_LIBERTY} />
      </p>
      {!levels.exchange && levels.exchange_energy_ev !== null && (
        <p className="caption">
          <strong>
            Exchange is worth{" "}
            {Math.abs(levels.exchange_energy_ev.value).toFixed(2)} eV of binding
          </strong>{" "}
          to this atom, the gap between the energy above and the Hartree-Fock
          one, both solved on the same mesh. Exchange binds because same-spin
          electrons keep out of each other's way, so they repel each other less
          than distinguishable ones would.{" "}
          {levels.exchange_energy_ev.value === 0
            ? "Zero here, and that is the answer rather than a missing one: no two electrons in this configuration share a spin, so there is no pair to exchange."
            : "The Pauli occupancies are untouched, nothing has piled into the 1s."}
        </p>
      )}
      {levels.collapse !== null && <PauliComparison collapse={levels.collapse} />}
      <Disclosure summary="What this model leaves out, and why the axis is logarithmic">
        <p className="caption">
          Self-consistent-field orbital energies (
          {levels.exchange
            ? "APPROXIMATION, see the badge for what Hartree-Fock leaves out, correlation above all"
            : "COUNTERFACTUAL, see the badge; this is not an approximation to the real atom"}
          ). Energy axis is logarithmic in binding energy{" "}
          <Badge provenance={HF_LADDER_AXIS_LIBERTY} /> because the 1s and the
          valence shell differ by more than two decades. Total energy{" "}
          {levels.total_energy_ev.value.toFixed(2)} eV
          {levels.exchange
            ? " is variational, unlike the screened model's sum of orbital energies."
            : " is stationary for this model, but it is not a variational bound on the real atom: a product wavefunction is not antisymmetric, so it is not an admissible trial function for electrons and the theorem does not apply to it."}
        </p>
      </Disclosure>
      <Disclosure summary="Solve diagnostics: how well the computation converged">
        <p className="caption">
          <strong>These describe the computation, not the atom</strong> (NUMERICAL):{" "}
          {levels.converged ? "converged" : "DID NOT CONVERGE"} in{" "}
          {levels.coarse_iterations} coarse + {levels.iterations} fine SCF
          iterations on {levels.grid_points} radial points. Virial ratio
          −〈V〉/〈T〉 = {virial.toFixed(6)}, which is exactly 2 for a converged
          solution of this Hamiltonian; the departure is a measure of the grid, not
          a property of the element.
        </p>
      </Disclosure>
    </div>
  );
}

// h in eV per MHz (h = 4.135667696e-15 eV·s, times 1e6 Hz/MHz), for turning a
// hyperfine energy split into its transition frequency, the payoff number.
const EV_PER_MHZ = 4.135667696e-9;

export function LevelsView() {
  const {
    n, l, system, fineStructure, dirac, setDirac, bField, setBField,
    eField, setEField, hyperfine, setHyperfine, levels, spectrum,
    loadLevels, loadSpectrum, model, config, hf, hfStatus, loadHF, error,
    exchange, pauli, setFineStructure, setQuantumNumbers,
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
  }, [wantHF, hf, hfStatus, system, config, exchange, pauli, loadHF]);

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
  /* Only transitions that actually cross a shell can be drawn on this ladder.
     A within-n fine-structure component (3p→3s, out at 9 cm) has both ends on
     the same rung, so it rendered as a zero-length arrow with a nine-digit
     wavelength printed across the middle of the plot. It is real physics and
     it is in the spectrum view where the axis can hold it; here it is a label
     with nothing under it, so it is counted and named rather than drawn. */
  const allArrows = spectrum ? arrowsFor(spectrum.lines, n, l) : [];
  const arrows = allArrows.filter((a) => a.n_upper !== a.n_lower);
  const withinN = allArrows.length - arrows.length;
  const grossE = new Map(levels.gross.map((g) => [g.n, g.energy_ev.value]));

  /* Rungs crowd toward the ionization limit because the energies go as -1/n²,
     so at n >= 4 the labels printed through one another. The rungs stay where
     the physics puts them and the text is nudged apart, with a leader line
     back to the rung it belongs to. */
  const rungY = levels.gross.map((g) => y(g.energy_ev.value));
  const labelY = spreadLabels(rungY, 13, 20, H - 28);
  const fineForN = levels.fine?.filter((f) => f.n === n) ?? [];
  // hyperfine shell for the selected n; the unavailable case is a single
  // sentinel entry (availability does not depend on n), so fall back to it.
  const hfShells = levels.hyperfine_shells ?? [];
  const hfShell =
    hfShells.find((s) => s.n === n) ??
    (hfShells.length === 1 && !hfShells[0].available ? hfShells[0] : undefined);

  /* One picker owns the magnifier, so an option that would do nothing says so
     on its face instead of being silently outranked. See lib/levels.ts. */
  const current = { fineStructure, bField, eField, hyperfine };
  const mode = detailMode(current);
  const applyMode = (next: DetailMode) => {
    const s = detailState(next, current);
    // Order matters only in that each setter clears `levels`; the effect above
    // refetches once React has batched them.
    if (s.fineStructure !== fineStructure) setFineStructure(s.fineStructure);
    if (s.bField !== bField) setBField(s.bField);
    if (s.eField !== eField) setEField(s.eField);
    if (s.hyperfine !== hyperfine) setHyperfine(s.hyperfine);
  };
  const hyperfineBlocked = hfShell !== undefined && !hfShell.available;

  return (
    <div className="view-wrap">
      <ViewIntro
        lead={VIEW_LEADS.levels}
        badge={<Badge provenance={levels.gross[0].energy.provenance} />}
      >
        <Disclosure summary="Why the rungs crowd together going up">
          <p className="caption">
            The energies go as −1/n², so the gap from n=1 to n=2 is far larger
            than the gap from n=5 to n=6. Every rung above the first is bunched
            toward the ionization limit at 0, where the electron is free. That
            crowding is why hydrogen's spectral series each pile up toward a
            short-wavelength edge instead of spreading out evenly.
          </p>
        </Disclosure>
      </ViewIntro>

      <ControlGroup
        title="Magnify one rung"
        hint="The right-hand column zooms in on what splits the selected shell. Only one at a time: they are four different magnifications of the same column."
        tone={mode === "none" ? "plain" : "active"}
      >
        <Choice<DetailMode>
          legend="detail"
          value={mode}
          onChange={applyMode}
          tourId="levels-detail"
          options={[
            {
              value: "none",
              label: "just the ladder",
              hint: "Gross levels only, exact under a reduced-mass Bohr model.",
            },
            {
              value: "fine",
              label: "fine structure",
              hint:
                "Relativity and spin-orbit coupling split each shell by about " +
                "a part in 100,000. The column magnifies it.",
            },
            {
              value: "zeeman",
              label: "magnetic field",
              hint: "A magnetic field splits each j-level into its 2j+1 orientations.",
            },
            {
              value: "stark",
              label: "electric field",
              hint: "An electric field fans each shell into n² parabolic states.",
            },
            {
              value: "hyperfine",
              label: "hyperfine",
              hint: hyperfineBlocked
                ? undefined
                : "The nucleus's own spin splits the s-states. This is where the 21 cm line comes from.",
              disabled: hyperfineBlocked,
              disabledReason: hfShell?.reason ?? undefined,
            },
          ]}
        />
        {mode === "fine" && (
          <Toggle
            label="Solve it exactly (Dirac) instead of to order α²"
            checked={dirac}
            onChange={setDirac}
            why={
              dirac
                ? "Dirac is exact for a point nucleus: the energy depends on n and j only, so 2s½ and 2p½ land on each other exactly. Reality splits them by the Lamb shift, which this model leaves out."
                : "The α² perturbation. Accurate to about a part in 10⁴ of the shift itself; switch to Dirac for the closed form."
            }
            tourId="dirac-toggle"
          />
        )}
        {mode === "zeeman" && (
          <Slider
            label="magnetic field B"
            readout={`${bField.toFixed(1)} T`}
            anchor={`${describeField(bField)} · µ_B·B = ${(
              (bField * 0.5) / 2.35051757e5 * 27.211386245e6
            ).toFixed(1)} µeV`}
            min={0}
            max={20}
            step={0.1}
            value={bField}
            atRest={bField === 0}
            onChange={setBField}
          />
        )}
        {mode === "stark" && (
          <Slider
            label="electric field F"
            readout={`${eField.toFixed(1)} MV/m`}
            anchor={describeEField(eField)}
            min={0}
            max={100}
            step={0.5}
            value={eField}
            atRest={eField === 0}
            onChange={setEField}
          />
        )}
      </ControlGroup>

      <p className="ladder-hint">
        Rungs are clickable: pick one to move the whole app to that shell.
        {arrows.length > 0
          ? ` The arrows are the ${arrows.length} transition${arrows.length === 1 ? "" : "s"} out of the state selected in the rail, labelled with the light they emit.`
          : ""}
        {withinN > 0
          ? ` ${withinN} more transition${withinN === 1 ? " starts" : "s start"} and end${withinN === 1 ? "s" : ""} on this same shell, so ${withinN === 1 ? "it has" : "they have"} no arrow to draw here; ${withinN === 1 ? "it is" : "they are"} out at microwave wavelengths in the spectrum view.`
          : ""}
      </p>

      <svg viewBox={`0 0 ${W} ${H}`} role="img" className="levels-svg">
        {levels.gross.map((g, i) => {
          const yr = rungY[i];
          const yl = labelY[i];
          const nudged = Math.abs(yl - yr) > 1;
          const pick = () => setQuantumNumbers(g.n, Math.min(l, g.n - 1), 0);
          return (
            <g
              key={g.n}
              className="rung-hit"
              role="button"
              tabIndex={0}
              aria-label={`select shell n=${g.n}`}
              onClick={pick}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  pick();
                }
              }}
            >
              {/* An invisible band on the label row, so the click target is a
                  row rather than a 1 px line. Without it the rungs are
                  technically clickable and practically not. */}
              <rect
                x={rungX1 - 40} y={Math.min(yr, yl) - 9}
                width={rungX2 - rungX1 + 100}
                height={Math.abs(yl - yr) + 18}
                className="rung-hit-area"
              />
              <line
                x1={rungX1} x2={rungX2} y1={yr} y2={yr}
                className={g.n === n ? "rung rung-active" : "rung"}
              />
              {nudged && (
                <>
                  <line x1={rungX1 - 30} x2={rungX1 - 6} y1={yl} y2={yr} className="leader" />
                  <line x1={rungX2 + 4} x2={rungX2 + 26} y1={yr} y2={yl} className="leader" />
                </>
              )}
              <text
                x={rungX1 - 32} y={yl} dy="0.32em"
                textAnchor="end" className="tick"
              >
                n={g.n}
              </text>
              <text x={rungX2 + 30} y={yl} dy="0.32em" className="tick">
                {g.energy_ev.value.toFixed(2)} eV · 2n²={g.degeneracy}
                {g.n === 1 ? " · ground state" : ""}
              </text>
            </g>
          );
        })}
        {arrows.map((a, i) => {
          if (!grossE.has(a.n_upper) || !grossE.has(a.n_lower)) return null;
          const ax = rungX1 + 30 + i * 26;
          const yTop = y(grossE.get(a.n_upper) ?? 0);
          const yBot = y(grossE.get(a.n_lower) ?? 0);
          return (
            <g key={`${a.n_lower}-${a.l_lower}-${i}`} className="arrow">
              <line x1={ax} x2={ax} y1={yTop} y2={yBot - 6} />
              <path d={`M${ax - 4},${yBot - 8} L${ax + 4},${yBot - 8} L${ax},${yBot} Z`} />
              {/* Staggered over three rows. The arrows are 26 px apart and a
                  wavelength label is about twice that wide, so with fine
                  structure on (which gives two arrows per shell pair at nearly
                  the same wavelength) every label printed over its neighbour. */}
              <text
                x={ax + 4}
                y={(yTop + yBot) / 2 + ((i % 3) - 1) * 13}
                className="tick arrow-label"
              >
                {a.wavelength_nm.value.toFixed(0)} nm
              </text>
            </g>
          );
        })}
        {mode === "fine" || mode === "zeeman"
          ? fineForN.length > 0 &&
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
                    ? `n=${n} Zeeman split [µeV]: APPROXIMATION`
                    : `n=${n} shifts [µeV], zoomed, ${dirac ? "EXACT" : "APPROXIMATION"}`}
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
          })()
          : null}
        {mode === "stark" &&
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
        {mode === "hyperfine" && hfShell &&
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
                  n={n}s hyperfine ({hfShell.nucleus}): APPROXIMATION
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

      {/* The headline for whatever the magnifier is showing, in one sentence.
          The full account is in the disclosure under it, unchanged. */}
      {mode === "hyperfine" && hfShell?.available && hfShell.levels.length > 1 && (
        <p className="caption">
          <strong>
            Δν ={" "}
            {(
              (Math.max(...hfShell.levels.map((x) => x.shift_ev.value)) -
                Math.min(...hfShell.levels.map((x) => x.shift_ev.value))) /
              EV_PER_MHZ
            ).toFixed(1)}{" "}
            MHz
          </strong>{" "}
          between the F states of {n}s.
          {hfShell.nucleus === "proton" && n === 1
            ? " That is the 21 cm line: radio astronomy's fingerprint of neutral hydrogen, and the reason we can map the galaxy's gas at all."
            : ""}{" "}
          <Badge provenance={hfShell.A!.provenance} />
        </p>
      )}

      <Disclosure summary="What the two scales mean, and what is left out" tone="caveat">
        <p className="caption">
          Gross levels are reduced-mass exact. The right column magnifies the{" "}
          {dirac ? "relativistic" : "α²"} shifts of the selected n, the two scales differ by ~10⁵
          and are labeled, never blended.{" "}
          {dirac
            ? "Dirac is exact for a point nucleus: the energy depends on n and j only, so 2s₁/₂ and 2p₁/₂ coincide exactly. Reality splits them by the Lamb shift (QED), which this model deliberately omits, see the badge assumptions."
            : "States with equal j coincide at this order (e.g. 2s₁/₂ and 2p₁/₂, the Lamb shift is beyond α² and honestly absent here)."}
          {mode === "zeeman" && (
            <>
              {" "}A magnetic field splits each j-level into 2j+1 m_j sublevels (anomalous
              Zeeman, spacing g_J·µ_B·B); as B rises they reorganize toward the Paschen-Back
              pattern where (m_l, m_s) become the good labels. Linear model, the diamagnetic
              B² term is omitted.
            </>
          )}
          {mode === "stark" && (
            <>
              {" "}An electric field splits each n-shell into n² parabolic (n₁,n₂,m)
              sublevels fanned by the electric quantum number k = n₁−n₂. The splitting is
              linear in F, which is hydrogen's accidental l-degeneracy showing itself: a
              first-order shift appears here that non-degenerate atoms (quadratic only) never
              get. Second-order model on the gross shells; the perturbation series is
              asymptotic and breaks down near field ionization, so read the badge.
            </>
          )}
          {mode === "hyperfine" && hfShell && (
            <>
              {" "}Hyperfine: the nuclear spin I couples to the electron's angular
              momentum J, splitting the {n}s level (J=½) into F = I+J states through the
              Fermi contact interaction.{" "}
              {hfShell.available
                ? hfShell.levels.length > 1
                  ? "s-states only (contact term); the l>0 orbital+dipolar channel is deferred. Non-relativistic, bound-state QED and nuclear structure (~0.01%) are omitted, see the badge."
                  : hfShell.note
                : hfShell.reason}
            </>
          )}
        </p>
      </Disclosure>
    </div>
  );
}
