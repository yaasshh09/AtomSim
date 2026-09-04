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
import { Notation, mathTspans } from "../lib/mathText";
import { useAppStore } from "../state/store";
import { withinView } from "../lib/zoom";
import { Badge } from "./Badge";
import { Disclosure } from "./Disclosure";
import { Choice, ControlGroup, Slider, Toggle } from "./Field";
import { OffWindowMarks, usePlotZoom, ZoomControls } from "./PlotZoom";
import { ViewIntro } from "./ViewIntro";

const W = 680;
const H = 460;
/** Where the hydrogen ladder's energy axis is drawn, top rung to bottom. */
const LADDER_RANGE: [number, number] = [H - 40, 24];

function ScreenedLadder({ levels }: { levels: ScreenedLevels }) {
  const all = levels.orbitals;
  const es = all.map((o) => o.energy_ev.value);
  const eMin = Math.min(...es);
  const eMax = Math.max(...es, 0); // the ionization threshold at 0 belongs on the scale
  const yRange: [number, number] = [H - 40, 24];
  const zoom = usePlotZoom({
    width: W,
    height: H,
    y: { domain: [eMin, eMax], range: yRange },
  });
  const y = scaleLinear(zoom.y, yRange);
  /* Only the rungs inside the window get their labels spread. Spreading a
     label for a subshell three screens above the frame would push every
     visible one aside to make room for something nobody can see, which is the
     opposite of what zooming into a crowded ladder is for. */
  const orbitals = withinView(all, (o) => o.energy_ev.value, zoom.y);
  const rungX1 = 90;
  const rungX2 = 340;
  /* Neon's virtual orbitals sit within a couple of eV of the ionization limit,
     which printed them straight through one another and through the limit's
     own label. The ionization line goes into the spread rather than staying
     out of it, because it is the row everything else crowds against: holding
     it fixed would have fixed the rungs and kept the collision that matters. */
  const limitShown = 0 >= Math.min(...zoom.y) && 0 <= Math.max(...zoom.y);
  const rowY = [
    ...(limitShown ? [y(0)] : []),
    ...orbitals.map((o) => y(o.energy_ev.value)),
  ];
  const labelY = spreadLabels(rowY, 13, 18, H - 26);
  const limitLabelY = limitShown ? labelY[0] : 0;
  const rungLabelY = limitShown ? labelY.slice(1) : labelY;
  return (
    <div className="view-wrap">
      <ViewIntro
        lead={{
          title: "Energy levels of a screened atom",
          lead:
            "Each rung is one subshell's energy in a fitted central field that " +
            "stands in for all the other electrons. Solid rungs hold " +
            "electrons. Dashed ones are empty, waiting for one.",
          notice:
            "Watch s sit below p below d at the same n. Hydrogen stacks those " +
            "on top of each other; screening is what pulls them apart.",
        }}
        badge={<Badge provenance={all[0].energy.provenance} />}
      >
        <p className="view-intro-config">
          {levels.config}
          {levels.is_ground ? " · ground configuration" : " · excited, not the ground state"}
        </p>
      </ViewIntro>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        className={`levels-svg plot-zoomable${zoom.dragging ? " plot-panning" : ""}`}
        ref={zoom.ref}
        {...zoom.handlers}
      >
        <OffWindowMarks
          above={all.filter((o) => o.energy_ev.value > Math.max(...zoom.y)).length}
          below={all.filter((o) => o.energy_ev.value < Math.min(...zoom.y)).length}
          x={rungX1} top={18} bottom={H - 14} noun="subshell"
        />
        {/* the ionization threshold */}
        {limitShown && (
          <>
            <line x1={rungX1} x2={rungX2} y1={y(0)} y2={y(0)} className="zero" />
            {Math.abs(limitLabelY - y(0)) > 1 && (
              <line
                x1={rungX2 + 4} x2={rungX2 + 26} y1={y(0)} y2={limitLabelY}
                className="leader"
              />
            )}
            <text x={rungX2 + 30} y={limitLabelY} dy="0.32em" className="tick">
              0: ionization limit
            </text>
          </>
        )}
        {orbitals.map((o, i) => {
          const filled = o.occupancy > 0;
          const yr = y(o.energy_ev.value);
          const yl = rungLabelY[i];
          const nudged = Math.abs(yl - yr) > 1;
          return (
            <g key={`${o.n}-${o.l}`}>
              <line
                x1={rungX1} x2={rungX2} y1={yr} y2={yr}
                className="rung"
                strokeWidth={filled ? 3 : 1.5}
                strokeDasharray={filled ? undefined : "4 4"}
                opacity={filled ? 1 : 0.5}
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
                {o.label}
                {filled ? <tspan dy="-0.5em">{o.occupancy}</tspan> : ""}
              </text>
              <text x={rungX2 + 30} y={yl} dy="0.32em" className="tick">
                {o.energy_ev.value.toFixed(2)} eV
                {filled ? "" : " · virtual"}
              </text>
            </g>
          );
        })}
      </svg>
      <ZoomControls zoom={zoom} what="the energy axis" />
      <p className="caption">
        Total energy {levels.total_energy_ev.value.toFixed(2)} eV.
      </p>
      <Disclosure summary="What this model is doing, and what it is not">
        <p className="caption">
          These are independent-particle orbital energies in the
          Green-Sellin-Zachor screened central field (APPROXIMATION). Screening
          lifts the hydrogenic l-degeneracy, which is why s sits below p sits
          below d at the same n. Filled subshells are solid and carry their
          occupancy; virtual ones are dashed. The total above,{" "}
          {levels.total_energy_ev.value.toFixed(2)} eV, is a sum of
          occupancy-weighted orbital energies, not a variational total. The
          badge says the rest.
        </p>
      </Disclosure>
    </div>
  );
}

/**
 * The self-consistent ladder.
 *
 * Deliberately not a variant of ScreenedLadder: it plots a different axis and
 * carries a convergence readout the fitted model has no analogue for. The two
 * are the same shape on screen and different claims underneath, which is the
 * whole point of being able to switch between them.
 */
/**
 * The real atom beside the collapsed one, as two numbers and what they mean.
 *
 * Both halves come off a single server-side comparison, so nothing is
 * subtracted here. Differencing two payloads would leave this component free
 * to mix a fine solve with a coarse one and call the remainder physics.
 */
function PauliComparison({ collapse }: { collapse: PauliCollapse }) {
  const gained = Math.abs(collapse.binding_change_ev.value);
  // Exactly zero for an atom whose ground configuration is already 1s^N, which
  // here is helium and nothing else. Not a missing number: the answer.
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
            <sup>{2}</sup>, so lifting the cap lifts nothing. Helium is the
            calibration case here, not the demonstration: the two solves agree
            to the last bit because they are the same solve.
          </>
        ) : (
          <>
, the real atom ({collapse.real_config}) sits at{" "}
            {collapse.real_total_energy_ev.value.toPrecision(6)} eV and the
            collapsed one at{" "}
            {(collapse.real_total_energy_ev.value + collapse.binding_change_ev.value)
              .toPrecision(6)}{" "}
            eV, both on the same mesh. Drop the cap and nothing holds the
            electrons out of the deep well.
          </>
        )}
      </p>
      <p className="caption">
        <strong>And it is what makes the atom big.</strong> ⟨r⟩ falls from{" "}
        {collapse.real_radius.value.toFixed(3)} to{" "}
        {collapse.collapsed_radius.value.toFixed(3)} bohr, a factor of{" "}
        {shrink.toFixed(3)}. With the cap on, atomic size rises and falls across
        a period, and that rise and fall is the periodic table. With it off, ⟨r⟩
        just shrinks forever as Z grows: every element a smaller copy of the
        last one, and no chemistry left to do.
      </p>
      <Disclosure summary="Checked against a closed form, not only against itself">
        <p className="caption">
          N electrons in a single 1s of exponent ζ minimize at ζ* ={" "}
          {collapse.variational_zeta.value.toFixed(4)}, giving{" "}
          {collapse.variational_energy_ev.value.toPrecision(6)} eV. The solve
          above optimizes the whole radial function rather than one exponent, so
          it searches a bigger space and has to land at or below that number.
          It does. The formula is textbook: at Z = N = 2 it is the variational
          helium result.
        </p>
      </Disclosure>
    </>
  );
}

function HFLadder({ levels }: { levels: HFLevels }) {
  const orbitals = levels.orbitals;
  // Binding energy, so the log is of a positive number. Every occupied HF
  // orbital is bound; the guard stays because a rung at exactly 0 would
  // silently become -Infinity and take the whole scale with it.
  const bind = orbitals.map((o) => Math.abs(o.energy_ev.value));
  const deepest = Math.max(...bind);
  const shallowest = Math.min(...bind.filter((b) => b > 0), deepest);
  // A quarter-decade of headroom: enough that the shallowest rung is not
  // welded to the frame edge, and deliberately no more. A generous gap above
  // the topmost level reads as "this one is far from ionization" when on a log
  // axis it is the one CLOSEST to it, and the empty band means nothing.
  /* Zoomed in log-binding-energy space, which is the space the axis is drawn
     in: one wheel notch then covers the same fraction of the picture wherever
     it lands, rather than a hundredth of a decade at the top and a whole one
     at the bottom. */
  const yRange: [number, number] = [40, H - 40];
  const yFull: [number, number] = [Math.log10(shallowest) - 0.25, Math.log10(deepest)];
  const zoom = usePlotZoom({ width: W, height: H, y: { domain: yFull, range: yRange } });
  const y = scaleLinear(zoom.y, yRange);
  const shown = withinView(orbitals, (o) => Math.log10(Math.abs(o.energy_ev.value)), zoom.y);
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
            "Each rung is one subshell, solved self-consistently: every electron " +
            "moves in the field of all the others, and the answer has to " +
            "reproduce itself.",
          notice:
            "The 1s sits more than two decades below the valence shell, so the " +
            "axis is logarithmic in binding energy. Deeper means further down.",
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
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        className={`levels-svg plot-zoomable${zoom.dragging ? " plot-panning" : ""}`}
        ref={zoom.ref}
        {...zoom.handlers}
      >
        {/* The ionization limit cannot be a rung here, see HF_LADDER_AXIS_LIBERTY. */}
        <text x={rungX1} y={16} className="tick" opacity={0.7}>
          ↑ 0 eV (ionization limit), off the top of a log axis
        </text>
        <OffWindowMarks
          above={orbitals.filter(
            (o) => Math.log10(Math.abs(o.energy_ev.value)) < Math.min(...zoom.y),
          ).length}
          below={orbitals.filter(
            (o) => Math.log10(Math.abs(o.energy_ev.value)) > Math.max(...zoom.y),
          ).length}
          x={rungX1} top={30} bottom={H - 14} noun="subshell"
        />
        {shown.map((o) => {
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
                {/* Three significant figures, across three decades of magnitude. */}
                {e.toPrecision(4)} eV
              </text>
            </g>
          );
        })}
      </svg>
      <ZoomControls zoom={zoom} what="the binding-energy axis" />
      <p className="caption">
        Total energy {levels.total_energy_ev.value.toFixed(2)} eV
        {levels.exchange
          ? ", and it is variational: the real atom sits at or below it."
          : ", stationary for this model, but not a bound on the real atom."}{" "}
        <Badge provenance={HF_LADDER_AXIS_LIBERTY} />
      </p>
      {!levels.exchange && levels.exchange_energy_ev !== null && (
        <p className="caption">
          <strong>
            Exchange is worth{" "}
            {Math.abs(levels.exchange_energy_ev.value).toFixed(2)} eV of binding
          </strong>{" "}
          to this atom: the gap between the energy above and the Hartree-Fock
          one, both solved on the same mesh. Exchange binds because same-spin
          electrons keep out of each other's way, so they repel each other less
          than distinguishable ones would.{" "}
          {levels.exchange_energy_ev.value === 0
            ? "Zero here, and that is the answer rather than a missing number: no two electrons in this configuration share a spin, so there is no pair to exchange."
            : "The Pauli occupancies are untouched, so nothing has piled into the 1s."}
        </p>
      )}
      {levels.collapse !== null && <PauliComparison collapse={levels.collapse} />}
      <Disclosure summary="What this model leaves out, and why the axis is logarithmic">
        <p className="caption">
          These are self-consistent-field orbital energies (
          {levels.exchange
            ? "APPROXIMATION, and the badge lists what Hartree-Fock leaves out, correlation above all"
            : "COUNTERFACTUAL, see the badge; this is not an approximation to the real atom"}
          ). The energy axis is logarithmic in binding energy{" "}
          <Badge provenance={HF_LADDER_AXIS_LIBERTY} /> because the 1s and the
          valence shell differ by more than two decades. The total above,{" "}
          {levels.total_energy_ev.value.toFixed(2)} eV,
          {levels.exchange
            ? " is variational, unlike the screened model's sum of orbital energies."
            : " is stationary for this model, but it is not a variational bound on the real atom: a product wavefunction is not antisymmetric, so it is not an admissible trial function for electrons, and the theorem simply does not apply to it."}
        </p>
      </Disclosure>
      <Disclosure summary="Solve diagnostics: how well the computation converged">
        <p className="caption">
          <strong>These describe the computation, not the atom</strong>{" "}
          (NUMERICAL): the solve{" "}
          {levels.converged ? "converged" : "DID NOT CONVERGE"} in{" "}
          {levels.coarse_iterations} coarse + {levels.iterations} fine SCF
          iterations on {levels.grid_points} radial points. The virial ratio
          −〈V〉/〈T〉 = {virial.toFixed(6)}, which is exactly 2 for a converged
          solution of this Hamiltonian. How far it sits from 2 measures the
          grid, not the element.
        </p>
      </Disclosure>
    </div>
  );
}

// h in eV per MHz (h = 4.135667696e-15 eV·s, times 1e6 Hz/MHz). This is what
// turns a hyperfine energy split into its transition frequency, which is the
// number worth printing.
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
    // costs seconds, so it never fires on the chance it is wanted.
    if (wantHF && hf === null && hfStatus === "idle") void loadHF();
  }, [wantHF, hf, hfStatus, system, config, exchange, pauli, loadHF]);

  /* Above the returns below, because it is a hook and they are conditional.
     The ladder it zooms exists only on the hydrogenic branch, so on every
     other one it holds a placeholder window that nothing draws. */
  const ladderZoom = usePlotZoom({
    width: W,
    height: H,
    y: {
      domain:
        levels && !isScreenedLevels(levels)
          ? [levels.gross[0].energy_ev.value, 0]
          : [-13.6, 0],
      range: LADDER_RANGE,
    },
  });

  if (wantHF) {
    if (hfStatus === "error") {
      return (
        <p className="hint-block">
          Hartree-Fock could not solve this atom: {error ?? "unknown reason"}
        </p>
      );
    }
    if (hf === null)
      return <p className="hint-block">Solving Hartree-Fock, this takes a few seconds…</p>;
    return <HFLadder levels={hf} />;
  }
  if (!levels) return <p className="hint-block">Loading the levels…</p>;
  if (isScreenedLevels(levels)) return <ScreenedLadder levels={levels} />;

  const y = scaleLinear(ladderZoom.y, LADDER_RANGE);
  const rungX1 = 70;
  const rungX2 = 320;
  /* Only transitions that actually cross a shell can be drawn on this ladder.
     A within-n fine-structure component (3p→3s, out at 9 cm) has both ends on
     the same rung, and it once rendered as a zero-length arrow with a
     nine-digit wavelength printed across the middle of the plot. It is real
     physics, and it stays in the spectrum view where the axis can hold it.
     Here it would be a label with nothing under it, so it gets counted and
     named instead of drawn. */
  const allArrows = spectrum ? arrowsFor(spectrum.lines, n, l) : [];
  const arrows = allArrows.filter((a) => a.n_upper !== a.n_lower);
  const withinN = allArrows.length - arrows.length;
  const grossE = new Map(levels.gross.map((g) => [g.n, g.energy_ev.value]));

  /* The rungs crowd toward the ionization limit because the energies go as
     -1/n², and past n = 4 the labels printed through one another. The rungs
     stay where the physics puts them; only the text gets nudged apart, with a
     leader line back to the rung each label belongs to. */
  /* The window decides which rungs are on screen, and only those get spread.
     Zoomed into the top of the ladder, that is what turns six labels printed
     through one another into six readable rows. */
  const gross = withinView(levels.gross, (g) => g.energy_ev.value, ladderZoom.y);
  const rungY = gross.map((g) => y(g.energy_ev.value));
  const labelY = spreadLabels(rungY, 13, 20, H - 28);
  const fineForN = levels.fine?.filter((f) => f.n === n) ?? [];
  // The hyperfine shell for the selected n. When it is unavailable the server
  // sends a single sentinel entry, since availability does not depend on n, so
  // fall back to that.
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
            The energies go as −1/n², so the gap from n=1 to n=2 dwarfs the gap
            from n=5 to n=6. Every rung above the first ends up bunched toward
            the ionization limit at 0, where the electron breaks free. That
            crowding is why hydrogen's spectral series each pile up against a
            short-wavelength edge instead of spreading out evenly.
          </p>
        </Disclosure>
      </ViewIntro>

      <ControlGroup
        title="Magnify one rung"
        hint="The right-hand column zooms in on what splits the selected shell. One at a time: these are four different magnifications of the same column."
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
              hint: "Gross levels only, and exact, under a reduced-mass Bohr model.",
            },
            {
              value: "fine",
              label: "fine structure",
              hint:
                "Relativity and spin-orbit coupling split each shell by roughly " +
                "a part in 100,000. The column magnifies that.",
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
                : "The nucleus has its own spin, and it splits the s-states. This is where the 21 cm line comes from.",
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
                ? "Dirac is exact for a point nucleus: the energy depends on n and j alone, so 2s½ and 2p½ land exactly on top of each other. Reality splits them by the Lamb shift, which is left out here."
                : "The α² perturbation, accurate to about a part in 10⁴ of the shift itself. Switch to Dirac for the closed form."
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
        The rungs are clickable. Pick one and every view moves to that shell.
        {arrows.length > 0
          ? ` The arrows are the ${arrows.length} transition${arrows.length === 1 ? "" : "s"} out of the state you selected in the rail, labelled with the light they emit.`
          : ""}
        {withinN > 0
          ? ` ${withinN} more transition${withinN === 1 ? " starts" : "s start"} and end${withinN === 1 ? "s" : ""} on this same shell, so there is no arrow to draw for ${withinN === 1 ? "it" : "them"} here. ${withinN === 1 ? "It is" : "They are"} out at microwave wavelengths in the spectrum view.`
          : ""}
      </p>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        className={`levels-svg plot-zoomable${ladderZoom.dragging ? " plot-panning" : ""}`}
        ref={ladderZoom.ref}
        {...ladderZoom.handlers}
      >
        <OffWindowMarks
          above={levels.gross.filter(
            (g) => g.energy_ev.value > Math.max(...ladderZoom.y),
          ).length}
          below={levels.gross.filter(
            (g) => g.energy_ev.value < Math.min(...ladderZoom.y),
          ).length}
          x={rungX1} top={18} bottom={H - 14} noun="shell"
        />
        {gross.map((g, i) => {
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
              {/* An invisible band over the label row, so the click target is a
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
        {/* Clipped, because an arrow can outlive its rung: zoom in on the top
            of the ladder and a Lyman transition still starts inside the window
            and ends far below it. The arrow is real and stays drawn; what it
            must not do is run out through the frame. */}
        <defs>
          <clipPath id="ladder-clip">
            <rect x={0} y={16} width={W} height={H - 46} />
          </clipPath>
        </defs>
        <g clipPath="url(#ladder-clip)">
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
                  structure on (two arrows per shell pair, at nearly the same
                  wavelength) every label printed over its neighbour. */}
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
        </g>
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
                                {mathTspans(`m_j = ${s.m_j}`)}
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
      <ZoomControls zoom={ladderZoom} what="the energy axis" />
      {gross.length < levels.gross.length && (
        <p className="caption">
          The right-hand magnifier keeps its own scale, which this zoom does not
          touch.
        </p>
      )}

      {/* One sentence on whatever the magnifier is showing. The full account
          stays in the disclosure underneath, unchanged. */}
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
            ? " That is the 21 cm line: radio astronomy's fingerprint for neutral hydrogen, and the reason anyone can map the galaxy's gas at all."
            : ""}{" "}
          <Badge provenance={hfShell.A!.provenance} />
        </p>
      )}

      <Disclosure summary="What the two scales mean, and what is left out" tone="caveat">
        <p className="caption">
          The gross levels are reduced-mass exact. The right column magnifies
          the {dirac ? "relativistic" : "α²"} shifts of the selected n. Those
          two scales differ by a factor of ~10⁵, so they are labelled
          separately and never blended.{" "}
          {dirac
            ? "Dirac is exact for a point nucleus: the energy depends on n and j alone, so 2s₁/₂ and 2p₁/₂ coincide exactly. Reality splits them by the Lamb shift (QED), which is deliberately left out; the badge assumptions say so."
            : "States with equal j coincide at this order. 2s₁/₂ and 2p₁/₂, for instance: the Lamb shift is beyond α², so it is honestly absent rather than quietly folded in."}
          {mode === "zeeman" && (
            <>
              {" "}A magnetic field splits each j-level into 2j+1 <Notation>{"m_j"}</Notation> sublevels:
              the anomalous Zeeman effect, spaced by g_J·µ_B·B. Push B higher
              and they reorganize toward the Paschen-Back pattern, where
              (m_l, m_s) become the good labels. The model here is linear, and
              the diamagnetic B² term is left out.
            </>
          )}
          {mode === "stark" && (
            <>
              {" "}An electric field splits each n-shell into n² parabolic
              (n₁,n₂,m) sublevels, fanned out by the electric quantum number
              k = n₁−n₂. The splitting is linear in F, which is hydrogen's
              accidental l-degeneracy showing itself: a first-order shift turns
              up here that non-degenerate atoms, quadratic only, never get. The
              work is second order on the gross shells, and the perturbation
              series is asymptotic, so it breaks down near field ionization.
              The badge has the details.
            </>
          )}
          {mode === "hyperfine" && hfShell && (
            <>
              {" "}Hyperfine: the nuclear spin I couples to the electron's
              angular momentum J, splitting the {n}s level (J=½) into F = I+J
              states through the Fermi contact interaction.{" "}
              {hfShell.available
                ? hfShell.levels.length > 1
                  ? "s-states only, through the contact term; the l>0 orbital and dipolar channels are deferred. The treatment is non-relativistic and leaves out bound-state QED and nuclear structure, worth about 0.01%. See the badge."
                  : hfShell.note
                : hfShell.reason}
            </>
          )}
        </p>
      </Disclosure>
    </div>
  );
}
