import { useEffect, useState } from "react";
import { isScreenedLevels } from "../api/client";
import { compareAvailable, gszAvailable, subshellAvailable } from "../lib/hfModel";
import type { NucleusMode } from "../lib/nucleus";
import { NUCLEUS_MODES } from "../lib/nucleus";
import { useAppStore } from "../state/store";
import type { ColorMode, ViewMode } from "../state/store";
import { ShowPhysics } from "./ShowPhysics";

const N_CHOICES = [1, 2, 3, 4, 5, 6];
const COUNT_CHOICES = [10_000, 50_000, 100_000, 250_000];

// One entry per view, appended as each one lands.
const VIEW_OPTIONS: { value: ViewMode; label: string }[] = [
  { value: "cloud", label: "3D point cloud" },
  { value: "plane", label: "2D cross-section" },
  { value: "radial", label: "Radial R(r), P(r)" },
  { value: "levels", label: "Energy levels" },
  { value: "spectrum", label: "Spectrum vs NIST" },
  { value: "whatif", label: "What-If: constants" },
  { value: "forcelaw", label: "What-If: force law" },
];

export function Controls() {
  const {
    n, l, m, count, status, progress, error, system, systems, basis, view,
    colorMode, fineStructure, nucleusMode, config, levels, model, exchange,
    pauli, hf, compare, meta,
    setQuantumNumbers, setCount, sample, setSystem, setBasis, setView,
    setColorMode, setFineStructure, setNucleusMode, setConfig, setModel,
    setExchange, setPauli, setCompare, loadSystems, ensureHF,
  } = useAppStore();
  useEffect(() => {
    if (systems.length === 0) void loadSystems();
  }, [systems.length, loadSystems]);
  // The picker greys the subshells the configuration does not occupy, and it
  // can only learn which those are from the solve. The request happens here
  // rather than in whichever view is open: Cloud samples on a button press, so
  // under that view nothing else would ever ask, and the picker would offer
  // every subshell right up until the job came back 422.
  useEffect(() => {
    if (model === "hf") void ensureHF();
  }, [model, ensureHF]);
  const lChoices = Array.from({ length: n }, (_, i) => i);
  const mChoices = Array.from({ length: 2 * l + 1 }, (_, i) => i - l);

  const hydrogenic = systems.filter((s) => s.kind === "hydrogenic");
  const screened = systems.filter((s) => s.kind === "screened");
  const isScreened = systems.find((s) => s.key === system)?.kind === "screened";
  // Sulfur and chlorine are real atoms the engine solves, but only one of the
  // two models has parameters for them. The radio gets disabled rather than
  // hidden, so the missing option stays visible with a reason next to it.
  const hasGsz = gszAvailable(systems, system);
  // The overlay needs both models, so it is offered exactly where the radio
  // above it has two live options.
  const canCompare = compareAvailable(systems, system);
  // The server echoes the resolved configuration back on the levels payload.
  const resolved = levels !== null && isScreenedLevels(levels) ? levels : null;

  // A local draft, so typing does not refetch physics on every keystroke. It
  // commits on submit or blur. `config` in the store is the committed value;
  // null means the Aufbau ground state.
  const [draft, setDraft] = useState(config ?? "");
  useEffect(() => setDraft(config ?? ""), [config]);
  const commitConfig = () => {
    const trimmed = draft.trim();
    setConfig(trimmed === "" ? null : trimmed);
  };

  return (
    <aside className="panel">
      <h2>System</h2>
      <label data-tour="system-picker">
        preset
        <select value={system} onChange={(e) => setSystem(e.target.value)}>
          {systems.length === 0 && <option value={system}>{system}</option>}
          {hydrogenic.length > 0 && (
            <optgroup label="Hydrogen-like">
              {hydrogenic.map((s) => (
                <option key={s.key} value={s.key}>
                  {s.name}
                </option>
              ))}
            </optgroup>
          )}
          {/* Not called "screened": two of these atoms have no screened
              model. */}
          {screened.length > 0 && (
            <optgroup label="Atoms (many-electron, approx.)">
              {screened.map((s) => (
                <option key={s.key} value={s.key}>
                  {s.name}
                </option>
              ))}
            </optgroup>
          )}
        </select>
      </label>
      {isScreened && (
        <div className="config-panel">
          <label>
            configuration
            <input
              type="text"
              value={draft}
              placeholder="Aufbau (ground)"
              spellCheck={false}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commitConfig}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitConfig();
              }}
            />
          </label>
          <p className="panel-hint">
            {config === null
              ? "Aufbau ground configuration (default)"
              : resolved
                ? `${resolved.config}${resolved.is_ground ? "" : " (excited, non-ground)"}`
                : "custom configuration"}
          </p>
          <button type="button" className="ghost" onClick={() => setConfig(null)}>
            Reset to Aufbau
          </button>
          <div className="radio-row" data-tour="model-picker">
            <label className="radio">
              <input
                type="radio"
                checked={model === "gsz"}
                disabled={!hasGsz}
                onChange={() => setModel("gsz")}
              />
              screened (GSZ)
            </label>
            <label className="radio">
              <input
                type="radio"
                checked={model === "hf"}
                onChange={() => setModel("hf")}
              />
              Hartree-Fock
            </label>
          </div>
          {!hasGsz && (
            <p className="panel-hint">
              Szydlik and Green never published neutral GSZ screening parameters
              for this element, so there is nothing to run the screened model
              on. Hartree-Fock builds its potential out of the orbitals it is
              solving for and needs no fitted table, which is the only reason
              this atom is here at all.
            </p>
          )}
          <label className="check" data-tour="compare-toggle">
            <input
              type="checkbox"
              checked={compare}
              disabled={!canCompare}
              onChange={(e) => setCompare(e.target.checked)}
            />
            Compare both models
          </label>
          <p className="panel-hint">
            {canCompare
              ? "Draws the total density under both models on one axis, with the number of electrons they place differently. The orbital plots stay on the model selected above."
              : "This needs both models, and only one of them has parameters for this element."}
          </p>
          <p className="panel-hint">
            {model === "gsz"
              ? "A fitted central field: one potential for every electron, and no self-consistency."
              : "A self-consistent field, solved per subshell with no fitted parameters, and drawn in every view: cloud, cross-section, radial and surface. What you see is one orbital, not the total density, which for these atoms is exactly spherical."}
          </p>
          {model === "hf" && (
            <>
              <label className="check" data-tour="exchange-toggle">
                <input
                  type="checkbox"
                  checked={!exchange}
                  // Disabled rather than hidden while the cap is off, and left
                  // ticked: the weaker counterfactual is contained in the
                  // stronger, so it is true and not available to change.
                  // Hiding it would suggest exchange had come back.
                  disabled={!pauli}
                  onChange={(e) => setExchange(!e.target.checked)}
                />
                distinguishable electrons
              </label>
              <p className="panel-hint">
                {!pauli
                  ? "The switch below forces this on: exchange energy comes from antisymmetry, and antisymmetry is the exclusion principle."
                  : exchange
                    ? "Exchange on. The wavefunction stays antisymmetric, as it is in this universe."
                    : "Counterfactual. Exchange is gone, so the wavefunction is a product instead of a determinant. The Pauli occupancies are untouched, so this is not electrons piling into 1s."}
              </p>
              <label className="check" data-tour="pauli-toggle">
                <input
                  type="checkbox"
                  checked={!pauli}
                  onChange={(e) => setPauli(!e.target.checked)}
                />
                no Pauli exclusion
              </label>
              <p className="panel-hint">
                {pauli
                  ? "Occupancies are capped at 2(2l+1), which is why the atom has shells and the periodic table has periods."
                  : "Counterfactual, and the stronger one. The cap is gone, so every electron falls into the 1s: one level, no shells, no chemistry. The Energy levels view compares the two energies and sizes."}
              </p>
            </>
          )}
        </div>
      )}
      <h2>View mode</h2>
      {/* A list, not a dropdown: these seven views are the whole instrument,
          and a closed <select> would hide six of them behind a click. It still
          behaves as one radio group, and `aria-pressed` says which is live. */}
      <div className="view-list" data-tour="view-list">
        {VIEW_OPTIONS.map((v) => (
          <button
            key={v.value}
            type="button"
            className={`view-option${view === v.value ? " view-option-on" : ""}`}
            aria-pressed={view === v.value}
            onClick={() => setView(v.value)}
          >
            {v.label}
          </button>
        ))}
      </div>
      <h2>Quantum no.</h2>
      <div className="qn-tiles">
        <label className="qn-tile" data-tour="n-picker">
          <span className="qn-tile-name">n</span>
          <select
            value={n}
            onChange={(e) => setQuantumNumbers(Number(e.target.value), l, m)}
          >
            {N_CHOICES.map((v) => (
              <option
                key={v}
                value={v}
                // A shell is offered when any subshell in it is occupied.
                // Under the screened model that is always, since a fitted
                // central field has a solution in every channel whether or not
                // an electron sits in it.
                disabled={
                  !Array.from({ length: v }, (_, i) => i).some((li) =>
                    subshellAvailable(hf, model, v, li),
                  )
                }
              >
                {v}
              </option>
            ))}
          </select>
        </label>
        <label className="qn-tile" data-tour="l-picker">
          <span className="qn-tile-name">ℓ</span>
          <select
            value={l}
            onChange={(e) => setQuantumNumbers(n, Number(e.target.value), m)}
          >
            {lChoices.map((v) => (
              <option key={v} value={v} disabled={!subshellAvailable(hf, model, n, v)}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label className="qn-tile" data-tour="m-picker">
          <span className="qn-tile-name">m</span>
          <select
            value={m}
            onChange={(e) => setQuantumNumbers(n, l, Number(e.target.value))}
          >
            {mChoices.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
      </div>
      {model === "hf" && hf !== null && (
        <p className="panel-hint">
          Subshells empty in {hf.config} are greyed out. Hartree-Fock builds one
          Fock operator per occupied subshell, so an empty one has no operator
          to be an eigenfunction of.
        </p>
      )}
      <h2>Physics</h2>
      <div className="radio-row" data-tour="basis-picker">
        <label className="radio">
          <input
            type="radio"
            checked={basis === "complex"}
            onChange={() => setBasis("complex")}
          />
          complex Y<sub>lm</sub>
        </label>
        <label className="radio">
          <input type="radio" checked={basis === "real"} onChange={() => setBasis("real")} />
          real S<sub>lm</sub>
        </label>
      </div>
      <label className="check" data-tour="fine-structure">
        <input
          type="checkbox"
          checked={fineStructure}
          onChange={(e) => setFineStructure(e.target.checked)}
        />
        fine structure (α² perturbation)
      </label>
      <h2>Sampling</h2>
      <label>
        points
        <select value={count} onChange={(e) => setCount(Number(e.target.value))}>
          {COUNT_CHOICES.map((v) => (
            <option key={v} value={v}>
              {v.toLocaleString()}
            </option>
          ))}
        </select>
      </label>
      <label>
        colour
        <select
          value={colorMode}
          onChange={(e) => setColorMode(e.target.value as ColorMode)}
        >
          <option value="solid">solid (accent)</option>
          <option value="density">density (inferno)</option>
          <option value="phase" disabled={basis === "real"}>
            phase as hue (complex only)
          </option>
        </select>
      </label>
      <label data-tour="nucleus-picker">
        nucleus
        <select
          value={nucleusMode}
          onChange={(e) => setNucleusMode(e.target.value as NucleusMode)}
        >
          {NUCLEUS_MODES.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </label>
      <button
        type="button"
        className="primary"
        disabled={status === "sampling"}
        onClick={() => void sample()}
      >
        {status === "sampling"
          ? `SAMPLING ${(progress * 100).toFixed(0)}%`
          : "▶ EXECUTE SAMPLE"}
      </button>
      {/* The engine log. The design had it reading "engine ready · CDF ok"
          before anything had run, which is a claim about a solver nobody has
          asked a question yet. This reports the job that actually happened,
          and says "awaiting input" when there has not been one. */}
      <div className="term" role="status" aria-live="polite">
        {status === "error" && error ? (
          <div className="term-line term-err">{error}</div>
        ) : status === "sampling" ? (
          <div className="term-line">
            sampling {(progress * 100).toFixed(0)}%
            <span className="term-cursor">_</span>
          </div>
        ) : meta ? (
          <div className="term-line">
            {meta.count.toLocaleString()} points drawn
            {/* The model is named only when there is a choice of one.
                `meta.model` reads "gsz" for hydrogen too, and printing that
                beside a one-electron atom would claim a screened field that is
                not there and could not be. */}
            {isScreened ? ` · ${meta.model}` : ""}
          </div>
        ) : (
          <div className="term-line">
            awaiting input<span className="term-cursor">_</span>
          </div>
        )}
      </div>
      <ShowPhysics />
    </aside>
  );
}
