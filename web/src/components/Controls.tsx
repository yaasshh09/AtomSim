import { useEffect, useState } from "react";
import { isScreenedLevels } from "../api/client";
import { gszAvailable, subshellAvailable } from "../lib/hfModel";
import type { NucleusMode } from "../lib/nucleus";
import { NUCLEUS_MODES } from "../lib/nucleus";
import { useAppStore } from "../state/store";
import type { ColorMode, ViewMode } from "../state/store";
import { ShowPhysics } from "./ShowPhysics";

const N_CHOICES = [1, 2, 3, 4, 5, 6];
const COUNT_CHOICES = [10_000, 50_000, 100_000, 250_000];

// Later tasks append entries as their views land.
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
    pauli, hf,
    setQuantumNumbers, setCount, sample, setSystem, setBasis, setView,
    setColorMode, setFineStructure, setNucleusMode, setConfig, setModel,
    setExchange, setPauli, loadSystems, ensureHF,
  } = useAppStore();
  useEffect(() => {
    if (systems.length === 0) void loadSystems();
  }, [systems.length, loadSystems]);
  // The picker greys subshells the configuration does not occupy, and it can
  // only know which those are from the solve. Asked for here rather than left
  // to whichever view is open: the Cloud samples on a button press, so under
  // that view nothing else would ever request it and the picker would offer
  // every subshell right up until the job came back 422.
  useEffect(() => {
    if (model === "hf") void ensureHF();
  }, [model, ensureHF]);
  const lChoices = Array.from({ length: n }, (_, i) => i);
  const mChoices = Array.from({ length: 2 * l + 1 }, (_, i) => i - l);

  const hydrogenic = systems.filter((s) => s.kind === "hydrogenic");
  const screened = systems.filter((s) => s.kind === "screened");
  const isScreened = systems.find((s) => s.key === system)?.kind === "screened";
  // Sulfur and chlorine are real atoms the engine solves; only one of the two
  // models has parameters for them. The radio is disabled rather than hidden,
  // so the missing option is visible and has a reason next to it.
  const hasGsz = gszAvailable(systems, system);
  // The server echoes the resolved configuration on the levels payload.
  const resolved = levels !== null && isScreenedLevels(levels) ? levels : null;

  // A local draft so typing does not refetch physics on every keystroke; commit
  // on submit/blur. `config` (store) is the committed value; null = Aufbau ground.
  const [draft, setDraft] = useState(config ?? "");
  useEffect(() => setDraft(config ?? ""), [config]);
  const commitConfig = () => {
    const trimmed = draft.trim();
    setConfig(trimmed === "" ? null : trimmed);
  };

  return (
    <aside className="panel">
      <h2>System</h2>
      <label>
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
          {/* Not "screened": two of these atoms have no screened model. */}
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
          <div className="radio-row">
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
              for this element, so the screened model has nothing to run on.
              Hartree-Fock builds its potential out of the orbitals it is
              solving for and needs no fitted table, which is why the atom is
              here at all.
            </p>
          )}
          <p className="panel-hint">
            {model === "gsz"
              ? "Fitted central field: one potential for every electron, no self-consistency."
              : "Self-consistent field, solved per subshell, with no fitted parameters. Every view draws it: cloud, cross-section, radial and surface. What you see is one orbital, not the total density, which for these atoms is exactly spherical."}
          </p>
          {model === "hf" && (
            <>
              <label className="check">
                <input
                  type="checkbox"
                  checked={!exchange}
                  // Disabled rather than hidden while the cap is off, and
                  // shown ticked: the weaker counterfactual is contained in
                  // the stronger, so it is true and not available to change.
                  // Hiding it would let a reader think exchange came back.
                  disabled={!pauli}
                  onChange={(e) => setExchange(!e.target.checked)}
                />
                distinguishable electrons
              </label>
              <p className="panel-hint">
                {!pauli
                  ? "Forced on by the switch below: exchange energy comes from antisymmetry, and antisymmetry is the exclusion principle."
                  : exchange
                    ? "Exchange on: the wavefunction is antisymmetric, as it is in this universe."
                    : "Counterfactual. Exchange removed, so the wavefunction is a product instead of a determinant. The Pauli occupancies are untouched, so this is not electrons piling into 1s."}
              </p>
              <label className="check">
                <input
                  type="checkbox"
                  checked={!pauli}
                  onChange={(e) => setPauli(!e.target.checked)}
                />
                no Pauli exclusion
              </label>
              <p className="panel-hint">
                {pauli
                  ? "Occupancies capped at 2(2l+1), which is why the atom has shells and the periodic table has periods."
                  : "Counterfactual, and the stronger one. The cap is gone, so every electron falls into the 1s: one level, no shells, no chemistry. Compare the two energies and sizes on the Energy levels view."}
              </p>
            </>
          )}
        </div>
      )}
      <h2>View</h2>
      <label>
        mode
        <select value={view} onChange={(e) => setView(e.target.value as ViewMode)}>
          {VIEW_OPTIONS.map((v) => (
            <option key={v.value} value={v.value}>
              {v.label}
            </option>
          ))}
        </select>
      </label>
      <h2>State</h2>
      <label>
        n
        <select value={n} onChange={(e) => setQuantumNumbers(Number(e.target.value), l, m)}>
          {N_CHOICES.map((v) => (
            <option
              key={v}
              value={v}
              // A shell is offered when any subshell in it is occupied. Under
              // the screened model that is always, since a fitted central
              // field has a solution in every channel whether or not an
              // electron is in it.
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
      <label>
        l
        <select value={l} onChange={(e) => setQuantumNumbers(n, Number(e.target.value), m)}>
          {lChoices.map((v) => (
            <option key={v} value={v} disabled={!subshellAvailable(hf, model, n, v)}>
              {v}
            </option>
          ))}
        </select>
      </label>
      {model === "hf" && hf !== null && (
        <p className="panel-hint">
          Greyed subshells are empty in {hf.config}. Hartree-Fock builds one Fock
          operator per occupied subshell, so an empty one has no operator to be an
          eigenfunction of.
        </p>
      )}
      <label>
        m
        <select value={m} onChange={(e) => setQuantumNumbers(n, l, Number(e.target.value))}>
          {mChoices.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </label>
      <h2>Physics</h2>
      <div className="radio-row">
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
      <label className="check">
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
      <label>
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
        {status === "sampling" ? `Sampling ${(progress * 100).toFixed(0)}%` : "Sample"}
      </button>
      {status === "error" && error && <p className="error">{error}</p>}
      <ShowPhysics />
    </aside>
  );
}
