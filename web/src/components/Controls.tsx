import { useEffect } from "react";
import { useAppStore } from "../state/store";
import type { ColorMode, ViewMode } from "../state/store";

const N_CHOICES = [1, 2, 3, 4, 5, 6];
const COUNT_CHOICES = [10_000, 50_000, 100_000, 250_000];

// Later tasks append entries as their views land.
const VIEW_OPTIONS: { value: ViewMode; label: string }[] = [
  { value: "cloud", label: "3D point cloud" },
  { value: "plane", label: "2D cross-section" },
  { value: "radial", label: "Radial R(r), P(r)" },
  { value: "levels", label: "Energy levels" },
];

export function Controls() {
  const {
    n, l, m, count, status, progress, error, system, systems, basis, view,
    colorMode, fineStructure,
    setQuantumNumbers, setCount, sample, setSystem, setBasis, setView,
    setColorMode, setFineStructure, loadSystems,
  } = useAppStore();
  useEffect(() => {
    if (systems.length === 0) void loadSystems();
  }, [systems.length, loadSystems]);
  const lChoices = Array.from({ length: n }, (_, i) => i);
  const mChoices = Array.from({ length: 2 * l + 1 }, (_, i) => i - l);
  return (
    <aside className="panel">
      <h2>System</h2>
      <label>
        preset
        <select value={system} onChange={(e) => setSystem(e.target.value)}>
          {systems.length === 0 && <option value={system}>{system}</option>}
          {systems.map((s) => (
            <option key={s.key} value={s.key}>
              {s.name}
            </option>
          ))}
        </select>
      </label>
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
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </label>
      <label>
        l
        <select value={l} onChange={(e) => setQuantumNumbers(n, Number(e.target.value), m)}>
          {lChoices.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </label>
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
      <button
        type="button"
        className="primary"
        disabled={status === "sampling"}
        onClick={() => void sample()}
      >
        {status === "sampling" ? `Sampling ${(progress * 100).toFixed(0)}%` : "Sample"}
      </button>
      {status === "error" && error && <p className="error">{error}</p>}
    </aside>
  );
}
