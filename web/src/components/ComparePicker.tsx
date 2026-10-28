import type { SystemInfo } from "../api/types";
import type { PlaneCompare } from "../lib/urlState";

/**
 * The second state for a side-by-side plane: n, l, m and a hydrogen-like system.
 *
 * Only hydrogen-like systems are offered: they are the ones a bare (n, l, m)
 * names unambiguously, with no configuration or many-electron model to pick.
 */
export function ComparePicker({
  value,
  systems,
  onChange,
}: {
  value: PlaneCompare;
  systems: SystemInfo[];
  onChange: (next: PlaneCompare) => void;
}) {
  const hydrogenic = systems.filter((s) => s.kind === "hydrogenic");
  const range = (from: number, to: number) =>
    Array.from({ length: to - from + 1 }, (_, i) => from + i);
  const set = (patch: Partial<PlaneCompare>) => {
    const next = { ...value, ...patch };
    // Keep the triple valid as n and l move under it.
    next.l = Math.min(next.l, next.n - 1);
    next.m = Math.max(-next.l, Math.min(next.l, next.m));
    onChange(next);
  };
  return (
    <div className="compare-picker">
      <label>
        n
        <select value={value.n} onChange={(e) => set({ n: Number(e.target.value) })}>
          {range(1, 7).map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </label>
      <label>
        l
        <select value={value.l} onChange={(e) => set({ l: Number(e.target.value) })}>
          {range(0, value.n - 1).map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </label>
      <label>
        m
        <select value={value.m} onChange={(e) => set({ m: Number(e.target.value) })}>
          {range(-value.l, value.l).map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </label>
      <label>
        system
        <select value={value.system} onChange={(e) => set({ system: e.target.value })}>
          {hydrogenic.map((s) => (
            <option key={s.key} value={s.key}>
              {s.name}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
