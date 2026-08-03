import { scaleLinear, scaleLog } from "d3-scale";
import { useEffect } from "react";
import type { FieldData, Quantity } from "../api/types";
import { HF_ORBITAL_CAPTION } from "../lib/hfModel";
import { linePath } from "../lib/plot";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 640;
const H = 240;
const M = { top: 16, right: 16, bottom: 34, left: 56 };

/** Powers of ten inside [lo, hi]. */
function decades([lo, hi]: [number, number]): number[] {
  const out: number[] = [];
  for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) {
    out.push(10 ** e);
  }
  return out;
}

/** "0.01", "1", "10" rather than "1e-2" — these are bohr, and readers know bohr. */
function decadeLabel(t: number): string {
  return t >= 0.01 ? String(Number(t.toFixed(2))) : t.toExponential(0);
}

/**
 * A radial curve.
 *
 * `logX` puts r on a log axis, which is not decoration and not a liberty: the
 * axis is labeled with the values it carries, so nothing is hidden or clipped.
 * It exists because a many-electron atom is logarithmic in r and a linear axis
 * cannot show it. Argon's solve box runs to 48 bohr while its K shell sits near
 * 0.06 and its M shell near 1.4, so on a linear axis all three shells pile into
 * the first 3% of the width and the plot shows one spike and 45 bohr of
 * nothing. The solver's own mesh is exponential for exactly this reason.
 */
function FieldPlot({
  field,
  marker,
  logX = false,
}: {
  field: FieldData;
  marker?: Quantity;
  logX?: boolean;
}) {
  const rMax = field.grid[field.grid.length - 1];
  const x = logX
    ? scaleLog(
        // The first grid point, not zero: log has no zero, and the mesh starts
        // where it does precisely so r = 0 is never needed.
        [Math.max(field.grid[0], 1e-4), rMax],
        [M.left, W - M.right],
      )
    : scaleLinear([0, rMax], [M.left, W - M.right]);
  // Decades only on a log axis. d3's own log ticks include every 2x, 3x, ... in
  // each decade, and at this width they overprint into an unreadable band.
  const xTicks = logX ? decades(x.domain() as [number, number]) : x.ticks(6);
  const lo = Math.min(0, ...field.values);
  const hi = Math.max(...field.values);
  const y = scaleLinear([lo, hi], [H - M.bottom, M.top]).nice();
  return (
    <figure className="plot">
      <figcaption>
        {field.label} [{field.unit}] <Badge provenance={field.provenance} />
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img">
        <line
          x1={M.left} y1={H - M.bottom} x2={W - M.right} y2={H - M.bottom}
          className="axis"
        />
        <line x1={M.left} y1={M.top} x2={M.left} y2={H - M.bottom} className="axis" />
        {xTicks.map((t) => (
          <g key={t} transform={`translate(${x(t)},${H - M.bottom})`}>
            <line y2="5" className="axis" />
            <text y="18" textAnchor="middle" className="tick">
              {logX ? decadeLabel(t) : t}
            </text>
          </g>
        ))}
        {y.ticks(4).map((t) => (
          <g key={t} transform={`translate(${M.left},${y(t)})`}>
            <line x2="-5" className="axis" />
            <text x="-8" dy="0.32em" textAnchor="end" className="tick">
              {t.toPrecision(2)}
            </text>
          </g>
        ))}
        {lo < 0 && (
          <line x1={M.left} x2={W - M.right} y1={y(0)} y2={y(0)} className="zero" />
        )}
        <path d={linePath(field.grid, field.values, x, y)} className="curve" />
        {marker && (
          <g>
            <line
              x1={x(marker.value)} x2={x(marker.value)} y1={M.top} y2={H - M.bottom}
              className="marker"
            />
            <text x={x(marker.value) + 4} y={M.top + 12} className="tick">
              ⟨r⟩
            </text>
          </g>
        )}
        <text
          x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle" className="tick"
        >
          r [{field.grid_unit}]{logX ? " (log)" : ""}
        </text>
      </svg>
    </figure>
  );
}

export function RadialView() {
  const { n, l, system, radial, stateInfo, loadRadial, model, config, exchange, pauli } =
    useAppStore();
  useEffect(() => {
    void loadRadial();
    // The Hartree-Fock inputs are dependencies too: each of them names a
    // different solve, so a curve fetched under one and left on screen under
    // another would be the stale render the store's INVALIDATED block exists
    // to prevent, arriving by a different door.
  }, [n, l, system, model, config, exchange, pauli, loadRadial]);
  if (!radial) return <p className="hint-block">loading radial functions…</p>;
  return (
    <div className="view-wrap">
      <FieldPlot field={radial.r_wavefunction} />
      <FieldPlot
        field={radial.radial_probability}
        marker={stateInfo?.mean_radius ?? undefined}
      />
      {model === "hf" && <p className="hint-block">{HF_ORBITAL_CAPTION}</p>}
      {radial.total_density && (
        <>
          <FieldPlot field={radial.total_density} logX />
          <p className="hint-block">
            This one is measurable. Each peak is a shell, the area under it is
            how many electrons that shell holds, and the whole curve integrates
            to {radial.system.n_electrons ?? "N"} — which is what makes it a
            density rather than a curve. Neon has two peaks and argon has three;
            that is the periodic table, drawn. r is on a log axis because the
            shells are decades apart in size, not to flatter the shape. The
            plots above are one orbital out of the basis this was summed from.
          </p>
        </>
      )}
    </div>
  );
}
