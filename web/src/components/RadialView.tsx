import { scaleLinear, scaleLog } from "d3-scale";
import { useEffect } from "react";
import type { FieldData, Quantity, ShellPeak } from "../api/types";
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

/** "0.01", "1", "10" rather than "1e-2": these are bohr, and readers know bohr. */
function decadeLabel(t: number): string {
  return t >= 0.01 ? String(Number(t.toFixed(2))) : t.toExponential(0);
}

/** Below which a peak's valley is too shallow to read as a shell boundary. */
const SHALLOW = 0.05;

/**
 * Decimals enough to carry the error bar to two significant figures.
 *
 * A fixed three decimals prints helium's 0.00034 electrons displaced as
 * "0.000 ± 0.000", which is a zero standing in for a measurement that is not
 * zero, and a resolved one: the bar there is 0.00018, so the number is about
 * twice it. The bar sets the precision because the bar is the thing that says
 * how much precision there is.
 */
function decimalsFor(bar: number): number {
  if (!(bar > 0)) return 4;
  return Math.max(3, 1 - Math.floor(Math.log10(bar)));
}

/**
 * The disagreement in words, or an admission that it is under the noise.
 *
 * A number smaller than its own error bar is not a measurement of anything,
 * and four decimals of it would read as precision the comparison does not
 * have. Helium is that case: 0.0003 electrons displaced against a bar of the
 * same size.
 */
export function displacedChargeText(q: Quantity, nElectrons: number | null): string {
  const bar = q.provenance.error_estimate ?? 0;
  const of = nElectrons === null ? "" : ` of the atom's ${nElectrons}`;
  const d = decimalsFor(bar);
  if (q.value <= bar) {
    return (
      `The two models agree to within the resolution of this comparison ` +
      `(${q.value.toFixed(d)} electrons displaced, against a ${bar.toFixed(d)} bar).`
    );
  }
  return (
    `The two models disagree about where ${q.value.toFixed(d)} ± ` +
    `${bar.toFixed(d)} electrons${of} are.`
  );
}

/** One row of the shell table, with "no peak" as an answer rather than a gap. */
export function shellCells(s: ShellPeak): {
  label: string;
  gsz: string;
  hf: string;
  note: string | null;
} {
  const cell = (r: number | null) => (r === null ? "no separate peak" : r.toFixed(3));
  const shallow = [s.gsz_depth, s.hf_depth].filter(
    (d): d is number => d !== null && d < SHALLOW,
  );
  return {
    label: s.label,
    gsz: cell(s.gsz_radius),
    hf: cell(s.hf_radius),
    note:
      shallow.length === 0
        ? null
        : `barely separated: the dip before it is ${(Math.min(...shallow) * 100).toFixed(1)}% deep`,
  };
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
  overlay,
}: {
  field: FieldData;
  marker?: Quantity;
  logX?: boolean;
  overlay?: { field: FieldData; label: string; selfLabel: string };
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
  // Both curves share one y domain, because two densities on separate scales
  // would show a disagreement neither model has.
  const all = overlay ? [...field.values, ...overlay.field.values] : field.values;
  const lo = Math.min(0, ...all);
  const hi = Math.max(...all);
  const y = scaleLinear([lo, hi], [H - M.bottom, M.top]).nice();
  return (
    <figure className="plot">
      <figcaption>
        {field.label} [{field.unit}] <Badge provenance={field.provenance} />
        {overlay && (
          <span className="legend-inline">
            <span className="swatch-line" /> {overlay.selfLabel}
            <span className="swatch-line dashed" /> {overlay.label}
          </span>
        )}
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
        {overlay && (
          <path
            d={linePath(overlay.field.grid, overlay.field.values, x, y)}
            className="curve curve-overlay"
          />
        )}
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
  const {
    n, l, system, radial, stateInfo, loadRadial, model, config, exchange, pauli, compare,
  } = useAppStore();
  useEffect(() => {
    void loadRadial();
    // The Hartree-Fock inputs are dependencies too: each of them names a
    // different solve, so a curve fetched under one and left on screen under
    // another would be the stale render the store's INVALIDATED block exists
    // to prevent, arriving by a different door.
    //
    // `compare` is here for the opposite reason. It names no different solve,
    // but it does name a field the payload is missing, and setCompare drops
    // the payload rather than the atom. Without it here the view would sit on
    // "loading" forever after the toggle.
  }, [n, l, system, model, config, exchange, pauli, compare, loadRadial]);
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
          {/* The primary curve stays the one the model radio selected, so the
              plot the reader was already looking at does not move under them. */}
          <FieldPlot
            field={radial.total_density}
            logX
            overlay={
              radial.density_comparison
                ? {
                    field:
                      model === "hf"
                        ? radial.density_comparison.gsz
                        : radial.density_comparison.hf,
                    label: model === "hf" ? "screened (GSZ)" : "Hartree-Fock",
                    selfLabel: model === "hf" ? "Hartree-Fock" : "screened (GSZ)",
                  }
                : undefined
            }
          />
          <p className="hint-block">
            This one is measurable. Each peak is a shell, integrating across
            one peak in r gives roughly how many electrons it holds, and the
            whole curve integrates to {radial.system.n_electrons ?? "N"}, which
            is what makes it a density rather than a curve. Roughly, because
            neighbouring shells overlap and the dip between two of them is not
            a wall: cut argon at its minima and the K shell comes out near 2.2,
            not 2, under either model. Neon has two peaks and argon has three;
            that is the periodic table, drawn.
          </p>
          <p className="hint-block">
            r is on a log axis because the shells are decades apart in size,
            not to flatter the shape. The price is that the areas visible here
            are not those integrals: the axis stretches the inner shell and
            squeezes the outer one, so read the peaks, and trust a number only
            where it is written down. The plots above are one orbital out of
            the basis this was summed from.
          </p>
          {model === "gsz" && (
            <p className="hint-block">
              GSZ was fitted to reproduce a potential rather than a density, and
              every shell here sees the same one. That turns out to cost less
              than it sounds: turn the comparison on and the two models place
              under 1.5% of the electrons differently for every atom they both
              cover, because the fit was made against Hartree-Fock in the first
              place. Where the fitted model gives out is the energy, not the
              shape.
            </p>
          )}
        </>
      )}
      {radial.density_comparison && (
        <div className="compare-block">
          <p className="hint-block">
            {displacedChargeText(
              radial.density_comparison.displaced_charge,
              radial.system.n_electrons ?? null,
            )}{" "}
            <Badge provenance={radial.density_comparison.provenance} />
          </p>
          <table className="shell-table">
            <caption>Shell peak radii [bohr]</caption>
            <thead>
              <tr>
                <th>shell</th>
                <th>GSZ</th>
                <th>Hartree-Fock</th>
              </tr>
            </thead>
            <tbody>
              {radial.density_comparison.shells.map(shellCells).map((c) => (
                <tr key={c.label}>
                  <th scope="row">{c.label}</th>
                  <td>{c.gsz}</td>
                  <td>{c.hf}</td>
                  {c.note && <td className="shell-note">{c.note}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
