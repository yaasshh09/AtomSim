import { scaleLinear } from "d3-scale";
import { useEffect } from "react";
import type { FieldData, Quantity } from "../api/types";
import { linePath } from "../lib/plot";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

const W = 640;
const H = 240;
const M = { top: 16, right: 16, bottom: 34, left: 56 };

function FieldPlot({ field, marker }: { field: FieldData; marker?: Quantity }) {
  const x = scaleLinear(
    [0, field.grid[field.grid.length - 1]],
    [M.left, W - M.right],
  );
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
        {x.ticks(6).map((t) => (
          <g key={t} transform={`translate(${x(t)},${H - M.bottom})`}>
            <line y2="5" className="axis" />
            <text y="18" textAnchor="middle" className="tick">
              {t}
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
          r [{field.grid_unit}]
        </text>
      </svg>
    </figure>
  );
}

export function RadialView() {
  const { n, l, system, radial, stateInfo, loadRadial } = useAppStore();
  useEffect(() => {
    void loadRadial();
  }, [n, l, system, loadRadial]);
  if (!radial) return <p className="hint-block">loading radial functions…</p>;
  return (
    <div className="view-wrap">
      <FieldPlot field={radial.r_wavefunction} />
      <FieldPlot
        field={radial.radial_probability}
        marker={stateInfo?.mean_radius ?? undefined}
      />
    </div>
  );
}
