import { scaleLinear } from "d3-scale";
import { useEffect, useState } from "react";
import { getFiniteSize } from "../api/client";
import type { FiniteSize } from "../api/types";
import { fixed } from "../lib/format";
import { usePlotWidth } from "../lib/plotSize";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { Disclosure } from "./Disclosure";

const LABELS: Record<string, string> = {
  muonic: "proton radius from muonic hydrogen",
  codata_2010: "proton radius from ordinary hydrogen (CODATA 2010)",
};
const L_LETTERS = "spd";

/**
 * The nucleus is not a point: the finite-size shift of the low levels, and the
 * proton radius puzzle that came out of measuring it in muonic hydrogen.
 * LevelsView mounts it on the hydrogen-like branch.
 */
export function ProtonRadiusPanel({ system }: { system: string }) {
  const [data, setData] = useState<FiniteSize | null>(null);
  const [refusal, setRefusal] = useState<string | null>(null);
  const { width: W, ref } = usePlotWidth(280);
  // The systems table already says which nuclei have no radius (positronium's
  // positron, a chosen nuclear mass), so those never send a request the server
  // can only refuse. undefined means the table has not arrived: wait, do not guess.
  const info = useAppStore((s) => s.systems.find((x) => x.key === system));
  const pointNucleus = info !== undefined && info.nuclear_radius === null;

  useEffect(() => {
    let live = true;
    setData(null);
    setRefusal(null);
    if (info === undefined) return;
    if (pointNucleus) {
      setRefusal(
        `${info.name} has no nuclear charge radius: its nucleus is a point particle or ` +
          "unidentified, so there is no size to shift anything.",
      );
      return;
    }
    getFiniteSize(system).then(
      (d) => live && setData(d),
      (e) => live && setRefusal(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      live = false;
    };
  }, [system, info, pointNucleus]);

  return (
    <section className="config-race" aria-labelledby="radius-title" ref={ref}>
      <h3 id="radius-title" className="readouts-head">
        The nucleus has a size: the proton radius puzzle
      </h3>
      {refusal && <p className="caption">{refusal}</p>}
      {data && <RadiusBody data={data} W={W} />}
    </section>
  );
}

function RadiusBody({ data, W }: { data: FiniteSize; W: number }) {
  const r = data.radius_fm;
  return (
    <>
      <p className="caption">
        Everything else in this app treats the nucleus as a point. It is really a cloud of
        charge with a radius of about {fixed(r.value, 3)} fm <Badge provenance={r.provenance} />,
        and an s electron spends a little time inside it, where the pull is weaker. That lifts
        the s levels slightly. p and d electrons avoid the centre, so they barely notice.
      </p>
      <table className="line-table">
        <thead>
          <tr>
            <th>Level</th>
            <th>Shift (MHz)</th>
            <th>Shift (meV)</th>
            <th>Share of the binding</th>
          </tr>
        </thead>
        <tbody>
          {data.levels.map((lv) => (
            <tr key={`${lv.n}${lv.l}`}>
              <td>
                {lv.n}
                {L_LETTERS[lv.l]} <Badge provenance={lv.shift_mev.provenance} />
              </td>
              <td>{lv.l ? "≈ 0" : lv.shift_mhz.value.toPrecision(3)}</td>
              <td>{lv.l ? "≈ 0" : lv.shift_mev.value.toPrecision(3)}</td>
              <td>{lv.l ? "≈ 0" : lv.fraction.toExponential(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="caption">
        The shift grows as the cube of the orbiting particle's (reduced) mass. Swap the
        electron for a muon and that mass is 186 times larger, so the 2s shift grows about
        six-million-fold. That is why muonic hydrogen measures the proton's size so well.
        This app's one-line formula gives {fixed(data.coefficient_ours.value, 3)} meV per fm²
        for muonic 2s <Badge provenance={data.coefficient_ours.provenance} />; the full
        published theory says {fixed(data.coefficient_published, 4)}.
      </p>
      <PuzzlePlot data={data} W={W} />
      <Disclosure summary="What the puzzle was, and where the numbers come from" tone="caveat">
        <p className="caption">
          In 2010 the muonic hydrogen 2S–2P energy came out where only a proton about 4%
          smaller than the accepted size could put it, seven standard deviations away. The
          plot shows why that was so hard to explain away: the two radii predict energies
          a third of a meV apart, many times either error bar. Later measurements on
          ordinary hydrogen moved toward the smaller value, and today's recommended radius
          (the one at the top, CODATA via scipy) sits beside the muonic one.
        </p>
        <p className="caption">
          The Lamb shift predictions use published QED theory, transcribed rather than
          computed here: {data.citation} (retrieved {data.retrieved}). Only the
          finite-size table is this app's own calculation, at leading order.
        </p>
      </Disclosure>
    </>
  );
}

function PuzzlePlot({ data, W }: { data: FiniteSize; W: number }) {
  const rowH = 46;
  const H = 30 + rowH * data.predictions.length;
  const err = (q: FiniteSize["predictions"][number]) => q.lamb_shift.provenance.error_estimate ?? 0;
  const lo = Math.min(...data.predictions.map((p) => p.lamb_shift.value - 2 * err(p)));
  const hi = Math.max(...data.predictions.map((p) => p.lamb_shift.value + 2 * err(p)));
  const pad = 0.15 * (hi - lo);
  const x = scaleLinear().domain([lo - pad, hi + pad]).range([24, W - 24]);
  return (
    <figure className="plot">
      <figcaption>
        <span className="plot-lead">The muonic hydrogen 2S–2P energy each radius predicts</span>
        <span className="plot-blurb">Dots are predictions; bars are one standard deviation.</span>
        <span className="plot-provenance">
          <Badge provenance={data.predictions[0].lamb_shift.provenance} />
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Lamb shift predicted by two proton radii">
        <line className="axis" x1={24} x2={W - 24} y1={H - 22} y2={H - 22} />
        {x.ticks(4).map((t) => (
          <text key={t} className="tick" x={x(t)} y={H - 6} textAnchor="middle">
            {fixed(t, 2)} meV
          </text>
        ))}
        {data.predictions.map((p, i) => {
          const y = 18 + i * rowH;
          const v = p.lamb_shift.value;
          const e = err(p);
          return (
            <g key={p.key}>
              <text className="tick" x={W / 2} y={y} textAnchor="middle">
                {LABELS[p.key] ?? p.key}: {p.radius_fm} fm
              </text>
              <line x1={x(v - e)} x2={x(v + e)} y1={y + 12} y2={y + 12} stroke="var(--text)" strokeWidth={2} />
              <circle cx={x(v)} cy={y + 12} r={4} fill="var(--accent)">
                <title>{`${fixed(v, 4)} ± ${e.toPrecision(2)} meV`}</title>
              </circle>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
