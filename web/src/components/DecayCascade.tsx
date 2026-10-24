import { scaleLinear, scaleLog } from "d3-scale";
import { useEffect, useState } from "react";
import { getCascade } from "../api/client";
import type { Cascade } from "../api/types";
import { endpoints, formatSeconds, grotrian, levelName, topBranches } from "../lib/cascade";
import { fixed } from "../lib/format";
import { CASCADE_GROTRIAN_LIBERTY } from "../lib/liberties";
import { usePlotWidth } from "../lib/plotSize";
import { Badge } from "./Badge";
import { Disclosure } from "./Disclosure";

const PAD = { left: 44, right: 16, top: 18, bottom: 30 };
const TABLE_ROWS = 8;

/**
 * Where the selected level goes when it is left alone: every E1 decay down to
 * the levels that cannot decay, with the photons it sends out on the way and
 * the populations in time. Hydrogen-like systems only; LevelsView mounts it on
 * that branch alone.
 */
export function DecayCascade({ n, l, system }: { n: number; l: number; system: string }) {
  const [data, setData] = useState<Cascade | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { width: W, ref } = usePlotWidth(280);

  useEffect(() => {
    let live = true;
    setData(null);
    setError(null);
    getCascade(n, l, system).then(
      (c) => live && setData(c),
      (e) => live && setError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      live = false;
    };
  }, [n, l, system]);

  const name = levelName(n, l);
  return (
    <section className="config-race" aria-labelledby="cascade-title" ref={ref}>
      <h3 id="cascade-title" className="readouts-head">
        Left alone, {name} decays: the cascade
      </h3>
      {error && <p className="error">{error}</p>}
      {!data && !error && <p className="hint">Computing the cascade…</p>}
      {data &&
        (n === 1 ? (
          <p className="caption">
            1s is the ground state. There is nothing lower to fall to, so it never decays.
          </p>
        ) : (
          <CascadeBody data={data} name={name} W={W} />
        ))}
    </section>
  );
}

function CascadeBody({ data, name, W }: { data: Cascade; name: string; W: number }) {
  const trapped = endpoints(data).find((e) => e.name === "2s");
  return (
    <>
      <p className="caption">
        An excited atom does not drop straight to the ground state. Each level empties at
        its own rate and splits its decays between the levels below it that a photon can
        reach (l changes by one). Starting in {name}, the atom settles in{" "}
        <strong>{formatSeconds(data.mean_settle_time.value)}</strong> on average{" "}
        <Badge provenance={data.mean_settle_time.provenance} />
        {trapped && (
          <>
            , and <strong>{fixed(100 * trapped.probability, 1)}%</strong> of the time it ends
            up stuck in 2s, which has no single-photon way down
          </>
        )}
        .
      </p>
      <Grotrian data={data} W={W} />
      <Populations data={data} W={W} />
      <table className="line-table">
        <thead>
          <tr>
            <th>Line</th>
            <th>λ (nm)</th>
            <th>Photons per atom</th>
            <th>Branching</th>
            <th>A (s⁻¹)</th>
          </tr>
        </thead>
        <tbody>
          {topBranches(data, TABLE_ROWS).map((b) => (
            <tr key={`${b.upper}-${b.lower}`}>
              <td>
                {levelName(...b.upper)} → {levelName(...b.lower)}
              </td>
              <td>{fixed(b.wavelength.value, 1)}</td>
              <td>{b.photons.value.toPrecision(3)}</td>
              <td>{fixed(100 * b.branching_ratio.value, 1)}%</td>
              <td>{b.einstein_A.value.toExponential(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {data.branches.length > TABLE_ROWS && (
        <p className="hint">
          The table shows the {TABLE_ROWS} brightest of {data.branches.length} lines; the
          diagram draws them all.
        </p>
      )}
      <Disclosure summary="What the cascade leaves out" tone="caveat">
        <p className="caption">
          The rates are hydrogen's exact one-electron dipole rates, so the branching ratios
          are as good as the model. The atom is isolated: no collisions, no light shining on
          it. 2s is shown as a dead end because it has no single-photon decay. A real 2s
          atom gets out by emitting two photons at once, about 8 times a second in hydrogen,
          some hundred million times slower than 2p. That channel is not modelled, and
          neither is fine structure. The badge lists the assumptions.
        </p>
      </Disclosure>
    </>
  );
}

function Grotrian({ data, W }: { data: Cascade; W: number }) {
  const H = Math.round(Math.min(360, Math.max(200, 60 * data.start[0])));
  const { nodes, arrows } = grotrian(data);
  const lMax = Math.max(1, ...nodes.map((nd) => nd.l));
  const x = scaleLinear().domain([0, 1]).range([PAD.left + 16, W - PAD.right - 24]);
  const y = scaleLinear().domain([0, 1]).range([PAD.top + 8, H - PAD.bottom - 12]);
  const half = Math.min(18, (W - PAD.left - PAD.right) / (lMax + 1) / 3);
  const rows = [...new Map(nodes.map((nd) => [nd.n, nd.y])).entries()];
  return (
    <figure className="plot">
      <figcaption>
        <span className="plot-lead">Which way it goes</span>
        <span className="plot-blurb">
          One column per l, one row per n. Arrow thickness is the expected number of photons
          on that line per excited atom.
        </span>
        <span className="plot-provenance">
          <Badge provenance={CASCADE_GROTRIAN_LIBERTY} />
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Grotrian diagram of the decay cascade">
        {rows.map(([n, ny]) => (
          <text key={n} className="tick" x={8} y={y(ny) + 4}>
            n={n}
          </text>
        ))}
        {Array.from({ length: lMax + 1 }, (_, l) => (
          <text key={l} className="tick" x={x(l / lMax)} y={H - 8} textAnchor="middle">
            {levelName(0, l).slice(1)}
          </text>
        ))}
        {arrows.map((a) => (
          <line
            key={`${a.from.n}${a.from.l}-${a.to.n}${a.to.l}`}
            x1={x(a.from.x)}
            y1={y(a.from.y)}
            x2={x(a.to.x)}
            y2={y(a.to.y)}
            stroke="var(--accent)"
            strokeWidth={0.6 + 5 * a.photons}
            strokeOpacity={0.35 + 0.6 * Math.min(1, a.photons)}
            strokeLinecap="round"
          >
            <title>
              {`${levelName(a.from.n, a.from.l)} → ${levelName(a.to.n, a.to.l)}: ${fixed(a.wavelengthNm, 1)} nm, ${a.photons.toPrecision(3)} photons per atom`}
            </title>
          </line>
        ))}
        {nodes.map((nd) => (
          <g key={`${nd.n}${nd.l}`}>
            <line
              x1={x(nd.x) - half}
              x2={x(nd.x) + half}
              y1={y(nd.y)}
              y2={y(nd.y)}
              stroke="var(--text)"
              strokeWidth={nd.terminal ? 3 : 2}
            />
            {/* Below the bar: every arrow arrives from above, so this side is clear. */}
            <text className="tick" x={x(nd.x)} y={y(nd.y) + 14} textAnchor="middle">
              {levelName(nd.n, nd.l)}
            </text>
            <title>
              {`${levelName(nd.n, nd.l)}: passed through ${fixed(100 * nd.visit, 1)}% of the time${nd.terminal ? ", and the cascade stops here" : ""}`}
            </title>
          </g>
        ))}
      </svg>
    </figure>
  );
}

function Populations({ data, W }: { data: Cascade; W: number }) {
  const H = 220;
  const grid = data.populations[0]?.grid ?? [];
  if (grid.length < 2) return null;
  const last = grid.length - 1;
  const x = scaleLog().domain([grid[0], grid[last]]).range([PAD.left, W - PAD.right]);
  const y = scaleLinear().domain([0, 1]).range([H - PAD.bottom, PAD.top]);
  const decades: number[] = [];
  for (let e = Math.ceil(Math.log10(grid[0])); 10 ** e <= grid[last]; e++) decades.push(10 ** e);
  const path = (values: number[]) =>
    values.map((v, i) => `${i ? "L" : "M"}${x(grid[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");
  return (
    <figure className="plot">
      <figcaption>
        <span className="plot-lead">Where the atom is, over time</span>
        <span className="plot-blurb">
          The chance of finding the atom in each level. The bold curve is{" "}
          {levelName(...data.start)}, dashed curves are the levels it ends in, and the faint
          ones are the stops on the way. Time runs on a log axis.
        </span>
        <span className="plot-provenance">
          N(t) <Badge provenance={data.populations[0].provenance} />
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Level populations against time">
        <line className="axis" x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom} />
        <line className="axis" x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={H - PAD.bottom} />
        {[0, 0.5, 1].map((v) => (
          <text key={v} className="tick" x={PAD.left - 6} y={y(v) + 4} textAnchor="end">
            {v}
          </text>
        ))}
        {decades.map((t) => (
          <text key={t} className="tick" x={x(t)} y={H - 10} textAnchor="middle">
            {formatSeconds(t, 1)}
          </text>
        ))}
        {data.populations.map((p, i) => {
          const lv = data.levels[i];
          const start = i === 0;
          const strong = start || lv.terminal;
          return (
            <path
              key={p.label}
              d={path(p.values)}
              fill="none"
              stroke={strong ? "var(--accent)" : "var(--muted)"}
              strokeWidth={start ? 2.2 : strong ? 1.6 : 1}
              strokeOpacity={strong ? 1 : 0.55}
              strokeDasharray={lv.terminal ? "5 3" : undefined}
            >
              <title>{levelName(lv.n, lv.l)}</title>
            </path>
          );
        })}
        {data.levels.map((lv, i) =>
          lv.terminal ? (
            <text
              key={`${lv.n}${lv.l}`}
              className="tick"
              x={W - PAD.right - 4}
              y={y(data.populations[i].values[last]) - 4}
              textAnchor="end"
            >
              {levelName(lv.n, lv.l)}
            </text>
          ) : null,
        )}
      </svg>
    </figure>
  );
}
