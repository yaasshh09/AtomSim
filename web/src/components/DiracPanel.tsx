import { scaleLinear } from "d3-scale";
import { useEffect, useState } from "react";
import { getDiracRadial } from "../api/client";
import type { DiracRadial } from "../api/types";
import { fixed } from "../lib/format";
import { Badge } from "./Badge";
import { Slider } from "./Field";
import { Disclosure } from "./Disclosure";

const PAD = { left: 44, right: 16, top: 16, bottom: 30 };
const DEBOUNCE_MS = 200;
const L_LETTERS = "spdfgh";

const jFraction = (j: number) => `${Math.round(2 * j)}/2`;

/**
 * The selected orbital under the Dirac equation, for a one-electron ion of any
 * charge: how relativity pulls it in and splits it by j. RadialView mounts it
 * for hydrogen-like systems only.
 */
export function DiracPanel({ n, l, width: W }: { n: number; l: number; width: number }) {
  const [z, setZ] = useState(80);
  const [data, setData] = useState<DiracRadial | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const timer = setTimeout(() => {
      getDiracRadial(n, l, z).then(
        (d) => {
          if (!live) return;
          setData(d);
          setError(null);
        },
        (e) => live && setError(e instanceof Error ? e.message : String(e)),
      );
    }, DEBOUNCE_MS);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [n, l, z]);

  const name = `${n}${L_LETTERS[l] ?? `(l=${l})`}`;
  return (
    <section className="config-race" aria-labelledby="dirac-title">
      <h3 id="dirac-title" className="readouts-head">
        {name} at relativistic speed: the Dirac equation
      </h3>
      <p className="caption">
        In heavy ions the inner electron moves at a large fraction of the speed of light,
        and Schrödinger's equation stops being right. Dirac's equation gives the exact
        answer for one electron around a point nucleus. Strip every electron but one from
        an atom of charge Z and this is what it looks like. At Z = 1 the curves sit on top
        of each other; slide toward uranium (Z = 92) to see them pull apart.
      </p>
      <Slider
        label="Nuclear charge Z"
        readout={`Z = ${z}`}
        min={1}
        max={118}
        step={1}
        value={z}
        onChange={setZ}
      />
      {error && <p className="error">{error}</p>}
      {data && data.z === z && data.n === n && data.l === l && <DiracBody data={data} W={W} />}
    </section>
  );
}

function DiracBody({ data, W }: { data: DiracRadial; W: number }) {
  const H = 240;
  const grid = data.schrodinger.grid;
  // Past where the Schrodinger curve has fallen to 1e-3 of its peak there is nothing to see.
  const peak = Math.max(...data.schrodinger.values);
  let end = grid.length - 1;
  while (end > 10 && data.schrodinger.values[end] < 1e-3 * peak) end--;
  const curves = [data.schrodinger.values, ...data.components.map((c) => c.probability.values)];
  const top = Math.max(...curves.flatMap((v) => v.slice(0, end + 1)));
  const x = scaleLinear().domain([0, grid[end]]).range([PAD.left, W - PAD.right]);
  const y = scaleLinear().domain([0, top * 1.05]).range([H - PAD.bottom, PAD.top]);
  const path = (v: number[]) =>
    v.slice(0, end + 1).map((p, i) => `${i ? "L" : "M"}${x(grid[i]).toFixed(1)},${y(p).toFixed(1)}`).join("");
  const eS = data.schrodinger_energy.value;
  const rS = data.schrodinger_mean_r.value;
  return (
    <>
      <figure className="plot">
        <figcaption>
          <span className="plot-lead">P(r), Dirac against Schrödinger</span>
          <span className="plot-blurb">
            Dashed: Schrödinger. Solid: Dirac, one curve per j (bold is j ={" "}
            {jFraction(data.l + 0.5)}). Each has area 1.
          </span>
          <span className="plot-provenance">
            r²(g² + f²) <Badge provenance={data.components[0].probability.provenance} />
          </span>
        </figcaption>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Dirac and Schrodinger radial probability">
          <line className="axis" x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom} />
          <line className="axis" x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={H - PAD.bottom} />
          {x.ticks(4).map((r) => (
            <text key={r} className="tick" x={x(r)} y={H - 10} textAnchor="middle">
              {r}
            </text>
          ))}
          <text className="tick" x={W - PAD.right} y={H - PAD.bottom - 6} textAnchor="end">
            r (bohr)
          </text>
          <path d={path(data.schrodinger.values)} className="curve curve-overlay" />
          {data.components.map((c) => (
            <path
              key={c.j}
              d={path(c.probability.values)}
              fill="none"
              stroke={c.j > data.l ? "var(--accent)" : "var(--text)"}
              strokeWidth={c.j > data.l ? 2 : 1.3}
            >
              <title>{`j = ${jFraction(c.j)}`}</title>
            </path>
          ))}
        </svg>
      </figure>
      <table className="line-table">
        <thead>
          <tr>
            <th>State</th>
            <th>Energy (hartree)</th>
            <th>Deeper than Bohr by</th>
            <th>⟨r⟩ vs Schrödinger</th>
            <th>Small part</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Schrödinger</td>
            <td>{fixed(eS, 4)}</td>
            <td>·</td>
            <td>{rS.toPrecision(4)} bohr</td>
            <td>·</td>
          </tr>
          {data.components.map((c) => (
            <tr key={c.j}>
              <td>
                j = {jFraction(c.j)} <Badge provenance={c.energy.provenance} />
              </td>
              <td>{fixed(c.energy.value, 4)}</td>
              <td>{((100 * (eS - c.energy.value)) / Math.abs(eS)).toPrecision(3)}%</td>
              <td>{fixed((100 * (c.mean_r.value - rS)) / rS, 2)}%</td>
              <td>{(100 * c.small_weight.value).toPrecision(3)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Disclosure summary="What the Dirac picture adds, and what it still leaves out" tone="caveat">
        <p className="caption">
          Relativity makes the electron heavier the faster it moves, so it sits deeper and
          closer in: the energy drops and ⟨r⟩ shrinks, most for j = 1/2. The "small part" is
          the share of probability in the second half of the Dirac spinor, which has the
          opposite parity and vanishes as c → ∞. For j = 1/2 the point-nucleus wavefunction
          even diverges gently at r = 0, which is why the grid starts just off the nucleus.
          A real nucleus has a size, and that smooths it out. Also left out: the Lamb shift
          and other QED corrections, which grow fast with Z, and nuclear recoil, since the
          nucleus here is infinitely heavy.
        </p>
      </Disclosure>
    </>
  );
}
