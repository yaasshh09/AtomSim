import { scaleLinear, scaleLog } from "d3-scale";
import { useEffect, useState } from "react";
import { getExchangeHole } from "../api/client";
import type { ExchangeHole } from "../api/types";
import { fixed } from "../lib/format";
import { linePath } from "../lib/plot";
import { usePlotWidth } from "../lib/plotSize";
import { Badge } from "./Badge";
import { Slider } from "./Field";

/** Closed-shell neutral atoms: an open shell here is an average over terms, not a determinant. */
export const CLOSED_SHELL_Z = [2, 4, 10, 12, 18, 20, 30, 36];

const REACH = 4; // bohr either side shown
const H = 260;
const PAD = { left: 52, right: 12, top: 12, bottom: 34 };

/**
 * Where the other electrons are, given one of them: the exchange hole drawn along
 * the line through the nucleus and a pinned electron.
 */
export function ExchangeHolePanel({ z }: { z: number }) {
  const [logA, setLogA] = useState(0); // log10 of a in bohr
  const [data, setData] = useState<ExchangeHole | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { width: W, ref } = usePlotWidth(280);
  const a = 10 ** logA;

  useEffect(() => {
    let live = true;
    const t = setTimeout(() => {
      getExchangeHole(z, a)
        .then((d) => {
          if (live) {
            setData(d);
            setError(null);
          }
        })
        .catch((e) => live && setError(e instanceof Error ? e.message : String(e)));
    }, 200);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [z, a]);

  return (
    <section className="config-race" aria-labelledby="xhole-title" ref={ref} data-tour="xhole-panel">
      <h3 id="xhole-title" className="readouts-head">
        The exchange hole: where the others are, given one electron
      </h3>
      <p className="caption">
        A density says where an electron is likely to be, not how electrons avoid each other.
        Pin one electron down and ask where the rest are. Electrons of the same spin cannot
        share a place, so around the pinned one the others thin out: a hole that follows it,
        holding exactly one electron's worth of charge.
      </p>
      <Slider
        label="Pinned electron's distance from the nucleus"
        readout={`${fixed(data?.a_bohr ?? a, 3)} bohr`}
        min={-1.3}
        max={0.6}
        step={0.01}
        value={logA}
        onChange={setLogA}
      />
      {data && <Badge provenance={data.conditional.provenance} />}
      {error && <p className="error">{error}</p>}
      {data && <HolePlot data={data} W={W} />}
      {data && (
        <>
          <p className="caption">
            Where the pinned electron sits, exactly half the density is left: the electrons
            of the other spin, which Hartree-Fock lets ignore it. The hole integrates to{" "}
            {fixed(data.sum_rule.value, 9)} electrons (NUMERICAL; −1 in theory), a check that
            the solved orbitals are orthonormal.
            {z === 2 &&
              " In helium the hole is the whole same-spin density, everywhere: there is one electron of each spin, so the other one has the opposite spin and ignores the first entirely."}
          </p>
          <p className="caption">
            What is missing is the Coulomb hole, the way opposite-spin electrons also keep
            apart because they repel. Hartree-Fock places them independently, and that
            missing avoidance is the correlation energy. Only closed-shell atoms are drawn: an
            open shell in this solver is an average over its terms, and an average has no
            single determinant with a hole to draw.
          </p>
        </>
      )}
    </section>
  );
}

function HolePlot({ data, W }: { data: ExchangeHole; W: number }) {
  const s = data.density.grid;
  const idx = s.flatMap((v, i) => (Math.abs(v) <= REACH ? [i] : []));
  const xs = idx.map((i) => s[i]);
  const rho = idx.map((i) => data.density.values[i]);
  const cond = idx.map((i) => data.conditional.values[i]);
  const top = Math.max(...rho);
  const floor = top * 1e-6;
  const x = scaleLinear([-REACH, REACH], [PAD.left, W - PAD.right]);
  const y = scaleLog([floor, top * 1.5], [H - PAD.bottom, PAD.top]).clamp(true);
  // Below the axis floor the curve is pinned to it rather than dropped, and the
  // floor is drawn as the axis, so nothing below it is claimed.
  const lifted = (v: number[]) => v.map((u) => Math.max(u, floor));
  return (
    <figure className="plot">
      <figcaption>
        <span className="plot-lead">Electron density along the line through the nucleus</span>
        <span className="plot-blurb">
          Faint: all {data.z} electrons. Bold: the other {data.z - 1}, seen from the pinned
          one at the dashed marker. Log axis, so the hole reads as a dip.
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Density with and without the exchange hole">
        <line className="axis" x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom} />
        <line className="axis" x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={H - PAD.bottom} />
        {y
          .ticks(4)
          .filter((t) => Number.isInteger(Math.log10(t)))
          .map((t) => (
            <text key={t} className="tick" x={PAD.left - 6} y={y(t) + 4} textAnchor="end">
              {t.toExponential(0)}
            </text>
          ))}
        {[-4, -2, 0, 2, 4].map((t) => (
          <text key={t} className="tick" x={x(t)} y={H - 16} textAnchor="middle">
            {t}
          </text>
        ))}
        <text className="tick" x={(PAD.left + W - PAD.right) / 2} y={H - 2} textAnchor="middle">
          position along the line [bohr], nucleus at 0; density in bohr⁻³
        </text>
        <line
          className="axis"
          x1={x(data.a_bohr)}
          x2={x(data.a_bohr)}
          y1={PAD.top}
          y2={H - PAD.bottom}
          strokeDasharray="3 3"
        />
        <path className="curve" d={linePath(xs, lifted(rho), x, y)} opacity={0.35} />
        <path className="curve" d={linePath(xs, lifted(cond), x, y)} strokeWidth={2} />
      </svg>
    </figure>
  );
}
