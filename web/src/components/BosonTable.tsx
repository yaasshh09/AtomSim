import { scaleLinear, scaleLog } from "d3-scale";
import { useRef, useState } from "react";
import { createHFJob, getJobMeta, isHFLevels, watchJob } from "../api/client";
import { fixed } from "../lib/format";
import { linePath } from "../lib/plot";
import { usePlotWidth } from "../lib/plotSize";
import { Badge } from "./Badge";
import type { Quantity } from "../api/types";

const ZS = Array.from({ length: 19 }, (_, i) => i + 2); // He to Ca
const SYMBOLS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca".split(" ");
const H = 240;
const PAD = { left: 48, right: 12, top: 12, bottom: 30 };

interface Row {
  z: number;
  real: Quantity;
  bosonic: Quantity;
}

/**
 * Every atom from helium to calcium twice: as it is, and with the exclusion
 * principle switched off so every electron sits in 1s. The real ionization
 * energies have the periodic table's sawtooth; the collapsed ones have none.
 */
export function BosonTable() {
  const [rows, setRows] = useState<Row[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const runId = useRef(0);
  const { width: W, ref } = usePlotWidth(280);

  const run = async () => {
    const id = ++runId.current;
    setRunning(true);
    setError(null);
    setRows([]);
    try {
      for (const z of ZS) {
        const job = await createHFJob({ z, n_electrons: z, exchange: false, pauli: false });
        await watchJob(job.id, () => {});
        const meta = await getJobMeta(job.id);
        if (!isHFLevels(meta) || !meta.collapse) throw new Error("expected a collapse comparison");
        if (id !== runId.current) return;
        const c = meta.collapse;
        setRows((r) => [...r, { z, real: c.real_ionization_ev, bosonic: c.collapsed_ionization_ev }]);
      }
    } catch (e) {
      if (id === runId.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (id === runId.current) setRunning(false);
    }
  };

  const x = scaleLinear([2, 20], [PAD.left, W - PAD.right]);
  const y = scaleLog([3, 1000], [H - PAD.bottom, PAD.top]).clamp(true);
  const zs = rows.map((r) => r.z);

  return (
    <section className="config-race" aria-labelledby="boson-title" ref={ref} data-tour="boson-table">
      <h3 id="boson-title" className="readouts-head">
        The periodic table without the exclusion principle
      </h3>
      <p className="caption">
        Every atom from helium to calcium, solved twice: as it is, and with the exclusion
        principle off, so every electron falls into 1s. Read the ionization energies across.
      </p>
      <div className="view-header">
        <button type="button" className="ghost" onClick={() => void run()} disabled={running}>
          {rows.length ? "Solve again" : `Solve ${ZS.length} atoms twice`}
        </button>
        {running && (
          <span className="hint">
            {rows.length} of {ZS.length} solved…
          </span>
        )}
        {rows[0] && <Badge provenance={rows[0].bosonic.provenance} />}
      </div>
      {error && <p className="error">{error}</p>}
      {rows.length > 1 && (
        <figure className="plot">
          <figcaption>
            <span className="plot-lead">Koopmans ionization energy against Z, log axis</span>
            <span className="plot-blurb">
              Solid: the real atoms. Dashed: every electron in 1s.
            </span>
          </figcaption>
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Ionization energy with and without the exclusion principle">
            <line className="axis" x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom} />
            <line className="axis" x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={H - PAD.bottom} />
            {[10, 100, 1000].map((t) => (
              <text key={t} className="tick" x={PAD.left - 6} y={y(t) + 4} textAnchor="end">
                {t} eV
              </text>
            ))}
            {[2, 10, 18].map((z) => (
              <text key={z} className="tick" x={x(z)} y={H - 10} textAnchor="middle">
                {SYMBOLS[z - 1]}
              </text>
            ))}
            <path className="curve" d={linePath(zs, rows.map((r) => r.real.value), x, y)} />
            <path
              className="curve curve-overlay"
              d={linePath(zs, rows.map((r) => r.bosonic.value), x, y)}
            />
          </svg>
        </figure>
      )}
      {rows.length > 0 && (
        <>
          <table className="line-table race-table">
            <thead>
              <tr>
                <th>atom</th>
                <th>real IE [eV]</th>
                <th>all in 1s [eV]</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.z}>
                  <td>{SYMBOLS[r.z - 1]}</td>
                  <td>{fixed(r.real.value, 2)}</td>
                  <td>{fixed(r.bosonic.value, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="caption">
            The real column drops sharply at every alkali, lithium, sodium, potassium, because the
            next electron has to start a new shell further out. The collapsed column only climbs:
            each added electron joins the same 1s, closer to a bigger nucleus, so nothing ever
            repeats. Shells, valence and the table's rows all come from the one rule switched off
            here. Both columns are Koopmans estimates (minus the outermost orbital energy), and
            the collapsed one is a counterfactual solved rigorously.
          </p>
        </>
      )}
    </section>
  );
}
