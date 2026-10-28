import { useRef, useState } from "react";
import { createXrayJob, getJobMeta, isKalpha, watchJob } from "../api/client";
import { fixed } from "../lib/format";
import { moseleyFit, xrayRow, type Kalpha, type XrayRow } from "../lib/xray";
import { Badge } from "./Badge";

/** Every other element, Ne to Kr: enough to draw Moseley's line in a minute or two. */
const ELEMENTS = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36];

/**
 * Moseley's law, and the K-alpha line done properly.
 *
 * Each row is two Hartree-Fock solves of a core-hole ion, assembled on the
 * server. Moseley's one-line formula sits beside it, and both sit beside NIST.
 */
export function XrayPanel() {
  const [rows, setRows] = useState<XrayRow[]>([]);
  const [first, setFirst] = useState<Kalpha | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const runId = useRef(0);

  const run = async () => {
    const id = ++runId.current;
    setRunning(true);
    setError(null);
    setRows([]);
    try {
      for (const z of ELEMENTS) {
        const job = await createXrayJob(z);
        await watchJob(job.id, () => {});
        const meta = await getJobMeta(job.id);
        if (!isKalpha(meta)) throw new Error("the server sent something other than a K-alpha line");
        if (id !== runId.current) return;
        setFirst((f) => f ?? meta);
        setRows((r) => [...r, xrayRow(meta)]);
      }
    } catch (e) {
      if (id === runId.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (id === runId.current) setRunning(false);
    }
  };

  const fit = moseleyFit(rows);

  return (
    <section className="config-race" aria-labelledby="xray-title" data-tour="xray-panel">
      <h3 id="xray-title" className="readouts-head">
        Moseley's law: the K-alpha X-ray line, neon to krypton
      </h3>
      <p className="caption">
        Knock out a 1s electron and a 2p electron drops into the hole, emitting an X-ray. In
        1913 Moseley found that the square root of its energy climbs in a straight line with
        an element's place in the table, which put the table in order of nuclear charge.
        Bohr's model gives the line at once: a 2 → 1 jump in a nucleus of charge Z − 1, the
        other 1s electron screening one unit, E = ¾ (Z − 1)² Ry.
      </p>
      <div className="view-header">
        <button type="button" className="ghost" onClick={() => void run()} disabled={running}>
          {rows.length ? "Solve again" : `Solve ${ELEMENTS.length} elements`}
        </button>
        {running && (
          <span className="hint">
            {rows.length} of {ELEMENTS.length} solved; each is two full solves, so a first
            pass takes a few minutes…
          </span>
        )}
        {first && <Badge provenance={first.delta_scf_ev.provenance} />}
      </div>
      {error && <p className="error">{error}</p>}
      {rows.length > 0 && (
        <>
          <table className="line-table race-table">
            <thead>
              <tr>
                <th>element</th>
                <th>Z</th>
                <th>measured [eV]</th>
                <th>Moseley [eV]</th>
                <th>Hartree-Fock [eV]</th>
                <th>+ relativity [eV]</th>
                <th>√(E/Ry)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.z}>
                  <td>{r.symbol}</td>
                  <td>{r.z}</td>
                  <td>{fixed(r.measured, 1)}</td>
                  <td>{fixed(r.moseley, 0)}</td>
                  <td>{fixed(r.hf, 1)}</td>
                  <td>{fixed(r.hfRel, 1)}</td>
                  <td>{fixed(r.sqrtRy, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {fit && (
            <p className="caption">
              A straight line through the measured √(E/Ry) has slope {fixed(fit.slope, 3)} and
              crosses zero at Z = {fixed(fit.zeroAt, 2)}. Bohr's model says 0.866 and 1.
            </p>
          )}
          <p className="caption">
            The Hartree-Fock column is a Delta-SCF calculation: the ion with a 1s hole and the
            ion with a 2p hole, each solved self-consistently so the other electrons relax
            around the hole, and the line is the difference. It is not relativistic, and it
            falls further short of the measurement with every row, because a krypton 1s
            electron moves at a quarter of the speed of light. The next column adds the
            leading Dirac correction for a hydrogenic ion of charge Z − 1, an estimate kept
            separate so the Hartree-Fock number stays what it is. With it every line lands
            within about 0.15% of the measurement, a little high; what is left is that
            estimate's rough screening, plus the correlation and QED that neither column
            includes. The measured value is
            NIST's (2 Kα₁ + Kα₂)/3, the average a calculation without spin-orbit splitting
            predicts.
          </p>
          {first && (
            <p className="caption">
              Measured: {first.citation}, retrieved {first.retrieved}, vendored in the
              repository.
            </p>
          )}
        </>
      )}
    </section>
  );
}
