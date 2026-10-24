import { useRef, useState } from "react";
import { createHFJob, getIonizationReference, getJobMeta, isHFLevels, watchJob } from "../api/client";
import {
  koopmansRow,
  meanAbsError,
  type IonizationReference,
  type KoopmansRow,
} from "../lib/hfVsExperiment";

/**
 * Every atom from helium to calcium, Hartree-Fock against the laboratory.
 *
 * Each row is a real ground-state solve (instant on a warm server, which
 * precomputes all of them) beside NIST's measured first ionization energy.
 * Rows fill in as they land, one solve at a time.
 */
export function HFvsExperiment() {
  const [ref, setRef] = useState<IonizationReference | null>(null);
  const [rows, setRows] = useState<KoopmansRow[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const runId = useRef(0);

  const run = async () => {
    const id = ++runId.current;
    setRunning(true);
    setError(null);
    setRows([]);
    try {
      const reference = ref ?? (await getIonizationReference());
      setRef(reference);
      for (const atom of reference.atoms) {
        const job = await createHFJob({ z: atom.z, n_electrons: atom.z });
        await watchJob(job.id, () => {});
        const meta = await getJobMeta(job.id);
        if (!isHFLevels(meta)) throw new Error("the server sent something other than a solve");
        if (id !== runId.current) return;
        setRows((r) => [...r, koopmansRow(meta, atom)]);
      }
    } catch (e) {
      if (id === runId.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (id === runId.current) setRunning(false);
    }
  };

  const high = rows.filter((r) => r.error > 0).map((r) => r.symbol);
  const low = rows.filter((r) => r.error < 0).map((r) => r.symbol);
  const split = [
    high.length ? `high for ${high.join(", ")}` : "",
    low.length ? `low for ${low.join(", ")}` : "",
  ]
    .filter(Boolean)
    .join("; ");

  return (
    <section className="config-race" aria-labelledby="hf-exp-title">
      <h3 id="hf-exp-title" className="readouts-head">
        Hartree-Fock against experiment, helium to calcium
      </h3>
      <p className="caption">
        Koopmans' theorem reads an ionization energy straight off a Hartree-Fock solve:
        minus the energy of the highest occupied orbital. Here it is for all nineteen
        atoms, beside the measured value.
      </p>
      <div className="view-header">
        <button type="button" className="ghost" onClick={() => void run()} disabled={running}>
          {rows.length ? "Solve again" : "Solve all 19 atoms"}
        </button>
        {running && (
          <span className="hint">
            {rows.length} of {ref?.atoms.length ?? 19} solved…
          </span>
        )}
      </div>
      {error && <p className="error">{error}</p>}
      {rows.length > 0 && (
        <>
          <table className="line-table race-table">
            <thead>
              <tr>
                <th>atom</th>
                <th>orbital</th>
                <th>measured IE [eV]</th>
                <th>Koopmans −ε [eV]</th>
                <th>Koopmans − measured [eV]</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.z}>
                  <td>{r.symbol}</td>
                  <td>{r.orbital}</td>
                  <td>{r.nist.toFixed(3)}</td>
                  <td>{r.koopmans.toFixed(3)}</td>
                  <td>{`${r.error > 0 ? "+" : ""}${r.error.toFixed(2)}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="caption">
            Mean |error| {meanAbsError(rows).toFixed(2)} eV over {rows.length} atoms
            {split ? `; Koopmans comes out ${split}` : ""}. Two neglected effects pull
            opposite ways. Freezing the orbitals ignores how the ion relaxes once an
            electron leaves, which overestimates the IE and dominates where removing an
            electron reshapes what is left (helium, neon). Hartree-Fock's missing
            correlation underestimates it, and shows through where relaxation is small:
            an alkali's lone electron leaves a closed core almost untouched. Hartree-Fock
            also averages over the terms of an open shell, while the measurement runs
            from one ground term to another.
          </p>
          {ref && (
            <p className="caption">
              Measured: {ref.citation}, retrieved {ref.retrieved}, vendored in the
              repository rather than queried live.
            </p>
          )}
        </>
      )}
    </section>
  );
}
