import { useRef, useState } from "react";
import { createHFJob, getIsoelectronic, getJobMeta, isHFLevels, watchJob } from "../api/client";
import { fixed } from "../lib/format";
import { isoRow, type IsoelectronicSequence, type IsoRow } from "../lib/isoelectronic";

const SEQUENCES = [
  { n: 2, label: "He-like" },
  { n: 3, label: "Li-like" },
  { n: 4, label: "Be-like" },
  { n: 10, label: "Ne-like" },
  { n: 11, label: "Na-like" },
];

const KIND_MARK: Record<IsoRow["kind"], string> = {
  measured: "",
  theoretical: " (theory)",
  "semi-empirical": " (interp.)",
};

/**
 * One electron count, the nuclear charge turned up: He, Li⁺, Be²⁺, and on.
 *
 * Each row is a real Hartree-Fock solve of one ion. The residual column takes
 * off the two orders perturbation theory gives exactly, and what is left should
 * settle onto a constant as Z grows.
 */
export function IsoelectronicPanel() {
  const [n, setN] = useState(2);
  const [seq, setSeq] = useState<IsoelectronicSequence | null>(null);
  const [rows, setRows] = useState<IsoRow[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const runId = useRef(0);

  const pick = (next: number) => {
    runId.current++;
    setN(next);
    setRows([]);
    setSeq(null);
    setRunning(false);
    setError(null);
  };

  const run = async () => {
    const id = ++runId.current;
    setRunning(true);
    setError(null);
    setRows([]);
    try {
      const s = await getIsoelectronic(n);
      if (id !== runId.current) return;
      setSeq(s);
      for (const ion of s.ions) {
        const job = await createHFJob({ z: ion.z, n_electrons: n });
        await watchJob(job.id, () => {});
        const meta = await getJobMeta(job.id);
        if (!isHFLevels(meta)) throw new Error("the server sent something other than a solve");
        if (id !== runId.current) return;
        setRows((r) => [...r, isoRow(meta, s, ion)]);
      }
    } catch (e) {
      if (id === runId.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (id === runId.current) setRunning(false);
    }
  };

  const total = seq?.ions.length ?? 11;

  return (
    <section className="config-race" aria-labelledby="iso-title">
      <h3 id="iso-title" className="readouts-head">
        Isoelectronic sequences: same electrons, more charge
      </h3>
      <p className="caption">
        Keep the electron count and turn up the nucleus. Perturbation theory in 1/Z gives
        the first two terms of the energy exactly: E(Z) = E₀Z² + E₁Z + E₂ + …, with E₀ the
        bare-nucleus energy and E₁ the electrons' repulsion at first order. Each ion below is
        a Hartree-Fock solve with those two terms taken off. What is left, E₂ and smaller,
        is where the electrons start reshaping each other's orbitals.
      </p>
      <div className="view-header">
        <div className="seg" role="group" aria-label="sequence">
          {SEQUENCES.map((s) => (
            <button
              key={s.n}
              type="button"
              className={s.n === n ? "seg-on" : ""}
              onClick={() => pick(s.n)}
            >
              {s.label}
            </button>
          ))}
        </div>
        <button type="button" className="ghost" onClick={() => void run()} disabled={running}>
          {rows.length ? "Solve again" : `Solve ${total} ions`}
        </button>
        {running && (
          <span className="hint">
            {rows.length} of {total} solved…
          </span>
        )}
      </div>
      {error && <p className="error">{error}</p>}
      {seq && rows.length > 0 && (
        <>
          <p className="caption">
            {seq.isoelectronic_with}-like, {seq.config}: E₀ = {fixed(seq.e0.value, 6)} hartree
            (EXACT), E₁ = {fixed(seq.e1.value, 5)} ±{" "}
            {seq.e1.provenance.error_estimate?.toExponential(0)} hartree (NUMERICAL quadrature
            of a closed form). E₁ is the same for Hartree-Fock and the real ion, because
            correlation first enters at second order.
          </p>
          <table className="line-table race-table">
            <thead>
              <tr>
                <th>ion</th>
                <th>Z</th>
                <th>E − E₀Z² − E₁Z [hartree]</th>
                <th>NIST IE [eV]</th>
                <th>Koopmans −ε [eV]</th>
                <th>Koopmans − NIST [eV]</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.z}>
                  <td>{r.label}</td>
                  <td>{r.z}</td>
                  <td>
                    {fixed(r.residual, 4)} ± {r.residualBar.toExponential(0)}
                  </td>
                  <td>
                    {r.nist.toFixed(3)}
                    {KIND_MARK[r.kind]}
                  </td>
                  <td>{fixed(r.koopmans, 3)}</td>
                  <td>{`${r.error > 0 ? "+" : ""}${fixed(r.error, 2)}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="caption">
            The residual changes less and less from row to row: it is closing on E₂ like 1/Z.
            For the helium sequence Hartree-Fock's E₂ is −0.1110 hartree and the exact atom's
            is −0.1577. The gap, −0.047 hartree, is the correlation energy a two-electron ion
            keeps however high Z goes. The error bar grows as Z² because the solver's mesh
            error scales with the energy.
          </p>
          <p className="caption">
            Koopmans' error can change sign along a sequence. Frozen orbitals push it high, by
            less and less as the nucleus dominates. Relativity pushes the other way: NIST's
            values include it and this solve does not, and it deepens a 1s orbital by about
            Z⁴α²/8 hartree, 0.7 eV at Z = 8 and 3.7 eV at Z = 12. In the helium sequence it
            takes over at oxygen.
          </p>
          <p className="caption">
            NIST values: {seq.citation}, retrieved {seq.retrieved}, vendored in the repository.
            "(theory)" and "(interp.)" mark NIST's theoretical and semi-empirical entries; for
            two-electron ions the calculations are more precise than any measurement.
          </p>
        </>
      )}
    </section>
  );
}
