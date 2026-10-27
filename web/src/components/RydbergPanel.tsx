import { useEffect, useState } from "react";
import { getRydberg } from "../api/client";
import { fixed } from "../lib/format";
import { bestMeasured, computedAt, type Rydberg } from "../lib/rydberg";
import { Badge } from "./Badge";

const ATOMS = [
  { key: "li", label: "Lithium" },
  { key: "na", label: "Sodium" },
];
const SHOWN_N = [5, 10, 20];

/**
 * An alkali's valence electron, pushed out to n = 20.
 *
 * Far from the core it sees one net charge, so its levels are hydrogen's with n
 * shifted by the quantum defect. The table shows the defect barely moving along
 * each series and collapsing with l.
 */
export function RydbergPanel({ initial }: { initial: string }) {
  const [atom, setAtom] = useState(initial === "li" ? "li" : "na");
  const [data, setData] = useState<Rydberg | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setData(null);
    setError(null);
    getRydberg(atom)
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, [atom]);

  return (
    <section className="config-race" aria-labelledby="rydberg-title">
      <h3 id="rydberg-title" className="readouts-head">
        Rydberg series: hydrogen with n shifted
      </h3>
      <p className="caption">
        Excite an alkali's lone electron far enough and it spends almost all its time outside
        the core, seeing one net charge, so its levels follow hydrogen's: E = −1 / (2(n − δ)²)
        hartree. The quantum defect δ is what the core does on the short stretch of each
        orbit the electron spends inside it.
      </p>
      <div className="view-header">
        <div className="seg" role="group" aria-label="atom">
          {ATOMS.map((a) => (
            <button
              key={a.key}
              type="button"
              className={a.key === atom ? "seg-on" : ""}
              onClick={() => setAtom(a.key)}
            >
              {a.label}
            </button>
          ))}
        </div>
        {data && <Badge provenance={data.series[0].computed[0].defect.provenance} />}
      </div>
      {error && <p className="error">{error}</p>}
      {!data && !error && <p className="hint">Solving four series to n = 20…</p>}
      {data && (
        <>
          <table className="line-table race-table">
            <thead>
              <tr>
                <th>series</th>
                {SHOWN_N.map((n) => (
                  <th key={n}>GSZ δ, n = {n}</th>
                ))}
                <th>NIST δ</th>
              </tr>
            </thead>
            <tbody>
              {data.series.map((s) => {
                const m = bestMeasured(s);
                return (
                  <tr key={s.l}>
                    <td>{"spdf"[s.l]}</td>
                    {SHOWN_N.map((n) => {
                      const q = computedAt(s, n);
                      return (
                        <td key={n}>
                          {q
                            ? `${fixed(q.value, 4)} ± ${(q.provenance.error_estimate ?? 0).toExponential(0)}`
                            : "—"}
                        </td>
                      );
                    })}
                    <td>
                      {m
                        ? `${fixed(m.defect, 4)} ± ${m.uncertainty.toExponential(0)} (n = ${m.n})`
                        : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="caption">
            Read across a row: δ hardly changes from n = 5 to n = 20. Every high orbital of
            one l has the same shape inside the core and only a smaller amplitude there, so
            the core shifts each one by the same fraction of a node. Read down: δ collapses
            with l, because the centrifugal barrier l(l+1)/2r² keeps high-l electrons out of
            the core.
          </p>
          <p className="caption">
            The screened model lands a few hundredths high for s and p. For sodium's f series
            it lands low, at about a third of the measured defect: an f electron barely enters the core, so
            most of its measured defect comes from its field polarizing the core from
            outside, which a fixed central potential cannot do. The computed values are GSZ
            orbital energies (APPROXIMATION), solved on an exponential mesh with each level
            polished by inverse iteration; the error bars are the mesh spread. NIST's terms
            are averaged over J, since a central field has no spin-orbit splitting to compare.
          </p>
          <p className="caption">
            Measured: {data.citation}, retrieved {data.retrieved}, vendored in the repository.
          </p>
        </>
      )}
    </section>
  );
}
