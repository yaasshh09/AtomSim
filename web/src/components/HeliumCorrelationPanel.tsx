import { useEffect, useState } from "react";
import { getHeliumCorrelation } from "../api/client";
import type { HeliumCorrelation } from "../api/types";
import { fixed } from "../lib/format";
import { Badge } from "./Badge";

/**
 * Helium beyond Hartree-Fock: Hylleraas's correlated wavefunction, the
 * correlation energy it measures, and the ionization energy it fixes.
 */
export function HeliumCorrelationPanel() {
  const [data, setData] = useState<HeliumCorrelation | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    getHeliumCorrelation()
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, []);

  const ieRows: [string, number][] = data
    ? [
        ["Hartree-Fock (Delta-SCF)", data.ie_hartree_fock_ev.value],
        ["correlated, nucleus held fixed", data.ie_correlated_ev.value],
        ["correlated, helium-4's reduced mass", data.ie_reduced_mass_ev.value],
      ]
    : [];

  return (
    <section className="config-race" aria-labelledby="corr-title">
      <h3 id="corr-title" className="readouts-head">
        Beyond Hartree-Fock: helium's correlation energy
      </h3>
      <p className="caption">
        Hartree-Fock lets each electron see the other only as an averaged cloud. The real
        two keep apart instant by instant, and that is worth energy. In 1929 Hylleraas
        wrote helium's wavefunction with the distance between the electrons, r₁₂, as one
        of its variables, so it can describe them avoiding each other. Each row below adds
        terms; each energy is an upper bound on the exact one, and the bounds fall toward it.
      </p>
      {error && <p className="error">{error}</p>}
      {!data && !error && <p className="hint">Solving the series…</p>}
      {data && (
        <>
          <table className="line-table race-table">
            <thead>
              <tr>
                <th>terms</th>
                <th>energy [hartree]</th>
                <th>above exact [hartree]</th>
                <th>−V/T</th>
              </tr>
            </thead>
            <tbody>
              {data.series.map((s) => (
                <tr key={s.omega}>
                  <td>{s.terms}</td>
                  <td>{fixed(s.energy.value, 7)}</td>
                  <td>{(s.energy.value - data.exact_energy).toExponential(1)}</td>
                  <td>{fixed(s.virial, 6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="caption">
            The one-term row is e^(−ζ(r₁ + r₂)): two hydrogen-like electrons, each shielding
            the nucleus from the other by 5/16 of a charge. The exact value,{" "}
            {fixed(data.exact_energy, 7)} hartree, is from {data.exact_citation}. −V/T is 2 at
            every row, as the virial theorem requires once the scale is optimized.
          </p>
          <table className="line-table race-table">
            <tbody>
              <tr>
                <td>Hartree-Fock energy (this solver)</td>
                <td>{fixed(data.hartree_fock_ev.value, 3)} eV</td>
                <td>
                  <Badge provenance={data.hartree_fock_ev.provenance} />
                </td>
              </tr>
              <tr>
                <td>correlated energy, {data.series[data.series.length - 1].terms} terms</td>
                <td>{fixed(data.correlated_ev.value, 3)} eV</td>
                <td>
                  <Badge provenance={data.correlated_ev.provenance} />
                </td>
              </tr>
              <tr>
                <td>
                  <strong>correlation energy</strong>
                </td>
                <td>
                  <strong>{fixed(data.correlation_ev.value, 3)} eV</strong>
                </td>
                <td>
                  <Badge provenance={data.correlation_ev.provenance} />
                </td>
              </tr>
            </tbody>
          </table>
          <table className="line-table race-table">
            <thead>
              <tr>
                <th>ionization energy</th>
                <th>eV</th>
                <th>minus measured [eV]</th>
              </tr>
            </thead>
            <tbody>
              {ieRows.map(([label, v]) => (
                <tr key={label}>
                  <td>{label}</td>
                  <td>{fixed(v, 4)}</td>
                  <td>{fixed(v - data.measured_ie_ev, 4)}</td>
                </tr>
              ))}
              <tr>
                <td>measured (NIST)</td>
                <td>{fixed(data.measured_ie_ev, 4)}</td>
                <td>—</td>
              </tr>
            </tbody>
          </table>
          <p className="caption">
            The correlation energy is about 1% of helium's total, and all of it lands on the
            ionization energy, because He⁺ has one electron and nothing to correlate.
            Hartree-Fock misses the measurement by over an electronvolt; the correlated value,
            once the nucleus is given its finite mass, is within a millielectronvolt. What
            remains is the mass-polarization term, relativity and QED, none of which are
            computed here. Measured: {data.measured_citation}, retrieved{" "}
            {data.measured_retrieved}.
          </p>
        </>
      )}
    </section>
  );
}
