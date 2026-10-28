import { useEffect, useState } from "react";
import { getDimensions } from "../api/client";
import type { Dimensions } from "../api/types";
import { fixed } from "../lib/format";
import { Badge } from "./Badge";

/**
 * Hydrogen's ladder with the same pull in two, three and four dimensions: what the
 * number of dimensions alone does to the levels and how many states share each.
 */
export function DimensionsPanel({ system }: { system: string }) {
  const [data, setData] = useState<Dimensions | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setData(null);
    setError(null);
    getDimensions(system)
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, [system]);

  return (
    <section className="config-race" aria-labelledby="dims-title" data-tour="dims-panel">
      <h3 id="dims-title" className="readouts-head">
        The same atom in two, three and four dimensions
      </h3>
      <p className="caption">
        Keep the −Z/r pull and change the number of dimensions. The Schrödinger equation still
        solves exactly, with n shifted by (D − 3)/2: in two dimensions the ground state is four
        times deeper, in four it is shallower, and the number of states sharing each level
        changes with the room there is to point in.
      </p>
      {error && <p className="error">{error}</p>}
      {data && (
        <>
          <table className="line-table race-table">
            <thead>
              <tr>
                <th>n</th>
                {data.dimensions.map((d) => (
                  <th key={d.dim}>{d.dim}-D energy [eV] (states)</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.dimensions[0].levels.map((_, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  {data.dimensions.map((d) => (
                    <td key={d.dim}>
                      {fixed(d.levels[i].energy_ev.value, 3)} ({d.levels[i].degeneracy})
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="caption">
            Only the three-dimensional column is a real atom{" "}
            <Badge provenance={data.dimensions[1].levels[0].energy_ev.provenance} />. The others
            are counterfactual twice over{" "}
            <Badge provenance={data.dimensions[0].levels[0].energy_ev.provenance} />: space has
            three dimensions, and in two or four Gauss's law would not give a 1/r pull at all. In
            two it gives ln r, which binds every state so none can escape; in four it gives 1/r²,
            and a 1/r² pull has no ground state to settle into. Keeping −Z/r isolates what the
            dimension alone does. The closed forms are checked against a numerical solve of the
            D-dimensional radial equation to a millionth of a hartree.
          </p>
        </>
      )}
    </section>
  );
}
