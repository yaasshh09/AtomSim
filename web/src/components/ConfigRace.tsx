import { useRef, useState } from "react";
import { createHFJob, getJobMeta, isHFLevels, watchJob } from "../api/client";
import type { HFLevels } from "../api/types";
import {
  ARGON_CORE,
  RACE_CASES,
  rankRace,
  raceVerdict,
  shortConfig,
  type RaceCase,
} from "../lib/orderingPuzzle";
import { Badge } from "./Badge";
import { fixed } from "../lib/format";

type Solved = { tail: string; levels: HFLevels }[];

/**
 * Which configuration wins past argon, and why orbital energies do not say.
 *
 * Every row is a real Hartree-Fock solve of that configuration, run one after
 * another (they share the job queue and the rate limit with everything else on
 * the page). Results are kept per case, so switching back is free.
 */
export function ConfigRace({ initial }: { initial: string }) {
  const [caseKey, setCaseKey] = useState(
    RACE_CASES.some((c) => c.key === initial) ? initial : "sc",
  );
  const [done, setDone] = useState<Record<string, Solved>>({});
  const [running, setRunning] = useState<string | null>(null);
  const [step, setStep] = useState("");
  const [error, setError] = useState<string | null>(null);
  const runId = useRef(0);
  const race = RACE_CASES.find((c) => c.key === caseKey) as RaceCase;
  const solved = done[caseKey];
  const rows = solved ? rankRace(solved) : null;

  const run = async (c: RaceCase) => {
    const id = ++runId.current;
    setRunning(c.key);
    setError(null);
    const out: Solved = [];
    try {
      for (const [i, tail] of c.tails.entries()) {
        setStep(`solving ${shortConfig(tail)} (${i + 1} of ${c.tails.length})`);
        const job = await createHFJob({ z: c.z, n_electrons: c.n, config: `${ARGON_CORE} ${tail}` });
        await watchJob(job.id, () => {});
        const meta = await getJobMeta(job.id);
        if (!isHFLevels(meta)) throw new Error("the server sent something other than a solve");
        out.push({ tail, levels: meta });
      }
      if (id === runId.current) setDone((d) => ({ ...d, [c.key]: out }));
    } catch (e) {
      if (id === runId.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (id === runId.current) setRunning(null);
    }
  };

  return (
    <section className="config-race" aria-labelledby="config-race-title">
      <h3 id="config-race-title" className="readouts-head">
        Past argon: does 4s or 3d fill first?
      </h3>
      <p className="caption">
        A textbook says 4s fills before 3d because 4s is lower. From scandium on, the 3d
        orbital is the lower one, and 4s still fills first. Pick an atom and solve each
        competing configuration with Hartree-Fock: the configuration with the lowest total
        energy wins, whatever its orbital energies say.
      </p>
      <div className="view-header">
        <div className="seg" role="group" aria-label="atom">
          {RACE_CASES.map((c) => (
            <button
              key={c.key}
              type="button"
              className={c.key === caseKey ? "seg-on" : ""}
              onClick={() => setCaseKey(c.key)}
            >
              {c.label}
            </button>
          ))}
        </div>
        <button
          type="button"
          className="ghost"
          disabled={running !== null}
          onClick={() => void run(race)}
        >
          {solved ? "Solve again" : `Solve ${race.tails.length} configurations`}
        </button>
        {running && <span className="hint">{step}…</span>}
      </div>
      {error && <p className="error">{error}</p>}
      {rows && solved && (
        <>
          <table className="line-table race-table">
            <thead>
              <tr>
                <th>configuration</th>
                <th>E − E(ground) [eV]</th>
                <th>ε(4s) [eV]</th>
                <th>ε(3d) [eV]</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.tail}>
                  <td>{shortConfig(r.tail)}</td>
                  <td>
                    {r.gapEv === 0 ? "0 (ground)" : `+${r.gapEv.toFixed(2)} ± ${r.gapErrEv.toFixed(2)}`}
                  </td>
                  <td>{r.eps4s === null ? "empty" : fixed(r.eps4s, 2)}</td>
                  <td>{r.eps3d === null ? "empty" : fixed(r.eps3d, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="caption">
            {raceVerdict(rows)} <Badge provenance={solved[0].levels.total_energy.provenance} />
          </p>
          <p className="caption">
            These are configuration averages, the energy the solver minimizes, not single
            terms. Hartree-Fock leaves out electron correlation, which can move a gap between
            3d and 4s configurations by close to an eV, so read the gaps as approximate; every
            winner here is the ground configuration NIST lists for that atom. The ± adds the
            two total energies' mesh error bars, a deliberately loose bound.
          </p>
        </>
      )}
    </section>
  );
}
