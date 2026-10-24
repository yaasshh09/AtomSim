import type { HFLevels } from "../api/types";
import { fixed } from "./format";

/** The argon core every configuration below shares. */
export const ARGON_CORE = "1s2 2s2 2p6 3s2 3p6";
const HARTREE_EV = 27.211386245988;

/**
 * One atom or ion and the configurations that compete for it.
 *
 * Each case is pinned by an engine test (tests/test_hf_past_argon.py) that
 * solves the same configurations and asserts the same winner, so the panel
 * cannot print an ordering the physics has stopped producing.
 */
export interface RaceCase {
  key: string;
  label: string;
  z: number;
  n: number;
  /** Outside the argon core, as the server parses them: "3d1 4s2". */
  tails: string[];
}

export const RACE_CASES: RaceCase[] = [
  { key: "k", label: "K", z: 19, n: 19, tails: ["4s1", "3d1"] },
  { key: "ca", label: "Ca", z: 20, n: 20, tails: ["4s2", "3d1 4s1", "3d2"] },
  { key: "sc", label: "Sc", z: 21, n: 21, tails: ["3d1 4s2", "3d2 4s1"] },
  { key: "sc+", label: "Sc⁺", z: 21, n: 20, tails: ["3d1 4s1", "3d2", "4s2"] },
];

/** "3d1 4s2" -> "[Ar] 3d 4s²", the way a chemistry book writes it. */
export function shortConfig(tail: string): string {
  const sup: Record<string, string> = { "1": "", "2": "²", "3": "³" };
  return (
    "[Ar] " +
    tail
      .split(" ")
      .map((p) => p.replace(/^(\d[spdf])(\d+)$/, (_, s: string, q: string) => s + (sup[q] ?? q)))
      .join(" ")
  );
}

export interface RaceRow {
  tail: string;
  /** E - E(winner), eV; 0 for the winner. */
  gapEv: number;
  /** A conservative bound on that gap: the two mesh error bars added. */
  gapErrEv: number;
  eps4s: number | null;
  eps3d: number | null;
}

function eps(levels: HFLevels, n: number, l: number): number | null {
  const o = levels.orbitals.find((x) => x.n === n && x.l === l);
  return o ? o.energy_ev.value : null;
}

/** The configurations ranked by total energy, lowest first. */
export function rankRace(results: { tail: string; levels: HFLevels }[]): RaceRow[] {
  const sorted = [...results].sort(
    (a, b) => a.levels.total_energy.value - b.levels.total_energy.value,
  );
  const best = sorted[0];
  const errOf = (lv: HFLevels) => lv.total_energy.provenance.error_estimate ?? 0;
  return sorted.map(({ tail, levels }) => ({
    tail,
    gapEv: (levels.total_energy.value - best.levels.total_energy.value) * HARTREE_EV,
    gapErrEv: levels === best.levels ? 0 : (errOf(levels) + errOf(best.levels)) * HARTREE_EV,
    eps4s: eps(levels, 4, 0),
    eps3d: eps(levels, 3, 2),
  }));
}

/**
 * What the ranking says, in words. The interesting case is the one where the
 * winner's own orbital energies point the other way: 3d below 4s, 4s filled.
 */
export function raceVerdict(rows: RaceRow[]): string {
  const win = rows[0];
  const lead = `${shortConfig(win.tail)} has the lowest total energy, so it is the ground configuration`;
  if (rows.length > 1) {
    const next = rows[1];
    const lead2 = `${lead}, ${next.gapEv.toFixed(2)} eV below ${shortConfig(next.tail)}.`;
    if (win.eps3d !== null && win.eps4s !== null && win.eps3d < win.eps4s) {
      return (
        `${lead2} Yet in that winner the 3d orbital (ε = ${fixed(win.eps3d, 2)} eV) lies ` +
        `below the 4s (ε = ${fixed(win.eps4s, 2)} eV). Filling the deeper orbital does not ` +
        "win, because putting a second electron into the compact 3d costs more repulsion " +
        "than the orbital energy gains: the total energy decides, not one orbital's ε."
      );
    }
    return `${lead2} Here the orbital energies agree with the total energies, so there is no puzzle yet.`;
  }
  return `${lead}.`;
}
