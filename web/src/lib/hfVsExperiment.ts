import type { HFLevels } from "../api/types";

/** One NIST entry, as /api/reference/ionization serves it. */
export interface NistIonization {
  z: number;
  symbol: string;
  ground_level: string;
  ie_ev: number;
  uncertainty_ev: number;
}

export interface IonizationReference {
  citation: string;
  retrieved: string;
  note: string;
  atoms: NistIonization[];
}

export interface KoopmansRow {
  z: number;
  symbol: string;
  nist: number;
  /** -epsilon of the highest occupied orbital, eV: Koopmans' ionization energy. */
  koopmans: number;
  /** Which orbital that was, "4s". */
  orbital: string;
  /** koopmans - nist, eV. Positive means Koopmans came out high. */
  error: number;
}

/** Koopmans' IE from a Hartree-Fock solve, beside the measured one. */
export function koopmansRow(levels: HFLevels, nist: NistIonization): KoopmansRow {
  const occupied = levels.orbitals.filter((o) => o.occupancy > 0);
  const top = occupied.reduce((a, b) => (b.energy.value > a.energy.value ? b : a));
  const koopmans = -top.energy_ev.value;
  return {
    z: nist.z,
    symbol: nist.symbol,
    nist: nist.ie_ev,
    koopmans,
    orbital: top.label,
    error: koopmans - nist.ie_ev,
  };
}

/** The mean of |error| over the rows, eV. */
export function meanAbsError(rows: KoopmansRow[]): number {
  return rows.length ? rows.reduce((s, r) => s + Math.abs(r.error), 0) / rows.length : 0;
}
