import type { HFLevels, Quantity } from "../api/types";
import { koopmansRow, type KoopmansRow } from "./hfVsExperiment";

/** One vendored NIST ion, as /api/isoelectronic serves it. */
export interface IsoelectronicIon {
  z: number;
  symbol: string;
  spectrum: string;
  ie_ev: number;
  uncertainty_ev: number;
  /** NIST's own marking for where the number came from. */
  kind: "measured" | "theoretical" | "semi-empirical";
}

export interface IsoelectronicSequence {
  n_electrons: number;
  isoelectronic_with: string;
  config: string;
  /** E(Z) = e0 Z^2 + e1 Z + E2 + ..., hartree. */
  e0: Quantity;
  e1: Quantity;
  ions: IsoelectronicIon[];
  citation: string;
  retrieved: string;
  note: string;
}

export interface IsoRow extends KoopmansRow {
  label: string;
  kind: IsoelectronicIon["kind"];
  /** E_HF - e0 Z^2 - e1 Z, hartree: what is left after the first two orders. */
  residual: number;
  /** Its error bar: the solve's own, plus Z times e1's. */
  residualBar: number;
}

const SUPERSCRIPT = "⁰¹²³⁴⁵⁶⁷⁸⁹";

/** "Li⁺", "C⁴⁺", or the bare symbol for the neutral atom. */
export function ionLabel(symbol: string, charge: number): string {
  if (charge === 0) return symbol;
  const digits = charge === 1 ? "" : [...String(charge)].map((d) => SUPERSCRIPT[+d]).join("");
  return `${symbol}${digits}⁺`;
}

/** One ion's Hartree-Fock solve, read against the 1/Z expansion and NIST. */
export function isoRow(levels: HFLevels, seq: IsoelectronicSequence, ion: IsoelectronicIon): IsoRow {
  const z = ion.z;
  const e = levels.total_energy;
  return {
    ...koopmansRow(levels, { ...ion, ground_level: "" }),
    label: ionLabel(ion.symbol, z - seq.n_electrons),
    kind: ion.kind,
    residual: e.value - seq.e0.value * z * z - seq.e1.value * z,
    residualBar:
      (e.provenance.error_estimate ?? 0) + z * (seq.e1.provenance.error_estimate ?? 0),
  };
}
