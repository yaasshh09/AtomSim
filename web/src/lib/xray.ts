import type { Quantity } from "../api/types";

export interface XrayLine {
  theory_ev: number;
  theory_uncertainty_ev: number;
  experiment_ev: number;
  experiment_uncertainty_ev: number;
  experiment_kind: string;
  blend: string | null;
}

/** One element's K-alpha line, as an /api/jobs/xray meta payload. */
export interface Kalpha {
  kind: "kalpha";
  z: number;
  symbol: string;
  hole_1s_config: string;
  hole_2p_config: string;
  delta_scf_ev: Quantity;
  relativity_ev: Quantity;
  moseley_ev: Quantity;
  ka1: XrayLine;
  ka2: XrayLine;
  citation: string;
  retrieved: string;
}

export interface XrayRow {
  z: number;
  symbol: string;
  /** (2 Kα1 + Kα2) / 3, eV: the measured line a calculation without spin-orbit predicts. */
  measured: number;
  moseley: number;
  hf: number;
  /** hf plus the Dirac estimate, eV. */
  hfRel: number;
  /** sqrt(measured / Ry), the quantity Moseley plotted against Z. */
  sqrtRy: number;
}

const RYDBERG_EV = 13.605693122994;

export function xrayRow(k: Kalpha): XrayRow {
  const measured = (2 * k.ka1.experiment_ev + k.ka2.experiment_ev) / 3;
  return {
    z: k.z,
    symbol: k.symbol,
    measured,
    moseley: k.moseley_ev.value,
    hf: k.delta_scf_ev.value,
    hfRel: k.delta_scf_ev.value + k.relativity_ev.value,
    sqrtRy: Math.sqrt(measured / RYDBERG_EV),
  };
}

/**
 * Least-squares line through (Z, sqrt(E/Ry)): Moseley's slope and intercept.
 * For a 2 -> 1 line Bohr predicts slope sqrt(3/4) = 0.866 and a zero at Z = 1.
 */
export function moseleyFit(rows: XrayRow[]): { slope: number; zeroAt: number } | null {
  if (rows.length < 2) return null;
  const n = rows.length;
  const mx = rows.reduce((s, r) => s + r.z, 0) / n;
  const my = rows.reduce((s, r) => s + r.sqrtRy, 0) / n;
  const sxy = rows.reduce((s, r) => s + (r.z - mx) * (r.sqrtRy - my), 0);
  const sxx = rows.reduce((s, r) => s + (r.z - mx) ** 2, 0);
  const slope = sxy / sxx;
  return { slope, zeroAt: mx - my / slope };
}
