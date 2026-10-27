import type { Quantity } from "../api/types";

export interface DefectPoint {
  n: number;
  energy: Quantity;
  defect: Quantity;
}

export interface MeasuredDefect {
  n: number;
  defect: number;
  uncertainty: number;
}

export interface DefectSeries {
  l: number;
  computed: DefectPoint[];
  measured: MeasuredDefect[];
}

export interface Rydberg {
  atom: string;
  symbol: string;
  series: DefectSeries[];
  citation: string;
  retrieved: string;
  note: string;
}

/** The computed defect at principal number n, or null when the series has none there. */
export function computedAt(series: DefectSeries, n: number): Quantity | null {
  return series.computed.find((p) => p.n === n)?.defect ?? null;
}

/**
 * The measured defect worth quoting for a series: the highest n whose
 * uncertainty is below `tolerance`. High n is where the Rydberg formula is
 * cleanest, but a high-n level known to 0.1 cm⁻¹ can carry a defect uncertainty
 * bigger than the defect itself (lithium's f series), so precision gates it.
 * Falls back to the most precise point when none clears the bar.
 */
export function bestMeasured(series: DefectSeries, tolerance = 1e-3): MeasuredDefect | null {
  if (series.measured.length === 0) return null;
  const precise = series.measured.filter((m) => m.uncertainty < tolerance);
  if (precise.length) return precise.reduce((a, b) => (b.n > a.n ? b : a));
  return series.measured.reduce((a, b) => (b.uncertainty < a.uncertainty ? b : a));
}
