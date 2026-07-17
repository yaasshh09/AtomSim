// Single source of truth for force-law preset parameters — mirrors the Python
// ParamSpec ranges in src/atomsim/numerics/force_law.py.
export type ForcePreset =
  | "powerlaw"
  | "yukawa"
  | "harmonic"
  | "finitewell"
  | "coulombcore";

export interface ParamSpec {
  name: string;
  min: number;
  max: number;
  default: number;
  unit: string;
  step: number;
}

export const PRESET_PARAMS: Record<ForcePreset, ParamSpec[]> = {
  powerlaw: [{ name: "p", min: 0.5, max: 1.5, default: 1.0, unit: "", step: 0.05 }],
  yukawa: [{ name: "lambda", min: 0.5, max: 20, default: 3, unit: "a₀", step: 0.5 }],
  harmonic: [{ name: "omega", min: 0.05, max: 1.0, default: 0.3, unit: "", step: 0.05 }],
  finitewell: [
    { name: "v0", min: 0.1, max: 5, default: 2, unit: "Ha", step: 0.1 },
    { name: "a", min: 0.5, max: 10, default: 3, unit: "a₀", step: 0.5 },
  ],
  coulombcore: [{ name: "core", min: 0, max: 1, default: 0.2, unit: "", step: 0.05 }],
};

export const PRESET_LABELS: Record<ForcePreset, string> = {
  powerlaw: "Power law  −Z/rᵖ",
  yukawa: "Yukawa / screened  −(Z/r)e^(−r/λ)",
  harmonic: "Harmonic  ½kr²",
  finitewell: "Finite well  −V₀ (r<a)",
  coulombcore: "Coulomb + core  −Z/r + c/r²",
};

export function defaultParams(preset: ForcePreset): Record<string, number> {
  const out: Record<string, number> = {};
  for (const spec of PRESET_PARAMS[preset]) out[spec.name] = spec.default;
  return out;
}

export function clampParam(spec: ParamSpec, value: number): number {
  return Math.min(Math.max(value, spec.min), spec.max);
}

/** classically-allowed r-span [rIn, rOut] where E > V(r) along the curve, or null. */
export function allowedSpan(
  r: number[],
  vEv: number[],
  energyEv: number,
): [number, number] | null {
  const inside: number[] = [];
  for (let i = 0; i < r.length; i++) if (energyEv > vEv[i]) inside.push(r[i]);
  if (inside.length === 0) return null;
  return [inside[0], inside[inside.length - 1]];
}
