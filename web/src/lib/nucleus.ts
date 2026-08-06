import type { SystemInfo } from "../api/types";

export type NucleusMode = "hidden" | "true-scale" | "marker";

export const NUCLEUS_MODES: { value: NucleusMode; label: string }[] = [
  { value: "marker", label: "visible marker" },
  { value: "true-scale", label: "true scale" },
  { value: "hidden", label: "hidden" },
];

/** VISUAL LIBERTY: the marker sphere radius is camera distance / MARKER_DIVISOR. */
export const MARKER_DIVISOR = 90;

export interface NucleusSphere {
  kind: "true-scale" | "marker";
  radius: number; // bohr (scene units)
  magnification: number; // radius / physical r_rms (1 for true scale)
}

/** What to draw at the origin, or null (hidden mode, or no measured nucleus). */
export function nucleusSphere(
  mode: NucleusMode,
  radiusBohr: number | null,
  cameraDistance: number,
): NucleusSphere | null {
  if (mode === "hidden" || radiusBohr === null || radiusBohr <= 0) return null;
  if (mode === "true-scale") {
    return { kind: "true-scale", radius: radiusBohr, magnification: 1 };
  }
  const radius = cameraDistance / MARKER_DIVISOR;
  return { kind: "marker", radius, magnification: radius / radiusBohr };
}

/** "6,300×", two significant figures, thousands separators. */
export function formatMagnification(x: number): string {
  const rounded = Number(x.toPrecision(2));
  return `${rounded.toLocaleString("en-US")}×`;
}

/** Honest one-line caption for the nucleus overlay; null when nothing to say. */
export function nucleusCaption(
  mode: NucleusMode,
  system: SystemInfo | null | undefined,
  sphere: NucleusSphere | null,
): string | null {
  if (mode === "hidden" || !system) return null;
  if (system.nuclear_radius === null || system.nuclear_radius_fm === null) {
    return "the “nucleus” here is a point lepton, no measured size to draw";
  }
  const fm = system.nuclear_radius_fm.value.toFixed(3);
  const bohr = system.nuclear_radius.value.toExponential(1).replace("e-", "e-");
  if (mode === "true-scale") {
    return (
      `nucleus at true scale: r_rms = ${fm} fm (${bohr} a₀), ` +
      "smaller than a pixel at this zoom; that IS the physics"
    );
  }
  if (!sphere) return null;
  return `nucleus marker drawn ${formatMagnification(sphere.magnification)} true size`;
}
