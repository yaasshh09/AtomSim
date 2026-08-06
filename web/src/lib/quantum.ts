const L_LETTERS = "spdfghik";

export function isValidState(n: number, l: number, m: number): boolean {
  return (
    Number.isInteger(n) &&
    Number.isInteger(l) &&
    Number.isInteger(m) &&
    n >= 1 &&
    l >= 0 &&
    l < n &&
    Math.abs(m) <= l
  );
}

export function stateLabel(n: number, l: number, m: number): string {
  const letter = L_LETTERS[l] ?? `(l=${l})`;
  return `${n}${letter} (m = ${m})`;
}

export function clampState(n: number, l: number, m: number): { n: number; l: number; m: number } {
  const cn = Math.max(1, Math.round(n));
  const cl = Math.min(Math.max(0, Math.round(l)), cn - 1);
  const cm = Math.min(Math.max(Math.round(m), -cl), cl);
  // + 0 normalizes -0 (from -cl when cl === 0) to +0
  return { n: cn, l: cl, m: cm + 0 };
}

const CHEMISTRY_LABELS: Record<string, string> = {
  "0,0": "s",
  "1,0": "p_z",
  "1,1": "p_x",
  "1,-1": "p_y",
  "2,0": "d_z2",
  "2,1": "d_xz",
  "2,-1": "d_yz",
  "2,2": "d_x2-y2",
  "2,-2": "d_xy",
  "3,0": "f_z3",
  "3,1": "f_xz2",
  "3,-1": "f_yz2",
  "3,2": "f_z(x2-y2)",
  "3,-2": "f_xyz",
  "3,3": "f_x(x2-3y2)",
  "3,-3": "f_y(3x2-y2)",
};

/** Mirror of atomsim.analytic.angular.real_orbital_label, keep in lockstep. */
export function realOrbitalLabel(l: number, m: number): string {
  const hit = CHEMISTRY_LABELS[`${l},${m}`];
  if (hit) return hit;
  const letter = L_LETTERS[l] ?? `(l=${l})`;
  if (m === 0) return `${letter}(m=0)`;
  const kind = m > 0 ? "cos" : "sin";
  const signed = m > 0 ? `+${m}` : `${m}`;
  return `${letter}(m=${signed}, ${kind})`;
}
