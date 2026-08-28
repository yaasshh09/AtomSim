// The pure classical-ghost trajectory and formatting helpers I use. No
// three.js, no store. The trajectory law I follow (see plan physics) is
// r(tau)=r0*(1-tau)^(1/3), theta(tau)=2*pi*N*(1-sqrt(1-tau)), with
// tau=t/t_collapse looping in [0,1).

export function ghostRadius(tau: number, r0: number): number {
  return r0 * Math.cbrt(1 - tau);
}

export function ghostAngle(tau: number, nOrbits: number): number {
  return 2 * Math.PI * nOrbits * (1 - Math.sqrt(1 - tau));
}

/** How many wall-clock seconds I stretch one real collapse over, which is my slow-mo factor. */
export function slowMotionFactor(collapseSeconds: number, wallSeconds = 5): number {
  return wallSeconds / collapseSeconds;
}

/** I loop tau in [0,1) from accumulated wall time. */
export function tauFromWall(wallElapsed: number, wallSeconds = 5): number {
  return (wallElapsed % wallSeconds) / wallSeconds;
}

/** The simulated (real) elapsed time in seconds I report for a given tau. */
export function simSeconds(tau: number, collapseSeconds: number): number {
  return tau * collapseSeconds;
}

export function formatSeconds(s: number): string {
  if (s >= 1e-9) return `${(s * 1e12).toFixed(1)} ps`;
  if (s >= 1e-12) return `${(s * 1e12).toFixed(2)} ps`;
  if (s >= 1e-15) return `${(s * 1e15).toFixed(1)} fs`;
  return `${s.toExponential(2)} s`;
}
