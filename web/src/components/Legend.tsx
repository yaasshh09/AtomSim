import { DENSITY_GAMMA, phaseColor } from "../lib/colormap";
import { INFERNO } from "../lib/luts";
import type { ColorMode } from "../state/store";

function infernoGradient(): string {
  const stops = [0, 32, 64, 96, 128, 160, 192, 224, 255].map((i) => {
    const [r, g, b] = INFERNO[i];
    return `rgb(${r},${g},${b}) ${((i / 255) * 100).toFixed(1)}%`;
  });
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

export function Legend({ mode }: { mode: ColorMode }) {
  if (mode === "density") {
    return (
      <div className="legend">
        <div className="legend-bar" style={{ background: infernoGradient() }} />
        <span>
          |ψ|² · brightness ∝ (ρ/ρ<sub>max</sub>)<sup>{DENSITY_GAMMA}</sup>
        </span>
      </div>
    );
  }
  if (mode === "phase") {
    // The same table the points are coloured from, sampled -pi to +pi, so the
    // key can never disagree with the picture it explains.
    const stops = Array.from({ length: 13 }, (_, i) => {
      const [r, g, b] = phaseColor(-Math.PI + (i / 12) * 2 * Math.PI);
      return `rgb(${r},${g},${b}) ${((i / 12) * 100).toFixed(1)}%`;
    }).join(", ");
    return (
      <div className="legend">
        <div
          className="legend-bar"
          style={{ background: `linear-gradient(to right, ${stops})` }}
        />
        <span>arg ψ: −π (left) → +π (right)</span>
      </div>
    );
  }
  return null;
}
