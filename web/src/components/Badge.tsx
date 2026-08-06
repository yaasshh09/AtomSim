import { useState } from "react";
import type { Provenance } from "../api/types";
import { formatErrorScale } from "../lib/liberties";

/* Tuned to the instrument palette: EXACT is the shell's own mint, so the tier
   the engine is proudest of is the colour the whole UI is built around.
   Counterfactual pink is unchanged on purpose, index.css uses that exact hue
   for the ghost HUD, the counterfactual banner and the counterfactual rungs,
   and a badge that drifted off it would stop matching the thing it labels. */
const COLORS: Record<string, string> = {
  exact: "#34e0a1",
  numerical: "#6cb7ff",
  approximation: "#fbbf24",
  counterfactual: "#f472b6",
  visual_liberty: "#b48bd9",
};

export function Badge({ provenance }: { provenance: Provenance }) {
  const [open, setOpen] = useState(false);
  const color = COLORS[provenance.fidelity] ?? "#ffffff";
  return (
    <span className="badge-wrap">
      <button
        type="button"
        className="badge"
        style={{ borderColor: color, color }}
        onClick={() => setOpen((v) => !v)}
      >
        {provenance.fidelity.replace("_", " ").toUpperCase()}
      </button>
      {open && (
        <div className="badge-inspector">
          <p>
            <strong>Method:</strong> {provenance.method}
          </p>
          {provenance.assumptions.length > 0 && (
            <ul>
              {provenance.assumptions.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          )}
          {provenance.error_estimate !== null && (
            <p>
              <strong>Error scale:</strong> {formatErrorScale(provenance.error_estimate)}
            </p>
          )}
          {provenance.refinement && (
            <p>
              <strong>To improve:</strong> {provenance.refinement}
            </p>
          )}
        </div>
      )}
    </span>
  );
}
