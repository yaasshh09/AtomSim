import { useState } from "react";
import type { Provenance } from "../api/types";
import { formatErrorScale } from "../lib/liberties";

const COLORS: Record<string, string> = {
  exact: "#4ade80",
  numerical: "#60a5fa",
  approximation: "#fbbf24",
  counterfactual: "#f472b6",
  visual_liberty: "#a78bfa",
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
