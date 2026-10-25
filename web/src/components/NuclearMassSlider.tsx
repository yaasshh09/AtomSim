import { useEffect, useMemo, useState } from "react";
import { Notation } from "../lib/mathText";
import { MASS_MAX, MASS_MIN, landmarks, nuclearMass, snapKey } from "../lib/nuclearMass";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { Slider } from "./Field";

/** How long the hand has to rest before the atom changes. A drag commits once,
 *  not once per pixel, which keeps it inside the API's per-action burst. */
const COMMIT_MS = 300;

/** Where a mass sits among the real particles, in words. */
function anchorFor(mass: number, marks: ReturnType<typeof landmarks>, key: string): string {
  const hit = marks.find((l) => l.key === key);
  if (hit) return `${hit.label}, a real atom`;
  const below = [...marks].reverse().find((l) => l.mass < mass);
  const above = marks.find((l) => l.mass > mass);
  if (below && above) return `between ${below.label} and ${above.label}; no such nucleus exists`;
  return `heavier than ${below?.label ?? "every landmark"}; no such nucleus exists`;
}

/**
 * The nuclear mass as a continuous dial, from positronium to past tritium.
 *
 * Log-scaled, because the interesting physics is in the ratio: mu/m_e = M/(1+M)
 * climbs from 0.5 to 0.91 between M = 1 and M = 10, and by 0.04% between
 * hydrogen and tritium.
 */
export function NuclearMassSlider() {
  const { system, systems, setSystem } = useAppStore();
  const marks = useMemo(() => landmarks(systems), [systems]);
  const current = nuclearMass(systems, system);
  const [pos, setPos] = useState(current === null ? Math.log10(MASS_MIN) : Math.log10(current));
  const [dirty, setDirty] = useState(false);

  // Follow the store when the system changes from elsewhere (the picker, a
  // tour, the back button), but never yank the thumb out from under a drag.
  useEffect(() => {
    if (!dirty && current !== null) setPos(Math.log10(current));
  }, [current, dirty]);

  useEffect(() => {
    if (!dirty) return;
    const timer = window.setTimeout(() => {
      setDirty(false);
      const key = snapKey(10 ** pos, marks);
      if (key !== system) setSystem(key);
    }, COMMIT_MS);
    return () => window.clearTimeout(timer);
  }, [dirty, pos, marks, system, setSystem]);

  const mass = 10 ** pos;
  const shownKey = snapKey(mass, marks);
  const shownMass = marks.find((l) => l.key === shownKey)?.mass ?? mass;
  // The server's description of the committed system, which is what the badge
  // and the reduced mass report: the thumb may be ahead of it mid-drag.
  const info = systems.find((s) => s.key === system);

  return (
    <div className="mass-slider" data-tour="mass-slider">
      <Slider
        label="nuclear mass M"
        readout={`${Number(shownMass.toPrecision(4))} m_e`}
        anchor={anchorFor(shownMass, marks, shownKey)}
        min={Math.log10(MASS_MIN)}
        max={Math.log10(MASS_MAX)}
        step={0.001}
        value={pos}
        onChange={(v) => {
          setPos(v);
          setDirty(true);
        }}
      />
      {info && !dirty && (
        <p className="panel-hint">
          <Notation>{`reduced mass mu = ${info.mu_ratio.value.toPrecision(6)} m_e`}</Notation>{" "}
          <Badge provenance={info.mu_ratio.provenance} />
        </p>
      )}
      <p className="panel-hint">
        <Notation>
          {"Every energy scales with mu = M/(1 + M) and every length with 1/mu, so " +
            "positronium (M = 1) is half as tightly bound as hydrogen and twice the " +
            "size. Away from the named atoms the nucleus is made up, and the badge " +
            "on mu says so."}
        </Notation>
      </p>
    </div>
  );
}
