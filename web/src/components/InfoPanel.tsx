import { useEffect } from "react";
import { realOrbitalLabel, stateLabel } from "../lib/quantum";
import { isHydrogenic } from "../lib/systemKind";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

export function InfoPanel() {
  const {
    n, l, m, basis, system, systems, fineStructure, stateInfo, meta, fps, view, model,
    loadStateInfo,
  } = useAppStore();
  const selected = systems.find((s) => s.key === system);
  const isScreened = selected?.kind === "screened";
  // /api/state is hydrogenic-only; screened atoms describe themselves via the
  // systems list and their dedicated level/radial/spectrum views. Gated on
  // knowing the system is hydrogenic rather than on it not being screened, for
  // the first-render reason spelled out in lib/systemKind.
  const hydrogenic = isHydrogenic(systems, system);
  useEffect(() => {
    if (hydrogenic) void loadStateInfo();
  }, [hydrogenic, n, l, m, system, fineStructure, loadStateInfo]);
  // Prefer the exact per-state system, else the selected preset from the list.
  const sys = stateInfo?.system ?? selected;
  // "…" means the table has not arrived. Once it has and the key is still not
  // in it, the ellipsis stops being a wait and becomes a lie: nothing further
  // is coming, because nothing is going to ask for it. A hand-edited ?system=
  // is the only way in, and leaving it looking like a slow load is the one
  // response that never resolves.
  const unknownSystem = systems.length > 0 && !sys;
  return (
    <aside className="panel">
      <h1 className="brand">atomsim</h1>
      <h2>{sys ? `${sys.name} (Z = ${sys.z})` : unknownSystem ? "Unknown system" : "…"}</h2>
      {sys && <p className="system-desc">{sys.description}</p>}
      {unknownSystem && (
        <p className="system-desc">
          No preset is called "{system}". Choose one under System to carry on.
        </p>
      )}
      {/*
        The description above is the preset's, and every screened preset names
        GSZ in it. With the Hartree-Fock model selected that sentence is
        describing a model this session is not using at all, so it gets
        corrected here rather than left to be read as the active one.

        The second half of this note used to read "other views still use the
        screened field", which was true until Phase 26 and is now false in
        every view. A correction that has itself gone stale is worse than no
        correction, because the reader has been told to trust it.
      */}
      {isScreened && model === "hf" && (
        <p className="system-desc">
          Every view in this session is solved by Hartree-Fock, not by the
          screened field named above.
        </p>
      )}
      <p className="state-label">
        {stateLabel(n, l, m)}
        {basis === "real" && (
          <span className="orbital-label"> · {realOrbitalLabel(l, m)}</span>
        )}
      </p>
      {stateInfo && (
        <dl className="readouts">
          <dt>
            Energy <Badge provenance={stateInfo.energy.provenance} />
          </dt>
          <dd>
            {stateInfo.energy.value.toFixed(6)} hartree
            <br />
            {stateInfo.energy_ev.value.toFixed(4)} eV
          </dd>
          {fineStructure && stateInfo.levels.length > 0 && (
            <>
              <dt>
                Fine structure <Badge provenance={stateInfo.levels[0].shift.provenance} />
              </dt>
              <dd>
                {stateInfo.levels.map((lev) => (
                  <span key={lev.j} className="fs-level">
                    j = {lev.j}: {(lev.shift_ev.value * 1e6).toFixed(2)} µeV
                    <br />
                  </span>
                ))}
              </dd>
            </>
          )}
          <dt>
            {"⟨r⟩"} <Badge provenance={stateInfo.mean_radius.provenance} />
          </dt>
          <dd>
            {stateInfo.mean_radius.value.toFixed(3)} a{"₀"} ·{" "}
            {stateInfo.mean_radius_pm.value.toFixed(1)} pm
          </dd>
          <dt>
            |L| <Badge provenance={stateInfo.angular_momentum.provenance} />
          </dt>
          <dd>{stateInfo.angular_momentum.value.toFixed(3)} ℏ</dd>
          <dt>Nodes</dt>
          <dd>
            {stateInfo.radial_nodes} radial · {stateInfo.angular_nodes} angular
          </dd>
          {sys?.nuclear_radius_fm ? (
            <>
              <dt>
                Nucleus r<sub>rms</sub>{" "}
                <Badge provenance={sys.nuclear_radius_fm.provenance} />
              </dt>
              <dd>{sys.nuclear_radius_fm.value.toFixed(3)} fm</dd>
            </>
          ) : (
            sys && (
              <>
                <dt>Nucleus</dt>
                <dd>point lepton — no measured size</dd>
              </>
            )
          )}
          {meta && (
            <>
              <dt>
                Sampled points <Badge provenance={meta.provenance} />
              </dt>
              <dd>{meta.count.toLocaleString()}</dd>
            </>
          )}
          {view === "cloud" && fps > 0 && (
            <>
              <dt>FPS (measured)</dt>
              <dd>{fps}</dd>
            </>
          )}
        </dl>
      )}
    </aside>
  );
}
