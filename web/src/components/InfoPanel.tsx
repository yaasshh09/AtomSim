import { useEffect } from "react";
import { realOrbitalLabel, stateLabel } from "../lib/quantum";
import { informativeEnd, sparkBars } from "../lib/spark";
import { isHydrogenic } from "../lib/systemKind";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

/** How many bars I put in the P(r) summary card. I size it so each one stays above a pixel wide. */
const SPARK_BARS = 22;

export function InfoPanel() {
  const {
    n, l, m, basis, system, systems, fineStructure, stateInfo, meta, model,
    radial, loadStateInfo, loadRadial,
  } = useAppStore();
  const selected = systems.find((s) => s.key === system);
  const isScreened = selected?.kind === "screened";
  // My /api/state is hydrogenic-only; screened atoms describe themselves
  // through my systems list and their own level, radial and spectrum views. I
  // gate on knowing the system is hydrogenic rather than on it not being
  // screened, for the first-render reason I spell out in lib/systemKind.
  const hydrogenic = isHydrogenic(systems, system);
  useEffect(() => {
    if (hydrogenic) void loadStateInfo();
  }, [hydrogenic, n, l, m, system, fineStructure, loadStateInfo]);
  // The P(r) card's data, on the same guard and for the same reason. Going
  // hydrogenic-only is not a style choice here: under Hartree-Fock my
  // `loadRadial` runs `ensureHF` first, so if I put it on the always-mounted
  // rail I would fire a full SCF solve from a panel that is only trying to
  // draw a thumbnail. For a hydrogen-like system the radial functions are
  // closed form and this is one cheap GET beside the /api/state call directly
  // above.
  useEffect(() => {
    if (hydrogenic) void loadRadial();
  }, [hydrogenic, n, l, system, loadRadial]);
  // I prefer the exact per-state system, and fall back to the preset you
  // selected from the list.
  const sys = stateInfo?.system ?? selected;
  // My "…" means the table has not arrived. Once it has and the key is still
  // not in it, that ellipsis stops being a wait and becomes a lie: nothing
  // further is coming, because nothing is going to ask for it. A hand-edited
  // ?system= is the only way in, and leaving it looking like a slow load is
  // the one response that never resolves.
  const unknownSystem = systems.length > 0 && !sys;
  // I keep `radial` in my store's INVALIDATED block, so I clear it the moment
  // (n, l, m, system) changes and it can never be a curve from another state
  // wearing this one's label. Empty means I have nothing to draw, and then I
  // do not draw the card at all.
  const pr = radial?.radial_probability ?? null;
  // I compute the window and the label together and from the same index, so
  // the r I print under the card is always the r the last bar sits at.
  const end = pr ? informativeEnd(pr.values) : 0;
  const bars = pr ? sparkBars(pr.values.slice(0, end), SPARK_BARS) : [];
  const rMax = pr && end > 0 ? pr.grid[end - 1] : null;
  return (
    <aside className="panel">
      <div className="state-card" data-tour="state-card">
        <div className="state-card-eyebrow">
          State vector{sys ? ` · ${sys.name} Z=${sys.z}` : ""}
        </div>
        <div className="state-card-title">
          {stateLabel(n, l, m)}
          {basis === "real" && (
            <span className="orbital-label"> · {realOrbitalLabel(l, m)}</span>
          )}
        </div>
        <div className="state-card-sub">
          n={n} ℓ={l} m={m} · {basis === "real" ? "real Sℓm" : "complex Yℓm"}
        </div>
      </div>
      {sys && <p className="system-desc">{sys.description}</p>}
      {unknownSystem && (
        <p className="system-desc">
          I have no preset called "{system}". Choose one under System and I will carry on.
        </p>
      )}
      {/*
        The description above is the preset's, and every screened preset names
        GSZ in it. With the Hartree-Fock model selected that sentence describes
        a model I am not using at all, so I correct it here rather than leaving
        it to be read as the active one.

        The second half of this note used to read "other views still use the
        screened field", which was true until Phase 26 and is now false in
        every view. A correction that has itself gone stale is worse than no
        correction, because I have told you to trust it.
      */}
      {isScreened && model === "hf" && (
        <p className="system-desc">
          I am solving every view in this session with Hartree-Fock, not with
          the screened field named above.
        </p>
      )}
      {stateInfo && (
        <dl className="readouts state-readouts">
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
                <dd>a point lepton, so I have no measured size</dd>
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
        </dl>
      )}
      {bars.length > 0 && pr && (
        <figure className="spark-card">
          <figcaption className="spark-head">
            Radial P(r) <Badge provenance={pr.provenance} />
          </figcaption>
          <div className="spark" aria-hidden="true">
            {bars.map((h, i) => (
              <div
                // I use the index as the identity here: these are fixed
                // positions on an axis, not a reorderable list.
                key={i}
                className="spark-bar"
                style={{ height: `${Math.max(h * 100, 1)}%` }}
              />
            ))}
          </div>
          <div className="spark-foot">
            <span>r = 0</span>
            <span>
              {rMax?.toFixed(rMax < 10 ? 1 : 0)} {pr.grid_unit}
            </span>
          </div>
        </figure>
      )}
    </aside>
  );
}
