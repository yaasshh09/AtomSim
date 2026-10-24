import { Fragment, useEffect } from "react";
import { hybridLabel } from "../lib/hybrids";
import { Notation } from "../lib/mathText";
import { nodeSummary, realOrbitalLabel, stateLabel } from "../lib/quantum";
import { informativeEnd, sparkBars } from "../lib/spark";
import { isHydrogenic } from "../lib/systemKind";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { fixed } from "../lib/format";

/** Engine labels write expectation brackets as < >; the panel shows ⟨ ⟩. */
export function bracketed(label: string): string {
  return label.replace(/</g, "⟨").replace(/>/g, "⟩");
}

/** Bars in the P(r) summary card, sized so each one stays above a pixel wide. */
const SPARK_BARS = 22;

export function InfoPanel() {
  const {
    n, l, m, basis, system, systems, fineStructure, stateInfo, meta, model, hybrid,
    radial, loadStateInfo, loadRadial,
  } = useAppStore();
  const selected = systems.find((s) => s.key === system);
  const isScreened = selected?.kind === "screened";
  // /api/state is hydrogenic-only. Screened atoms describe themselves through
  // the systems list and their own level, radial and spectrum views. The gate
  // is on knowing the system is hydrogenic rather than on it not being
  // screened, for the first-render reason spelled out in lib/systemKind.
  const hydrogenic = isHydrogenic(systems, system);
  useEffect(() => {
    if (hydrogenic) void loadStateInfo();
  }, [hydrogenic, n, l, m, system, fineStructure, loadStateInfo]);
  // The P(r) card's data, on the same guard and for the same reason.
  // Hydrogenic-only is not a style choice here: under Hartree-Fock,
  // `loadRadial` runs `ensureHF` first, so putting it on the always-mounted
  // rail would fire a full SCF solve from a panel that is only trying to draw
  // a thumbnail. For a hydrogen-like system the radial functions are closed
  // form and this is one cheap GET beside the /api/state call directly above.
  useEffect(() => {
    if (hydrogenic) void loadRadial();
  }, [hydrogenic, n, l, system, loadRadial]);
  // Prefer the exact per-state system, and fall back to the preset selected
  // from the list.
  const sys = stateInfo?.system ?? selected;
  // The "…" means the table has not arrived. Once it has and the key is still
  // not in it, that ellipsis stops being a wait and becomes a lie: nothing
  // further is coming, because nothing is going to ask for it. A hand-edited
  // ?system= is the only way in, and leaving it looking like a slow load is
  // the one response that never resolves.
  const unknownSystem = systems.length > 0 && !sys;
  // `radial` sits in the store's INVALIDATED block, so it clears the moment
  // (n, l, m, system) changes and can never be a curve from another state
  // wearing this one's label. Empty means there is nothing to draw, and then
  // the card does not render at all.
  const pr = radial?.radial_probability ?? null;
  // The window and the label come from the same index, so the r printed under
  // the card is always the r the last bar sits at.
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
          {hybrid ? (
            hybridLabel(hybrid)
          ) : (
            <>
              {stateLabel(n, l, m)}
              {basis === "real" && (
                <span className="orbital-label"> · {realOrbitalLabel(l, m)}</span>
              )}
            </>
          )}
        </div>
        <div className="state-card-sub">
          n={n} ℓ={l} m={m} · {basis === "real" ? "real Sℓm" : "complex Yℓm"}
        </div>
      </div>
      {sys && <p className="system-desc">{sys.description}</p>}
      {hybrid && (
        <p className="system-desc">
          The energy below is the hybrid's own: 2s and 2p share E₂ exactly. The other
          readouts are for its 2p component.
        </p>
      )}
      {unknownSystem && (
        <p className="system-desc">
          There is no preset called "{system}". Choose one under System to carry on.
        </p>
      )}
      {/*
        The description above is the preset's, and every screened preset names
        GSZ in it. With the Hartree-Fock model selected, that sentence
        describes a model nothing here is using, so it gets corrected rather
        than left to be read as the active one.

        The second half of this note used to read "other views still use the
        screened field", which was true until Phase 26 and is now false in
        every view. A correction that has itself gone stale is worse than no
        correction, because the reader was told to trust it.
      */}
      {isScreened && model === "hf" && (
        <p className="system-desc">
          Every view in this session is solved with Hartree-Fock, not with the
          screened field named above.
        </p>
      )}
      {stateInfo && (
        <dl className="readouts state-readouts">
          <dt>
            Energy <Badge provenance={stateInfo.energy.provenance} />
          </dt>
          <dd>
            {fixed(stateInfo.energy.value, 6)} hartree
            <br />
            {fixed(stateInfo.energy_ev.value, 4)} eV
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
          <dd>{nodeSummary(n, l, m, basis)}</dd>
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
                <dd>a point lepton, so there is no measured size</dd>
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
      {stateInfo && stateInfo.expectations.length > 0 && (
        <details className="expectations" data-tour="expectations">
          <summary>Expectation values</summary>
          <dl className="readouts state-readouts">
            {stateInfo.expectations.map((q) => (
              <Fragment key={q.label}>
                <dt>
                  <Notation>{bracketed(q.label)}</Notation> <Badge provenance={q.provenance} />
                </dt>
                <dd>
                  {q.value.toPrecision(6)} <Notation>{q.unit}</Notation>
                </dd>
              </Fragment>
            ))}
          </dl>
          {stateInfo.virial && (
            <p className="virial-check">
              <span className="virial-head">
                Virial check, from ψ itself <Badge provenance={stateInfo.virial.residual.provenance} />
              </span>
              ⟨T⟩ = {stateInfo.virial.kinetic.value.toPrecision(6)} and ⟨V⟩ ={" "}
              {stateInfo.virial.potential.value.toPrecision(6)} hartree, integrated from the
              wavefunction's shape without using its energy. The virial theorem for 1/r says
              −⟨V⟩/⟨T⟩ = 2; here it is{" "}
              {(-stateInfo.virial.potential.value / stateInfo.virial.kinetic.value).toFixed(6)}.
            </p>
          )}
        </details>
      )}
      {bars.length > 0 && pr && (
        <figure className="spark-card">
          <figcaption className="spark-head">
            Radial P(r) <Badge provenance={pr.provenance} />
          </figcaption>
          <div className="spark" aria-hidden="true">
            {bars.map((h, i) => (
              <div
                // The index is the identity here: these are fixed positions on
                // an axis, not a reorderable list.
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
      <nav className="legal-links" aria-label="site policies">
        <a href="/terms.html">Terms</a>
        <a href="/privacy.html">Privacy</a>
        <a href="/api.html">API</a>
        <a href="/changelog.html">What's new</a>
        <a href="https://github.com/yaasshh09/AtomSim/issues">Feedback</a>
      </nav>
    </aside>
  );
}
