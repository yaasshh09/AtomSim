import { stateLabel } from "../lib/quantum";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { TourMenu } from "./TourMenu";

/**
 * The session header: what is loaded, and what tier its energy came out of.
 *
 * Everything here is measured or it is absent. The design this implements also
 * carried a session id, a memory figure and a frame rate as constants; a
 * fabricated telemetry field is a lie about the run even when nobody reads it,
 * so the id and the memory figure are gone and the frame rate is the one the
 * renderer actually counts (`FpsMeter` in CloudView), shown only while the
 * view that counts it is open.
 */
export function TopBar() {
  const { n, l, m, system, systems, basis, model, stateInfo, fps, view } =
    useAppStore();
  const sys = stateInfo?.system ?? systems.find((s) => s.key === system);
  const isScreened =
    systems.find((s) => s.key === system)?.kind === "screened";
  // For a many-electron atom the basis is not the interesting half of the
  // sentence, which of the two models is solving it is.
  const method = isScreened
    ? model === "hf"
      ? "Hartree-Fock"
      : "screened (GSZ)"
    : basis === "real"
      ? "real Slm"
      : "complex Ylm";
  return (
    <header className="topbar">
      <div className="topbar-left">
        <span className="topbar-brand">atomsim</span>
        <span className="topbar-rule" />
        <span className="topbar-crumb">
          {sys ? sys.name : system} · {stateLabel(n, l, m)} · {method}
        </span>
      </div>
      <div className="topbar-right">
        {view === "cloud" && fps > 0 && (
          <span className="topbar-stat">FPS {fps}</span>
        )}
        {stateInfo && <Badge provenance={stateInfo.energy.provenance} />}
        <TourMenu />
      </div>
    </header>
  );
}
