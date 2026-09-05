import { useEffect, useState } from "react";
import { stateLabel } from "../lib/quantum";
import { isNarrow, useViewport } from "../lib/viewport";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";
import { Shortcuts } from "./Shortcuts";
import { TourMenu } from "./TourMenu";

/**
 * Hands over the link to whatever is on screen.
 *
 * Every state is already addressable (lib/urlState), but until now the only
 * way to get one of those links was the address bar, which is the worst
 * surface on the device where sharing actually happens. The URL in the bar is
 * live and canonical, so this copies it rather than rebuilding it.
 *
 * A failure says so. The clipboard is refused outright in a few places (an
 * insecure origin, a browser that wants a user gesture it did not see), and
 * "copied" over an empty clipboard is a small lie about something the reader
 * is about to rely on.
 *
 * The label loses a word on a phone. The right-hand group of the top bar does
 * not shrink, so at 320px "copy link" is 33px that come off the end of the
 * tours button beside it; "link" over an icon because the point of the button
 * is what it hands you, and a glyph would need the word underneath it anyway.
 */
function CopyLink({ narrow }: { narrow: boolean }) {
  const [said, setSaid] = useState<"idle" | "copied" | "failed">("idle");
  useEffect(() => {
    if (said === "idle") return;
    const t = setTimeout(() => setSaid("idle"), 1600);
    return () => clearTimeout(t);
  }, [said]);
  return (
    <button
      className="topbar-btn"
      type="button"
      title="Copy a link to this exact state"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(window.location.href);
          setSaid("copied");
        } catch {
          setSaid("failed");
        }
      }}
    >
      {said === "idle" ? (narrow ? "link" : "copy link") : said === "copied" ? "copied" : "failed"}
    </button>
  );
}

/**
 * The session header: what is loaded, and what tier its energy came out of.
 *
 * Everything here is measured, or it is left out. The design this was built
 * from also carried a session id, a memory figure and a frame rate, all of
 * them constants. A fabricated telemetry field is a lie about the run even
 * when nobody reads it, so the id and the memory figure went. The frame rate
 * is the one the renderer actually counts (`FpsMeter` in CloudView), shown
 * only while the view that counts it is open.
 */
export function TopBar() {
  const { width } = useViewport();
  const { n, l, m, system, systems, basis, model, stateInfo, fps, view } =
    useAppStore();
  const sys = stateInfo?.system ?? systems.find((s) => s.key === system);
  const isScreened =
    systems.find((s) => s.key === system)?.kind === "screened";
  // For a many-electron atom the basis is not the interesting half of the
  // sentence. Which of the two models is solving it is.
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
        <CopyLink narrow={isNarrow(width)} />
        <Shortcuts />
        <TourMenu />
      </div>
    </header>
  );
}
