import { thumbnailUrl } from "../api/client";
import type { Basis } from "../api/client";
import { galleryStates } from "../lib/gallery";
import { THUMBNAIL_LIBERTY } from "../lib/liberties";
import { stateLabel } from "../lib/quantum";
import { useAppStore } from "../state/store";
import { Badge } from "./Badge";

export function GalleryStrip() {
  const { n, l, m, system, systems, basis, setQuantumNumbers } = useAppStore();
  // Positive knowledge only. /api/thumbnail renders a hydrogenic plane inline
  // and 422s on a screened atom, and before `systems` arrives we cannot tell
  // which this is — so asking anyway means a 422 and a broken image icon on
  // every screened atom. Absence of evidence is not evidence of hydrogen.
  //
  // Not fixed by rendering screened thumbnails instead: measured, a screened
  // plane grid costs 2.5 to 3.5 s even at 96 px, and this strip asks for up to
  // 36 at once. Serving them means a job and a cache, not an inline render.
  const hasThumbnails = systems.find((s) => s.key === system)?.kind === "hydrogenic";
  return (
    <div className="gallery">
      <div className="gallery-head">
        <span>n = {n} states</span>
        {hasThumbnails && <Badge provenance={THUMBNAIL_LIBERTY} />}
      </div>
      <div className="gallery-scroll">
        {galleryStates(n).map((s) => {
          const active = s.l === l && s.m === m;
          return (
            <button
              key={`${s.l},${s.m}`}
              type="button"
              className={active ? "thumb thumb-active" : "thumb"}
              title={stateLabel(s.n, s.l, s.m)}
              onClick={() => setQuantumNumbers(s.n, s.l, s.m)}
            >
              {hasThumbnails ? (
                <img
                  src={thumbnailUrl(s.n, s.l, s.m, system, basis as Basis, 96)}
                  alt={stateLabel(s.n, s.l, s.m)}
                  width={72}
                  height={72}
                  loading="lazy"
                />
              ) : (
                // The buttons still pick the state, which is what this strip is
                // for. Only the picture is missing, so only the picture goes.
                <span className="thumb-blank" aria-hidden="true" />
              )}
              <span>{stateLabel(s.n, s.l, s.m)}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
