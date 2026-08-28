import katex from "katex";
// I ship this in this chunk, not the entry. The stylesheet is only meaningful
// once I have typeset maths on the page, and pulling it into the entry dragged
// KaTeX's 260 kB of JS along with it for every visitor who never opened the
// panel.
import "katex/dist/katex.min.css";
import { PHYSICS_CONTENT } from "../physics/content";
import type { ViewMode } from "../state/store";

function MathBlock({ tex }: { tex: string }) {
  // I give KaTeX my own static strings only; no user input reaches it.
  const html = katex.renderToString(tex, { displayMode: true, throwOnError: false });
  return <div className="math" dangerouslySetInnerHTML={{ __html: html }} />;
}

/** The typeset half of "Show the physics", which I load the first time you open it. */
export default function PhysicsBody({ view }: { view: ViewMode }) {
  const content = PHYSICS_CONTENT[view];
  return (
    <>
      <h3>{content.title}</h3>
      {content.blocks.map((b) => (
        <div key={b.tex}>
          <MathBlock tex={b.tex} />
          <p className="physics-note">{b.note}</p>
        </div>
      ))}
    </>
  );
}
