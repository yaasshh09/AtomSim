import type { ReactNode } from "react";
import type { ViewLead } from "../lib/explain";
import { Notation } from "../lib/mathText";
import { isNarrow } from "../lib/viewport";

/**
 * The card every instrument view now opens with.
 *
 * Three lines before the first axis: what this is, what question it answers,
 * and what to look at. That ordering is deliberate. Someone who does not know
 * what "P(r)" stands for cannot be helped by a better-labelled y axis, and
 * every view used to start at the axis.
 *
 * `children` is where the disclosures go, so the honest detail sits under the
 * plain sentence rather than in place of it.
 *
 * On a phone the card is the same card, closed. Three sentences and two
 * disclosures are about 400px of a 844px screen, and with the top bar and the
 * sheet's peek bar around them the plot started below the fold in every view:
 * the app opened on a paragraph about a picture nobody had seen yet. Closed,
 * the title and its badge stay on screen, which is the part that says what the
 * plot is, and the rest is one tap away. Nothing is deleted on either shell,
 * and the reader can close it on a desktop too.
 */
export function ViewIntro({
  lead,
  badge,
  children,
}: {
  lead: ViewLead;
  /** The provenance badge for the headline quantity, when the view has one. */
  badge?: ReactNode;
  children?: ReactNode;
}) {
  /* Read rather than subscribed to. Crossing MIN_WIDTH swaps the whole shell
     and remounts this, so a resize listener here would only ever re-run for
     resizes that already remount it. */
  return (
    <details className="view-intro" open={!isNarrow(window.innerWidth)}>
      <summary className="view-intro-title">
        <Notation>{lead.title}</Notation>
        {badge ? <span className="view-intro-badge">{badge}</span> : null}
      </summary>
      <p className="view-intro-lead">
        <Notation>{lead.lead}</Notation>
      </p>
      {lead.notice && (
        <p className="view-intro-notice">
          <span className="view-intro-notice-mark" aria-hidden="true">
            ◇
          </span>
          <Notation>{lead.notice}</Notation>
        </p>
      )}
      {children}
    </details>
  );
}
