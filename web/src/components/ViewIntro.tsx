import type { ReactNode } from "react";
import type { ViewLead } from "../lib/explain";

/**
 * The card every instrument view now opens with.
 *
 * Three lines before the first axis: what this is, what question it answers,
 * and what to look at. That ordering is deliberate, a reader who does not know
 * what "P(r)" stands for cannot be helped by a better-labelled y axis, and
 * every view in this app previously began at the axis.
 *
 * `children` is where the disclosures go, so the honest detail sits under the
 * plain sentence rather than in place of it.
 */
export function ViewIntro({
  lead,
  badge,
  children,
}: {
  lead: ViewLead;
  /** Provenance badge for the headline quantity, when the view has one. */
  badge?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <section className="view-intro">
      <h2 className="view-intro-title">
        {lead.title}
        {badge ? <span className="view-intro-badge">{badge}</span> : null}
      </h2>
      <p className="view-intro-lead">{lead.lead}</p>
      {lead.notice && (
        <p className="view-intro-notice">
          <span className="view-intro-notice-mark" aria-hidden="true">
            ◇
          </span>
          {lead.notice}
        </p>
      )}
      {children}
    </section>
  );
}
