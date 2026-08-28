import type { ReactNode } from "react";
import type { ViewLead } from "../lib/explain";

/**
 * The card I now open every instrument view with.
 *
 * Three lines before the first axis: what this is, what question I answer with
 * it, and what to look at. I chose that ordering deliberately, because if you
 * do not know what "P(r)" stands for a better-labelled y axis cannot help you,
 * and I used to start every view at the axis.
 *
 * `children` is where my disclosures go, so the honest detail sits under the
 * plain sentence rather than in place of it.
 */
export function ViewIntro({
  lead,
  badge,
  children,
}: {
  lead: ViewLead;
  /** The provenance badge for my headline quantity, when the view has one. */
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
