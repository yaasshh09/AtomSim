import type { ReactNode } from "react";

/**
 * A section you open, rather than one you have to scroll past.
 *
 * This is the whole mechanism by which the app got friendlier without getting
 * less honest. Nothing that used to be on screen was deleted or softened; the
 * dense paragraphs moved inside these, verbatim, behind a summary that says
 * what they are about. A caveat one click away is still disclosed. A caveat
 * that has driven the reader off the page is not.
 *
 * Built on <details> on purpose: it is keyboard operable, screen readers
 * announce it as expandable, and browser find-in-page opens it to show a
 * match. A div with an onClick gives none of that.
 */
export function Disclosure({
  summary,
  children,
  tone = "note",
  open = false,
}: {
  summary: string;
  children: ReactNode;
  /** `note` is neutral; `caveat` marks a limit of the model, not a decoration. */
  tone?: "note" | "caveat";
  open?: boolean;
}) {
  return (
    <details className={`disclosure disclosure-${tone}`} open={open}>
      <summary>{summary}</summary>
      <div className="disclosure-body">{children}</div>
    </details>
  );
}
