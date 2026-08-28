import type { ReactNode } from "react";

/**
 * A section you open rather than one you have to scroll past.
 *
 * This is the whole mechanism by which I got friendlier without getting less
 * honest. I deleted and softened nothing that used to be on screen; I moved
 * the dense paragraphs inside these, verbatim, behind a summary that says what
 * they are about. A caveat one click away is still disclosed; a caveat that
 * has driven you off the page is not.
 *
 * I build it on <details> on purpose: it is keyboard operable, screen readers
 * announce it as expandable, and browser find-in-page opens it to show a
 * match. A div with an onClick would give me none of those.
 */
export function Disclosure({
  summary,
  children,
  tone = "note",
  open = false,
}: {
  summary: string;
  children: ReactNode;
  /** `note` is neutral; `caveat` marks a limit of my model, not a decoration. */
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
