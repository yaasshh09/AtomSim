import { describe, expect, it } from "vitest";

/**
 * `text-transform: uppercase` is unsafe in this app, and keeps proving it.
 *
 * CSS uppercasing is a text transform, not a typographic one: it maps α to A
 * and ² to 2, so a rule applied to a string containing a physical symbol
 * silently renames the quantity. The What-If banner read "DERIVED A = 1/137",
 * naming a constant that does not exist, and the rail's checkbox labels once
 * printed "fine structure (α² perturbation)" as "A² PERTURBATION".
 *
 * The rule is not "never uppercase". It is "uppercase only strings that are
 * guaranteed to be plain words". Section headings qualify because they are
 * written here and are always words like "System" or "Sampling". Anything that
 * interpolates a value, a unit or a symbol does not, and a heading that starts
 * naming symbols has to leave this list.
 *
 * The scan reads the stylesheet through Vite's raw glob for the same reason
 * the tour anchor scan does: no node:fs, and it sees exactly the file the
 * bundle is built from.
 */
const CSS = import.meta.glob("./*.css", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

/** Selectors allowed to uppercase, each because it can only ever hold words. */
const ALLOWED = new Set([
  // Rail section headings: "System", "View mode", "Sampling", "Physics".
  ".panel h2",
  // Group headings inside a view, written in this repo and word-only.
  ".ctl-group-title",
  // The left rail's state card eyebrow: a preset name and its Z.
  ".state-card-eyebrow",
  // The gallery strip's own heading.
  ".gallery-head",
  // The classical-ghost banner over the 3-D stage. One hardcoded sentence in
  // CloudView.tsx, interpolating nothing: "Counterfactual: a classical
  // electron would spiral in; real atoms do not". Safe while it stays that.
  ".ghost-banner",
]);

function uppercasingSelectors(css: string): string[] {
  const out: string[] = [];
  // Blocks are "selector { body }", so splitting on the closing brace leaves
  // each rule's selector as the tail of the preceding chunk.
  for (const chunk of css.split("}")) {
    const brace = chunk.indexOf("{");
    if (brace < 0) continue;
    const body = chunk.slice(brace + 1);
    if (!/text-transform:\s*uppercase/.test(body)) continue;
    const selector = chunk
      .slice(0, brace)
      // Drop comments, which is where this file explains each decision.
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .trim()
      .replace(/\s+/g, " ");
    if (selector) out.push(selector);
  }
  return out;
}

describe("uppercase safety", () => {
  const css = Object.values(CSS)[0];

  it("reads the stylesheet at all, so a broken scan cannot pass vacuously", () => {
    expect(css).toBeDefined();
    expect(css.length).toBeGreaterThan(1000);
    expect(css).toContain("text-transform");
  });

  it("uppercases only selectors that can hold nothing but plain words", () => {
    for (const selector of uppercasingSelectors(css)) {
      expect(
        ALLOWED.has(selector),
        `${selector} uppercases text. CSS uppercasing maps a to A and squared ` +
          `to 2, so it may only be applied where the string is guaranteed to ` +
          `be plain words. Add it to ALLOWED only after checking that.`,
      ).toBe(true);
    }
  });

  it("does not uppercase the counterfactual banner, which carries a derived alpha", () => {
    expect(uppercasingSelectors(css)).not.toContain(".counterfactual-banner");
  });

  it("finds the selectors it is meant to be guarding", () => {
    // Without this the scan could silently match nothing and the guard above
    // would pass on an empty list forever.
    expect(uppercasingSelectors(css).length).toBeGreaterThan(0);
  });
});
