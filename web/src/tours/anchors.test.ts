import { describe, expect, it } from "vitest";
import { TOURS } from "./registry";

/**
 * Every anchor a tour points at has to exist in the app.
 *
 * With no jsdom there is nothing to query, so this reads the component sources
 * and extracts the data-tour literals. A source scan is the right tool anyway:
 * it catches an anchor deleted in a refactor, which is the failure being
 * guarded, and it cannot be fooled by a component that happens not to render.
 *
 * The sources arrive through Vite's own raw glob rather than node:fs, because
 * this project carries no @types/node and one test is a poor reason to add a
 * toolchain dependency. It also means the scan sees exactly the files the
 * bundle is built from.
 */
const SOURCES = import.meta.glob("../components/*.tsx", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

function declaredAnchors(): Set<string> {
  const out = new Set<string>();
  for (const src of Object.values(SOURCES)) {
    for (const m of src.matchAll(/data-tour="([^"]+)"/g)) out.add(m[1]);
  }
  return out;
}

describe("spotlight anchors", () => {
  it("every anchor a tour names exists in a component", () => {
    const have = declaredAnchors();
    for (const t of TOURS) {
      for (const s of t.steps) {
        if (s.spotlight) {
          expect(have, `${t.id}/${s.id} points at a missing anchor`).toContain(s.spotlight);
        }
      }
    }
  });

  it("reads the component sources at all, so a broken scan cannot pass vacuously", () => {
    expect(Object.keys(SOURCES).length).toBeGreaterThan(0);
    expect(declaredAnchors().size).toBeGreaterThan(0);
  });
});
