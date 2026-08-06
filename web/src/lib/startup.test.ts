import { describe, expect, it } from "vitest";
import type { SurfaceMode, ViewMode } from "../state/store";
import { shouldAutoSample } from "./startup";

const opening = (over: {
  tour?: string | null;
  view?: ViewMode;
  surfaceMode?: SurfaceMode;
}) => ({
  tour: over.tour ?? null,
  view: over.view ?? ("cloud" as ViewMode),
  surfaceMode: over.surfaceMode ?? ("cloud" as SurfaceMode),
});

describe("shouldAutoSample", () => {
  it("samples the default landing view", () => {
    expect(shouldAutoSample(opening({}))).toBe(true);
  });

  it("stays out of the way of a tour link", () => {
    // startTour applies the step's own physics and clears what is derived from
    // it. A racing sample would be discarded, or drawn under a caption
    // describing different physics than produced it.
    expect(shouldAutoSample(opening({ tour: "hydrogen-honestly" }))).toBe(false);
  });

  it("does not fire for views that have no empty cloud to fill", () => {
    const others: ViewMode[] = [
      "plane",
      "radial",
      "levels",
      "spectrum",
      "whatif",
      "forcelaw",
    ];
    for (const view of others) {
      expect(shouldAutoSample(opening({ view }))).toBe(false);
    }
  });

  it("leaves the surface mode to fetch its own mesh", () => {
    expect(shouldAutoSample(opening({ surfaceMode: "surface" }))).toBe(false);
  });

  it("does not fire for the combined mode either", () => {
    // "both" draws a surface too, and loadIso is what that view waits on.
    // Sampling here would half-fill it and look like a bug.
    expect(shouldAutoSample(opening({ surfaceMode: "both" }))).toBe(false);
  });

  it("treats an empty tour id as no tour", () => {
    expect(shouldAutoSample(opening({ tour: "" }))).toBe(true);
    expect(shouldAutoSample(opening({ tour: undefined }))).toBe(true);
  });
});
