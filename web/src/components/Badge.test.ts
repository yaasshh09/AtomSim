import { describe, expect, it } from "vitest";
import { placeInspector } from "./Badge";

/** A 1280x800 window, which is where every case below is measured. */
const VIEW = { width: 1280, height: 800 };
/** The inspector at its declared width and a typical three-assumption height. */
const PANEL = { width: 280, height: 220 };

/** Badge rects as the browser reports them: left, top and bottom in px. */
function badge(left: number, top: number) {
  return { left, top, bottom: top + 18 };
}

describe("placeInspector", () => {
  it("hangs under the badge when there is room", () => {
    const at = placeInspector(badge(320, 120), PANEL, VIEW);
    expect(at.left).toBe(320);
    expect(at.top).toBe(120 + 18 + 6);
  });

  it("flips above a badge sitting on the canvas floor", () => {
    // The Cloud view's liberty badges live in .canvas-overlay, 1rem off the
    // bottom of the stage. Opening downward is what used to run the panel off
    // the window and stretch the page under it.
    const at = placeInspector(badge(40, 750), PANEL, VIEW);
    expect(at.top).toBe(750 - 6 - 220);
    expect(at.top).toBeGreaterThanOrEqual(8);
  });

  it("slides a right-rail panel back inside the window", () => {
    // A badge 60px from the right edge would put 220px of a 280px panel
    // outside the window, which is the clipping this function exists to stop.
    const at = placeInspector(badge(1220, 200), PANEL, VIEW);
    expect(at.left).toBe(1280 - 8 - 280);
    expect(at.left + PANEL.width).toBeLessThanOrEqual(1280 - 8);
  });

  it("never places a panel off the left or top edge", () => {
    // Both clamps have to survive a window smaller than the panel: the min/max
    // pair is ordered so the margin wins, otherwise a short window produced a
    // negative top and hid the first line of the method.
    const tiny = { width: 200, height: 150 };
    const at = placeInspector(badge(4, 10), PANEL, tiny);
    expect(at.left).toBe(8);
    expect(at.top).toBeGreaterThanOrEqual(8);
  });

  it("keeps a tall panel anchored at the top rather than centred on nothing", () => {
    // Taller than the window: neither side fits, so the panel starts at the
    // margin and the CSS max-height scrolls the rest. Anything else would cut
    // off the top of the disclosure, which is the part naming the method.
    const tall = { width: 280, height: 900 };
    const at = placeInspector(badge(100, 400), tall, VIEW);
    expect(at.top).toBe(8);
  });
});
