import { describe, expect, it } from "vitest";
import { nextSnap, PEEK_HEIGHT, snapHeight } from "./sheet";

const TALL = 844; // iPhone 15 portrait
const SHORT = 390; // the same phone, on its side

describe("snapHeight", () => {
  it("keeps the peek bar the same on any screen", () => {
    // It holds one line of text. A fraction of the viewport would make that
    // line float in the middle of a band on a tall screen.
    expect(snapHeight("collapsed", TALL)).toBe(PEEK_HEIGHT);
    expect(snapHeight("collapsed", SHORT)).toBe(PEEK_HEIGHT);
  });

  it("leaves the picture visible at half, on both orientations", () => {
    expect(snapHeight("half", TALL)).toBeLessThan(TALL / 2);
    expect(snapHeight("half", SHORT)).toBeLessThan(SHORT / 2);
  });

  it("shrinks both open snaps on a landscape phone", () => {
    // 90% of 390px is the whole instrument gone. The fractions drop below
    // SHORT_VIEWPORT so that "full" still leaves something to look at.
    expect(snapHeight("full", SHORT) / SHORT).toBeLessThan(0.65);
    expect(snapHeight("full", TALL) / TALL).toBeGreaterThan(0.85);
  });
});

describe("nextSnap", () => {
  it("stays put when nothing much happened", () => {
    expect(nextSnap("half", 0, 0, TALL)).toBe("half");
    expect(nextSnap("collapsed", -8, 0, TALL)).toBe("collapsed");
  });

  it("lands on the nearest snap to where it was let go", () => {
    // Dragged up most of the way from half: the height it is being held at is
    // nearer full than half.
    const rise = snapHeight("full", TALL) - snapHeight("half", TALL);
    expect(nextSnap("half", -rise * 0.8, 0, TALL)).toBe("full");
    expect(nextSnap("half", -rise * 0.2, 0, TALL)).toBe("half");
  });

  it("takes a flick as intent, however short", () => {
    expect(nextSnap("collapsed", -4, -1.2, TALL)).toBe("half");
    expect(nextSnap("half", -4, -1.2, TALL)).toBe("full");
    expect(nextSnap("full", 4, 1.2, TALL)).toBe("half");
    expect(nextSnap("half", 4, 1.2, TALL)).toBe("collapsed");
  });

  it("has nowhere further to go at either end", () => {
    expect(nextSnap("full", -200, -2, TALL)).toBe("full");
    expect(nextSnap("collapsed", 200, 2, TALL)).toBe("collapsed");
  });

  it("cannot be dragged below the peek bar", () => {
    // A long downward drag from full overshoots collapsed by a long way, and
    // collapsed is still the nearest thing to where it ended up.
    expect(nextSnap("full", 5000, 0, TALL)).toBe("collapsed");
  });
});
