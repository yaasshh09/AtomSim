import { describe, expect, it } from "vitest";
import { MIN_WIDTH, needsWiderScreen } from "./NarrowNotice";

describe("needsWiderScreen", () => {
  it("admits the threshold itself", () => {
    // MIN_WIDTH is the narrowest width that WORKS, not the widest that fails.
    // Off by one here shows the notice to a viewport the app fits in.
    expect(needsWiderScreen(MIN_WIDTH)).toBe(false);
    expect(needsWiderScreen(MIN_WIDTH - 1)).toBe(true);
  });

  it("turns away the phones this exists for", () => {
    // iPhone 15 portrait, Pixel 8 portrait, iPad mini portrait.
    for (const width of [390, 412, 744]) {
      expect(needsWiderScreen(width)).toBe(true);
    }
  });

  it("lets real desktops through", () => {
    for (const width of [1024, 1280, 1440, 2560]) {
      expect(needsWiderScreen(width)).toBe(false);
    }
  });

  it("keeps the threshold at the width the rails actually need", () => {
    // 300px rail + 300px centre + 300px rail. If someone narrows the rails in
    // index.css, this is the number that has to move with them.
    expect(MIN_WIDTH).toBe(900);
  });
});
