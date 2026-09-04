import { describe, expect, it } from "vitest";
import { isNarrow, MIN_WIDTH } from "./viewport";

describe("isNarrow", () => {
  it("admits the threshold itself", () => {
    // MIN_WIDTH is the narrowest width the three columns FIT in, not the
    // widest that fails. Off by one here hands the stacked shell to a viewport
    // the full one fits in.
    expect(isNarrow(MIN_WIDTH)).toBe(false);
    expect(isNarrow(MIN_WIDTH - 1)).toBe(true);
  });

  it("sends the phones and small tablets to the stacked shell", () => {
    // iPhone 15 portrait, Pixel 8 portrait, iPad mini portrait, and a
    // landscape phone, which is the width that gets the shell and the height
    // that shrinks its sheet.
    for (const width of [390, 412, 744, 844]) {
      expect(isNarrow(width)).toBe(true);
    }
  });

  it("leaves real desktops on the three-column shell", () => {
    for (const width of [1024, 1280, 1440, 2560]) {
      expect(isNarrow(width)).toBe(false);
    }
  });

  it("keeps the threshold at the width the rails actually need", () => {
    // 300px rail + 300px centre + 300px rail. If someone narrows the rails in
    // index.css, this is the number that has to move with them.
    expect(MIN_WIDTH).toBe(900);
  });
});
