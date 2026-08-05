// spotlight.ts is tested here as pure geometry; the DOM read happens in the
// component, which this project cannot test without jsdom.
import { describe, expect, it } from "vitest";
import { spotlightBox } from "./spotlight";

describe("spotlightBox", () => {
  it("pads the ring out from the control", () => {
    const b = spotlightBox({ left: 10, top: 20, width: 100, height: 40 }, 6);
    expect(b).toEqual({ x: 4, y: 14, w: 112, h: 52 });
  });

  it("returns null for a control with no box", () => {
    // An anchor that is display:none, unmounted, or inside a collapsed
    // <details> measures 0x0. Ringing it would draw a dot in the corner.
    expect(spotlightBox({ left: 0, top: 0, width: 0, height: 0 }, 6)).toBeNull();
  });
});
