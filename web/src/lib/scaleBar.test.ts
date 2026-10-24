import { describe, expect, it } from "vitest";
import { scaleBar, sizeAnchor } from "./scaleBar";

const BOHR_PM = 52.9177210544;

describe("the real-size scale bar", () => {
  it("picks a 1-2-5 length in picometres for hydrogen", () => {
    // 2p frame: half-extent 10 bohr, so 20 bohr = 1058 pm wide; 22% is 233 pm.
    const bar = scaleBar(20, BOHR_PM);
    expect(bar.label).toBe("200 pm");
    expect(bar.bohr).toBeCloseTo(200 / BOHR_PM, 9);
  });

  it("switches to femtometres for muonic hydrogen and nanometres for big frames", () => {
    expect(scaleBar(5 / 186, BOHR_PM).label).toBe("200 fm");
    expect(scaleBar(400, BOHR_PM).label).toBe("2 nm");
  });

  it("anchors the frame to green light", () => {
    expect(sizeAnchor(20, BOHR_PM)).toBe(
      "This frame is 1.06 nm across: one wavelength of green light (530 nm) would span about 500 of them.",
    );
  });
});
