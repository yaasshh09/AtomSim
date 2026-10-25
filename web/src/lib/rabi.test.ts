import { describe, expect, it } from "vitest";
import { formatIntensity, RABI_TRANSITIONS } from "./rabi";

describe("rabi helpers", () => {
  it("formats intensities with an SI prefix and two figures", () => {
    expect(formatIntensity(1)).toBe("1 W/m²");
    expect(formatIntensity(3.2e7)).toBe("32 MW/m²");
    expect(formatIntensity(1e12)).toBe("1 TW/m²");
    expect(formatIntensity(0.5)).toBe("0.5 W/m²");
  });

  it("only offers dipole-allowed transitions that go up in n", () => {
    for (const t of RABI_TRANSITIONS) {
      expect(Math.abs(t.l2 - t.l1)).toBe(1);
      expect(t.n2).toBeGreaterThan(t.n1);
      expect(t.value).toBe(`${t.n1}${"spd"[t.l1]}-${t.n2}${"spd"[t.l2]}`);
    }
  });
});
