import { describe, expect, it } from "vitest";
import type { SystemInfo } from "../api/types";
import { CHALLENGES, solvedBy } from "./challenges";
import { URL_DEFAULTS } from "./urlState";

// The two numbers the size challenge reads: Z and the reduced mass ratio.
const sys = (key: string, z: number, mu: number) =>
  ({ key, z, mu_ratio: { value: mu } }) as unknown as SystemInfo;
const SYSTEMS = [sys("h", 1, 0.999456), sys("he+", 2, 0.999863), sys("mu-h", 1, 185.84)];

describe("challenges", () => {
  it("are each solved by their stated solution", () => {
    for (const c of CHALLENGES) {
      expect(c.check({ ...URL_DEFAULTS, ...c.solution }, SYSTEMS), c.id).toBe(true);
    }
  });

  it("are none of them solved by the app's opening state", () => {
    expect([...solvedBy(URL_DEFAULTS, SYSTEMS)]).toEqual([]);
  });

  it("have unique ids and no dashes in their prose", () => {
    expect(new Set(CHALLENGES.map((c) => c.id)).size).toBe(CHALLENGES.length);
    for (const c of CHALLENGES) {
      for (const text of [c.prompt, c.hint, c.why]) expect(text, c.id).not.toMatch(/[—–]/);
    }
  });

  it("read the system's size, so hydrogen itself does not count as smaller", () => {
    const size = CHALLENGES.find((c) => c.id === "smaller-than-hydrogen")!;
    expect(size.check({ ...URL_DEFAULTS, n: 1, system: "h" }, SYSTEMS)).toBe(false);
    expect(size.check({ ...URL_DEFAULTS, n: 1, system: "mu-h" }, SYSTEMS)).toBe(true);
    expect(size.check({ ...URL_DEFAULTS, n: 1, system: "he+" }, [])).toBe(false);
  });
});
