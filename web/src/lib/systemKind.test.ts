import { describe, expect, it } from "vitest";
import type { SystemInfo } from "../api/types";
import { isHydrogenic, isScreened, systemKind } from "./systemKind";

const sys = (key: string, kind: SystemInfo["kind"]): SystemInfo =>
  ({ key, name: key, kind }) as never;

const TABLE = [sys("h", "hydrogenic"), sys("mu-h", "hydrogenic"), sys("na", "screened")];

describe("systemKind", () => {
  it("reports the kind for a system in the table", () => {
    expect(systemKind(TABLE, "h")).toBe("hydrogenic");
    expect(systemKind(TABLE, "na")).toBe("screened");
  });

  it("is null before the table has loaded, not a guess either way", () => {
    // The first render of the app. Every component that gates a request on the
    // system's kind sees this, and the bug it caused was that both gates read
    // it as "not screened" and fired a hydrogenic-only request at sodium.
    expect(systemKind([], "na")).toBeNull();
    expect(systemKind([], "h")).toBeNull();
  });

  it("is null for a key the table does not carry", () => {
    expect(systemKind(TABLE, "unobtanium")).toBeNull();
  });
});

describe("isHydrogenic / isScreened", () => {
  it("both answer false while the kind is unknown", () => {
    // The property that matters: they are not negations of each other. Asking
    // either question before the table arrives gets you "no", so a caller that
    // fetches on isHydrogenic and explains on isScreened does neither, which
    // is the correct behaviour for one render.
    expect(isHydrogenic([], "na")).toBe(false);
    expect(isScreened([], "na")).toBe(false);
    expect(isHydrogenic([], "h")).toBe(false);
    expect(isScreened([], "h")).toBe(false);
  });

  it("disagree once the kind is known", () => {
    expect(isHydrogenic(TABLE, "h")).toBe(true);
    expect(isScreened(TABLE, "h")).toBe(false);
    expect(isHydrogenic(TABLE, "na")).toBe(false);
    expect(isScreened(TABLE, "na")).toBe(true);
  });
});
