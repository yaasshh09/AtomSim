import { describe, expect, it } from "vitest";
import { parseAppUrl, serializeAppUrl } from "../lib/urlState";
import { FLAGSHIP_TOUR_ID, TOURS, tourById } from "./registry";
import { stepState } from "./step";
import { CLAIM_KINDS } from "./types";

describe("registry", () => {
  it("loads every tour with at least one step", () => {
    expect(TOURS.length).toBeGreaterThan(0);
    for (const t of TOURS) expect(t.steps.length).toBeGreaterThan(0);
  });

  it("finds a tour by id and drops an unknown one", () => {
    expect(tourById(TOURS[0].id)?.id).toBe(TOURS[0].id);
    expect(tourById("no-such-tour")).toBeNull();
  });

  it("has unique tour ids", () => {
    const ids = TOURS.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("has unique step ids inside each tour", () => {
    for (const t of TOURS) {
      const ids = t.steps.map((s) => s.id);
      expect(new Set(ids).size, `duplicate step id in ${t.id}`).toBe(ids.length);
    }
  });

  it("gives every step a title and a non-empty body", () => {
    for (const t of TOURS) {
      for (const s of t.steps) {
        expect(s.title.length, `${t.id}/${s.id}`).toBeGreaterThan(0);
        expect(s.body.length, `${t.id}/${s.id}`).toBeGreaterThan(0);
        for (const p of s.body) expect(p.trim().length).toBeGreaterThan(0);
      }
    }
  });

  it("round-trips every step's state through the URL", () => {
    // A step that cannot survive serialisation is a step whose deep link
    // silently shows something else, which is the whole contract broken.
    for (const t of TOURS) {
      for (const s of t.steps) {
        const want = stepState(s);
        const got = { ...want, ...parseAppUrl(serializeAppUrl(want)) };
        expect(got, `${t.id}/${s.id}`).toEqual(want);
      }
    }
  });

  it("only claims quantities the Python resolver implements", () => {
    for (const t of TOURS) {
      for (const s of t.steps) {
        for (const c of s.claims ?? []) {
          expect(CLAIM_KINDS, `${t.id}/${s.id}`).toContain(c.of);
        }
      }
    }
  });

  it("gives every claim an explicit tolerance", () => {
    for (const t of TOURS) {
      for (const s of t.steps) {
        for (const c of s.claims ?? []) {
          expect(c.tol, `${t.id}/${s.id}/${c.of}`).toBeGreaterThan(0);
        }
      }
    }
  });

  it("names Hartree-Fock on any atom the screened model cannot draw", () => {
    // Sulfur and chlorine have no GSZ parameters. A step landing on one with
    // the default model selected asks the server for a refusal.
    for (const t of TOURS) {
      for (const s of t.steps) {
        if (s.state.system === "s" || s.state.system === "cl") {
          expect(s.state.model, `${t.id}/${s.id}`).toBe("hf");
        }
      }
    }
  });

  it("resolves the flagship the invitation offers", () => {
    // The invitation opens this id by name. A rename in the JSON has to fail
    // here rather than at a button a first-time reader clicks once.
    expect(tourById(FLAGSHIP_TOUR_ID)).not.toBeNull();
  });

  it("never writes an em dash", () => {
    for (const t of TOURS) {
      for (const s of t.steps) {
        for (const p of [s.title, ...s.body]) {
          expect(p, `${t.id}/${s.id}`).not.toContain("—");
        }
      }
    }
  });
});
