import { describe, expect, it } from "vitest";
import { isTypingTarget, matchShortcut } from "./shortcuts";

function key(k: string, mods: Partial<Record<"shiftKey" | "ctrlKey" | "metaKey" | "altKey", boolean>> = {}) {
  return { key: k, shiftKey: false, ctrlKey: false, metaKey: false, altKey: false, ...mods };
}

describe("the keyboard map", () => {
  it("walks n vertically and l horizontally", () => {
    expect(matchShortcut(key("ArrowUp"))).toEqual({ kind: "quantum", axis: "n", delta: 1 });
    expect(matchShortcut(key("ArrowDown"))).toEqual({ kind: "quantum", axis: "n", delta: -1 });
    expect(matchShortcut(key("ArrowRight"))).toEqual({ kind: "quantum", axis: "l", delta: 1 });
    expect(matchShortcut(key("ArrowLeft"))).toEqual({ kind: "quantum", axis: "l", delta: -1 });
  });

  it("shift moves the inner number instead", () => {
    expect(matchShortcut(key("ArrowRight", { shiftKey: true }))).toEqual({
      kind: "quantum",
      axis: "m",
      delta: 1,
    });
    expect(matchShortcut(key("ArrowLeft", { shiftKey: true }))).toEqual({
      kind: "quantum",
      axis: "m",
      delta: -1,
    });
  });

  it("numbers pick a view, zero-indexed into the seven", () => {
    expect(matchShortcut(key("1"))).toEqual({ kind: "view", index: 0 });
    expect(matchShortcut(key("7"))).toEqual({ kind: "view", index: 6 });
    // Only seven views exist, so an eighth digit must not name one.
    expect(matchShortcut(key("8"))).toBeNull();
    expect(matchShortcut(key("0"))).toBeNull();
  });

  it("leaves the browser's and the OS's combinations alone", () => {
    for (const mod of ["ctrlKey", "metaKey", "altKey"] as const) {
      expect(matchShortcut(key("ArrowUp", { [mod]: true }))).toBeNull();
      expect(matchShortcut(key("1", { [mod]: true }))).toBeNull();
    }
  });

  it("opens and closes the list", () => {
    expect(matchShortcut(key("?"))).toEqual({ kind: "help" });
    expect(matchShortcut(key("Escape"))).toEqual({ kind: "close" });
  });

  it("ignores keys it does not claim", () => {
    expect(matchShortcut(key("a"))).toBeNull();
    expect(matchShortcut(key(" "))).toBeNull();
  });
});

describe("focus wins", () => {
  it("hands the keystroke to a field that is being typed in", () => {
    // A range slider answers arrows itself, and a select answers digits.
    expect(isTypingTarget({ tagName: "INPUT" })).toBe(true);
    expect(isTypingTarget({ tagName: "select" })).toBe(true);
    expect(isTypingTarget({ tagName: "TEXTAREA" })).toBe(true);
    expect(isTypingTarget({ tagName: "DIV", isContentEditable: true })).toBe(true);
  });

  it("keeps it for the app otherwise", () => {
    expect(isTypingTarget({ tagName: "BUTTON" })).toBe(false);
    expect(isTypingTarget({ tagName: "BODY" })).toBe(false);
    expect(isTypingTarget(null)).toBe(false);
  });
});
