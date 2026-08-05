import { afterEach, describe, expect, it } from "vitest";
import {
  noMemory,
  parseMemory,
  readMemory,
  shouldInvite,
  withCompleted,
  writeMemory,
} from "./seen";

/** A Storage that lives in a Map, for the tests that need one to exist. */
function fakeStorage(seed: Record<string, string> = {}): Storage {
  const map = new Map(Object.entries(seed));
  return {
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
    clear: () => map.clear(),
    key: (i: number) => [...map.keys()][i] ?? null,
    get length() {
      return map.size;
    },
  } as Storage;
}

/** Install a window with this storage; vitest runs in node, so there is none. */
function withWindow(storage: unknown) {
  (globalThis as { window?: unknown }).window = { localStorage: storage };
}

afterEach(() => {
  delete (globalThis as { window?: unknown }).window;
});

describe("parseMemory", () => {
  it("reads nothing as a reader who has never been here", () => {
    expect(parseMemory(null)).toEqual({ dismissed: false, completed: [] });
  });

  it("survives a value that is not JSON", () => {
    // localStorage is a shared namespace the reader can edit by hand. A throw
    // here would take the whole app down on load, so garbage reads as absent.
    expect(parseMemory("{not json")).toEqual({ dismissed: false, completed: [] });
    expect(parseMemory("null")).toEqual({ dismissed: false, completed: [] });
    expect(parseMemory("42")).toEqual({ dismissed: false, completed: [] });
  });

  it("takes only the right type from each field", () => {
    const m = parseMemory(JSON.stringify({ dismissed: "yes", completed: "hydrogen" }));
    expect(m.dismissed).toBe(false);
    expect(m.completed).toEqual([]);
  });

  it("drops non-string entries rather than the whole list", () => {
    const m = parseMemory(JSON.stringify({ dismissed: true, completed: ["a", 7, null, "b"] }));
    expect(m).toEqual({ dismissed: true, completed: ["a", "b"] });
  });

  it("round-trips what writeMemory writes", () => {
    const storage = fakeStorage();
    withWindow(storage);
    const m = { dismissed: true, completed: ["hydrogen-honestly"] };
    writeMemory(m);
    expect(readMemory()).toEqual(m);
  });
});

describe("shouldInvite", () => {
  it("offers the tour to a reader who has never been offered one", () => {
    expect(shouldInvite(noMemory())).toBe(true);
  });

  it("never offers again once the reader has answered", () => {
    expect(shouldInvite({ dismissed: true, completed: [] })).toBe(false);
    expect(shouldInvite({ dismissed: true, completed: ["hydrogen-honestly"] })).toBe(false);
  });
});

describe("withCompleted", () => {
  it("records the tour and retires the invitation", () => {
    // Finishing a tour answers the invitation's question, so the two move
    // together. Otherwise a reader who took a tour from the menu would still
    // be asked whether they would like one.
    expect(withCompleted(noMemory(), "many-electrons")).toEqual({
      dismissed: true,
      completed: ["many-electrons"],
    });
  });

  it("does not list the same tour twice", () => {
    const once = withCompleted(noMemory(), "many-electrons");
    expect(withCompleted(once, "many-electrons").completed).toEqual(["many-electrons"]);
  });

  it("keeps the tours already finished", () => {
    const m = withCompleted({ dismissed: true, completed: ["a"] }, "b");
    expect(m.completed).toEqual(["a", "b"]);
  });

  it("does not mutate what it was given", () => {
    const before = noMemory();
    withCompleted(before, "a");
    expect(before).toEqual({ dismissed: false, completed: [] });
  });
});

describe("noMemory", () => {
  it("hands out a fresh object every time", () => {
    // Shared structure here would let one caller's push land in everyone's
    // "never been here" record.
    const a = noMemory();
    a.completed.push("a");
    expect(noMemory().completed).toEqual([]);
  });
});

describe("storage that is not there", () => {
  it("reads as a first visit when there is no window at all", () => {
    expect(readMemory()).toEqual({ dismissed: false, completed: [] });
  });

  it("does not throw when writing with no window at all", () => {
    expect(() => writeMemory({ dismissed: true, completed: [] })).not.toThrow();
  });

  it("survives a localStorage that throws on access", () => {
    // Privacy modes throw on the property itself, not just on the call.
    withWindow(undefined);
    Object.defineProperty(globalThis.window, "localStorage", {
      get() {
        throw new Error("denied");
      },
    });
    expect(readMemory()).toEqual({ dismissed: false, completed: [] });
    expect(() => writeMemory({ dismissed: true, completed: [] })).not.toThrow();
  });

  it("survives a storage that refuses to read or write", () => {
    withWindow({
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {
        throw new Error("quota");
      },
    });
    expect(readMemory()).toEqual({ dismissed: false, completed: [] });
    expect(() => writeMemory({ dismissed: true, completed: [] })).not.toThrow();
  });
});
