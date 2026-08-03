import { describe, expect, it } from "vitest";
import { decodeFloats, decodeIndices, decodePositions, thumbnailUrl } from "./client";

describe("decodePositions", () => {
  it("decodes interleaved xyz float32", () => {
    const src = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = decodePositions(src.buffer);
    expect(out).toHaveLength(6);
    expect(out[4]).toBe(5);
  });
  it("rejects buffers that are not whole xyz triplets", () => {
    expect(() => decodePositions(new ArrayBuffer(10))).toThrow(/multiple of 12/);
  });
});

describe("decodeFloats", () => {
  it("decodes float32 buffers", () => {
    const buf = new Float32Array([1.5, -2.5]).buffer;
    expect(Array.from(decodeFloats(buf))).toEqual([1.5, -2.5]);
  });
  it("rejects lengths that are not multiples of 4", () => {
    expect(() => decodeFloats(new ArrayBuffer(5))).toThrow(/multiple of 4/);
  });
});

describe("decodeIndices", () => {
  it("decodes triangles as uint32, not as floats", () => {
    const src = new Uint32Array([0, 1, 2, 2, 1, 3]);
    const out = decodeIndices(src.buffer);
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([0, 1, 2, 2, 1, 3]);
  });
  it("rejects buffers that are not whole triangles", () => {
    expect(() => decodeIndices(new ArrayBuffer(8))).toThrow(/multiple of 12/);
  });
  it("would have read index data as garbage through the float decoder", () => {
    // The reason this decoder exists: the same bytes are valid float32 and
    // decode without error, so a wrong call is silent rather than loud.
    const src = new Uint32Array([0, 1, 2]);
    expect(decodeFloats(src.buffer)[1]).not.toBe(1);
  });
});

describe("thumbnailUrl", () => {
  it("builds the endpoint url", () => {
    expect(thumbnailUrl(2, 1, -1, "mu-h", "real", 96)).toBe(
      "/api/thumbnail/2/1/-1?system=mu-h&basis=real&size=96",
    );
  });
});
