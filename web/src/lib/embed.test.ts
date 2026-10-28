import { describe, expect, it } from "vitest";
import { embedSnippet, fullAppHref } from "./embed";
import { URL_DEFAULTS, parseAppUrl } from "./urlState";

describe("embed", () => {
  const state = {
    ...URL_DEFAULTS,
    n: 3,
    l: 2,
    m: 1,
    view: "plane" as const,
    tour: "many-electrons",
    step: 2,
  };

  it("embeds the same state with the embed flag and no tour", () => {
    const html = embedSnippet("https://atomsim.fly.dev", state);
    const src = /src="([^"]+)"/.exec(html)![1].replace(/&amp;/g, "&");
    const parsed = parseAppUrl(src.slice(src.indexOf("?")));
    expect(parsed).toMatchObject({ n: 3, l: 2, m: 1, view: "plane", embed: true });
    expect(parsed.tour).toBeUndefined();
    expect(html).toContain('loading="lazy"');
  });

  it("links back to the full app without the embed flag", () => {
    const href = fullAppHref({ ...state, embed: true });
    expect(href).not.toContain("embed");
    expect(parseAppUrl(href.slice(1))).toMatchObject({ n: 3, l: 2, m: 1 });
  });
});
