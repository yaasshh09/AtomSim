import { serializeAppUrl, type UrlState } from "./urlState";

/**
 * An iframe that shows this exact state, embedded in someone else's page.
 *
 * The embedded app draws the view alone, with a link back to the full app,
 * so the picture keeps its labels and badges and can be checked at the source.
 * Tours never travel into an embed: a guided walk belongs to the full app.
 */
export function embedSnippet(origin: string, state: UrlState, width = 640, height = 480): string {
  const query = serializeAppUrl({ ...state, embed: true, present: false, tour: null, step: 0 });
  const src = `${origin}/${query}`;
  const title = `AtomSim: n = ${state.n}, l = ${state.l}, m = ${state.m}, ${state.system}`;
  return (
    `<iframe src="${src.replace(/&/g, "&amp;")}" width="${width}" height="${height}" ` +
    `title="${title}" loading="lazy" style="border:0"></iframe>`
  );
}

/** The full app's link for an embedded state: the same state with the embed switched off. */
export function fullAppHref(state: UrlState): string {
  return `/${serializeAppUrl({ ...state, embed: false, tour: null, step: 0 })}`;
}
