/** How I count visitors: a headcount, and deliberately nothing finer.
 *
 * The question I answer here is "did anyone open it, and roughly how many",
 * over a span of months. Fly's proxy cannot answer it for me: it counts
 * requests rather than people, and one visit is 20 to 40 requests once you add
 * up the bundle, the fonts, the websocket and a job POST. Its Prometheus also
 * retains about 15 days, so a two-month question asked in two months has no
 * data behind it unless something started recording first.
 *
 * GoatCounter does the counting for me. It sets no cookie and stores no
 * address: it derives a daily-rotating hash from the address and user agent,
 * which is what makes "unique visitors" a number rather than a fiction. The
 * cost, which I state plainly because I do not hide costs, is that I report a
 * visit to a host that is not this one.
 *
 * **I build the beacon here rather than letting GoatCounter's count.js do
 * it**, and I have a measured reason. I tried count.js first, with its `path`
 * setting overridden to report `/`. Against production it sent
 * `?p=%2F&...&q=%3Fn%3D3%26l%3D1%26m%3D-1%26view%3Dplane`: the override
 * governs the recorded page, and `q: location.search` is hardcoded in its
 * get_data with no setting that reaches it. Since every store change rewrites
 * my URL, that field was the visitor's live state, n, l, m, the system, the
 * view, any open tour step, going out on every arrival. I do not need it for a
 * headcount and the visitor did not agree to it.
 *
 * So I call the /count endpoint directly, which GoatCounter documents as a
 * supported integration ("the /count endpoint returns a small 1x1 GIF on GET
 * requests... or you can build your own JavaScript integration"). What I give
 * up is their client-side bot heuristic, so expect a little more bot noise in
 * my total than count.js would leave. What I buy is that the only fields
 * leaving this page are the ones I name in `beaconUrl` below, and a change to
 * a third-party script cannot quietly widen them again.
 *
 * Two limits belong on any number I produce. A shared network makes many
 * people look like one, and a phone moving between wifi and cellular makes one
 * person look like several. So my count is an estimate in both directions, and
 * the honest way to quote it is "roughly".
 */

/** How I mark the beacon as sent; it doubles as my "already counted" probe.
 * I put an attribute on <html>, because unlike a module variable it survives a
 * dev-server module reload, where I would otherwise count one arrival twice. */
const MARKER = "data-atomsim-counted";

/** Hosts I must never report a visit from, whatever the configuration says.
 * count.js used to refuse localhost on its own ("not counting because of:
 * localhost"), and rolling the beacon by hand means I inherit that duty. My
 * config gate below already covers the normal case, since VITE_GOATCOUNTER is
 * empty in a dev server and a local build. This is the belt for those braces. */
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]", ""]);

/** What I let a single arrival say about itself.
 *
 * The whole privacy question lives in this shape: I cannot send a field that
 * is absent here, and `location.search` is absent on purpose.
 */
export interface VisitFacts {
  /** The page path, never the query string. */
  pathname: string;
  /** The document title. Constant for me, and what the dashboard lists. */
  title: string;
  /** Where the visitor came from. Useful, and about the referring site rather
   * than about anything the visitor did here. */
  referrer: string;
  /** Screen width, which is how GoatCounter splits phone from desktop. */
  screenWidth: number;
  /** Whether this is an automated browser, so I can mark scripted checks
   * (Playwright, headless smoke tests) rather than count them as people. */
  automated: boolean;
  /** A cache-buster. An <img> GET is cacheable, and a cached beacon is a visit
   * that never reaches the server. I take it as an argument so this function
   * stays pure. */
  nonce: string;
}

/** My counting endpoint, or `null` when I am not counting.
 *
 * Off is my default and the whole point of the null: `VITE_GOATCOUNTER` is
 * unset in a dev server and in a plain `npm run build`, so I report nothing
 * while someone is working on me, and only the deploy that sets it counts.
 *
 * I return null for anything that is not an https URL rather than repairing
 * it. The value is the destination of an outbound beacon, so a typo should
 * switch my counting off and be noticed, never point it somewhere unintended.
 */
export function analyticsEndpoint(raw: unknown): string | null {
  if (typeof raw !== "string") return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;

  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    return null;
  }
  // I take https only. My page is served over https and force_https is set in
  // fly.toml, so an http endpoint would be blocked as mixed content anyway;
  // failing here makes that a configuration error instead of a console warning
  // nobody reads.
  if (url.protocol !== "https:") return null;

  // The dashboard shows you the origin, and "/count" is the part you have to
  // know to append. I accept either, so the obvious paste works.
  if (url.pathname === "/") url.pathname = "/count";
  // A query string or fragment here is always a paste accident, and if I
  // passed one along I would tack it onto every beacon.
  url.search = "";
  url.hash = "";
  return url.toString();
}

/** The exact URL I report a single arrival to.
 *
 * The field names are GoatCounter's: p path, t title, r referrer, s screen
 * width, b bot, rnd cache-buster. I name everything I send right here, which
 * is the property my tests pin.
 */
export function beaconUrl(endpoint: string, visit: VisitFacts): string {
  const url = new URL(endpoint);
  url.searchParams.set("p", visit.pathname);
  if (visit.title) url.searchParams.set("t", visit.title);
  // Empty is the common case (someone typed the address, or the referrer was
  // stripped) and if I sent an empty r it would be noise.
  if (visit.referrer) url.searchParams.set("r", visit.referrer);
  if (visit.screenWidth > 0) url.searchParams.set("s", String(visit.screenWidth));
  url.searchParams.set("b", visit.automated ? "1" : "0");
  url.searchParams.set("rnd", visit.nonce);
  return url.toString();
}

/** I report one arrival, and say whether it went out.
 *
 * I return false when counting is off, when the page is local, or when I have
 * already counted this document. None of those are failures; all three are
 * ordinary.
 */
export function installAnalytics(raw: unknown, doc: Document): boolean {
  const endpoint = analyticsEndpoint(raw);
  if (endpoint === null) return false;
  if (LOCAL_HOSTS.has(doc.location.hostname)) return false;
  if (doc.documentElement.hasAttribute(MARKER)) return false;

  const win = doc.defaultView;
  if (!win) return false;

  doc.documentElement.setAttribute(MARKER, "");
  const beacon = new win.Image();
  beacon.src = beaconUrl(endpoint, {
    // I send pathname, never location.search: see my note at the top of this file.
    pathname: doc.location.pathname,
    title: doc.title,
    referrer: doc.referrer,
    screenWidth: win.screen?.width ?? 0,
    automated: win.navigator?.webdriver === true,
    nonce: Math.random().toString(36).slice(2),
  });
  return true;
}
