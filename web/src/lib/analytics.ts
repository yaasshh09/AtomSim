/** Visitor counting: a headcount, and deliberately nothing finer.
 *
 * The question this answers is "did anyone open it, and roughly how many",
 * over a span of months. Fly's proxy cannot answer it: it counts requests
 * rather than people, and one visit is 20 to 40 requests once the bundle, the
 * fonts, the websocket and a job POST are counted. Its Prometheus also retains
 * about 15 days, so a two-month question asked in two months has no data
 * behind it unless something started recording first.
 *
 * GoatCounter does the counting. It sets no cookie and stores no address: it
 * derives a daily-rotating hash from the address and user agent, which is what
 * makes "unique visitors" a number rather than a fiction. The cost, stated
 * plainly because this project does not hide costs, is that a visit is
 * reported to a host that is not this one.
 *
 * **The beacon is built here rather than by GoatCounter's count.js**, and the
 * reason is a measured one. count.js was tried first, with its `path` setting
 * overridden to report `/`. Against production it sent
 * `?p=%2F&...&q=%3Fn%3D3%26l%3D1%26m%3D-1%26view%3Dplane`: the override
 * governs the recorded page, and `q: location.search` is hardcoded in its
 * get_data with no setting that reaches it. Since every store change rewrites
 * this app's URL, that field was the visitor's live state, n, l, m, the
 * system, the view, any open tour step, going out on every arrival. A
 * headcount does not need it and a visitor did not agree to it.
 *
 * So the /count endpoint is called directly, which GoatCounter documents as a
 * supported integration ("the /count endpoint returns a small 1x1 GIF on GET
 * requests... or you can build your own JavaScript integration"). What that
 * gives up is their client-side bot heuristic, so expect a little more bot
 * noise in the total than count.js would leave. What it buys is that the
 * fields leaving this page are only the ones named in `beaconUrl` below, and
 * a change to a third-party script cannot quietly widen them again.
 *
 * Two limits belong on any number that comes out of it. A shared network makes
 * many people look like one, and a phone moving between wifi and cellular
 * makes one person look like several. So the count is an estimate in both
 * directions, and the honest way to quote it is "roughly".
 */

/** Marks the beacon as sent, and doubles as the "already counted" probe.
 * An attribute on <html>, because unlike a module variable it survives a
 * dev-server module reload, where one arrival would otherwise count twice. */
const MARKER = "data-atomsim-counted";

/** Hosts that must never report a visit, whatever the configuration says.
 * count.js used to refuse localhost on its own ("not counting because of:
 * localhost"); rolling the beacon by hand means inheriting that duty. The
 * config gate below already covers the normal case, since VITE_GOATCOUNTER is
 * empty in a dev server and a local build. This is the belt for that braces. */
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]", ""]);

/** What a single arrival is allowed to say about itself.
 *
 * The whole privacy question lives in this shape: a field absent here cannot
 * be sent, and `location.search` is absent on purpose.
 */
export interface VisitFacts {
  /** Page path, never the query string. */
  pathname: string;
  /** Document title. Constant for this app, and what the dashboard lists. */
  title: string;
  /** Where the visitor came from. Useful, and about the referring site rather
   * than about anything the visitor did here. */
  referrer: string;
  /** Screen width, which is how GoatCounter splits phone from desktop. */
  screenWidth: number;
  /** Whether this is an automated browser, so scripted checks (Playwright,
   * headless smoke tests) can be marked rather than counted as people. */
  automated: boolean;
  /** Cache-buster. An <img> GET is cacheable, and a cached beacon is a visit
   * that never reaches the server. Passed in so this function stays pure. */
  nonce: string;
}

/** The counting endpoint, or `null` when counting is off.
 *
 * Off is the default and the whole point of the null: `VITE_GOATCOUNTER` is
 * unset in a dev server and in a plain `npm run build`, so nothing is reported
 * while the app is being worked on, and only the deploy that sets it counts.
 *
 * Anything that is not an https URL returns null rather than being repaired.
 * The value is the destination of an outbound beacon, so a typo should switch
 * counting off and be noticed, never point it somewhere unintended.
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
  // https only. The page is served over https and force_https is set in
  // fly.toml, so an http endpoint would be blocked as mixed content anyway;
  // failing here makes that a configuration error instead of a console
  // warning nobody reads.
  if (url.protocol !== "https:") return null;

  // The dashboard shows you the origin, and "/count" is the part you have to
  // know to append. Accept either, so the obvious paste works.
  if (url.pathname === "/") url.pathname = "/count";
  // A query string or fragment here is always a paste accident, and passing
  // one along would tack it onto every beacon.
  url.search = "";
  url.hash = "";
  return url.toString();
}

/** The exact URL a single arrival reports itself to.
 *
 * Field names are GoatCounter's: p path, t title, r referrer, s screen width,
 * b bot, rnd cache-buster. Everything sent is named right here, which is the
 * property the tests pin.
 */
export function beaconUrl(endpoint: string, visit: VisitFacts): string {
  const url = new URL(endpoint);
  url.searchParams.set("p", visit.pathname);
  if (visit.title) url.searchParams.set("t", visit.title);
  // Empty is the common case (someone typed the address, or the referrer was
  // stripped) and sending an empty r would be noise.
  if (visit.referrer) url.searchParams.set("r", visit.referrer);
  if (visit.screenWidth > 0) url.searchParams.set("s", String(visit.screenWidth));
  url.searchParams.set("b", visit.automated ? "1" : "0");
  url.searchParams.set("rnd", visit.nonce);
  return url.toString();
}

/** Report one arrival, and say whether it went out.
 *
 * Returns false when counting is off, when the page is local, or when this
 * document already counted. None of those are failures; all three are
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
    // pathname, never location.search: see the note at the top of this file.
    pathname: doc.location.pathname,
    title: doc.title,
    referrer: doc.referrer,
    screenWidth: win.screen?.width ?? 0,
    automated: win.navigator?.webdriver === true,
    nonce: Math.random().toString(36).slice(2),
  });
  return true;
}
