/** Visitor counting: a headcount, and deliberately nothing finer.
 *
 * The question this answers is "did anyone open it, and roughly how many",
 * over a span of months. Fly's proxy cannot answer it: it counts requests
 * rather than people, and one visit is 20 to 40 requests once the bundle, the
 * fonts, the websocket and a job POST are counted. Its Prometheus also retains
 * about 15 days, so a two-month question asked in two months has no data
 * behind it unless something started recording first.
 *
 * GoatCounter does the counting instead. It sets no cookie and stores no
 * address: it derives a daily-rotating hash from the address and user agent,
 * which is what makes "unique visitors" a number rather than a fiction. The
 * cost, stated plainly because this project does not hide costs, is that a
 * visit is reported to a host that is not this one.
 *
 * Two limits belong on any number that comes out of it. A shared network makes
 * many people look like one, and a phone moving between wifi and cellular
 * makes one person look like several. So the count is an estimate in both
 * directions, and the honest way to quote it is "roughly".
 */

/** Where count.js is fetched from. GoatCounter's own CDN, pinned here rather
 * than passed in configuration: the endpoint is the deployment's business, the
 * script's origin is this file's. */
const SCRIPT_SRC = "https://gc.zgo.at/count.js";

/** Marks the injected tag, and doubles as the "already installed" probe. */
const MARKER = "script[data-goatcounter]";

/** The counting endpoint, or `null` when counting is off.
 *
 * Off is the default and the whole point of the null: `VITE_GOATCOUNTER` is
 * unset in a dev server and in a plain `npm run build`, so nothing is reported
 * while the app is being worked on, and only the deploy that sets it counts.
 *
 * Anything that is not an https URL returns null rather than being repaired.
 * The value ends up as the src-adjacent target of a third-party script, so a
 * typo should switch counting off and be noticed, never point the beacon
 * somewhere unintended.
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

/** Append the counting script, and report whether it went in.
 *
 * Returns false when counting is off or a tag is already present, which is not
 * a failure: both are ordinary. The already-present guard is cheap insurance
 * against a second tag under a dev-server module reload, where one arrival
 * would otherwise be counted as two visits.
 */
export function installAnalytics(raw: unknown, doc: Document): boolean {
  const endpoint = analyticsEndpoint(raw);
  if (endpoint === null) return false;
  if (doc.querySelector(MARKER) !== null) return false;

  // Report the path and not the query string. Every store change rewrites the
  // URL (see main.tsx), so the address carries the live state: n, l, m, the
  // system, the view, an open tour step. Sending that would report what each
  // visitor was looking at, which is finer than a headcount needs and finer
  // than a visitor agreed to, and it would shard one page into thousands of
  // distinct rows in the dashboard. Set before the script loads, because
  // count.js reads this object on load and counts immediately.
  const win = doc.defaultView as (Window & { goatcounter?: unknown }) | null;
  if (win) win.goatcounter = { path: () => doc.location.pathname };

  const script = doc.createElement("script");
  script.async = true;
  script.src = SCRIPT_SRC;
  script.dataset.goatcounter = endpoint;
  doc.head.appendChild(script);
  return true;
}
