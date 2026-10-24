/**
 * Browser-side crash reports, sent to /api/client-error.
 *
 * What goes: the error message, its stack, the page path without the query
 * string, and the app version. Nothing else, no identifier of any kind, and
 * the privacy page says so. Each distinct message goes once, and a page sends
 * at most MAX_REPORTS, so an error inside an animation frame cannot become a
 * flood. Production builds only: in development the console is the report.
 */
const MAX_REPORTS = 5;

export interface ErrorReport {
  message: string;
  stack: string | null;
  page: string;
  version: string;
}

/** The report for one error, trimmed to what the server accepts. */
export function buildReport(err: unknown, path: string, version: string): ErrorReport {
  const e = err instanceof Error ? err : new Error(String(err));
  return {
    message: `${e.name}: ${e.message}`.slice(0, 500),
    stack: e.stack ? e.stack.slice(0, 4000) : null,
    page: path.split("?")[0].slice(0, 200),
    version: version.slice(0, 40),
  };
}

/** A gate that lets each distinct message through once, and at most `max` in all. */
export function reportGate(max = MAX_REPORTS): (message: string) => boolean {
  const seen = new Set<string>();
  return (message) => {
    if (seen.has(message) || seen.size >= max) return false;
    seen.add(message);
    return true;
  };
}

export function installErrorReporter(version: () => Promise<string>): void {
  const allow = reportGate();
  const send = (err: unknown) => {
    void version().then((v) => {
      const report = buildReport(err, window.location.pathname, v);
      if (!allow(report.message)) return;
      void fetch("/api/client-error", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(report),
        keepalive: true,
      }).catch(() => {});
    }, () => {});
  };
  window.addEventListener("error", (e) => send(e.error ?? e.message));
  window.addEventListener("unhandledrejection", (e) => send(e.reason));
}
