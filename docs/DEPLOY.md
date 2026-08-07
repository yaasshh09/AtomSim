# Deploying atomsim

Public URL: <https://atomsim.fly.dev>
Fly app `atomsim`, region `fra` (Frankfurt), one `shared-cpu-1x` machine with
1GB, suspended when idle.

Design: `docs/specs/2026-08-07-web-hosting-design.md`.

## One machine, always

`JobStore` and `TokenBucket` are per-process in-memory state. A second machine
would hand clients job ids that do not exist on the instance they reach next,
and would give every client a second full rate-limit bucket. `ratelimit.py`
says it outright: counting per process is "wrong the moment there are two".

Deploys pass `--ha=false`. **`fly status` showing two machines is a correctness
bug, not a cost inefficiency.** Fix it with `fly scale count 1` immediately.

## What the environment variables prevent

All are set in the `Dockerfile`, so they travel with the image rather than
living in host configuration that a rebuild elsewhere would lose.

| Variable | Value | Without it |
|---|---|---|
| `ATOMSIM_WEB_DIST` | `/app/web/dist` | The interface silently does not mount. `parents[3]` resolves into site-packages once installed, so `/` returns 404 while the API answers normally |
| `ATOMSIM_CLIENT_IP_HEADER` | `fly-client-ip` | Every visitor on the internet shares one token bucket. Verified below |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` | `1` | OpenBLAS spawns a thread per core underneath two job workers on one shared core, and they contend |
| `MPLCONFIGDIR` | `/tmp/matplotlib` | matplotlib has no writable cache directory |
| `PYTHONUNBUFFERED` | `1` | Log lines appear at shutdown rather than when they happen |

## Region: why not Mumbai

`bom` was the first choice, being nearest the author. It ran out of capacity
mid-deploy on 2026-08-07: the deploy destroyed the running machine, could not
create its replacement, and **the app was down until the region changed**.
Retrying did not help.

The lesson is that a deploy is exactly the moment capacity is needed, so a
constrained region is an availability risk and not only a latency choice. `fra`
is the next closest with room. If it is ever worth moving back, change
`primary_region` in `fly.toml` and deploy, but treat capacity as the deciding
factor rather than distance.

## Measured, 2026-08-07

Taken against `shared-cpu-1x` / 1GB while the app was in `bom`, from outside
the region, so every figure includes roughly 0.4 s of network round trip. The
move to `fra` changes that network component but not the wake mechanism, which
is what the cold-versus-resume comparison is about. Warm latency measured the
same 0.43 s from the same client after the move.

| Case | Time to `/api/health` |
|---|---|
| Warm | **0.41 s** (median of 5: 349, 354, 409, 412, 416 ms) |
| Resume from suspend | **1.57 s** |
| True cold start | **22.49 s** |

The machine suspends after about **240 s** of idle.

Argon (Z=18), the slowest solve offered, finishes in **2.8 s** on this machine
and returns the expected total energy of -526.815 hartree at tier
`approximation`.

### No waiting page, and why

The spec deferred that decision to these numbers. A resume costs 1.57 s
including the round trip, roughly 1.2 s of actual wake, which is an ordinary
page load rather than a wait that needs explaining. A splash screen at that
duration would flash and distract rather than reassure.

`suspend` is what buys this. A plain `stop` would pay the 22.49 s cold start on
every first visit, because it re-imports numpy, scipy and matplotlib from
nothing; `cli.py` measures that import at 5.4 s on a 14-core laptop and this
machine has one shared core. The snapshot already contains the imported
process.

The 22.49 s path still exists in two cases: when Fly discards the snapshot
(host migration or capacity pressure), and on the first visit after a deploy.
The deploy case is handled by warming the machine at the end of the deploy
workflow, so the one predictably cold moment never faces a visitor. The
snapshot-eviction case is rare and left unhandled deliberately.

If that ever stops being true, the waiting page is scoped in the spec: a static
page on an always-warm host that polls `/api/health` and forwards. It must not
show a progress bar, because boot progress is not measurable from outside the
machine and an animated percentage would be inventing a number.

### Mid-solve suspension

Not a hazard at these numbers. The longest job is about 3 s and the idle timer
is about 240 s, an 80x margin, and the client holds a websocket open for the
whole job anyway.

## The websocket needs a library nothing imports

`uvicorn` ships no websocket implementation. Without `websockets` or `wsproto`
installed it declines the upgrade, the request stays plain HTTP, no route
matches `/ws/jobs/{id}`, and the client gets a 404. Job progress then never
arrives and the app appears to do nothing.

This shipped on 2026-08-07 and was fixed the same day. It had been latent since
the project began: `pyproject.toml` declared `uvicorn>=0.30` rather than
`uvicorn[standard]`, and the conda env supplied `websockets` transitively, so
the first environment that did not supply it was production.

**The pytest suite cannot catch this.** `TestClient.websocket_connect` speaks
websockets in process and never reaches uvicorn, so
`test_websocket_streams_progress_to_done` passes against a server that has no
websocket support at all. `scripts/smoke_container.sh` now asks the real server
for a 101, and `test_a_websocket_implementation_is_installed` asserts the
dependency is declared.

If job progress ever stalls again, look for this in the logs first:

    WARNING:  No supported WebSocket library detected.

## Forwarded-address verification

The spec marked this must-verify rather than assume, because `_client_key`
trusts whatever header it is told to.

- **The app sees `172.16.3.90` as `request.client.host`**, a private address:
  Fly's internal proxy. Without a trusted header every visitor would therefore
  share a single bucket, and the symptom would be indistinguishable from
  ordinary popularity.
- **`Fly-Client-IP` arrives and carries the visitor's real public address.**
  Confirmed by exhausting the bucket and reading the refusal line, which named
  a public IPv6 address rather than the `172.16` proxy.
- **A forged `Fly-Client-IP` does not mint a fresh bucket.** With the real
  bucket drained, ten requests with no headers were refused ten times, and ten
  requests each claiming a different `Fly-Client-IP` were refused ten times
  too. Fly's proxy overwrites the header.

Note when re-running this: the bucket refills at one token per three seconds,
so any test with a gap in it will show forgery "working" when the bucket has
simply refilled. Drain and test back to back.

## Deploying

Automatic on push to `main`, once `python`, `web` and `container` all pass. By
hand:

    fly deploy --remote-only --ha=false

The image builds the frontend itself, so no local `npm run build` is needed.
`VITE_SITE_URL` is a build argument in `fly.toml`: it is substituted into the
Open Graph tags and baked into the bundle, not read at runtime. Changing the
public URL means changing it there and rebuilding, or link previews will point
at the old origin.

## Counting visitors

Fly cannot answer "how many people used this". It counts requests, and one
visit is 20 to 40 of them once the bundle, the fonts, the websocket and a job
POST are counted. Its Prometheus also retains **about 15 days**, so a question
asked two months from now has nothing behind it. That is the whole reason
something else does the counting, and the reason it had to start before the
data was wanted rather than when it was.

GoatCounter does it. Set up once:

1. Create the site at <https://www.goatcounter.com/signup>, which gives a code
   and the URL `https://<code>.goatcounter.com`.
2. Put `https://<code>.goatcounter.com/count` in `VITE_GOATCOUNTER` under
   `[build.args]` in `fly.toml`.
3. Deploy. The value is baked into the bundle at build time, so it takes a
   rebuild, not a restart.

It is public, not a secret: every visitor downloads it inside the bundle. That
is why it sits in `fly.toml` and not in `fly secrets`, which would be neither
secret nor readable at build time.

Empty is the default everywhere else, so a dev server, a local `npm run build`
and the CI container image all report nothing. Only a build that passes the
value counts.

**What is reported**, and this is the complete list, built by `beaconUrl` in
`web/src/lib/analytics.ts`: the path (always `/`), the document title, the
referrer when there is one, the screen width, and a bot flag. GoatCounter adds
a daily-rotating hash of address and user agent at its end, which is what makes
"unique visitors" a number instead of a guess. No cookie is set and no address
is stored.

**What is deliberately not reported.** The query string. Every store change
rewrites the URL, so the address carries n, l, m, the system, the view and any
open tour step. A headcount does not need it and a visitor did not agree to it.

That is also why the beacon is built here instead of by GoatCounter's
`count.js`. count.js was tried first, with its `path` setting overridden to
report `/`, and against production it sent
`?p=%2F&…&q=%3Fn%3D3%26l%3D1%26m%3D-1%26view%3Dplane`. The override governs the
recorded page; `q: location.search` is hardcoded in its `get_data` with no
setting that reaches it. Calling `/count` directly is a supported integration
(GoatCounter documents the endpoint as a 1×1 GIF on GET, and says outright that
you can build your own). The cost is their client-side bot heuristic, so expect
slightly more bot noise in the total. What it buys is that the fields leaving
the page are only the ones named in `beaconUrl`, and a change to a third-party
script cannot quietly widen them.

Because the beacon is sent by JavaScript, **the deployed HTML contains no trace
of it**. Grepping the page source for "goatcounter" returns nothing however
well it is working; verifying it needs a browser.

**How to read the number.** The dashboard's unique-visitor count over a date
range is the figure to quote. Two limits belong on it in both directions: a
shared network makes many people look like one, and a phone moving between wifi
and cellular makes one person look like several. So it is an estimate, and
"roughly N people" is the honest form. Traffic in the first days is mostly this
project's own deploy verification and smoke tests rather than visitors.

## Costs

`shared-cpu-1x` with 1GB is $5.92/month if it never sleeps; suspended when idle
it should land far below that. Shared IPv4, IPv6 and TLS certificates are free.
Egress is $0.02/GB with inbound free.

A hard spending limit is set on the organisation. `/api/jobs/{id}/data` is not
rate limited (only `POST /api/jobs/` is), so one token buys repeated fetches of
a result up to 20MB until it is evicted. That is bounded by the spending limit
rather than by code.

## Adding a custom domain

    fly certs add <domain>

Then the DNS record it prints. Afterwards update `VITE_SITE_URL` in `fly.toml`
and the `container` job in `.github/workflows/ci.yml`, and redeploy so the
Open Graph tags follow. No code change.
