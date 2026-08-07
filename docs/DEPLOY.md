# Deploying atomsim

Public URL: <https://atomsim.fly.dev>
Fly app `atomsim`, region `bom` (Mumbai), one `shared-cpu-1x` machine with 1GB,
suspended when idle.

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

## Measured, 2026-08-07

Against `shared-cpu-1x` / 1GB in `bom`, measured from outside the region, so
every figure includes roughly 0.4 s of network round trip.

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
