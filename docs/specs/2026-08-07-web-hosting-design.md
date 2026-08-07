# Web hosting: atomsim on a public URL

**Date:** 2026-08-07
**Status:** design approved, implementation pending

## Goal

Put the instrument on a public URL that anyone can open, without changing what
it computes or what it admits about its own physics. The deliverable is a
portable container plus the configuration that makes a single-process compute
server safe to expose.

## Non-goals

- **No horizontal scaling.** The architecture forbids it; see "One process is a
  requirement" below. Anything that would run two replicas is out of scope.
- **No persistence.** There is no state worth keeping across a restart. Jobs are
  ephemeral by design and the physics is deterministic given (n, l, m, system).
- **No auth.** The whole point is a link that opens.
- **No product changes.** The million-point cloud, every view, and every atom
  stay exactly as they are in public. Sizing is chosen to fit the app rather
  than the app trimmed to fit a host.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Host | Fly.io | Custom domain with free TLS, fixed-price ceiling, and a container that ports elsewhere unchanged |
| URL | `atomsim.fly.dev` first | Verifies the deploy before DNS is a variable; custom domain is two commands later with no redeploy |
| Machine | `shared-cpu-1x`, 1GB | 1GB is what `JobStore` actually needs; see "Sizing" |
| Idle | Scale to zero, `suspend` | Pay only for running seconds. Suspend snapshots RAM, so resume skips the engine import that dominates a cold start |
| Deploy | GitHub Action on push to `main` | Matches the existing main-only workflow |

### Rejected alternatives

**Hugging Face Docker Space (free, 2 vCPU / 16GB).** Better hardware than
anything in budget and genuinely viable. Rejected only because free hardware
cannot have a custom domain, and a showcase URL is the point. Worth remembering
as the fallback if Fly's bill ever stops being worth it: the same Dockerfile
runs there with a port change.

**Vercel.** Not a Python limitation. WebSockets have been supported since June
2026 and Python functions get 500MB (5GB with Large Functions), so the usual
objections no longer apply. It fails on three specifics in this codebase:

1. `_dispatch` returns a job id and lets the solve continue in a thread pool.
   Serverless freezes the instance when the response is sent, so the job would
   never run.
2. `JobStore` is a per-process `OrderedDict`. The POST, the `/ws/jobs/{id}`
   watch, and the `/data` fetch are three invocations, and Vercel does not
   guarantee they reach the same instance.
3. `TokenBucket` is per-process. `ratelimit.py` already says it is "wrong the
   moment there are two."

There is also a cost-shape argument: Vercel bills Active CPU and this app *is*
pinned CPU, so a crawler generates an unbounded metered bill. A fixed-price box
has a ceiling no amount of traffic can exceed.

**Render.** Free tier sleeps after 15 minutes, so nearly every showcase visitor
would eat a cold start. Paid entry is $7, above budget, on weaker hardware.

**Split static frontend on a CDN plus a sleeping API.** Every API call site is
currently a relative URL and the websocket uses `location.host`, so this needs a
base-URL layer, CORS origins, and cross-origin WSS. And a page that paints
instantly but cannot compute is arguably worse than an honest wait.

## Architecture

Unchanged, which is the point. One uvicorn process serves `/api/*`,
`/ws/jobs/{id}`, and the static `web/dist` at `/`, all same origin. The
frontend needs no changes: relative URLs and `location.host` already do the
right thing behind any proxy.

### One process is a requirement

Two pieces of server state live in process memory and have no distributed form:

- `JobStore` holds job status, progress, and results. A job created on one
  instance is invisible to another, which breaks the POST then watch then fetch
  flow.
- `TokenBucket` counts per process. N instances means N full buckets, so the
  80-token burst silently becomes 80N.

**`fly launch` creates two machines by default for high availability.** That
configuration breaks this app in exactly the way `ratelimit.py` warns about.
The deploy must pin the count to one, and a test should assert the deployed
machine count rather than trusting the config file.

## Sizing

Driven by `JobStore`, not by the framework.

| Component | Memory |
|---|---|
| Python with numpy, scipy, matplotlib imported | ~250MB |
| 16 retained jobs at the 1,000,000-point ceiling (~20MB each) | ~320MB |
| Headroom | remainder |

`SampleRequest.count` is already hard-capped at 1,000,000 by Pydantic, and
plane resolution at 1024, and isosurface resolution to the `GRID_SIZES` set. The
worst case is therefore bounded, and 1GB holds it. 512MB would not, which is why
the cheaper machine is rejected rather than compensated for with a lower cap.

## The container

Multi-stage, because `web/dist` is gitignored and must be built:

1. **`node:22-slim`**: `npm ci` then `npm run build` (which includes
   `tsc --noEmit`), producing `/web/dist`.
2. **`python:3.12-slim`**: `pip install .` for manylinux wheels with no
   compiler, then copy `web/dist` from stage 1.

Runs as a non-root user. `CMD` invokes uvicorn directly rather than
`atomsim serve`, because the CLI hardcodes `host="127.0.0.1"` and opens a
browser:

```
uvicorn atomsim.server.app:create_app --factory --host 0.0.0.0 --port 8080
```

## Server changes

Three, and each one closes a way the deploy could lie.

### 1. `ATOMSIM_WEB_DIST` override (required)

`WEB_DIST` is `Path(__file__).resolve().parents[3] / "web" / "dist"`, which
assumes a source checkout. Installed into site-packages, `parents[3]` resolves
to the Python lib directory and the path does not exist, so `app.mount` is
skipped. The failure is silent: the API answers normally, `/` returns 404, and
nothing says why.

Add an env override, and log at startup where the UI mounted from or that it did
not. A deploy that ships no UI should say so in its first ten lines of output.

### 2. Trustworthy client IP

Set `ATOMSIM_CLIENT_IP_HEADER=fly-client-ip`.

`_client_key` takes `forwarded.split(",")[0]`, the leftmost entry. Its own
docstring says trusting a client-settable address "is the same as having no
limiter at all", and with `X-Forwarded-For` that is exactly what happens: a proxy
appends, so a client sending `X-Forwarded-For: whatever` produces
`whatever, <real-ip>` and the leftmost entry is the attacker's own string.
Rotate it per request and the limiter is bypassed entirely.

`Fly-Client-IP` is set by Fly's proxy, single-valued, and overwritten rather
than appended, so the existing leftmost parse is correct for it.

**Must verify against the live deploy**, not assumed: confirm the header
arrives, is single-valued, and that a client-supplied `Fly-Client-IP` does not
survive. If any of that is false, fall back to parsing the rightmost
`X-Forwarded-For` entry with the hop count measured rather than guessed.

### 3. Thread pinning

Set `OMP_NUM_THREADS=1` so OpenBLAS does not oversubscribe the shared core
across two job workers. `_job_worker_count()` needs no override: a 1-vCPU
Firecracker VM reports 1 core, giving the intended floor of 2 workers, which the
code already documents as correct ("on a single core they simply timeshare").

## Runtime configuration

`fly.toml`, the essentials:

| Setting | Value | Why |
|---|---|---|
| `internal_port` | 8080 | Matches the uvicorn `CMD` |
| `force_https` | true | |
| `auto_stop_machines` | `suspend` | Resume from a RAM snapshot in a few hundred ms rather than re-importing the engine |
| `auto_start_machines` | true | Wake on request |
| `min_machines_running` | 0 | The scale-to-zero decision |
| `[[vm]] size` | `shared-cpu-1x` | |
| `[[vm]] memory` | `1gb` | See "Sizing" |
| machine count | exactly 1 | Enforced separately; the config file alone does not guarantee it |

Environment:

| Variable | Value | Why |
|---|---|---|
| `ATOMSIM_WEB_DIST` | path to the copied `web/dist` | Without it the UI silently does not mount |
| `ATOMSIM_CLIENT_IP_HEADER` | `fly-client-ip` | Without it the whole internet shares one token bucket |
| `OMP_NUM_THREADS` | 1 | Stop OpenBLAS oversubscribing the shared core |
| `MPLCONFIGDIR` | a writable temp path | `HOME` may not be writable; matplotlib is already pinned to Agg in `thumbnails.py` |
| `PYTHONUNBUFFERED` | 1 | Logs appear when they happen |

## Deploy pipeline

GitHub Action on push to `main`, gated on the existing `python` and `web` jobs
passing, using `flyctl deploy --remote-only` with `FLY_API_TOKEN` as a repo
secret. Deploying a broken build to a public URL because a test was red is the
one outcome worth blocking.

## Cold start and the waiting page

A sleeping site makes a visitor wait, and the visitor cannot be told why by the
server that is asleep. Anything that explains the wait has to be hosted
somewhere that never sleeps.

`suspend` is expected to make this moot: resume restores a snapshot in which the
engine is already imported, so the wait should be a few hundred milliseconds
rather than the 10 to 20 seconds a cold start costs on a shared vCPU. The 5.4s
figure `cli.py` records for importing the server stack was measured on a
14-core laptop and is the thing suspend skips.

**Decision: measure before building anything.** Deploy, then time three cases:

1. Resume from suspend (the common case).
2. True cold start after a deploy (the guaranteed-slow case).
3. A warm request (the baseline).

If resume lands under a second, there is no wait to explain and no page to
build. If it does not, the waiting page becomes a work item, scoped as: a static
page on an always-warm free host that polls `/api/health` and forwards once the
engine answers, needing CORS on that one endpoint and nothing else.

If it is ever built, it must not show a progress bar. Boot progress is not
measurable from outside the machine, so an animated 0-to-100% would be inventing
a number. An indeterminate spinner or an elapsed-seconds counter states only what
is actually known, which is the same standard every other number in this project
is held to.

## Testing

- **Container smoke test:** build the image, run it, then assert `/api/health`
  answers, `/` serves the built index rather than 404, and a job POST plus
  websocket watch plus data fetch completes end to end. The last one is what
  actually proves the container works, because it exercises the thread pool and
  the cross-request job state together.
- **Linux CI job:** the current CI is Windows-only and would never catch a
  Dockerfile break. Add an `ubuntu-latest` job that builds the image and runs
  the smoke test.
- **Machine count assertion:** verify exactly one machine is running after
  deploy.
- **Wake-time measurement:** the three timings in "Cold start and the waiting
  page", recorded in the deploy doc as numbers rather than adjectives. They are
  what decides whether the waiting page gets built.

## Risks and open verifications

| Risk | Handling |
|---|---|
| Auto-stop kills a job mid-solve | Background threads are invisible to the proxy's idle detection. In practice the client opens the websocket immediately after the POST, so a connection is always in flight. Verify with a long job (HF solve on argon) and confirm the machine stays up. |
| Health checks defeat scale-to-zero | A Fly http check may keep the machine running forever. Default to no health check; if one is added, verify the machine still stops when idle. |
| Unmetered result re-download | `/api/jobs/{id}/data` is not rate limited (only `POST /api/jobs/` is), so one token buys unlimited 20MB re-fetches until eviction. Low severity. Covered by a hard spend cap rather than new code. |
| Egress from a determined client | Sustained abuse at the limiter's ceiling is bounded but nonzero at $0.02/GB. Set a Fly spending limit and a budget alert. |
| Suspend snapshots are not durable | Host migration or capacity pressure discards the snapshot and the next visit pays a full cold start. This is the case the waiting page would exist for; see "Cold start and the waiting page". |
| Clock is briefly wrong after resume | `TokenBucket` reads `time.monotonic()`, so buckets do not refill across a suspension. That is stricter than intended rather than looser, and no requests arrive while suspended, so it is harmless. Noted so it is not rediscovered as a bug. |
| Region choice | `bom` is nearest to the author; verify against `fly platform regions` rather than trusting a remembered list. |

## Cost

| Item | Cost |
|---|---|
| shared-cpu-1x 1GB, scale to zero | ~$1-2/mo at showcase traffic ($5.92 if it never slept) |
| Shared IPv4, IPv6, TLS certificates | $0 |
| Egress | $0.02/GB, inbound free |

A hard spending limit on the Fly organisation is part of the deploy, not an
afterthought. It is the only control that bounds a metered bill against traffic
nobody predicted.
