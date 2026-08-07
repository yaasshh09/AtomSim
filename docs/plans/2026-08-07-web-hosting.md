# Web Hosting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put atomsim on a public URL as a single containerised process on
Fly.io that suspends when idle, without changing what it computes or what it
admits.

**Architecture:** One uvicorn process serves `/api/*`, `/ws/jobs/{id}` and the
built frontend at `/`, all same origin, exactly as it does on a laptop. A
multi-stage Dockerfile builds `web/dist` with Node and runs it under Python.
Two server changes close ways the deploy could lie: where the UI mounted from,
and which client a compute job is charged to.

**Tech Stack:** Docker (multi-stage, `node:22-slim` then `python:3.12-slim`),
Fly.io Machines, FastAPI/uvicorn, pytest, GitHub Actions.

**Spec:** `docs/specs/2026-08-07-web-hosting-design.md`

## Global Constraints

- Engine-internal math stays in Hartree atomic units. This plan adds no physics
  and must change no number any endpoint returns.
- `ruff check .` must pass. Line length 100. E741 is ignored project-wide.
- Python 3.12, Node 22.
- Every value crossing a module boundary carries its `Provenance`. Nothing in
  this plan touches that boundary; if a change here would strip or synthesise a
  provenance, the change is wrong.
- **One process, always.** `JobStore` and `TokenBucket` are per-process
  in-memory state. Any configuration that could run two replicas is a defect.
- Commit messages carry no AI attribution and no tooling names.
- Work commits directly to `main`, one logical change per commit.
- No em dashes in prose or comments.

---

### Task 1: Say where the UI mounted from

`WEB_DIST` is `Path(__file__).resolve().parents[3] / "web" / "dist"`, which
assumes a source checkout with the package beside `src/`. Installed into
site-packages that resolves into the Python library directory, nothing is
there, and `app.mount` is skipped without a word: the API answers normally and
`/` returns 404 forever. The container needs the override, and the silence is a
bug independent of the container.

**Files:**
- Modify: `src/atomsim/server/app.py:134` (the `WEB_DIST` constant)
- Modify: `src/atomsim/server/app.py:2203-2204` (the mount site)
- Test: `tests/test_server_static.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `atomsim.server.app._web_dist() -> pathlib.Path`, honouring the
  `ATOMSIM_WEB_DIST` environment variable. Task 3 sets that variable in the
  image.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_server_static.py`:

```python
"""Where the built frontend came from, and what happens when it did not.

The mount is conditional, so a wrong path does not fail: it serves an API with
no application in front of it and says nothing. These tests pin both halves,
the override that a container needs and the disclosure that a silent 404 was
missing.
"""

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import _web_dist, create_app


@pytest.fixture()
def built(tmp_path):
    """A directory shaped like a real `vite build` output."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<div id=\"root\"></div>", encoding="utf-8")
    return dist


def test_the_default_is_the_source_checkout(monkeypatch):
    # An override in the ambient environment would make this pass for the
    # wrong reason, so clear it: this test is about the fallback.
    monkeypatch.delenv("ATOMSIM_WEB_DIST", raising=False)
    # parents[3] of src/atomsim/server/app.py is the repo root, and parents[1]
    # of this file is the same directory.
    expected = Path(__file__).resolve().parents[1] / "web" / "dist"
    assert _web_dist() == expected


def test_the_override_names_where_the_build_is(monkeypatch, built):
    monkeypatch.setenv("ATOMSIM_WEB_DIST", str(built))
    assert _web_dist() == built

    with TestClient(create_app()) as client:
        served = client.get("/")
    assert served.status_code == 200
    assert "id=\"root\"" in served.text


def test_a_missing_build_is_said_out_loud(monkeypatch, tmp_path, caplog):
    absent = tmp_path / "never-built"
    monkeypatch.setenv("ATOMSIM_WEB_DIST", str(absent))

    with caplog.at_level(logging.WARNING, logger="atomsim.server.app"):
        with TestClient(create_app()) as client:
            served = client.get("/")

    assert served.status_code == 404
    assert str(absent) in caplog.text


def test_a_mounted_build_is_said_out_loud(monkeypatch, built, caplog):
    monkeypatch.setenv("ATOMSIM_WEB_DIST", str(built))

    with caplog.at_level(logging.INFO, logger="atomsim.server.app"):
        create_app()

    assert str(built) in caplog.text


def test_the_api_still_answers_without_a_build(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOMSIM_WEB_DIST", str(tmp_path / "never-built"))
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["status"] == "ok"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_server_static.py -v`
Expected: FAIL with `ImportError: cannot import name '_web_dist'`

- [ ] **Step 3: Replace the constant with a function**

In `src/atomsim/server/app.py`, add `import logging` to the standard-library
imports at the top (alphabetically it sits between `dataclasses` and `math`),
and replace line 134:

```python
WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
```

with:

```python
logger = logging.getLogger(__name__)


def _web_dist() -> Path:
    """Where the built frontend lives.

    The default assumes a source checkout: `parents[3]` is the repo root, the
    directory holding both `src/` and `web/`. Installed into site-packages the
    same expression resolves into the Python library directory, where there is
    no `web/dist` and never will be, and the mount below is skipped. That
    failure is silent by construction, so the override exists to let a
    container state where it put the build rather than hope.
    """
    override = os.environ.get("ATOMSIM_WEB_DIST")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "web" / "dist"
```

- [ ] **Step 4: Make the mount disclose itself**

Replace lines 2203-2204 (now shifted):

```python
    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
```

with:

```python
    web_dist = _web_dist()
    if web_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")
        logger.info("UI mounted from %s", web_dist)
    else:
        logger.warning("no UI at %s; serving the API only", web_dist)
```

`is_dir` rather than `exists`: a file at that path would pass `exists` and then
fail inside `StaticFiles` with a less useful message.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_server_static.py -v`
Expected: PASS, 5 tests

- [ ] **Step 6: Run the full suite and the linter**

Run: `pytest -q && ruff check .`
Expected: PASS. Nothing else reads `WEB_DIST`; the only other hits are in
`docs/plans/2026-07-05-phase1-m1-walking-skeleton.md`, which is a historical
record and must not be edited.

- [ ] **Step 7: Commit**

```bash
git add src/atomsim/server/app.py tests/test_server_static.py
git commit -m "Say which directory the interface was served from"
```

---

### Task 2: Charge the client the proxy vouched for

`_client_key` reads `forwarded.split(",")[0]`, the leftmost entry. Its own
docstring says trusting a client-settable address "is the same as having no
limiter at all", and with an appending proxy that is exactly what the leftmost
entry is. A forwarding proxy appends, so a caller sending
`X-Forwarded-For: anything` arrives as `anything, <real address>`. Rotate that
string per request and every request gets a fresh full bucket.

Only the last hop was written by the proxy we chose to trust.

**Files:**
- Modify: `src/atomsim/server/app.py:181-194` (`_client_key`)
- Test: `tests/test_server_ratelimit.py` (extend; its module docstring already
  claims to pin "that the proxy header is trusted only when it has been named")

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_client_key` unchanged in signature,
  `(request, header: str | None) -> str`. Task 3 sets
  `ATOMSIM_CLIENT_IP_HEADER=fly-client-ip` in the image.

- [ ] **Step 1: Write the failing tests**

First harden the existing `limited` fixture, which is about to be used to prove
that an *unnamed* header is ignored. If the developer's environment happens to
name one, it would prove the opposite. Add one line to it:

```python
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT_PERIOD", "600")
    monkeypatch.delenv("ATOMSIM_CLIENT_IP_HEADER", raising=False)  # add this
```

Then append to `tests/test_server_ratelimit.py`:

```python
@pytest.fixture
def behind_a_proxy(monkeypatch):
    """A limiter that trusts one named forwarding header, two jobs deep."""
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT", "on")
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT_BURST", "2")
    monkeypatch.setenv("ATOMSIM_RATE_LIMIT_PERIOD", "600")
    monkeypatch.setenv("ATOMSIM_CLIENT_IP_HEADER", "x-forwarded-for")
    with TestClient(create_app()) as client:
        yield client


def test_a_named_header_separates_clients(behind_a_proxy):
    for _ in range(2):
        assert behind_a_proxy.post(
            "/api/jobs/sample", json=SAMPLE, headers={"x-forwarded-for": "1.1.1.1"}
        ).status_code == 200
    assert behind_a_proxy.post(
        "/api/jobs/sample", json=SAMPLE, headers={"x-forwarded-for": "1.1.1.1"}
    ).status_code == 429
    # A different client still has its own bucket.
    assert behind_a_proxy.post(
        "/api/jobs/sample", json=SAMPLE, headers={"x-forwarded-for": "2.2.2.2"}
    ).status_code == 200


def test_a_spoofed_prefix_does_not_buy_a_fresh_bucket(behind_a_proxy):
    """The caller writes the left of the list; the proxy appends the right.

    A client that could mint a new bucket per request by varying what it sends
    would not be rate limited at all, which is the failure this header was
    introduced to avoid rather than to cause.
    """
    for attempt in range(2):
        assert behind_a_proxy.post(
            "/api/jobs/sample",
            json=SAMPLE,
            headers={"x-forwarded-for": f"spoof-{attempt}, 3.3.3.3"},
        ).status_code == 200

    refused = behind_a_proxy.post(
        "/api/jobs/sample",
        json=SAMPLE,
        headers={"x-forwarded-for": "spoof-2, 3.3.3.3"},
    )
    assert refused.status_code == 429


def test_an_unnamed_header_is_ignored(limited):
    """`limited` names no header, so a forwarded address must not be believed."""
    for attempt in range(2):
        assert limited.post(
            "/api/jobs/sample",
            json=SAMPLE,
            headers={"x-forwarded-for": f"{attempt}.{attempt}.{attempt}.{attempt}"},
        ).status_code == 200

    refused = limited.post(
        "/api/jobs/sample", json=SAMPLE, headers={"x-forwarded-for": "9.9.9.9"}
    )
    assert refused.status_code == 429
```

- [ ] **Step 2: Run the tests to verify the spoof test fails**

Run: `pytest tests/test_server_ratelimit.py -v`
Expected: `test_a_spoofed_prefix_does_not_buy_a_fresh_bucket` FAILS with
`assert 200 == 429`, because each varied prefix currently mints a new bucket.
The other three PASS.

- [ ] **Step 3: Take the rightmost entry**

Replace the body of `_client_key` in `src/atomsim/server/app.py`:

```python
def _client_key(request, header: str | None) -> str:
    """Who to charge for this request.

    Behind a proxy every request carries the proxy's address, so without the
    header the whole internet shares one bucket and the first busy visitor
    locks out the rest. The header is opt-in by name rather than assumed,
    because trusting a forwarded address that a client can set is the same as
    having no limiter at all.

    The rightmost entry is the one to charge, and the distinction is not
    cosmetic. A forwarding proxy appends, so a caller that sends its own
    `X-Forwarded-For: someone-else` arrives as `someone-else, <real address>`.
    Charging the leftmost entry charges a string the caller typed, and varying
    it per request buys an unlimited supply of full buckets. Only the last hop
    was written by the proxy we chose to trust.

    This assumes exactly one trusted proxy in front, which is what the
    deployment has. Behind two, the rightmost entry is the inner proxy and
    every client would share its bucket: wrong in the safe direction, and worth
    re-deriving rather than inheriting if another hop is ever added.
    """
    if header:
        forwarded = request.headers.get(header)
        if forwarded:
            candidate = forwarded.split(",")[-1].strip()
            if candidate:
                return candidate
    return request.client.host if request.client else "unknown"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_server_ratelimit.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Run the full suite and the linter**

Run: `pytest -q && ruff check .`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/atomsim/server/app.py tests/test_server_ratelimit.py
git commit -m "Charge the address the proxy wrote, not the one the caller sent"
```

---

### Task 3: Containerise

`web/dist` is gitignored, so the image builds it. Two stages: Node produces the
bundle, Python runs it.

Note `web/index.html` substitutes `%VITE_SITE_URL%` into the Open Graph and
Twitter image URLs at build time. Left unset they stay root-relative, which the
file's own comment says "most scrapers resolve but none are obliged to". For a
link meant to be shared, the build passes the deployed origin in.

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `scripts/smoke_container.sh`

**Interfaces:**
- Consumes: `_web_dist()` from Task 1 via `ATOMSIM_WEB_DIST=/app/web/dist`, and
  `_client_key` from Task 2 via `ATOMSIM_CLIENT_IP_HEADER=fly-client-ip`.
- Produces: an image listening on port 8080, and
  `scripts/smoke_container.sh <image-tag>` which exits non-zero if the image is
  not a complete application. Tasks 4 and 5 both call it.

- [ ] **Step 1: Write `.dockerignore`**

Without this the build context includes `web/node_modules`, which is hundreds
of megabytes and would also overwrite the `npm ci` output inside the image.

```
.git
.github
.gitignore

**/node_modules
**/__pycache__
**/*.py[cod]
*.egg-info

web/dist
dist
build
out

.pytest_cache
.ruff_cache
.coverage
htmlcov
tests

docs
scripts
environment.yml

.claude
.superpowers
.gstack
```

`README.md` is deliberately not excluded: `pyproject.toml` declares
`readme = "README.md"` and the install fails without it.

- [ ] **Step 2: Write the `Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1

# --------------------------------------------------------------- web build --
FROM node:22-slim AS web

# Substituted into index.html's Open Graph tags. Left empty the image URLs stay
# root-relative, which is legal and worse: a scraper is not obliged to resolve
# them, so a shared link loses its preview.
ARG VITE_SITE_URL=""
ENV VITE_SITE_URL=$VITE_SITE_URL

WORKDIR /build

# Copied before the sources so a source edit does not reinstall the tree.
COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
# `npm run build` is `tsc --noEmit && vite build`, so a type error fails the
# image rather than shipping.
RUN npm run build

# ----------------------------------------------------------------- runtime --
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# numpy, scipy and matplotlib all publish manylinux wheels, so no compiler is
# needed and none is installed.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install .

COPY --from=web /build/dist ./web/dist

# One shared core runs two job workers. OpenBLAS defaulting to a thread per
# core underneath them only makes them contend.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    ATOMSIM_WEB_DIST=/app/web/dist \
    ATOMSIM_CLIENT_IP_HEADER=fly-client-ip

RUN useradd --create-home --uid 1000 atomsim \
    && mkdir -p /tmp/matplotlib \
    && chown atomsim:atomsim /tmp/matplotlib
USER atomsim

EXPOSE 8080

# Not `atomsim serve`: that binds 127.0.0.1, which is unreachable from outside
# the container, and opens a browser that does not exist here.
CMD ["uvicorn", "atomsim.server.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 3: Write the smoke test**

Create `scripts/smoke_container.sh`. This is the test for this task: a build
that succeeds proves nothing, because the failure this image is most likely to
have is serving an API with no interface in front of it.

```bash
#!/usr/bin/env bash
# Prove the image is the whole application rather than an engine that imports.
#
# Three questions, in order of what actually breaks: does it answer at all, is
# the interface really mounted (the silent failure Task 1 exists for), and does
# a job survive the round trip that spans three requests and a thread pool.
set -euo pipefail

IMAGE="${1:-atomsim:local}"
PORT="${PORT:-8080}"
NAME="atomsim-smoke-$$"
BASE="http://127.0.0.1:${PORT}"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run -d --name "$NAME" -p "${PORT}:8080" "$IMAGE" >/dev/null

echo "--- waiting for the engine to import"
for _ in $(seq 1 90); do
  if curl -fsS "${BASE}/api/health" >/dev/null 2>&1; then break; fi
  sleep 1
done

echo "--- health"
curl -fsS "${BASE}/api/health"
echo

echo "--- the interface is mounted"
curl -fsS "${BASE}/" | grep -q 'id="root"'

echo "--- the startup said where it mounted from"
docker logs "$NAME" 2>&1 | grep -q "UI mounted from /app/web/dist"

echo "--- a job runs to completion"
JOB=$(curl -fsS -X POST "${BASE}/api/jobs/sample" \
        -H 'content-type: application/json' \
        -d '{"n":2,"l":1,"m":0,"count":5000}' \
      | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')
test -n "$JOB"

for _ in $(seq 1 60); do
  STATUS=$(curl -fsS "${BASE}/api/jobs/${JOB}" \
           | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')
  case "$STATUS" in
    done) break ;;
    error) echo "job failed"; docker logs "$NAME"; exit 1 ;;
  esac
  sleep 1
done
test "$STATUS" = "done"

curl -fsS "${BASE}/api/jobs/${JOB}/meta" | grep -q '"kind":"sample"'

echo "--- ok"
```

- [ ] **Step 4: Build the image**

Run from the repo root:

```bash
docker build --build-arg VITE_SITE_URL=https://atomsim.fly.dev -t atomsim:local .
```

Expected: a successful build. The Node stage runs `tsc --noEmit` then `vite
build`; the Python stage installs wheels only. Expect several minutes the first
time and a large image, roughly 700MB to 1GB, dominated by scipy and
matplotlib.

- [ ] **Step 5: Run the smoke test**

```bash
chmod +x scripts/smoke_container.sh
./scripts/smoke_container.sh atomsim:local
```

Expected: every section prints, ending with `--- ok`.

If `--- the interface is mounted` fails, `ATOMSIM_WEB_DIST` and the
`COPY --from=web` destination disagree; check `docker logs` for the warning
Task 1 added, which names the path it looked at.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore scripts/smoke_container.sh
git commit -m "Build the interface and the engine into one image"
```

---

### Task 4: Build the image in CI

CI is Windows-only today and would never notice a broken Dockerfile.

**Files:**
- Modify: `.github/workflows/ci.yml` (add a job)

**Interfaces:**
- Consumes: `scripts/smoke_container.sh` from Task 3.
- Produces: a `container` job that Task 6's deploy job depends on.

- [ ] **Step 1: Add the job**

Append to `.github/workflows/ci.yml`, at the same indentation as the existing
`python` and `web` jobs:

```yaml
  container:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4
      - name: Build
        run: docker build --build-arg VITE_SITE_URL=https://atomsim.fly.dev -t atomsim:ci .
      - name: Smoke test
        run: bash scripts/smoke_container.sh atomsim:ci
```

`ubuntu-latest`, not `windows-latest`: the other two jobs are on Windows
because that is where this project is developed, but Linux containers do not
build on a Windows runner.

- [ ] **Step 2: Verify the workflow parses**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "Build the container on Linux so a broken image fails the build"
```

- [ ] **Step 4: Confirm the job runs green**

Push and watch the run. Expected: `container` passes alongside `python` and
`web`. Do not continue to Task 5 with a red `container` job.

---

### Task 5: First deploy, and measure it

Steps here touch a real account and a real bill. The `fly` commands that create
or pay for things are for the repository owner to run, not for an agent to run
unattended.

**Files:**
- Create: `fly.toml`
- Create: `docs/DEPLOY.md`

**Interfaces:**
- Consumes: the image from Task 3.
- Produces: a running app at `https://atomsim.fly.dev`, and measured wake
  timings recorded in `docs/DEPLOY.md` which decide whether the waiting page
  described in the spec gets built at all.

- [ ] **Step 1: Install flyctl and sign in**

`flyctl` is not on this machine. In PowerShell:

```powershell
iwr https://fly.io/install.ps1 -useb | iex
```

Then, in the interactive terminal (prefix with `!` in Claude Code so the output
lands in the session):

```
fly auth signup   # or: fly auth login
```

A credit card is required. Fly has no free tier; the machine below is billed
per second it runs.

- [ ] **Step 2: Set a spending limit before anything is running**

In the Fly dashboard, set a hard spending limit and a budget alert on the
organisation. This is the only control that bounds a metered bill against
traffic nobody predicted, and it belongs on the account before the first
machine exists rather than after the first surprise.

- [ ] **Step 3: Write `fly.toml`**

Do not run `fly launch`: it writes its own config, and it provisions **two**
machines for high availability, which splits `JobStore` and `TokenBucket` and
breaks the application in exactly the way `ratelimit.py` warns about.

```toml
# atomsim: one process, suspended when nobody is looking.
#
# One machine is a requirement, not a preference. JobStore and TokenBucket are
# per-process in-memory state, so a second machine would hand clients job ids
# that do not exist on the instance they reach next, and would give every
# client a second full rate-limit bucket. Deploys pass --ha=false for the same
# reason.

app = "atomsim"
primary_region = "bom"

[build]
  [build.args]
    # Substituted into index.html's Open Graph tags at build time. Update this
    # when a custom domain replaces the fly.dev name, and rebuild: the value is
    # baked into the bundle, not read at runtime.
    VITE_SITE_URL = "https://atomsim.fly.dev"

[http_service]
  internal_port = 8080
  force_https = true
  auto_start_machines = true
  min_machines_running = 0
  # "suspend" snapshots the whole VM, so a resume skips re-importing numpy,
  # scipy and matplotlib. cli.py measures that import at 5.4 s on a 14-core
  # laptop, which is the cost a plain "stop" would pay on every visit.
  auto_stop_machines = "suspend"

# Deliberately no [[http_service.http_checks]]. A periodic health check is a
# request, and a machine receiving requests never goes idle, which would defeat
# the suspension this file is built around.

[[vm]]
  size = "shared-cpu-1x"
  memory = "1gb"
```

`memory` is 1GB because `JobStore` retains 16 finished jobs and
`SampleRequest.count` is capped at 1,000,000 points, roughly 20MB each. Worst
case is about 320MB of held results on top of roughly 250MB for Python with the
scientific stack loaded. 512MB can be OOM-killed by an ordinary click-storm.

Confirm `bom` is a real region before deploying: `fly platform regions`. It is
the closest to the author; pick another if the audience is elsewhere.

- [ ] **Step 4: Create the app and deploy**

```
fly apps create atomsim
fly deploy --remote-only --ha=false
```

If the name is taken, choose another and update `app` in `fly.toml` and
`VITE_SITE_URL` in both `fly.toml` and `.github/workflows/ci.yml`.

- [ ] **Step 5: Verify exactly one machine exists**

```
fly status
```

Expected: exactly one machine. If there are two, `fly scale count 1`
immediately. Two machines is a correctness bug, not a cost inefficiency.

- [ ] **Step 6: Verify the deployed app end to end**

```
./scripts/smoke_container.sh
```

will not reach a remote host, so check by hand:

```bash
curl -fsS https://atomsim.fly.dev/api/health
curl -fsS https://atomsim.fly.dev/ | grep -q 'id="root"' && echo "ui ok"
fly logs | grep "UI mounted from"
```

Then open `https://atomsim.fly.dev` in a browser and confirm a cloud renders, a
plane renders, and provenance badges open. The websocket is the part no curl
covers.

- [ ] **Step 7: Verify the rate limiter charges real addresses**

This is the check the spec marks as must-verify rather than assume. Confirm
Fly's proxy sets the header, that it is single-valued, and that a client cannot
inject their own:

```bash
fly logs &
curl -s -H 'Fly-Client-IP: 1.2.3.4' https://atomsim.fly.dev/api/health >/dev/null
```

Add a temporary debug echo of the header if needed, or inspect via
`fly ssh console`. The requirement: the value the app sees is the caller's real
address, not `1.2.3.4`.

If `Fly-Client-IP` turns out not to arrive, switch
`ATOMSIM_CLIENT_IP_HEADER` to `x-forwarded-for` in the `Dockerfile`. Task 2
made the rightmost entry the charged one, so that fallback is already correct
for a single proxy hop.

- [ ] **Step 8: Measure the three wake times**

The spec defers the waiting-page decision to these numbers. Record them, not
adjectives.

```bash
# warm baseline
curl -o /dev/null -s -w 'warm: %{time_total}s\n' https://atomsim.fly.dev/api/health

# resume from suspend: wait for the machine to idle out, confirm with
# `fly status` showing it suspended, then
curl -o /dev/null -s -w 'resume: %{time_total}s\n' https://atomsim.fly.dev/api/health

# true cold start
fly machine stop <machine-id>
curl -o /dev/null -s -w 'cold: %{time_total}s\n' https://atomsim.fly.dev/api/health
```

- [ ] **Step 9: Verify a long job survives the idle timer**

The spec flags this risk and it is the one failure that would corrupt results
rather than merely annoy. `_dispatch` returns the job id immediately and the
solve continues on a background thread, which the Fly proxy cannot see. If the
proxy counts the machine as idle while a solve is running, it could suspend
mid-computation.

In practice the client opens `/ws/jobs/{id}` immediately after the POST, so a
connection is in flight for the whole job. Confirm that rather than assume it.
Open the deployed app in a browser, switch to an atom whose Hartree-Fock solve
is slow (argon), and confirm the result arrives and the badge reports the
expected tier. Then check the machine did not restart underneath it:

```bash
fly logs | grep -i "suspend\|resume\|starting"
```

Expected: no suspend between the job starting and finishing. If one appears,
record it in `docs/DEPLOY.md` as a known hazard and open a follow-up rather
than patching around it here.

- [ ] **Step 10: Write `docs/DEPLOY.md`**

Fill in the measured values; do not leave the placeholders:

```markdown
# Deploying atomsim

Public URL: https://atomsim.fly.dev
Fly app: `atomsim`, region `bom`, one `shared-cpu-1x` machine with 1GB.

## One machine, always

`JobStore` and `TokenBucket` are per-process in-memory state. A second machine
hands clients job ids that do not exist on the instance they reach next, and
gives every client a second full rate-limit bucket. Deploys pass `--ha=false`,
and `fly status` showing two machines is a bug to fix immediately.

## What the environment variables prevent

| Variable | Value | Without it |
|---|---|---|
| `ATOMSIM_WEB_DIST` | `/app/web/dist` | The interface silently does not mount and `/` returns 404 |
| `ATOMSIM_CLIENT_IP_HEADER` | `fly-client-ip` | Every visitor shares one rate-limit bucket |
| `OMP_NUM_THREADS` | `1` | OpenBLAS oversubscribes the shared core |
| `MPLCONFIGDIR` | `/tmp/matplotlib` | matplotlib fails to cache against a read-only home |

## Measured wake times

Measured <DATE> against `shared-cpu-1x` / 1GB in `bom`:

| Case | Time to `/api/health` |
|---|---|
| Warm | <WARM> |
| Resume from suspend | <RESUME> |
| True cold start | <COLD> |

Conclusion: <state plainly whether a waiting page is needed, and why>.

## Forwarded-address verification

<record what the app actually saw when a caller sent their own Fly-Client-IP>

## Deploying

Automatic on push to `main` once `python`, `web` and `container` pass. By hand:

    fly deploy --remote-only --ha=false
```

State plainly whether the measured resume time means the waiting page is
needed. If resume is under about a second, say so and say the page is not being
built. If it is not, open a follow-up plan rather than improvising the page
here.

- [ ] **Step 11: Commit**

```bash
git add fly.toml docs/DEPLOY.md
git commit -m "Run one suspended machine and record what waking it costs"
```

---

### Task 6: Deploy on push to main

**Files:**
- Modify: `.github/workflows/ci.yml` (add a deploy job)

**Interfaces:**
- Consumes: the `python`, `web` and `container` jobs.
- Produces: nothing later tasks depend on. This is the last task.

- [ ] **Step 1: Add the Fly token as a repository secret**

```
fly tokens create deploy -x 999999h
```

Copy the output, including the `FlyV1 ` prefix, into GitHub under Settings,
Secrets and variables, Actions, as `FLY_API_TOKEN`.

- [ ] **Step 2: Resolve the action SHA**

The repository pins actions by commit, not by tag. Get the current one:

```bash
gh api repos/superfly/flyctl-actions/commits/master --jq .sha
```

Use that value in the next step in place of `<SHA>`, keeping the `# master`
comment so a future reader can tell what was pinned.

- [ ] **Step 3: Add the deploy job**

Append to `.github/workflows/ci.yml`:

```yaml
  deploy:
    runs-on: ubuntu-latest
    needs: [python, web, container]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    concurrency:
      group: deploy-atomsim
      cancel-in-progress: false
    steps:
      - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4
      - uses: superfly/flyctl-actions/setup-flyctl@<SHA> # master
      # --ha=false: a second machine would split JobStore and TokenBucket.
      - run: flyctl deploy --remote-only --ha=false
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
```

`needs` is what stops a red test suite reaching a public URL. `concurrency`
with `cancel-in-progress: false` stops two pushes deploying over each other.
The `if` keeps pull requests from deploying.

- [ ] **Step 4: Verify the workflow parses**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit and confirm a real deploy**

```bash
git add .github/workflows/ci.yml
git commit -m "Deploy from main once the suite has passed"
git push
```

Watch the run. Expected: `python`, `web` and `container` pass, then `deploy`
runs and the site serves the new build. Confirm with `fly status` that there is
still exactly one machine.

---

## Deferred, deliberately

- **The waiting page.** Scoped in the spec, gated on Task 5 Step 8. Build it
  only if the measured resume time says there is a wait worth explaining, and
  if built, without a progress bar: boot progress is not measurable from
  outside the machine, and an animated percentage would be inventing a number.
- **Custom domain.** `fly certs add <domain>` plus one DNS record, then update
  `VITE_SITE_URL` in `fly.toml` and `.github/workflows/ci.yml` and redeploy so
  the Open Graph tags follow. No code change.
- **Unmetered result re-download.** `/api/jobs/{id}/data` is not rate limited;
  one token buys repeated 20MB fetches until eviction. Bounded by the spending
  limit from Task 5 Step 2 rather than by new code.
