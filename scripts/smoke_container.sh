#!/usr/bin/env bash
# Prove the image is the whole application rather than an engine that imports.
#
# Three questions, in order of what actually breaks: does it answer at all, is
# the interface really mounted (the silent failure `_web_dist` exists for), and
# does a job survive the round trip that spans three requests and a thread
# pool. A build that merely succeeds proves none of them.
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

STATUS=""
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
