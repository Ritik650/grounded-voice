#!/usr/bin/env bash
# Restart the dev server on a known port, killing whatever holds it first.
# Windows will happily leave an orphaned uvicorn bound to 8000, and the new process
# then fails to bind while every client silently keeps talking to the stale build.
set -u
PORT="${1:-8000}"
LOG="${2:-/tmp/uvicorn.log}"
cd "$(dirname "$0")/.."

for pid in $(netstat -ano | grep ":${PORT} " | grep LISTENING | awk '{print $5}' | sort -u); do
  taskkill //F //PID "$pid" > /dev/null 2>&1 && echo "killed stale server (pid ${pid})"
done

rm -f "$LOG"
.venv/Scripts/python.exe -m uvicorn app.server:app --host 127.0.0.1 --port "$PORT" \
  --log-level warning > "$LOG" 2>&1 &

until curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; do
  if grep -q "error while attempting to bind" "$LOG" 2>/dev/null; then
    echo "FAILED to bind port ${PORT}"; cat "$LOG"; exit 1
  fi
  sleep 2
done
echo "server ready on ${PORT}"
