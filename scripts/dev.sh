#!/usr/bin/env bash
# Start the llm-geometry dev stack: FastAPI backend on :8000 + Vite frontend on :5173.
#
#   sh scripts/dev.sh          # start both (backgrounded), wait until healthy, tail nothing
#   sh scripts/dev.sh stop     # stop whatever this script started (pid files)
#
# Guards against the stale-server trap from the 2026-06-15 session notes: a previously
# leaked uvicorn bound to :8000 shadows the fresh one, so we refuse to start if the
# port is already taken and say who owns it.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$ROOT/.devservers"
mkdir -p "$RUN_DIR"

stop() {
  for f in "$RUN_DIR"/*.pid; do
    [ -e "$f" ] || continue
    pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      echo "stopped $(basename "$f" .pid) (pid $pid)"
    fi
    rm -f "$f"
  done
}

if [ "${1:-}" = "stop" ]; then
  stop
  exit 0
fi

for port in 8000 5173; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "ERROR: port $port is already in use:" >&2
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >&2
    echo "Run 'sh scripts/dev.sh stop' (or kill the stale process) first." >&2
    exit 1
  fi
done

cd "$ROOT/code/backend"
# shellcheck disable=SC1091
. .venv/bin/activate

# The Lexicon Lab trains in the BROWSER in both modes, so it reads its corpus from the
# build-time export rather than from the API. Without this, `npm run dev` alone leaves the
# tab correctly refusing to invent a corpus — which is right, but is not a working dev
# stack. Generate it once if it is missing; the real export overwrites it in CI and in the
# Pages build. (--only lex keeps this to ~1 s instead of re-exporting arch's weight tiles.)
LEX_CORPUS="$ROOT/code/frontend/public/static-data/lex/corpus.json"
if [ ! -f "$LEX_CORPUS" ]; then
  echo -n "exporting the Lexicon Lab corpus (first run only) "
  if (cd "$ROOT" && python scripts/export_static_assets.py --quick --only lex \
        >"$RUN_DIR/lex-export.log" 2>&1); then
    echo "ok"
  else
    echo "FAILED — see $RUN_DIR/lex-export.log" >&2
    echo "The Lexicon tab will say its corpus is missing rather than fabricate one." >&2
  fi
fi

python -m uvicorn llm_geometry.api.app:app --port 8000 >"$RUN_DIR/backend.log" 2>&1 &
echo $! >"$RUN_DIR/backend.pid"

cd "$ROOT/code/frontend"
npm run dev >"$RUN_DIR/frontend.log" 2>&1 &
echo $! >"$RUN_DIR/frontend.pid"

echo -n "waiting for backend :8000 "
for _ in $(seq 1 60); do
  if curl -sf http://localhost:8000/api/health >/dev/null 2>&1; then break; fi
  echo -n "."
  sleep 1
done
curl -sf http://localhost:8000/api/health >/dev/null || { echo " backend failed; see $RUN_DIR/backend.log" >&2; exit 1; }
echo " ok"

echo -n "waiting for frontend :5173 "
for _ in $(seq 1 30); do
  if curl -sf http://localhost:5173 >/dev/null 2>&1; then break; fi
  echo -n "."
  sleep 1
done
curl -sf http://localhost:5173 >/dev/null || { echo " frontend failed; see $RUN_DIR/frontend.log" >&2; exit 1; }
echo " ok"

echo "dev stack up: http://localhost:5173 (frontend) -> http://localhost:8000 (api)"
echo "logs: $RUN_DIR/{backend,frontend}.log · stop with: sh scripts/dev.sh stop"
