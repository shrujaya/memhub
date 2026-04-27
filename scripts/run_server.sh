#!/usr/bin/env bash
# scripts/run_server.sh — Start the MemHub FastAPI server
#
# Usage:
#   bash scripts/run_server.sh          # development (auto-reload)
#   PORT=9000 bash scripts/run_server.sh
#   MEMHUB_DISABLE_AUTH=1 bash scripts/run_server.sh  # no ACL (testing)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# ── Optional: activate virtualenv if present ──────────────────────────────────
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
    echo "[run_server] virtualenv activated."
fi

echo "[run_server] Starting MemHub on http://${HOST}:${PORT} ..."
echo "[run_server] Docs: http://localhost:${PORT}/docs"

exec uvicorn api.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --reload
