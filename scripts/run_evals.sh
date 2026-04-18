#!/usr/bin/env bash
# scripts/run_evals.sh — Trigger the MemHub benchmark suite and generate charts
#
# Usage:
#   bash scripts/run_evals.sh
#   RESULTS_DIR=eval/results OUT_DIR=eval/charts bash scripts/run_evals.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="${RESULTS_DIR:-eval/results}"
OUT_DIR="${OUT_DIR:-eval/charts}"

# ── Optional virtualenv ───────────────────────────────────────────────────────
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
    echo "[run_evals] virtualenv activated."
fi

mkdir -p "$RESULTS_DIR" "$OUT_DIR"

echo "[run_evals] Running MemHub benchmark suite…"
python - <<'PYEOF'
import asyncio
import sys
sys.path.insert(0, ".")
from eval.benchmark_tasks import BenchmarkSuite
import json, pathlib

async def main():
    suite = BenchmarkSuite(output_dir="'"$RESULTS_DIR"'")
    results = await suite.run_all()
    for r in results:
        out = pathlib.Path("'"$RESULTS_DIR"'") / f"{r.task_name}.json"
        data = {
            "run_id":  r.task_name,
            "passed":  r.passed,
            "notes":   r.notes,
            "error":   r.error,
            "summary": r.summary.__dict__,
        }
        out.write_text(json.dumps(data, indent=2))
        print(f"  {'✓' if r.passed else '✗'} {r.task_name} → {out}")

asyncio.run(main())
PYEOF

echo "[run_evals] Generating charts…"
python -m eval.visualize --results-dir "$RESULTS_DIR" --out-dir "$OUT_DIR"

echo "[run_evals] Done. Charts saved to: $OUT_DIR"
ls -1 "$OUT_DIR"/*.png 2>/dev/null || echo "  (no charts generated)"
