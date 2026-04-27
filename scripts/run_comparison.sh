#!/bin/bash
# scripts/run_comparison.sh — Run MemHub Comparative Analysis Suite

echo "[run_comparison] Running side-by-side analysis (MemHub vs Baseline)..."

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Check for virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set PYTHONPATH to include the current directory
export PYTHONPATH=$PYTHONPATH:.

# Run the comparison script
python eval/run_comparison.py

echo "[run_comparison] Done. Results are in eval/results/ and charts in eval/charts/"
