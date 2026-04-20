import asyncio
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json

# Ensure project root is in path
sys.path.insert(0, ".")

from eval.benchmark_tasks import BenchmarkSuite

async def main():
    print("=== STARTING MEMHUB VS BASELINE COMPARISON ===")
    
    # 1. Run Baseline (Policies OFF)
    print("\n[1/3] Running Baseline (No MemHub Policies)...")
    baseline_suite = BenchmarkSuite(output_dir="eval/results/baseline", use_policies=False)
    await baseline_suite.run_all()
    
    # 2. Run MemHub (Policies ON)
    print("\n[2/3] Running MemHub (All Policies ON)...")
    memhub_suite = BenchmarkSuite(output_dir="eval/results/memhub", use_policies=True)
    await memhub_suite.run_all()
    
    # 3. Generate Comparison Charts
    print("\n[3/3] Generating Comparison Charts...")
    generate_comparison_charts()
    
    print("\n=== COMPARISON COMPLETE ===")
    print("Charts saved to: eval/charts/comparison_*.png")

def generate_comparison_charts():
    baseline_dir = Path("eval/results/baseline")
    memhub_dir = Path("eval/results/memhub")
    out_dir = Path("eval/charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # --- Chart 1: Latency Comparison (Single Agent Recall) ---
    try:
        b_recall = json.loads((baseline_dir / "single_agent_recall.json").read_text())
        m_recall = json.loads((memhub_dir / "single_agent_recall.json").read_text())
        
        latency_data = {
            "System": ["Baseline", "MemHub"],
            "Mean Latency (ms)": [
                b_recall["summary"]["latency_mean"]["retrieve"],
                m_recall["summary"]["latency_mean"]["retrieve"]
            ]
        }
        df_lat = pd.DataFrame(latency_data)
        plt.figure(figsize=(8, 5))
        sns.barplot(x="System", y="Mean Latency (ms)", data=df_lat, palette=["#95a5a6", "#3498db"])
        plt.title("Retrieval Latency: Baseline vs MemHub")
        plt.savefig(out_dir / "comparison_latency.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting latency: {e}")

    # --- Chart 2: Throughput (Ops/sec) ---
    try:
        throughput_data = {
            "System": ["Baseline", "MemHub"],
            "Ops/Second": [
                b_recall["summary"]["ops_per_second"],
                m_recall["summary"]["ops_per_second"]
            ]
        }
        df_thru = pd.DataFrame(throughput_data)
        plt.figure(figsize=(8, 5))
        sns.barplot(x="System", y="Ops/Second", data=df_thru, palette=["#95a5a6", "#2ecc71"])
        plt.title("Throughput: Baseline vs MemHub")
        plt.savefig(out_dir / "comparison_throughput.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting throughput: {e}")

    # --- Chart 3: Token Management (Long Session) ---
    try:
        # We compare Baseline's long_session_no_policy vs MemHub's eviction_lru
        b_long = json.loads((baseline_dir / "long_session_no_policy.json").read_text())
        m_long = json.loads((memhub_dir / "eviction_lru.json").read_text())
        
        token_data = {
            "System": ["Baseline (No Eviction)", "MemHub (With Eviction)"],
            "Tokens After session": [
                b_long["summary"]["tokens_before_demotion"], # In baseline, tokens_before is just the total
                m_long["summary"]["tokens_after_demotion"]
            ]
        }
        df_tokens = pd.DataFrame(token_data)
        plt.figure(figsize=(8, 5))
        sns.barplot(x="System", y="Tokens After session", data=df_tokens, palette=["#e74c3c", "#2ecc71"])
        plt.axhline(y=2000, color='r', linestyle='--', label='Context Budget (2000)')
        plt.title("Memory Budget Control (80 Memories)")
        plt.legend()
        plt.savefig(out_dir / "comparison_tokens.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting tokens: {e}")

if __name__ == "__main__":
    asyncio.run(main())
