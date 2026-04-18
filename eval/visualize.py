"""
eval/visualize.py  — MemHub Benchmark Chart Generator
======================================================

Reads JSON result files produced by :class:`~eval.benchmark_tasks.BenchmarkSuite`
and generates publication-quality charts for:

  1. Latency comparison   — p50 / p90 / p99 bar chart per eviction strategy
  2. Throughput comparison— ops/second bar chart across all tasks
  3. Token compression    — before/after token counts with reduction % label
  4. Tier hit rates       — Tier-1 vs Tier-2 stacked bar chart per task
  5. Policy sweep cost    — latency cost of LRU vs FIFO vs LFU sweeps

All charts are saved as high-DPI PNG files and can optionally be shown
interactively.

Usage
─────
  # Generate all charts from a results directory:
  python -m eval.visualize --results-dir eval/results --out-dir eval/charts

  # In a notebook:
  from eval.visualize import Visualizer
  viz = Visualizer(results_dir="eval/results")
  viz.plot_all(show=True)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (safe for scripts)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed — charts will not be generated.")


# ── Style constants ────────────────────────────────────────────────────────────

PALETTE = {
    "tier1":    "#4C72B0",
    "tier2":    "#DD8452",
    "lru":      "#55A868",
    "fifo":     "#C44E52",
    "lfu":      "#8172B2",
    "p50":      "#4C72B0",
    "p90":      "#DD8452",
    "p99":      "#C44E52",
    "before":   "#E8A838",
    "after":    "#55A868",
    "bg":       "#F8F9FA",
    "grid":     "#DDDDDD",
}

DPI = 150
FIGURE_SIZE = (10, 5)


def _apply_style(ax: "plt.Axes", title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent MemHub chart styling to an Axes object."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_results(results_dir: Path) -> Dict[str, dict]:
    """
    Load all JSON result files from *results_dir*.

    Returns:
        Mapping of task_name → result payload dict.
    """
    payloads: Dict[str, dict] = {}
    for fp in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text())
            name = data.get("run_id") or fp.stem
            payloads[name] = data
        except Exception as exc:
            logger.warning("Could not load '%s': %s", fp, exc)
    return payloads


# ── Chart functions ───────────────────────────────────────────────────────────


def plot_latency_percentiles(
    results: Dict[str, dict],
    out_dir: Path,
    show: bool = False,
) -> Optional[Path]:
    """
    Grouped bar chart of p50 / p90 / p99 retrieve latency per task.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    tasks, p50s, p90s, p99s = [], [], [], []

    for name, data in results.items():
        summary = data.get("summary", {})
        retrieve_p50 = summary.get("latency_p50", {}).get("retrieve", 0.0)
        retrieve_p90 = summary.get("latency_p90", {}).get("retrieve", 0.0)
        retrieve_p99 = summary.get("latency_p99", {}).get("retrieve", 0.0)
        if retrieve_p50 or retrieve_p90 or retrieve_p99:
            tasks.append(name.replace("_", "\n"))
            p50s.append(retrieve_p50)
            p90s.append(retrieve_p90)
            p99s.append(retrieve_p99)

    if not tasks:
        logger.info("No retrieve latency data available for percentile chart.")
        return None

    import numpy as np
    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    ax.bar(x - width, p50s, width, label="p50", color=PALETTE["p50"], zorder=3)
    ax.bar(x,         p90s, width, label="p90", color=PALETTE["p90"], zorder=3)
    ax.bar(x + width, p99s, width, label="p99", color=PALETTE["p99"], zorder=3)

    _apply_style(ax, "Retrieve Latency Percentiles by Task", "Task", "Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=8)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = out_dir / "latency_percentiles.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    if show:
        plt.show()
    logger.info("Saved: %s", out)
    return out


def plot_throughput(
    results: Dict[str, dict],
    out_dir: Path,
    show: bool = False,
) -> Optional[Path]:
    """
    Horizontal bar chart of ops/second per task.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    tasks = []
    ops   = []
    for name, data in results.items():
        summary = data.get("summary", {})
        ops_per_s = summary.get("ops_per_second", 0.0)
        tasks.append(name)
        ops.append(ops_per_s)

    if not tasks:
        return None

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    colours = [PALETTE["lru"] if "lru" in t
               else PALETTE["fifo"] if "fifo" in t
               else PALETTE["lfu"] if "lfu" in t
               else PALETTE["tier1"]
               for t in tasks]
    bars = ax.barh(tasks, ops, color=colours, zorder=3)

    # Annotate bar ends
    for bar, val in zip(bars, ops):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", ha="left", fontsize=8,
        )

    _apply_style(ax, "Throughput: Operations per Second", "ops/sec", "Task")
    ax.invert_yaxis()
    fig.tight_layout()

    out = out_dir / "throughput.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    if show:
        plt.show()
    logger.info("Saved: %s", out)
    return out


def plot_token_compression(
    results: Dict[str, dict],
    out_dir: Path,
    show: bool = False,
) -> Optional[Path]:
    """
    Paired before/after bar chart showing token reduction from DemotionPolicy.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    import numpy as np

    tasks, befores, afters, pcts = [], [], [], []
    for name, data in results.items():
        summary = data.get("summary", {})
        before = summary.get("tokens_before_demotion", 0)
        after  = summary.get("tokens_after_demotion",  0)
        pct    = summary.get("compression_pct", 0.0)
        if before > 0:
            tasks.append(name.replace("_", "\n"))
            befores.append(before)
            afters.append(after)
            pcts.append(pct)

    if not tasks:
        logger.info("No demotion data available for compression chart.")
        return None

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    ax.bar(x - width / 2, befores, width, label="Before",
           color=PALETTE["before"], zorder=3)
    ax.bar(x + width / 2, afters,  width, label="After",
           color=PALETTE["after"], zorder=3)

    # Annotate compression percentage
    for xi, pct in zip(x, pcts):
        ax.text(xi, max(befores) * 1.02,
                f"↓{pct:.1f}%", ha="center", fontsize=9, color="#333333",
                fontweight="bold")

    _apply_style(ax, "Token Compression via DemotionPolicy", "Task", "Tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=8)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = out_dir / "token_compression.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    if show:
        plt.show()
    logger.info("Saved: %s", out)
    return out


def plot_tier_hit_rates(
    results: Dict[str, dict],
    out_dir: Path,
    show: bool = False,
) -> Optional[Path]:
    """
    Stacked bar chart of Tier-1 vs Tier-2 hit counts per task.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    import numpy as np

    tasks, t1_hits, t2_hits = [], [], []
    for name, data in results.items():
        summary = data.get("summary", {})
        t1 = summary.get("tier1_hits", 0)
        t2 = summary.get("tier2_hits", 0)
        if t1 + t2 > 0:
            tasks.append(name.replace("_", "\n"))
            t1_hits.append(t1)
            t2_hits.append(t2)

    if not tasks:
        return None

    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    ax.bar(x, t1_hits, label="Tier 1 (SQLite)", color=PALETTE["tier1"], zorder=3)
    ax.bar(x, t2_hits, bottom=t1_hits, label="Tier 2 (ChromaDB)",
           color=PALETTE["tier2"], zorder=3)

    _apply_style(ax, "Memory Tier Hit Distribution", "Task", "Retrieval Hits")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=8)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = out_dir / "tier_hit_rates.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    if show:
        plt.show()
    logger.info("Saved: %s", out)
    return out


def plot_policy_strategy_comparison(
    results: Dict[str, dict],
    out_dir: Path,
    show: bool = False,
) -> Optional[Path]:
    """
    Bar chart comparing policy sweep latency (ms) for LRU, FIFO, and LFU.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    strategy_latency: Dict[str, float] = {}
    for name, data in results.items():
        for strategy in ("lru", "fifo", "lfu"):
            if strategy in name.lower():
                summary = data.get("summary", {})
                lat = summary.get("latency_mean", {}).get("policy", 0.0)
                strategy_latency[strategy.upper()] = lat

    if not strategy_latency:
        logger.info("No policy sweep data for strategy comparison chart.")
        return None

    fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
    labels = list(strategy_latency.keys())
    values = list(strategy_latency.values())
    colours = [PALETTE.get(k.lower(), PALETTE["tier1"]) for k in labels]

    bars = ax.bar(labels, values, color=colours, zorder=3, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f} ms", ha="center", va="bottom", fontsize=9,
        )

    _apply_style(ax, "Eviction Policy Sweep Latency: LRU vs FIFO vs LFU",
                 "Eviction Strategy", "Mean Sweep Latency (ms)")
    fig.tight_layout()

    out = out_dir / "policy_comparison.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    if show:
        plt.show()
    logger.info("Saved: %s", out)
    return out


# ── Visualizer facade ─────────────────────────────────────────────────────────


class Visualizer:
    """
    Convenience wrapper that loads results from a directory and calls all
    chart generators.

    Args:
        results_dir: Directory containing ``*.json`` result files.
        out_dir:     Directory to write chart PNG files to.
    """

    def __init__(
        self,
        results_dir: str = "eval/results",
        out_dir: str = "eval/charts",
    ) -> None:
        self.results_dir = Path(results_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def plot_all(self, show: bool = False) -> List[Optional[Path]]:
        """
        Generate all charts and return the list of output file paths.

        Args:
            show: If True, display charts interactively (requires a display).
        """
        results = _load_results(self.results_dir)
        if not results:
            logger.warning(
                "No result files found in '%s'. Run BenchmarkSuite first.",
                self.results_dir,
            )
            return []

        logger.info(
            "Generating charts from %d result file(s) in '%s'…",
            len(results),
            self.results_dir,
        )

        outputs = [
            plot_latency_percentiles(results, self.out_dir, show),
            plot_throughput(results, self.out_dir, show),
            plot_token_compression(results, self.out_dir, show),
            plot_tier_hit_rates(results, self.out_dir, show),
            plot_policy_strategy_comparison(results, self.out_dir, show),
        ]

        generated = [p for p in outputs if p is not None]
        logger.info(
            "Chart generation complete: %d chart(s) saved to '%s'.",
            len(generated),
            self.out_dir,
        )
        return outputs


# ── CLI entry point ───────────────────────────────────────────────────────────


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MemHub benchmark charts from JSON result files."
    )
    parser.add_argument(
        "--results-dir",
        default="eval/results",
        help="Directory containing *.json result files (default: eval/results)",
    )
    parser.add_argument(
        "--out-dir",
        default="eval/charts",
        help="Output directory for PNG charts (default: eval/charts)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display charts interactively (requires a display).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    viz = Visualizer(results_dir=args.results_dir, out_dir=args.out_dir)
    viz.plot_all(show=args.show)


if __name__ == "__main__":
    _cli()
