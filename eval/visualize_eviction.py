"""
eval/visualize_eviction.py  — Chart generator for the eviction benchmark
=======================================================================

Reads `eval/results/eviction/eviction_differential.json` and produces:

  1. retention_matrix.png       — heatmap of retention % per strategy × category
  2. differential_retention.png — grouped bars: per-strategy retention across categories
  3. sweep_latency_scaling.png  — sweep-latency mean & p90 vs workload size
  4. budget_adherence.png       — tokens_before vs tokens_after vs budget

Run:
    PYTHONPATH=. python eval/visualize_eviction.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viz_eviction")

INPUT_JSON = Path("eval/results/eviction/eviction_differential.json")
OUTPUT_DIR = Path("eval/charts/eviction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_ORDER = ["lru", "fifo", "lfu"]
CATEGORY_ORDER = ["A_cold", "B_recent_hit", "C_frequent_hit", "D_fresh"]
CATEGORY_LABELS = {
    "A_cold":          "A: Cold\n(old + unused)",
    "B_recent_hit":    "B: Recent-hit\n(old + new access)",
    "C_frequent_hit":  "C: Frequent-hit\n(mid + high count)",
    "D_fresh":         "D: Fresh\n(new + low count)",
}

PALETTE = {
    "lru":  "#55A868",
    "fifo": "#C44E52",
    "lfu":  "#8172B2",
    "budget": "#E8A838",
    "before": "#BBBBBB",
    "after":  "#4C72B0",
}

BG = "#F8F9FA"
GRID = "#DDDDDD"
DPI = 150


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def _load() -> Dict:
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Result file not found: {INPUT_JSON}. "
            "Run `PYTHONPATH=. python eval/eviction_policy_eval.py` first."
        )
    return json.loads(INPUT_JSON.read_text())


# ── Chart 1: Retention Heatmap ───────────────────────────────────────────────

def plot_retention_matrix(data: Dict) -> Path:
    diff = data["differential_retention"]
    oracle = data["oracle"]

    matrix = [
        [diff[s]["retention"][c] for c in CATEGORY_ORDER]
        for s in STRATEGY_ORDER
    ]

    fig, ax = plt.subplots(figsize=(6.65, 4.2))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "retention", ["#FCE4E4", "#F3A7A7", "#55A868"], N=256
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORY_ORDER], fontsize=9)
    ax.set_yticks(range(len(STRATEGY_ORDER)))
    ax.set_yticklabels([s.upper() for s in STRATEGY_ORDER], fontsize=11, fontweight="bold")

    for i, s in enumerate(STRATEGY_ORDER):
        for j, c in enumerate(CATEGORY_ORDER):
            val = matrix[i][j]
            in_oracle = c in oracle[s]
            marker = " *" if in_oracle else ""
            txt_color = "white" if val > 55 else "#333333"
            ax.text(j, i, f"{val:.0f}%{marker}", ha="center", va="center",
                    color=txt_color, fontsize=11, fontweight="bold")

    ax.set_title(
        "Retention per Category × Eviction Strategy  (★ = oracle-favored)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Workload Category", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Retention %", fontsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / "retention_matrix.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


# ── Chart 2: Differential Retention (grouped bars) ───────────────────────────

def plot_differential_retention(data: Dict) -> Path:
    diff = data["differential_retention"]

    fig, ax = plt.subplots(figsize=(10, 5))
    n_cats = len(CATEGORY_ORDER)
    width = 0.26
    x = list(range(n_cats))

    for i, s in enumerate(STRATEGY_ORDER):
        values = [diff[s]["retention"][c] for c in CATEGORY_ORDER]
        offsets = [xi + (i - 1) * width for xi in x]
        bars = ax.bar(offsets, values, width=width,
                      color=PALETTE[s], label=s.upper(),
                      edgecolor="white", linewidth=0.8, zorder=2)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 1,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORY_ORDER], fontsize=9)
    ax.set_ylim(0, 115)
    _style(ax, "Differential Retention: which memories survive under each policy?",
           "Workload Category", "Retention %")
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    plt.tight_layout()
    out = OUTPUT_DIR / "differential_retention.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


# ── Chart 3: Sweep-latency Scaling ───────────────────────────────────────────

def plot_sweep_latency_scaling(data: Dict) -> Path:
    scaling = data["scale_sweep"]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    for s in STRATEGY_ORDER:
        rows = scaling[s]
        xs = [r["total_items"] for r in rows]
        means = [r["mean_ms"] for r in rows]
        p90s = [r["p90_ms"] for r in rows]
        ax.plot(xs, means, marker="o", linewidth=2, color=PALETTE[s],
                label=f"{s.upper()} mean")
        ax.fill_between(xs, means, p90s, alpha=0.15, color=PALETTE[s],
                        label=f"{s.upper()} mean→p90")

    _style(ax, "Sweep-latency scaling vs working-set size",
           "Total rows seeded (4 categories)", "Sweep latency (ms)")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    ax.set_xticks([r["total_items"] for r in scaling["lru"]])

    plt.tight_layout()
    out = OUTPUT_DIR / "sweep_latency_scaling.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


# ── Chart 4: Budget Adherence ────────────────────────────────────────────────

def plot_budget_adherence(data: Dict) -> Path:
    diff = data["differential_retention"]
    budget = data["budget_tokens"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = list(range(len(STRATEGY_ORDER)))
    width = 0.38

    before_vals = [diff[s]["tokens_before"] for s in STRATEGY_ORDER]
    after_vals = [diff[s]["tokens_after"] for s in STRATEGY_ORDER]

    b1 = ax.bar([xi - width / 2 for xi in x], before_vals, width=width,
                color=PALETTE["before"], label="Before eviction",
                edgecolor="white", zorder=2)
    b2 = ax.bar([xi + width / 2 for xi in x], after_vals, width=width,
                color=PALETTE["after"], label="After eviction",
                edgecolor="white", zorder=2)

    for b, v in list(zip(b1, before_vals)) + list(zip(b2, after_vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + 40,
                f"{v}", ha="center", va="bottom", fontsize=9)

    ax.axhline(budget, color=PALETTE["budget"], linestyle="--",
               linewidth=2, zorder=3, label=f"Budget = {budget}")

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in STRATEGY_ORDER], fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(before_vals) * 1.15)
    _style(ax, "Budget restoration: tokens before vs after eviction",
           "Strategy", "Token count")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    plt.tight_layout()
    out = OUTPUT_DIR / "budget_adherence.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def main():
    data = _load()
    plot_retention_matrix(data)
    plot_differential_retention(data)
    plot_sweep_latency_scaling(data)
    plot_budget_adherence(data)
    logger.info("All charts written to %s/", OUTPUT_DIR)


if __name__ == "__main__":
    main()
