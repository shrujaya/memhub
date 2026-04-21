"""
eval/eviction_policy_eval.py  — Differential Eviction-Policy Benchmark
=====================================================================

The stock `long_session_eviction` task in `benchmark_tasks.py` only asks:
"did the policy restore the token budget?" — which every policy trivially
passes. It doesn't test whether each policy *kept the right memories*.

This benchmark fixes that by seeding a workload with four disjoint
categories of memory, engineered so each eviction policy should preserve
a **distinct, predictable pair** of categories:

    Category        created_at   last_accessed   access_count
    ─────────────────────────────────────────────────────────
    A_cold          T-1000       T-1000          0
    B_recent_hit    T-1000       T-10            5
    C_frequent_hit  T-500        T-500           100
    D_fresh         T-10         T-10            1

    Budget forces ~50% eviction (100 items, each ~40 tokens, budget=2000).

    Expected retention (correctness oracle):
                      A       B       C       D
        LRU           0%      100%    0%      100%   (newest last_accessed)
        FIFO          0%      0%      100%    100%   (newest created_at)
        LFU           0%      100%    100%    0%     (highest access_count)

Every strategy evicts A (no one wants cold data), and each keeps a
unique pair of {B,C,D}. A run is a PASS when the observed retention
matches the oracle exactly.

Additionally, the script runs a **scale sweep** across n_memories to
characterise sweep-latency scaling per policy.

Usage
─────
    PYTHONPATH=. python eval/eviction_policy_eval.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb  # type: ignore

from core.policies import (
    EvictionPolicy,
    EvictionStrategy,
    WORKING_MEMORY_TOKEN_BUDGET,
)
from core.summarization import count_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eviction_eval")


# ── Workload definition ──────────────────────────────────────────────────────

# Per-category profile: how far in the past each field should be (seconds)
# and the initial access_count.
CATEGORY_PROFILE: Dict[str, Dict[str, float]] = {
    "A_cold":         {"created_offset": 1000, "last_offset": 1000, "access_count": 0},
    "B_recent_hit":   {"created_offset": 1000, "last_offset": 10,   "access_count": 5},
    "C_frequent_hit": {"created_offset": 500,  "last_offset": 500,  "access_count": 100},
    "D_fresh":        {"created_offset": 10,   "last_offset": 10,   "access_count": 1},
}

# Oracle: for each strategy, which two categories SHOULD survive.
EXPECTED_RETAINED: Dict[str, set] = {
    "lru":  {"B_recent_hit",   "D_fresh"},          # sort by last_accessed DESC
    "fifo": {"C_frequent_hit", "D_fresh"},          # sort by created_at DESC
    "lfu":  {"B_recent_hit",   "C_frequent_hit"},   # sort by access_count DESC
}

# Content template sized to produce ~40 tokens/item so that 100 items
# ≈ 4_000 tokens (2× the 2_000 budget → ~50 % eviction).
_CONTENT_TEMPLATE = (
    "[{cat} #{idx}] memory-row payload describing workload segment "
    "with decision marker {marker} and status tag {status} recorded "
    "for eviction-policy differential testing."
)


# ── Fixture helpers ──────────────────────────────────────────────────────────


def _make_ephemeral_db() -> sqlite3.Connection:
    from api.main import _SCHEMA_SQL
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


def _make_ephemeral_chroma() -> chromadb.Client:
    return chromadb.EphemeralClient()


def _insert_row(
    db: sqlite3.Connection,
    agent_id: str,
    content: str,
    category: str,
    created_at: float,
    last_accessed: float,
    access_count: int,
) -> str:
    row_id = str(uuid.uuid4())
    db.execute(
        """
        INSERT INTO working_memory
                (id, agent_id, content, created_at, last_accessed,
                 access_count, namespace)
        VALUES  (?,  ?,        ?,       ?,          ?,
                 ?,            ?)
        """,
        (row_id, agent_id, content, created_at, last_accessed,
         access_count, f"cat:{category}"),
    )
    db.commit()
    return row_id


def seed_workload(
    db: sqlite3.Connection,
    agent_id: str,
    n_per_category: int,
) -> Dict[str, List[str]]:
    """
    Populate the working-memory table with `n_per_category` items per
    category. Returns a mapping category → list of row-ids.
    """
    now = time.time()
    ids_by_category: Dict[str, List[str]] = {c: [] for c in CATEGORY_PROFILE}

    for category, profile in CATEGORY_PROFILE.items():
        for idx in range(n_per_category):
            content = _CONTENT_TEMPLATE.format(
                cat=category,
                idx=idx,
                marker=idx * 7,
                status="OK" if idx % 2 == 0 else "PENDING",
            )
            # Add a tiny intra-category jitter so sorts within a group are
            # deterministic but the category-level ordering is preserved.
            jitter = idx * 0.001
            row_id = _insert_row(
                db=db,
                agent_id=agent_id,
                content=content,
                category=category,
                created_at=now - profile["created_offset"] - jitter,
                last_accessed=now - profile["last_offset"] - jitter,
                access_count=int(profile["access_count"]),
            )
            ids_by_category[category].append(row_id)

    return ids_by_category


# ── Single-run result ─────────────────────────────────────────────────────────


@dataclass
class PolicyRunResult:
    strategy: str
    n_per_category: int
    tokens_before: int
    tokens_after: int
    sweep_latency_ms: float
    rows_evicted: int
    retention: Dict[str, float] = field(default_factory=dict)   # category → %
    retained_counts: Dict[str, int] = field(default_factory=dict)
    category_totals: Dict[str, int] = field(default_factory=dict)
    oracle_match: bool = False
    oracle_retained: List[str] = field(default_factory=list)
    oracle_observed: List[str] = field(default_factory=list)


async def run_single(
    strategy: EvictionStrategy,
    n_per_category: int = 25,
) -> PolicyRunResult:
    """Seed a fresh fixture, run eviction, compute retention per category."""
    db = _make_ephemeral_db()
    chroma = _make_ephemeral_chroma()
    agent_id = f"bench-{strategy.value}-{uuid.uuid4().hex[:6]}"

    ids_by_cat = seed_workload(db, agent_id, n_per_category)
    category_of: Dict[str, str] = {
        rid: cat for cat, rids in ids_by_cat.items() for rid in rids
    }

    tokens_before = sum(
        count_tokens(r[0])
        for r in db.execute(
            "SELECT content FROM working_memory WHERE agent_id = ?", (agent_id,)
        ).fetchall()
    )

    policy = EvictionPolicy(db=db, chroma=chroma, strategy=strategy)
    t0 = time.perf_counter()
    result = await policy.run(agent_id=agent_id)
    latency_ms = (time.perf_counter() - t0) * 1_000

    surviving_ids = {
        r[0]
        for r in db.execute(
            "SELECT id FROM working_memory WHERE agent_id = ?", (agent_id,)
        ).fetchall()
    }
    retained_counts = {c: 0 for c in CATEGORY_PROFILE}
    for rid in surviving_ids:
        retained_counts[category_of[rid]] += 1

    retention_pct = {
        c: 100.0 * retained_counts[c] / max(n_per_category, 1)
        for c in CATEGORY_PROFILE
    }

    # Oracle comparison: the two "mostly-retained" categories (>= 80%)
    # should match the expected set for this strategy.
    observed_retained = {
        c for c, pct in retention_pct.items() if pct >= 80.0
    }
    expected = EXPECTED_RETAINED[strategy.value]
    oracle_match = observed_retained == expected

    return PolicyRunResult(
        strategy=strategy.value,
        n_per_category=n_per_category,
        tokens_before=tokens_before,
        tokens_after=result.tokens_after,
        sweep_latency_ms=round(latency_ms, 3),
        rows_evicted=len(result.evicted_ids),
        retention=retention_pct,
        retained_counts=retained_counts,
        category_totals={c: n_per_category for c in CATEGORY_PROFILE},
        oracle_match=oracle_match,
        oracle_retained=sorted(expected),
        oracle_observed=sorted(observed_retained),
    )


# ── Scale sweep ──────────────────────────────────────────────────────────────


async def run_scale_sweep(
    sizes: Tuple[int, ...] = (12, 25, 50, 100),
    reps: int = 3,
) -> Dict[str, List[Dict]]:
    """
    Run each strategy at each size, `reps` times, to characterise the
    sweep-latency distribution as the working set grows.

    Returns:
        {strategy: [{"n_per_category", "mean_ms", "p90_ms"} …]}
    """
    out: Dict[str, List[Dict]] = {s.value: [] for s in EvictionStrategy}

    for size in sizes:
        for strategy in EvictionStrategy:
            lats: List[float] = []
            for _ in range(reps):
                r = await run_single(strategy, n_per_category=size)
                lats.append(r.sweep_latency_ms)
            lats.sort()
            p90_idx = max(0, min(int(len(lats) * 0.90), len(lats) - 1))
            out[strategy.value].append({
                "n_per_category": size,
                "total_items": size * len(CATEGORY_PROFILE),
                "mean_ms": round(sum(lats) / len(lats), 3),
                "p90_ms":  round(lats[p90_idx], 3),
                "min_ms":  round(min(lats), 3),
                "max_ms":  round(max(lats), 3),
            })

    return out


# ── Orchestrator ─────────────────────────────────────────────────────────────


async def run_full_benchmark(
    output_dir: str = "eval/results/eviction",
) -> Dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Part 1: differential retention at the canonical size (25/cat = 100) ──
    logger.info("=" * 60)
    logger.info("PART 1 — Differential retention test (25 items × 4 cats)")
    logger.info("=" * 60)

    per_strategy: Dict[str, PolicyRunResult] = {}
    for s in EvictionStrategy:
        r = await run_single(s, n_per_category=25)
        per_strategy[s.value] = r
        logger.info(
            "  %-4s  tokens %d → %d,  evicted=%d,  sweep=%.2fms,  oracle_match=%s",
            s.value.upper(),
            r.tokens_before,
            r.tokens_after,
            r.rows_evicted,
            r.sweep_latency_ms,
            r.oracle_match,
        )
        logger.info("        retention=%s",
                    {k: f"{v:.0f}%" for k, v in r.retention.items()})

    # ── Part 2: scale sweep ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PART 2 — Sweep-latency scaling (4 sizes × 3 reps × 3 strategies)")
    logger.info("=" * 60)
    scaling = await run_scale_sweep(sizes=(12, 25, 50, 100), reps=3)
    for strategy, rows in scaling.items():
        for row in rows:
            logger.info(
                "  %-4s  n=%-4d  total=%-4d  mean=%-6.2fms  p90=%-6.2fms",
                strategy.upper(),
                row["n_per_category"],
                row["total_items"],
                row["mean_ms"],
                row["p90_ms"],
            )

    # ── Persist ──────────────────────────────────────────────────────────────
    payload = {
        "budget_tokens": WORKING_MEMORY_TOKEN_BUDGET,
        "workload_profile": CATEGORY_PROFILE,
        "oracle": {k: sorted(v) for k, v in EXPECTED_RETAINED.items()},
        "differential_retention": {k: asdict(v) for k, v in per_strategy.items()},
        "scale_sweep": scaling,
    }
    result_file = out_path / "eviction_differential.json"
    result_file.write_text(json.dumps(payload, indent=2))
    logger.info("Saved results to %s", result_file)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'STRATEGY':<10} {'A_cold':<10} {'B_recent':<10} {'C_frequent':<12} {'D_fresh':<10} {'ORACLE':<8}")
    print("=" * 78)
    for s, r in per_strategy.items():
        print(
            f"{s.upper():<10} "
            f"{r.retention['A_cold']:<10.0f} "
            f"{r.retention['B_recent_hit']:<10.0f} "
            f"{r.retention['C_frequent_hit']:<12.0f} "
            f"{r.retention['D_fresh']:<10.0f} "
            f"{'MATCH' if r.oracle_match else 'MISS':<8}"
        )
    print("=" * 78)

    return payload


if __name__ == "__main__":
    asyncio.run(run_full_benchmark())
