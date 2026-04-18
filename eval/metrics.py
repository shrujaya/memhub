"""
eval/metrics.py  — MemHub Performance Metrics Collector
========================================================

Tracks latency, throughput, token usage, and memory-tier hit rates
for every MemHub operation during an evaluation run.

Metrics collected
─────────────────
  Per-operation:
    • wall-clock latency (ms)
    • token count of the stored/retrieved content
    • which tier returned the result (Tier 1 / Tier 2)
    • policy sweep durations

  Aggregated:
    • p50 / p90 / p99 latencies per operation type
    • requests-per-second throughput
    • tier-1 vs tier-2 hit rate
    • total tokens processed
    • compression ratio achieved by DemotionPolicy

Usage
─────
  from eval.metrics import MetricsCollector

  collector = MetricsCollector(run_id="lru_vs_fifo_001")

  with collector.measure("store"):
      store_memory(agent_id=..., content=...)

  collector.record_retrieval(tier="tier1", latency_ms=4.2, tokens=120)
  collector.save("eval/results/run_001.json")
  summary = collector.summary()
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────


class OperationType(str, Enum):
    STORE    = "store"
    RETRIEVE = "retrieve"
    POLICY   = "policy"
    LLM_CALL = "llm_call"


@dataclass
class OperationRecord:
    """A single timed operation sample."""
    op_type:    str
    latency_ms: float
    tokens:     int = 0
    tier:       Optional[str] = None   # "tier1" | "tier2" | None
    success:    bool = True
    error:      Optional[str] = None
    timestamp:  float = field(default_factory=time.time)
    metadata:   Dict = field(default_factory=dict)


@dataclass
class RunSummary:
    """Aggregated statistics for a complete evaluation run."""
    run_id:              str
    total_operations:    int = 0
    total_tokens:        int = 0
    elapsed_seconds:     float = 0.0
    ops_per_second:      float = 0.0

    # Per-operation-type latency percentiles (ms)
    latency_p50:  Dict[str, float] = field(default_factory=dict)
    latency_p90:  Dict[str, float] = field(default_factory=dict)
    latency_p99:  Dict[str, float] = field(default_factory=dict)
    latency_mean: Dict[str, float] = field(default_factory=dict)

    tier1_hits:   int = 0
    tier2_hits:   int = 0
    tier1_hit_rate: float = 0.0

    # Token compression stats
    tokens_before_demotion: int = 0
    tokens_after_demotion:  int = 0
    compression_pct:        float = 0.0

    error_count: int = 0


# ── MetricsCollector ──────────────────────────────────────────────────────────


class MetricsCollector:
    """
    Thread-safe (GIL-protected) metrics collector for a single eval run.

    Args:
        run_id:  Human-readable label for this run (e.g. "lru_k5_run1").
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._records: List[OperationRecord] = []
        self._run_start: float = time.perf_counter()
        self._compression_before: int = 0
        self._compression_after: int = 0
        logger.info("MetricsCollector initialised for run '%s'.", run_id)

    # ── Context manager for timing ────────────────────────────────────────────

    @contextmanager
    def measure(
        self,
        op_type: str,
        tokens: int = 0,
        tier: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Generator[None, None, None]:
        """
        Context manager that times the enclosed block and records a sample.

        Args:
            op_type:  One of 'store', 'retrieve', 'policy', 'llm_call', or custom.
            tokens:   Token count associated with this operation.
            tier:     Storage tier ('tier1' or 'tier2') if applicable.
            metadata: Any extra key-value pairs to attach to the record.

        Example::

            with collector.measure("store", tokens=count_tokens(content)):
                store_memory(agent_id=..., content=content)
        """
        t0 = time.perf_counter()
        error_msg: Optional[str] = None
        try:
            yield
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1_000
            self._records.append(
                OperationRecord(
                    op_type=op_type,
                    latency_ms=round(latency_ms, 3),
                    tokens=tokens,
                    tier=tier,
                    success=error_msg is None,
                    error=error_msg,
                    metadata=metadata or {},
                )
            )

    # ── Manual recording helpers ──────────────────────────────────────────────

    def record_retrieval(
        self,
        tier: str,
        latency_ms: float,
        tokens: int = 0,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Record a retrieval result (used when not wrapping with ``measure``)."""
        self._records.append(
            OperationRecord(
                op_type=OperationType.RETRIEVE,
                latency_ms=latency_ms,
                tokens=tokens,
                tier=tier,
                metadata=metadata or {},
            )
        )

    def record_policy_sweep(
        self,
        latency_ms: float,
        tokens_before: int,
        tokens_after: int,
        strategy: str = "lru",
    ) -> None:
        """Record a full policy sweep and update compression counters."""
        self._records.append(
            OperationRecord(
                op_type=OperationType.POLICY,
                latency_ms=latency_ms,
                metadata={"strategy": strategy,
                          "tokens_before": tokens_before,
                          "tokens_after": tokens_after},
            )
        )
        self._compression_before += tokens_before
        self._compression_after  += tokens_after

    def record_llm_call(
        self,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        model: str = "unknown",
    ) -> None:
        """Record an LLM API call (for summarisation / agent reasoning)."""
        self._records.append(
            OperationRecord(
                op_type=OperationType.LLM_CALL,
                latency_ms=latency_ms,
                tokens=tokens_in + tokens_out,
                metadata={"model": model,
                          "tokens_in": tokens_in,
                          "tokens_out": tokens_out},
            )
        )

    # ── Aggregation ───────────────────────────────────────────────────────────

    def summary(self) -> RunSummary:
        """
        Compute and return a :class:`RunSummary` for this run.

        Latency percentiles are computed per ``op_type`` using the standard
        library ``statistics.quantiles`` to avoid a NumPy dependency.
        """
        elapsed = time.perf_counter() - self._run_start
        total   = len(self._records)

        # Group latencies by op_type
        latencies_by_type: Dict[str, List[float]] = {}
        for r in self._records:
            latencies_by_type.setdefault(r.op_type, []).append(r.latency_ms)

        def _percentile(data: List[float], pct: int) -> float:
            if not data:
                return 0.0
            if len(data) == 1:
                return data[0]
            # statistics.quantiles uses n=100 for percentile-style results
            idx = max(0, min(int(len(data) * pct / 100), len(data) - 1))
            return sorted(data)[idx]

        p50, p90, p99, mean = {}, {}, {}, {}
        for op, lats in latencies_by_type.items():
            p50[op]  = round(_percentile(lats, 50), 3)
            p90[op]  = round(_percentile(lats, 90), 3)
            p99[op]  = round(_percentile(lats, 99), 3)
            mean[op] = round(statistics.mean(lats), 3) if lats else 0.0

        tier1_hits = sum(1 for r in self._records if r.tier == "tier1")
        tier2_hits = sum(1 for r in self._records if r.tier == "tier2")
        total_hits = tier1_hits + tier2_hits

        compression_pct = (
            round(100 * (1 - self._compression_after / max(self._compression_before, 1)), 2)
            if self._compression_before > 0 else 0.0
        )

        return RunSummary(
            run_id=self.run_id,
            total_operations=total,
            total_tokens=sum(r.tokens for r in self._records),
            elapsed_seconds=round(elapsed, 3),
            ops_per_second=round(total / max(elapsed, 0.001), 2),
            latency_p50=p50,
            latency_p90=p90,
            latency_p99=p99,
            latency_mean=mean,
            tier1_hits=tier1_hits,
            tier2_hits=tier2_hits,
            tier1_hit_rate=round(tier1_hits / max(total_hits, 1), 4),
            tokens_before_demotion=self._compression_before,
            tokens_after_demotion=self._compression_after,
            compression_pct=compression_pct,
            error_count=sum(1 for r in self._records if not r.success),
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """
        Serialise all raw records and the run summary to a JSON file.

        Args:
            path: Output file path (created along with any missing dirs).

        Returns:
            The resolved :class:`~pathlib.Path` of the written file.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "run_id":  self.run_id,
            "summary": asdict(self.summary()),
            "records": [asdict(r) for r in self._records],
        }
        out.write_text(json.dumps(payload, indent=2))
        logger.info("Metrics saved to '%s' (%d records).", out, len(self._records))
        return out

    def reset(self) -> None:
        """Clear all collected records and restart the run timer."""
        self._records.clear()
        self._run_start = time.perf_counter()
        self._compression_before = 0
        self._compression_after  = 0
        logger.debug("MetricsCollector reset for run '%s'.", self.run_id)

    def __repr__(self) -> str:
        return (
            f"MetricsCollector(run_id={self.run_id!r}, "
            f"records={len(self._records)})"
        )
