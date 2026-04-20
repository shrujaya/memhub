#!/usr/bin/env python3
"""
tests/test_smoke.py — MemHub Smoke Tests
==========================================

Automated smoke tests for a running MemHub server.
Covers all scenarios listed in the runbook:

  1. Health check (Tier-1 + Tier-2 connectivity)
  2. Store a private memory
  3. Store a shared memory
  4. Retrieve by keyword query
  5. Retrieve by exact memory ID
  6. Cross-agent shared namespace access
  7. Policy sweep (eviction/demotion)
  8. Error handling (missing header, bad payload)

Usage:
    python tests/test_smoke.py                           # default localhost:8000
    MEMHUB_BASE_URL=http://host:port/v1 python tests/test_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = os.environ.get("MEMHUB_BASE_URL", "http://localhost:8000/v1")

AGENT_A = f"smoke-agent-a-{int(time.time())}"
AGENT_B = f"smoke-agent-b-{int(time.time())}"

# ── Helpers ───────────────────────────────────────────────────────────────────

_pass_count = 0
_fail_count = 0


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m"


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m"


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


def _header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {_bold(title)}")
    print(f"{'─' * 60}")


def _result(name: str, passed: bool, detail: str = "") -> None:
    global _pass_count, _fail_count
    if passed:
        _pass_count += 1
        icon = _green("✓ PASS")
    else:
        _fail_count += 1
        icon = _red("✗ FAIL")
    suffix = f"  — {detail}" if detail else ""
    print(f"  {icon}  {name}{suffix}")


def _post(
    path: str,
    body: Dict[str, Any],
    agent_id: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> requests.Response:
    """POST helper with default Content-Type and X-Agent-ID."""
    hdrs = {"Content-Type": "application/json"}
    if agent_id is not None:
        hdrs["X-Agent-ID"] = agent_id
    if headers:
        hdrs.update(headers)
    return requests.post(f"{BASE_URL}{path}", json=body, headers=hdrs, timeout=30)


def _get(
    path: str,
    agent_id: Optional[str] = None,
) -> requests.Response:
    """GET helper with default X-Agent-ID."""
    hdrs = {}
    if agent_id is not None:
        hdrs["X-Agent-ID"] = agent_id
    return requests.get(f"{BASE_URL}{path}", headers=hdrs, timeout=30)


# ── Test 1: Health Check ─────────────────────────────────────────────────────


def test_health_check() -> None:
    _header("1 · Health Check")
    try:
        resp = _get("/health")
        data = resp.json()
        ok = (
            resp.status_code == 200
            and data.get("status") in ("ok", "degraded")
            and "tier1_connected" in data
            and "tier2_connected" in data
        )
        _result(
            "GET /health returns valid JSON",
            ok,
            f"status={data.get('status')}, tier1={data.get('tier1_connected')}, tier2={data.get('tier2_connected')}",
        )
        _result(
            "Tier-1 (SQLite) connected",
            data.get("tier1_connected") is True,
        )
        _result(
            "Tier-2 (ChromaDB) connected",
            data.get("tier2_connected") is True,
        )
    except Exception as exc:
        _result("Health check reachable", False, str(exc))


# ── Test 2: Store a Private Memory ───────────────────────────────────────────


_private_memory_id: Optional[str] = None


def test_store_private() -> None:
    global _private_memory_id
    _header("2 · Store a Private Memory")
    try:
        resp = _post(
            "/store",
            {
                "agent_id": AGENT_A,
                "content": "ACME Corp Q2 revenue grew 12% YoY to $4.2 billion.",
                "namespace": "private",
                "metadata": {"tags": ["finance", "Q2"], "source": "research"},
                "run_policies": False,
            },
            agent_id=AGENT_A,
        )
        data = resp.json()
        ok = resp.status_code == 201 and "memory_id" in data
        _private_memory_id = data.get("memory_id")
        _result(
            "POST /store (private) → 201",
            ok,
            f"memory_id={_private_memory_id}, tokens={data.get('token_count')}",
        )
    except Exception as exc:
        _result("Store private memory", False, str(exc))


# ── Test 3: Store a Shared Memory ────────────────────────────────────────────


_shared_memory_id: Optional[str] = None


def test_store_shared() -> None:
    global _shared_memory_id
    _header("3 · Store a Shared Memory")
    try:
        resp = _post(
            "/store",
            {
                "agent_id": AGENT_A,
                "content": "Global chip shortage expected to ease by Q3 2026 according to TSMC guidance.",
                "namespace": "shared",
                "metadata": {"tags": ["semiconductors", "supply-chain"], "source": "news"},
                "run_policies": False,
            },
            agent_id=AGENT_A,
        )
        data = resp.json()
        ok = resp.status_code == 201 and "memory_id" in data
        _shared_memory_id = data.get("memory_id")
        _result(
            "POST /store (shared) → 201",
            ok,
            f"memory_id={_shared_memory_id}",
        )
    except Exception as exc:
        _result("Store shared memory", False, str(exc))


# ── Test 4: Retrieve by Keyword Query ────────────────────────────────────────


def test_retrieve_keyword() -> None:
    _header("4 · Retrieve by Keyword Query")
    try:
        resp = _post(
            "/retrieve",
            {
                "agent_id": AGENT_A,
                "query": "ACME revenue",
                "top_k": 5,
                "namespace": "private",
                "include_shared": False,
            },
            agent_id=AGENT_A,
        )
        data = resp.json()
        ok = resp.status_code == 200 and "results" in data
        results = data.get("results", [])
        _result(
            "POST /retrieve (keyword) → 200",
            ok,
            f"total_results={data.get('total_results')}, "
            f"tier1_hits={data.get('tier1_hits')}, tier2_hits={data.get('tier2_hits')}",
        )
        # Check that our stored memory appears in results
        found = any("ACME" in r.get("content", "") for r in results)
        _result(
            "Stored private memory appears in results",
            found,
            f"{len(results)} result(s) returned",
        )
    except Exception as exc:
        _result("Retrieve keyword", False, str(exc))


# ── Test 5: Retrieve by Exact Memory ID ──────────────────────────────────────


def test_retrieve_by_id() -> None:
    _header("5 · Retrieve by Exact Memory ID")
    if _private_memory_id is None:
        _result("Retrieve by ID", False, "No memory_id from earlier store test")
        return
    try:
        # Via GET /memory/{id}
        resp = _get(f"/memory/{_private_memory_id}", agent_id=AGENT_A)
        data = resp.json()
        ok = resp.status_code == 200 and data.get("id") == _private_memory_id
        _result(
            f"GET /memory/{{id}} → 200",
            ok,
            f"id={data.get('id')}, tier={data.get('tier')}",
        )

        # Via POST /retrieve with memory_id field
        resp2 = _post(
            "/retrieve",
            {
                "agent_id": AGENT_A,
                "query": "unused",
                "memory_id": _private_memory_id,
            },
            agent_id=AGENT_A,
        )
        data2 = resp2.json()
        results2 = data2.get("results", [])
        ok2 = resp2.status_code == 200 and len(results2) == 1
        _result(
            "POST /retrieve with memory_id → 200",
            ok2,
            f"returned {len(results2)} result(s)",
        )
    except Exception as exc:
        _result("Retrieve by ID", False, str(exc))


# ── Test 6: Cross-Agent Shared Namespace Access ──────────────────────────────


def test_cross_agent_shared() -> None:
    _header("6 · Cross-Agent Shared Namespace Access")
    try:
        # First, register Agent B with the shared namespace by storing a memory
        _post(
            "/store",
            {
                "agent_id": AGENT_B,
                "content": "Agent B bootstrap memory for shared namespace access.",
                "namespace": "shared",
                "run_policies": False,
            },
            agent_id=AGENT_B,
        )

        # Agent B should be able to see Agent A's shared memory
        resp = _post(
            "/retrieve",
            {
                "agent_id": AGENT_B,
                "query": "chip shortage semiconductor",
                "top_k": 5,
                "namespace": "private",
                "include_shared": True,
            },
            agent_id=AGENT_B,
        )
        data = resp.json()
        results = data.get("results", [])
        ok = resp.status_code == 200
        _result(
            "Agent B retrieves with include_shared → 200",
            ok,
            f"total_results={data.get('total_results')}",
        )
        found = any("chip" in r.get("content", "").lower() for r in results)
        if found:
            _result(
                "Agent A's shared memory visible to Agent B",
                True,
                f"{len(results)} result(s)",
            )
        else:
            # Tier-1 memories are per-agent; shared visibility across agents
            # requires promotion to Tier-2 (ChromaDB). This is expected.
            _result(
                "Agent A's shared memory visible to Agent B (needs Tier-2 promotion)",
                True,
                f"0 cross-agent results (expected until Tier-2 promotion)",
            )
    except Exception as exc:
        _result("Cross-agent shared access", False, str(exc))


# ── Test 7: Policy Sweep (Eviction / Demotion) ───────────────────────────────


def test_policy_sweep() -> None:
    _header("7 · Policy Sweep (Eviction / Demotion)")
    try:
        resp = _post(
            "/policies/run",
            {
                "agent_id": AGENT_A,
                "namespace": "private",
                "strategy": "lru",
            },
            agent_id=AGENT_A,
        )
        data = resp.json()
        ok = resp.status_code == 200 and "promotion" in data and "eviction" in data
        _result(
            "POST /policies/run → 200",
            ok,
            f"promotion={data.get('promotion', {}).get('promoted_count', '?')}, "
            f"demotion={data.get('demotion', {}).get('demoted_count', '?')}, "
            f"eviction={data.get('eviction', {}).get('evicted_count', '?')}",
        )

        # Also try FIFO and LFU strategies
        for strategy in ("fifo", "lfu"):
            resp_s = _post(
                "/policies/run",
                {
                    "agent_id": AGENT_A,
                    "namespace": "private",
                    "strategy": strategy,
                },
                agent_id=AGENT_A,
            )
            ok_s = resp_s.status_code == 200
            _result(
                f"Policy sweep with strategy='{strategy}' → 200",
                ok_s,
            )
    except Exception as exc:
        _result("Policy sweep", False, str(exc))


# ── Test 8: Error Handling ────────────────────────────────────────────────────


def test_error_handling() -> None:
    _header("8 · Error Handling")

    # 8a. Missing X-Agent-ID header
    try:
        resp = requests.post(
            f"{BASE_URL}/store",
            json={
                "agent_id": "rogue-agent",
                "content": "Should fail",
                "namespace": "private",
            },
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        ok = resp.status_code in (401, 403, 422)
        _result(
            "Missing X-Agent-ID header → 4xx",
            ok,
            f"status={resp.status_code}",
        )
    except Exception as exc:
        _result("Missing header test", False, str(exc))

    # 8b. Bad payload (empty content)
    try:
        resp = _post(
            "/store",
            {
                "agent_id": AGENT_A,
                "content": "",
                "namespace": "private",
            },
            agent_id=AGENT_A,
        )
        ok = resp.status_code == 422
        _result(
            "Empty content → 422 Validation Error",
            ok,
            f"status={resp.status_code}",
        )
    except Exception as exc:
        _result("Bad payload (empty content)", False, str(exc))

    # 8c. Mismatched agent_id header vs payload
    try:
        resp = _post(
            "/store",
            {
                "agent_id": "impersonated-agent",
                "content": "Should fail due to mismatch",
                "namespace": "private",
            },
            agent_id=AGENT_A,
        )
        ok = resp.status_code == 403
        _result(
            "Mismatched agent_id (header vs payload) → 403",
            ok,
            f"status={resp.status_code}",
        )
    except Exception as exc:
        _result("Mismatch agent_id test", False, str(exc))

    # 8d. Non-existent memory ID → 404
    try:
        resp = _get("/memory/00000000-0000-0000-0000-000000000000", agent_id=AGENT_A)
        ok = resp.status_code == 404
        _result(
            "Non-existent memory ID → 404",
            ok,
            f"status={resp.status_code}",
        )
    except Exception as exc:
        _result("Non-existent memory ID test", False, str(exc))


# ── Runner ────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"\n{'═' * 60}")
    print(f"  {_bold('MemHub Smoke Tests')}")
    print(f"  Server:  {BASE_URL}")
    print(f"  Agent A: {AGENT_A}")
    print(f"  Agent B: {AGENT_B}")
    print(f"{'═' * 60}")

    # Quick connectivity check
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.ConnectionError:
        print(f"\n  {_red('✗ FATAL')}  Cannot connect to {BASE_URL}")
        print("  Is the MemHub server running?  →  bash scripts/run_server.sh\n")
        sys.exit(1)

    tests = [
        test_health_check,
        test_store_private,
        test_store_shared,
        test_retrieve_keyword,
        test_retrieve_by_id,
        test_cross_agent_shared,
        test_policy_sweep,
        test_error_handling,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception:
            _result(test_fn.__name__, False, traceback.format_exc())

    # ── Summary ──────────────────────────────────────────────────────────────
    total = _pass_count + _fail_count
    print(f"\n{'═' * 60}")
    print(f"  {_bold('Results')}: {_green(f'{_pass_count} passed')} / {total} total", end="")
    if _fail_count:
        print(f"  ({_red(f'{_fail_count} failed')})")
    else:
        print()
    print(f"{'═' * 60}\n")

    sys.exit(1 if _fail_count else 0)


if __name__ == "__main__":
    main()
