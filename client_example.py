#!/usr/bin/env python3
"""
client_example.py — MemHub Remote Client Demo
===============================================

Run this on any machine that can reach the MemHub server.
It demonstrates store, retrieve, and health-check operations
against a remote MemHub deployment.

Usage:
    # Point at your server (default: localhost:8000)
    export MEMHUB_SERVER=http://<server-ip>:8000

    python client_example.py
"""

import os
import json
import requests

SERVER = os.getenv("MEMHUB_SERVER", "http://localhost:8000")
BASE   = f"{SERVER}/v1"
AGENT  = "remote-agent-01"

HEADERS = {
    "Content-Type": "application/json",
    "X-Agent-ID":   AGENT,
}


def health() -> dict:
    """Check server health."""
    r = requests.get(f"{BASE}/health", timeout=5)
    r.raise_for_status()
    return r.json()


def store(content: str, namespace: str = "private", tags: list | None = None) -> dict:
    """Store a memory on the remote server."""
    payload = {
        "agent_id":  AGENT,
        "content":   content,
        "namespace": namespace,
        "metadata":  {"tags": tags or [], "source": "client_example"},
    }
    r = requests.post(f"{BASE}/store", json=payload, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def retrieve(query: str, top_k: int = 5, include_shared: bool = True) -> dict:
    """Search memories on the remote server."""
    payload = {
        "agent_id":       AGENT,
        "query":          query,
        "top_k":          top_k,
        "include_shared": include_shared,
    }
    r = requests.post(f"{BASE}/retrieve", json=payload, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def pp(data: dict) -> None:
    """Pretty-print JSON."""
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    print(f"MemHub server: {SERVER}\n")

    # 1. Health check
    print("── Health Check ──────────────────────────────")
    pp(health())

    # 2. Store some memories
    print("\n── Storing Memories ──────────────────────────")
    r1 = store("ACME Corp Q2 revenue grew 12% YoY to $4.2B.", "shared", ["finance"])
    print(f"Stored: {r1['memory_id']}  ({r1['token_count']} tokens)")

    r2 = store("Project Phoenix Phase 2 is on track for July delivery.", "shared", ["project"])
    print(f"Stored: {r2['memory_id']}  ({r2['token_count']} tokens)")

    r3 = store("Internal note: review budget allocation before Friday.", "private")
    print(f"Stored: {r3['memory_id']}  ({r3['token_count']} tokens)")

    # 3. Retrieve
    print("\n── Retrieving Memories ───────────────────────")
    results = retrieve("ACME revenue and financial performance")
    print(f"Found {results['total_results']} result(s) "
          f"(tier1={results['tier1_hits']}, tier2={results['tier2_hits']}, "
          f"latency={results['latency_ms']}ms)\n")

    for item in results["results"]:
        print(f"  [{item['tier'].upper()}] score={item['score']:.4f}  {item['content'][:80]}")

    print("\n── Done ─────────────────────────────────────")
