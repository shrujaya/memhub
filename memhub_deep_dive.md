# MemHub — Deep Dive: How It Works

---

## The Problem MemHub Solves

When you run a team of LLM-powered agents (via AutoGen, LangGraph, CrewAI, etc.), every agent has a **context window** — the text the LLM can see during one inference call. This creates three problems:

1. **Context overflow** — a long-running agent accumulates so many observations, decisions, and tool outputs that its context window fills up. Older information silently drops off the end, and the agent "forgets."

2. **No cross-agent memory** — Agent A discovers a critical fact, but Agent B has no idea. Each agent's context is its own isolated bubble. There's no shared knowledge base.

3. **No persistence** — if you restart the system, everything is gone. Agents can't pick up where they left off.

MemHub solves all three by acting as a **centralised memory service** that sits between the agents and their LLM calls:

```
┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Orchestrator│  │Researcher│  │ Analyst  │  │  Critic  │
└─────┬──────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘
      │               │             │             │
      └───────────────┼─────────────┼─────────────┘
                      │    HTTP API
                      ▼
              ┌──────────────┐
              │   MemHub     │
              │   (FastAPI)  │
              └──────┬───────┘
           ┌─────────┴─────────┐
           ▼                   ▼
    ┌────────────┐      ┌────────────┐
    │  Tier 1    │      │  Tier 2    │
    │  SQLite    │      │  ChromaDB  │
    │  (fast,    │      │  (semantic │
    │   exact)   │      │   vector)  │
    └────────────┘      └────────────┘
```

---

## The Two-Tier Storage Architecture

The core design insight is that not all memories are equally valuable at all times. MemHub splits storage into two tiers optimised for different access patterns:

### Tier 1 — SQLite (Working Memory)

| Property | Detail |
|----------|--------|
| **What's stored** | An agent's immediately relevant context: recent observations, tool outputs, decisions, LLM-generated summaries |
| **How it's searched** | Keyword matching (SQL `LIKE`) with a hybrid relevance score |
| **Access speed** | Sub-millisecond (local file, in-process) |
| **Schema** | `working_memory` table: `id`, `agent_id`, `content`, `created_at`, `last_accessed`, `access_count`, `namespace` |
| **Analogy** | A person's short-term / working memory — what you're actively thinking about |

### Tier 2 — ChromaDB (Long-Term Memory)

| Property | Detail |
|----------|--------|
| **What's stored** | Archived memories that were demoted from Tier 1 after being summarised. Raw historical text that's too old/large for the context window but might be useful later. |
| **How it's searched** | Semantic vector similarity (cosine distance on embeddings) |
| **Access speed** | ~5–50ms depending on collection size |
| **Schema** | ChromaDB collection `memhub_longterm`: document + metadata (agent_id, namespace, access_count, created_at) |
| **Analogy** | A person's long-term memory — facts you can recall if prompted, but aren't actively thinking about |

### Why two tiers?

A single vector database would work, but it's suboptimal:
- Vector search is slower than SQLite for exact/keyword matches on recent data.
- You'd still need a way to manage context window size — you can't dump 10,000 ChromaDB results into a prompt.
- The two-tier design lets the **PolicyEngine** make intelligent decisions about what stays "hot" and what gets archived.

---

## Key Features

### Feature 1: Automatic Memory Lifecycle (PolicyEngine)

This is the heart of MemHub. After every `/store` call, the **PolicyEngine** runs three policies in order:

```
Promote  →  Demote  →  Evict
```

#### 1. Promotion (Tier 2 → Tier 1)

**What**: If an agent keeps retrieving the same Tier-2 chunk over and over, it's clearly important — promote it back to Tier 1 for faster access.

**How it works**:
- Every time a Tier-2 chunk is returned in a `/retrieve` response, its `access_count` is incremented in ChromaDB metadata.
- When `access_count >= 3` (configurable), `PromotionPolicy` moves it back to SQLite.
- The original Tier-2 copy is deleted to avoid duplication.

**Why it runs first**: If we evicted before promoting, we might throw away a Tier-1 row that's about to be replaced by a promoted chunk — wasting the promotion.

#### 2. Demotion (Tier 1 → Tier 2, with LLM summarisation)

**What**: When an agent's working memory exceeds the token budget (default: 2,000 tokens), compress the oldest memories using an LLM and archive the originals.

**How it works**:
1. Load all Tier-1 rows for the agent, ordered oldest-first.
2. Check total tokens. If under budget, skip.
3. Check minimum age (default: 60s). Brand-new verbose memories aren't eligible — they might be highly relevant despite being large.
4. Take the **oldest 70%** of memories and send them to the local LLM with a prompt like:

   ```
   Summarize the following chronological events for Agent researcher-01.
   Focus on key decisions, facts, relationships, and entities.
   Discard redundant conversational filler.
   ```

5. The LLM returns a compressed summary (e.g., 2000 tokens → 200 tokens).
6. **Embed the raw original chunks into ChromaDB** (Tier 2) — they're archived, not deleted.
7. **Insert the summary as a new Tier-1 row** — the agent now sees a compact version.
8. **Delete the original rows from SQLite** — they've been replaced by the summary.

The **newest 30%** of memories are left completely untouched — the agent keeps its most recent context intact.

**Why this matters**: Without this, an agent's context window fills up and old information silently disappears. With demotion, old information is *compressed* (not lost) and the originals are searchable via semantic vector search in Tier 2.

#### 3. Eviction (destructive removal from Tier 1)

**What**: A last-resort cleanup when the token budget is still exceeded after demotion. Eviction is **destructive** — rows are deleted without archiving.

**Three strategies** (configurable per agent):

| Strategy | Evicts by | Best for |
|----------|-----------|----------|
| **LRU** (Least Recently Used) | Oldest `last_accessed` timestamp | Conversational agents that revisit recent topics |
| **FIFO** (First In First Out) | Oldest `created_at` timestamp | Simple baseline; good for benchmarking |
| **LFU** (Least Frequently Used) | Lowest `access_count` | Research agents that build on frequently-cited facts |

The policy walks rows in order, removing them one-by-one until the token budget is satisfied.

---

### Feature 2: Hybrid Retrieval (Keyword + Semantic Search)

When an agent calls `/retrieve`, MemHub searches **both tiers simultaneously** and merges the results:

```
Agent query: "What is ACME's Q2 revenue?"
       │
  ┌────┴─────────────────────────────┐
  ▼                                  ▼
Tier-1 (SQLite)                Tier-2 (ChromaDB)
SQL LIKE keyword scan          Cosine similarity on embeddings
  │                                  │
  │ Score = keyword_hits             │ Score = 1 / (1 + distance)
  │       + recency_decay            │
  │       + log(access_count)        │
  └────────────┬─────────────────────┘
               ▼
        _merge_and_rank()
        normalize each tier to [0,1]
        weight: Tier-1 × 0.6, Tier-2 × 0.4
        deduplicate (Tier-1 wins on ID collision)
        sort descending, take top-k
               │
               ▼
        Increment access_count on all results
        (feeds PromotionPolicy)
               │
               ▼
        Return ranked MemoryItem list
```

**Why a 60/40 Tier-1 weighting?** Tier-1 memories are the agent's active context — they're more immediately relevant. Tier-2 results are valuable but represent archived history. The weights can be tuned.

**Why normalize before merging?** Tier-1 scores are additive (keyword count + recency + frequency) and unbounded. Tier-2 scores are cosine similarities in (0, 1]. Without normalization, one tier would dominate.

---

### Feature 3: Namespace-Based Access Control

Every memory belongs to a **namespace**:

| Namespace | Visibility | Use case |
|-----------|-----------|----------|
| `private` | Only the owning agent can read/write | Personal scratchpad, intermediate reasoning, draft conclusions |
| `shared` | All agents in the team can read; only the owner can write | Broadcasting findings, team-wide facts, coordination signals |

The `agent_acl` SQLite table maps each agent to its permitted namespaces with `can_read` / `can_write` flags. New agents are **auto-registered** on first contact (for dev ergonomics).

**Example scenario**: The Researcher stores a finding as `shared`:
```python
store_memory(agent_id="researcher", content="ACME Q2 revenue = $4.2B", namespace="shared")
```
The Analyst can now retrieve it:
```python
query_team_memory(agent_id="analyst", query="ACME revenue", include_shared=True)
```
The Researcher's private reasoning steps remain invisible to the Analyst. MemHub enforces this by filtering Tier-1 and Tier-2 results: if `include_shared=True`, it returns `private` records for the requesting agent PLUS all `shared` records from any agent.

---

### Feature 4: Transparent Context Injection (Interceptor)

The `MemHubInterceptor` wraps an AutoGen agent's `generate_reply` method so that MemHub works **without the agent needing to explicitly call tools every turn**:

```
Normal AutoGen flow:
  User message → LLM → reply

With MemHubInterceptor:
  User message
       ↓
  [PRE-CALL] fetch top-5 relevant memories from MemHub
             inject as a system message:
             "## MemHub Working Memory (top-5 relevant chunks)
              [TIER1 | score=0.82] ACME Q2 revenue grew 12%...
              [TIER2 | score=0.45] Historical: ACME Q1 was $3.8B..."
       ↓
  LLM sees: system prompt + injected memories + user message
       ↓
  LLM reply: "Based on the data, revenue grew 12%...
              [REMEMBER: ACME showed consistent YoY growth pattern]"
       ↓
  [POST-CALL] regex extracts "[REMEMBER: ...]" annotations
              auto-stores each as a new private memory
              (no explicit tool call needed by the LLM)
```

The `[REMEMBER: <fact>]` pattern gives the LLM a natural way to say "I want to remember this" without making a formal tool call. The interceptor silently persists it.

**Token ceiling guard**: If the injected context would exceed 1,500 tokens, the interceptor silently triggers a `trigger_policy_sweep` before injecting — ensuring the agent never receives an overflowing context.

---

### Feature 5: Token-Aware Management

MemHub uses `tiktoken` (the same tokenizer OpenAI uses) to count tokens precisely. This is important because:

- The **token budget** (default: 2,000) is measured in actual LLM tokens, not characters or words. A 2,000-token budget maps directly to how much context the LLM can see.
- The **summarisation threshold** uses the same tokenizer, so the system knows exactly when compression will save meaningful context space.
- The **compression ratio** metric (in benchmarks) measures real token reduction, not just character counts.

---

### Feature 6: Benchmarking & Visualisation

The `eval/` module provides a complete benchmarking framework:

#### MetricsCollector

Every operation goes through a `with collector.measure("store")` context manager that records:
- Wall-clock latency (ms) at nanosecond precision
- Token count of the content
- Which tier served the result
- Success/failure status

From these raw samples, it computes: p50/p90/p99/mean latencies per operation type, ops/second, tier hit rates, and compression ratios — all without NumPy (stdlib only).

#### Seven Benchmark Tasks

Each task creates an **isolated in-memory environment** (no server needed):

| # | Task | What it validates |
|---|------|-------------------|
| 1 | **Single Agent Recall** | Store 20 facts → keyword retrieve top-5 → verify accuracy |
| 2 | **Multi-Agent Shared Collaboration** | 4 agents write to `shared` → each reads all others → verify cross-read count |
| 3 | **Long Session Eviction (LRU)** | Fill 60 memories past budget → LRU eviction → verify budget restored |
| 4 | **Long Session Eviction (FIFO)** | Same as #3 but FIFO strategy |
| 5 | **Long Session Eviction (LFU)** | Same as #3 but LFU strategy |
| 6 | **Demotion Compression** | 30 old memories → DemotionPolicy → measure token compression ratio |
| 7 | **Tiered Promotion** | Seed a Tier-2 chunk at the hit threshold → PromotionPolicy → verify it appears in Tier 1 |

#### Five Generated Charts

| Chart | What it shows |
|-------|---------------|
| Latency Percentiles | p50/p90/p99 grouped bars per task — identifies tail latency issues |
| Throughput | ops/second horizontal bars — identifies bottlenecks |
| Token Compression | Before/after paired bars with reduction % — proves summarisation works |
| Tier Hit Rates | Stacked Tier-1/Tier-2 bars — shows where data actually lives |
| Policy Comparison | LRU vs FIFO vs LFU sweep latency — helps choose a strategy |

#### 8. Baseline Comparison (Comparative Mode)

A specialized benchmarking mode that runs the suite with policies **disabled**. This serves as the "Control" group to measure the absolute benefits of MemHub's tiered architecture against a standard flat-file memory system. It visualizes the "sawtooth" token compression pattern that prevents context overflow.

---

## End-to-End Scenarios

### Scenario 1: Simple Store → Retrieve

The most basic flow. A single agent writes a memory and reads it back.

```
Agent-01 ──POST /store──► MemHub ──INSERT──► SQLite (Tier 1)
                                                 │
Agent-01 ──POST /retrieve──► MemHub ──SELECT──► SQLite
                                     ──query──► ChromaDB (nothing found)
                                     ──merge──► return 1 result
```

**What happens under the hood**:
1. `/store` validates the payload via Pydantic, checks ACL, generates a UUID, inserts into SQLite.
2. `PolicyEngine.run_all()` fires: Promote (no candidates), Demote (under budget), Evict (under budget) → all skip.
3. `/retrieve` searches Tier 1 with keyword matching, Tier 2 with vector search (empty), merges, increments `access_count`, returns.

---

### Scenario 2: Multi-Agent Research Collaboration

Four agents working on a financial analysis task:

```
                    ┌─────────────────────────────────┐
                    │         Orchestrator             │
                    │  "Break this into subtasks"      │
                    └──────────┬──────────────────────┘
                               │ assigns
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
         ┌──────────┐   ┌──────────┐   ┌──────────┐
         │Researcher│   │ Analyst  │   │  Critic  │
         │  finds   │   │synthesises│  │  reviews │
         │  facts   │   │ findings │   │  quality │
         └────┬─────┘   └────┬─────┘   └────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    MemHub (shared namespace)
```

**Turn-by-turn walkthrough**:

1. **Orchestrator** calls `query_team_memory("past ACME analysis")` → empty results. Assigns: "Researcher: find ACME Q2 financials. Analyst: wait for data."

2. **Researcher** discovers ACME Q2 revenue = $4.2B. Calls `store_shared_finding(content="ACME Q2 revenue=$4.2B", tags=["finance"])`. MemHub returns `memory_id=abc-123`.

3. **Analyst** calls `query_team_memory("ACME Q2 financials")` → gets the Researcher's finding from `shared` namespace. Performs analysis, stores conclusion: `store_shared_finding(content="ACME revenue grew 12% YoY, above sector avg of 8%")`.

4. **Critic** calls `query_team_memory("ACME revenue analysis")` → gets both findings. Cross-checks. Calls `store_shared_finding(content="APPROVED: Revenue analysis is consistent with public filings")`.

5. **Orchestrator** reads all shared memories, compiles final report.

**Key MemHub behaviours during this scenario**:
- Each `store_shared_finding` triggers `PolicyEngine.run_all()`, but the budget is unlikely to be exceeded in a short task.
- The `include_shared=True` flag in retrieve calls is what enables cross-agent visibility.
- Each agent's private reasoning (stored as `namespace=private`) remains invisible to others.

---

### Scenario 3: Long Session with Memory Pressure

An agent runs for hundreds of turns, accumulating observations:

```
Turn 1-50:    Agent stores 50 observations    (total: ~3,000 tokens)
              PolicyEngine fires at turn ~30:
                → DemotionPolicy activates (over 2,000-token budget)
                → Takes oldest 70% (21 memories)
                → LLM summarises: "Turns 1-21: Agent explored dataset,
                   found 3 outliers in columns A/C/F, decided to use
                   median imputation..."  (~100 tokens)
                → Raw 21 memories → embedded in ChromaDB
                → Summary → inserted in SQLite
                → Agent now has: summary + turns 22-30 (~600 tokens)

Turn 50-100:  More observations. Budget exceeded again.
              → DemotionPolicy runs again, compresses turns 22-60.
              → Original summary + new summary + turns 61-100 in Tier 1
              → Everything older is in Tier 2 (searchable by meaning)

Turn 100:     Agent queries "what outliers did I find earlier?"
              → Tier-1 keyword search: finds summary mentioning "outliers"
              → Tier-2 semantic search: finds original raw observations
                about columns A, C, F from turns 5-15
              → Merged result: agent recovers detail it "compressed" away
```

**This is the core value proposition**: the agent never loses information. It just gets compressed in Tier 1 and archived in Tier 2. A good semantic query can recover details from hundreds of turns ago.

---

### Scenario 4: Hot Memory Promotion

A Tier-2 chunk keeps getting retrieved:

```
Turn 1:   Memory "ACME Q2 = $4.2B" is demoted to Tier 2 (old, compressed)

Turn 20:  Agent asks about revenue → Tier-2 search returns chunk
          access_count: 0 → 1

Turn 35:  Agent asks about financials → chunk returned again
          access_count: 1 → 2

Turn 50:  Agent asks about quarterly data → chunk returned again
          access_count: 2 → 3  ← hits PROMOTION_HIT_THRESHOLD

Turn 51:  PolicyEngine.run_all() fires
          PromotionPolicy finds chunk with access_count=3
          → Copies it back to SQLite (Tier 1)
          → Deletes from ChromaDB (Tier 2)
          → Now it's in fast keyword-search path again
```

---

### Scenario 5: Eviction Strategy Comparison (Benchmarking)

You want to know which eviction strategy works best for your agents:

```bash
bash scripts/run_evals.sh
```

This runs `long_session_eviction` three times with LRU, FIFO, and LFU, then generates `policy_comparison.png`:

```
┌──────────────────────────────────────────────────┐
│  Eviction Policy Sweep Latency: LRU vs FIFO vs LFU │
│                                                   │
│  LRU   ████████████████  12.3 ms                  │
│  FIFO  ██████████████    10.1 ms                  │
│  LFU   ████████████████████  15.7 ms              │
│                                                   │
│  (LFU is slowest because it sorts by access_count │
│   which requires scanning more metadata)          │
└──────────────────────────────────────────────────┘
```

The benchmark also measures:
- Whether each strategy successfully restores the token budget (pass/fail)
- How many rows were evicted to reach the budget
- Tokens before vs tokens after

---

### Scenario 6: Remote Deployment (Server + Client Split)

```
┌───────────────────────────┐       ┌──────────────────────────┐
│     Server (Docker)       │       │   Client (any machine)   │
│                           │       │                          │
│  docker compose up -d     │       │  pip install requests    │
│  memhub:8000 ◄────────────────────── MEMHUB_SERVER=http://.. │
│  ollama:11434             │  HTTP │  python client_example.py│
│                           │       │  or: agents/tools.py     │
│  volumes:                 │       │  or: AutoGen team        │
│    memhub-data (SQLite)   │       │                          │
│    ollama-models (LLM)    │       │  No Docker needed.       │
└───────────────────────────┘       └──────────────────────────┘
```

The client only needs `requests` (or `curl`). All storage, policy execution, and LLM summarisation happen server-side. The client is stateless.

---

## How Each File Contributes (Execution Flow)

Here's a single `/store` request traced through every file it touches:

```
Client
  │
  ├─ POST /v1/store { agent_id, content, namespace }
  │  Header: X-Agent-ID: agent-01
  │
  ▼
api/main.py
  │  app = create_app()  ← lifespan opened db + chroma
  │  router mounted at /v1
  │
  ▼
api/routes.py → store_memory()
  │  1. Depends(get_agent_id) → extracts X-Agent-ID header
  │  2. Validates payload matches header
  │
  ▼
api/auth.py → ensure_agent_registered()
  │  Checks agent_acl table; auto-inserts if new
  │
  ▼
api/routes.py (continued)
  │  3. count_tokens(content) → via core/summarization.py
  │  4. INSERT INTO working_memory
  │  5. if run_policies: PolicyEngine(db, chroma).run_all()
  │
  ▼
core/policies.py → PolicyEngine.run_all()
  │
  ├─ PromotionPolicy.run()
  │    → queries ChromaDB for access_count >= 3
  │    → moves matching chunks to SQLite
  │
  ├─ DemotionPolicy.run()
  │    → loads Tier-1 rows, checks token budget
  │    → if over budget:
  │        ▼
  │      core/summarization.py → summarize_working_memory()
  │        → count_tokens() per memory
  │        → split 70/30 oldest/newest
  │        → _fetch_summary_from_llm()
  │            → POST http://localhost:11434/v1/chat/completions
  │            → LLM returns compressed summary
  │        → return { needs_update, new_summary, retained, demoted }
  │        ▼
  │      DemotionPolicy (continued)
  │        → _embed_to_tier2(demoted_records)  ← ChromaDB .add()
  │        → _insert_summary_to_tier1(summary) ← SQLite INSERT
  │        → _delete_from_tier1(demoted_ids)   ← SQLite DELETE
  │
  ├─ EvictionPolicy.run()
  │    → if still over budget: walk rows in LRU/FIFO/LFU order
  │    → DELETE until budget satisfied
  │
  ▼
api/routes.py → return StoreResponse
  { memory_id, agent_id, token_count, policies_triggered }
```

---

## Summary Table

| Feature | Files Involved | Purpose |
|---------|---------------|---------|
| REST API | `api/main.py`, `api/routes.py`, `api/models.py` | HTTP interface for store/retrieve |
| ACL Enforcement | `api/auth.py` | Agent identity + namespace permissions |
| Working Memory | `core/retrieval.py` (Tier 1 search) | Fast keyword + recency search in SQLite |
| Long-Term Memory | `core/retrieval.py` (Tier 2 search) | Semantic vector search in ChromaDB |
| Hybrid Merge | `core/retrieval.py` (`_merge_and_rank`) | Normalize, weight, dedup, sort |
| Token Counting | `core/summarization.py` (`count_tokens`) | Precise tiktoken-based token counting |
| LLM Summarisation | `core/summarization.py` (`summarize_working_memory`) | Compress oldest 70% via local LLM |
| Eviction (LRU/FIFO/LFU) | `core/policies.py` (`EvictionPolicy`) | Destructive Tier-1 cleanup |
| Demotion | `core/policies.py` (`DemotionPolicy`) | Compress → archive → clean pipeline |
| Promotion | `core/policies.py` (`PromotionPolicy`) | Tier-2 → Tier-1 for hot chunks |
| Policy Orchestration | `core/policies.py` (`PolicyEngine`) | Promote → Demote → Evict in order |
| Agent Tools | `agents/tools.py` | `requests`-based tool functions for AutoGen |
| Team Config | `agents/team_config.py` | 4-agent AutoGen team with tool schemas |
| Context Injection | `agents/interceptor.py` | Auto-inject memories + auto-store [REMEMBER] |
| Metrics | `eval/metrics.py` | Latency/throughput/compression tracking |
| Benchmarks | `eval/benchmark_tasks.py` | 7 parameterised test scenarios |
| Charts | `eval/visualize.py` | 5 matplotlib chart generators |
