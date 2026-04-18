# MemHub

**Centralized Memory-as-a-Service for Multi-Agent Systems**

MemHub provides a two-tier shared memory store for AutoGen / LangGraph agent teams, with automatic eviction, demotion, promotion, and LLM-based summarisation policies.

| Tier | Store | Role |
|------|-------|------|
| **1** | SQLite | Working memory — fast keyword search, immediate agent context |
| **2** | ChromaDB | Long-term memory — semantic vector search, archived knowledge |

```
         ┌─────────────────────────────────────┐
         │        Agent Team (AutoGen)          │
         │  Orchestrator · Researcher · Analyst │
         └──────────────┬──────────────────────┘
                        │  REST API
                        ▼
         ┌──────────────────────────────────────┐
         │          MemHub API (FastAPI)         │
         │   /store · /retrieve · /policies/run  │
         └────────┬─────────────────┬───────────┘
                  │                 │
         ┌────────▼──────┐  ┌──────▼────────┐
         │  Tier 1       │  │  Tier 2        │
         │  SQLite       │  │  ChromaDB      │
         │  (working)    │  │  (long-term)   │
         └───────────────┘  └───────────────┘
```

**Python 3.12+**

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [File Reference](#file-reference)
  - [api/](#api--rest-api-layer)
  - [core/](#core--memory-engine)
  - [agents/](#agents--autogen-integration)
  - [eval/](#eval--benchmarking--visualisation)
  - [scripts/](#scripts--utilities)
  - [db/](#db--local-storage)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Running Benchmarks](#running-benchmarks)
- [Docker Deployment](#docker-deployment)

---

## Installation

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd memhub
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up a local LLM (required for summarisation)

MemHub calls a local, OpenAI-compatible LLM endpoint for memory summarisation. Install [Ollama](https://ollama.com/) and pull a model:

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3
```

Ollama serves on `http://localhost:11434` by default — no extra configuration needed.

---

## Quick Start

### Start the MemHub server

```bash
bash scripts/run_server.sh
```

This launches the FastAPI server on `http://localhost:8000` with auto-reload enabled. The SQLite database and ChromaDB storage are created automatically under `db/` on first run.

### Verify it's running

```bash
curl http://localhost:8000/v1/health
```

Expected response:

```json
{"status": "ok", "tier1_connected": true, "tier2_connected": true, "version": "0.1.0"}
```

### Store a memory

```bash
curl -X POST http://localhost:8000/v1/store \
  -H "Content-Type: application/json" \
  -H "X-Agent-ID: agent-01" \
  -d '{
    "agent_id": "agent-01",
    "content": "ACME Corp Q2 revenue grew 12% YoY to $4.2B.",
    "namespace": "shared",
    "metadata": {"tags": ["finance", "Q2"], "source": "research"}
  }'
```

### Retrieve memories

```bash
curl -X POST http://localhost:8000/v1/retrieve \
  -H "Content-Type: application/json" \
  -H "X-Agent-ID: agent-01" \
  -d '{
    "agent_id": "agent-01",
    "query": "ACME revenue",
    "top_k": 5,
    "include_shared": true
  }'
```

### Interactive API docs

Open `http://localhost:8000/docs` in a browser for the full Swagger UI.

---

## Project Structure

```
memhub/
├── .gitignore
├── requirements.txt
├── README.md
│
├── db/                       # Local storage (git-ignored)
│   ├── chroma_db/            # ChromaDB vector files (Tier 2)
│   └── memhub.db             # SQLite database (Tier 1 + ACLs)
│
├── api/                      # REST API layer
│   ├── __init__.py
│   ├── main.py               # FastAPI application entry point
│   ├── routes.py             # /store, /retrieve, /policies/run, /health endpoints
│   ├── models.py             # Pydantic request/response schemas
│   └── auth.py               # Agent identity & namespace ACL enforcement
│
├── core/                     # Memory engine
│   ├── __init__.py
│   ├── policies.py           # Eviction, promotion, and demotion policies
│   ├── summarization.py      # LLM-based memory compression
│   └── retrieval.py          # Hybrid Tier-1 + Tier-2 ranked search
│
├── agents/                   # AutoGen integration
│   ├── __init__.py
│   ├── team_config.py        # Agent role definitions & GroupChat setup
│   ├── tools.py              # Callable tool functions (store, query, etc.)
│   └── interceptor.py        # Transparent context injection & auto-store
│
├── eval/                     # Benchmarking & visualisation
│   ├── __init__.py
│   ├── benchmark_tasks.py    # Multi-step benchmark scenarios
│   ├── metrics.py            # Latency, throughput, and token usage tracking
│   └── visualize.py          # Chart generation (matplotlib)
│
└── scripts/                  # Utilities
    ├── run_server.sh         # Start the FastAPI service
    └── run_evals.sh          # Run benchmarks and generate charts
```

---

## File Reference

### `api/` — REST API Layer

#### `api/main.py` — Application Entry Point

- Creates and configures the **FastAPI application**.
- Manages the full application **lifespan**: opens SQLite (with WAL mode) and ChromaDB connections at startup, closes them at shutdown.
- Bootstraps the database schema automatically on first run — creates the `working_memory` and `agent_acl` tables.
- Mounts all routes under the `/v1` version prefix.
- Configures CORS middleware.
- Run with: `uvicorn api.main:app --reload` or `bash scripts/run_server.sh`.

#### `api/routes.py` — Endpoint Definitions

| Endpoint | Method | Description |
|---|---|---|
| `/v1/store` | POST | Write a memory to Tier-1 (SQLite). Optionally triggers the PolicyEngine (eviction/demotion) after the write. |
| `/v1/retrieve` | POST | Hybrid Tier-1 keyword + Tier-2 semantic search. Supports `include_shared` for cross-agent retrieval and `memory_id` for exact-ID lookup. |
| `/v1/policies/run` | POST | Manually trigger a full PolicyEngine sweep: Promote → Demote → Evict. Intended for admin use. |
| `/v1/memory/{id}` | GET | Fetch a single memory by UUID, searching Tier 1 then Tier 2. |
| `/v1/health` | GET | Liveness probe. Checks SQLite and ChromaDB connectivity. |

All endpoints require the `X-Agent-ID` header. The header value is validated against the ACL table.

#### `api/models.py` — Pydantic Schemas

Defines fully typed and validated request/response models:

- **`StoreRequest`** — `agent_id`, `content`, `namespace`, `metadata` (tags, source, priority), `run_policies` flag.
- **`RetrieveRequest`** — `agent_id`, `query`, `top_k`, `namespace`, `include_shared`, optional `tier_filter` and `memory_id`.
- **`PolicyRunRequest`** — `agent_id`, `namespace`, eviction `strategy` (LRU/FIFO/LFU).
- **`StoreResponse`** — returns the new `memory_id`, `token_count`, and whether policies were triggered.
- **`RetrieveResponse`** — ranked list of `MemoryItem` objects with `tier1_hits`, `tier2_hits`, `latency_ms`.
- **`PolicyRunResponse`** — per-policy stats: evicted/demoted/promoted counts, token compression %, summary preview.
- **`HealthResponse`** — `status`, `tier1_connected`, `tier2_connected`, `version`.

#### `api/auth.py` — ACL Enforcement

- Extracts and validates the **`X-Agent-ID` header** on every request.
- Maintains an **`agent_acl` SQLite table** mapping each agent to its permitted namespaces with read/write flags.
- **Auto-registers** unknown agents on first call so there's no separate registration step needed for development.
- Provides FastAPI dependency functions: `get_agent_id()`, `require_write_access(namespace)`, `require_read_access(namespace)`, `require_shared_read`.
- Can be fully disabled with `MEMHUB_DISABLE_AUTH=1` for testing.

---

### `core/` — Memory Engine

#### `core/summarization.py` — LLM-Based Compression

- **`count_tokens(text, model)`** — Returns the token count for a string using `tiktoken`. Falls back to `cl100k_base` encoding for local models.
- **`summarize_working_memory(agent_id, memories, threshold)`** — Async function that checks if the total tokens across a list of memories exceeds `threshold`. If so, it takes the **oldest 70%** of memories, sends them to the local LLM with a structured prompt, and returns the compressed summary. The **newest 30%** are retained untouched so the agent keeps immediate context.
- **`summarize_content(text)`** — Synchronous single-text summarisation using the Ollama Python client directly. Used by the store endpoint for inline compression.
- Makes async HTTP calls to a local OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`).

#### `core/policies.py` — Memory Lifecycle Policies

Implements three policy classes that govern how memories flow between tiers:

- **`EvictionPolicy`** — Destructively removes Tier-1 rows when the token budget is exceeded. Supports three strategies:
  - **LRU** — evicts by oldest `last_accessed` timestamp (best for conversational agents).
  - **FIFO** — evicts by oldest `created_at` timestamp (good baseline).
  - **LFU** — evicts by lowest `access_count` (preserves frequently-used knowledge).

- **`PromotionPolicy`** — Queries ChromaDB for Tier-2 chunks whose `access_count` has reached `PROMOTION_HIT_THRESHOLD` (default: 3). Moves them back into Tier-1 SQLite. Includes duplicate detection.

- **`DemotionPolicy`** — The full compress → archive → clean-up pipeline:
  1. Check token budget + minimum age guard.
  2. Call `summarize_working_memory()` to compress oldest 70%.
  3. Embed raw demoted chunks into ChromaDB (Tier 2).
  4. Insert the LLM summary as a new Tier-1 row.
  5. Delete originals from Tier 1.

- **`PolicyEngine`** — Unified entry point that runs the correct order: **Promote → Demote → Evict**. Called automatically after every `/store` call or manually via `/policies/run`.

Also defines shared data models: `MemoryRecord`, `PolicyResult`, `EvictionStrategy`.

#### `core/retrieval.py` — Hybrid Search

- **`MemoryRetriever`** — Unified search interface:
  - **Tier-1 search**: SQLite `LIKE` scan with a hybrid scoring formula: keyword match bonus + exponential recency decay + log-scaled access frequency.
  - **Tier-2 search**: ChromaDB `query_texts` cosine similarity, with distance-to-similarity conversion (`1 / (1 + distance)`).
  - **`_merge_and_rank()`**: Normalises each tier's scores to `[0,1]`, applies tier weights (60% Tier-1, 40% Tier-2), deduplicates (Tier-1 wins on ID collision), and returns sorted top-k.
  - **`_increment_access_counts()`**: Bumps `access_count` and `last_accessed` for all returned results in both SQLite and ChromaDB metadata. This is the **promotion hook** — when a Tier-2 chunk reaches the hit threshold, `PromotionPolicy` will promote it on the next sweep.
- **`retrieve_working_memory()`** — Full dump of all Tier-1 memories for an agent (used by DemotionPolicy).
- **`get_by_id()`** — Exact UUID lookup with ACL enforcement.

---

### `agents/` — AutoGen Integration

#### `agents/tools.py` — Callable Tool Functions

Six synchronous `requests`-based wrappers that AutoGen agents call as tools:

| Function | API Endpoint | Purpose |
|---|---|---|
| `store_memory()` | POST `/v1/store` | Store a private or shared memory |
| `query_team_memory()` | POST `/v1/retrieve` | Hybrid Tier-1 + Tier-2 search |
| `store_shared_finding()` | POST `/v1/store` | Convenience: always writes `namespace=shared` |
| `get_memory_by_id()` | GET `/v1/memory/{id}` | Exact UUID lookup |
| `trigger_policy_sweep()` | POST `/v1/policies/run` | Manual memory compaction |
| `check_memhub_health()` | GET `/v1/health` | Verify MemHub is reachable |

All functions set the `X-Agent-ID` header automatically and raise `RuntimeError` on failures.

#### `agents/team_config.py` — Team Definitions

- Defines four specialised **`ConversableAgent`** roles:
  - **Orchestrator** — breaks goals into subtasks, coordinates turn-taking, stores final conclusions.
  - **Researcher** — finds information, stores raw facts to the shared namespace.
  - **Analyst** — synthesises findings, performs quantitative reasoning, stores intermediate steps privately.
  - **Critic** — reviews conclusions, identifies gaps, issues APPROVED/REJECTED verdicts.
- **UserProxy** — executes tool calls on behalf of agents.
- All agents are registered with OpenAI-style tool schemas for the six MemHub functions.
- **`build_team(task)`** factory function: creates agents, registers tools, sets up `GroupChat` with `GroupChatManager`, and runs a MemHub health check before returning.
- LLM config reads from env vars (`MEMHUB_LLM_MODEL`, `MEMHUB_LLM_BASE_URL`), defaults to local Ollama.

#### `agents/interceptor.py` — Context Injection

- **`MemHubInterceptor`** — wraps an agent's `generate_reply` method to transparently:
  - **Pre-call**: fetch the top-k relevant memories from MemHub and inject them as a synthetic `system` message at the start of the prompt. The agent sees its memory context without making explicit tool calls.
  - **Post-call**: parse `[REMEMBER: <fact>]` tags from the LLM's output and auto-store each one via `store_memory()`.
  - **Token ceiling guard**: if injected context exceeds `CONTEXT_TOKEN_CEILING` (default 1500), silently triggers a policy sweep before injecting.
- **`attach_interceptors_to_team(team, agent_name_to_id)`** — convenience function to wire up interceptors for all speaking agents in a team at once.

---

### `eval/` — Benchmarking & Visualisation

#### `eval/metrics.py` — Performance Metrics

- **`MetricsCollector`** — collects per-operation timing samples:
  - `with collector.measure("store", tokens=n)` context manager for automatic latency recording.
  - `record_retrieval()`, `record_policy_sweep()`, `record_llm_call()` manual recording helpers.
  - **`summary()`** — computes `RunSummary` with p50/p90/p99/mean latencies per op type, tier hit rates, compression ratios, ops/sec.
  - **`save(path)`** — serialises all raw records + summary to JSON.

#### `eval/benchmark_tasks.py` — Benchmark Scenarios

Seven parameterised benchmark tasks using in-memory SQLite + ephemeral ChromaDB:

| Task | What It Tests |
|---|---|
| `single_agent_recall` | Store N facts → keyword retrieval accuracy |
| `multi_agent_shared_collab` | 4 agents cross-read via shared namespace |
| `long_session_eviction` (LRU) | Budget restoration after LRU eviction |
| `long_session_eviction` (FIFO) | Budget restoration after FIFO eviction |
| `long_session_eviction` (LFU) | Budget restoration after LFU eviction |
| `demotion_compression` | Token reduction ratio from LLM summarisation |
| `tiered_promotion` | Tier-2 → Tier-1 promotion after hit threshold |

- **`BenchmarkSuite`** — runs all tasks, prints a summary table, and saves JSON results.

#### `eval/visualize.py` — Chart Generation

Generates five publication-quality PNG charts from JSON result files:

| Chart | File |
|---|---|
| Retrieve latency p50/p90/p99 | `latency_percentiles.png` |
| Operations per second | `throughput.png` |
| Token reduction before/after | `token_compression.png` |
| Tier-1 vs Tier-2 hit counts | `tier_hit_rates.png` |
| LRU vs FIFO vs LFU sweep cost | `policy_comparison.png` |

Also runnable as a CLI: `python -m eval.visualize --results-dir eval/results --out-dir eval/charts`.

---

### `scripts/` — Utilities

#### `scripts/run_server.sh`

Starts the FastAPI server with sensible defaults:

```bash
bash scripts/run_server.sh                    # localhost:8000, auto-reload
PORT=9000 bash scripts/run_server.sh          # custom port
MEMHUB_DISABLE_AUTH=1 bash scripts/run_server.sh  # no ACL (testing)
```

Auto-activates `.venv` if present.

#### `scripts/run_evals.sh`

Runs the full benchmark suite and generates charts:

```bash
bash scripts/run_evals.sh
```

Results are saved to `eval/results/` (JSON) and `eval/charts/` (PNG).

---

### `db/` — Local Storage

Created automatically on first server start. **Git-ignored.**

| File | Description |
|---|---|
| `memhub.db` | SQLite database containing the `working_memory` and `agent_acl` tables (Tier 1). |
| `chroma_db/` | ChromaDB persistent storage directory for vector embeddings (Tier 2). |

---

## API Endpoints

All endpoints are versioned under `/v1` and require the `X-Agent-ID` header.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/store` | Store a memory in Tier-1 working memory |
| `POST` | `/v1/retrieve` | Hybrid keyword + semantic search across both tiers |
| `POST` | `/v1/policies/run` | Manually run Promote → Demote → Evict policies |
| `GET`  | `/v1/memory/{id}` | Fetch a single memory by UUID |
| `GET`  | `/v1/health` | Service health check |

Full interactive documentation: `http://localhost:8000/docs` (Swagger) or `http://localhost:8000/redoc` (ReDoc).

---

## Configuration

All settings are controlled via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MEMHUB_DB_PATH` | `db/memhub.db` | Path to the SQLite database file |
| `MEMHUB_CHROMA_PATH` | `db/chroma_db` | Path to the ChromaDB storage directory |
| `MEMHUB_DISABLE_AUTH` | `0` | Set to `1` to bypass ACL checks (dev only) |
| `MEMHUB_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `MEMHUB_BASE_URL` | `http://localhost:8000/v1` | Base URL for agent tool functions |
| `MEMHUB_TIMEOUT` | `15` | HTTP timeout (seconds) for tool calls |
| `MEMHUB_LLM_MODEL` | `llama3` | Model name for AutoGen LLM calls |
| `MEMHUB_LLM_BASE_URL` | `http://localhost:11434/v1` | LLM API endpoint |
| `MEMHUB_LLM_API_KEY` | `ollama` | API key for the LLM endpoint |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

---

## Running Benchmarks

```bash
# 1. Make sure MemHub server is NOT required for benchmarks
#    (benchmarks use in-memory SQLite + ephemeral ChromaDB)

# 2. Run the full suite
bash scripts/run_evals.sh

# 3. View results
ls eval/results/     # JSON files per task
ls eval/charts/      # PNG charts

# Or run from Python:
python -c "
import asyncio
from eval.benchmark_tasks import BenchmarkSuite
asyncio.run(BenchmarkSuite().run_all())
"
```

> **Note:** The `demotion_compression` benchmark requires Ollama to be running for LLM-based summarisation. All other benchmarks run without an LLM.

---

## Docker Deployment

MemHub ships with a `Dockerfile` and `docker-compose.yml` for deploying the **server** on a remote machine while running **clients** (agents, scripts, notebooks) separately from any other machine.

### Architecture

```
┌───────────────────────────────────────────┐
│          Server Machine (Docker)          │
│                                           │
│  ┌─────────────┐    ┌──────────────────┐  │
│  │  memhub     │    │  ollama          │  │
│  │  (FastAPI)  │◄──►│  (local LLM)     │  │
│  │  :8000      │    │  :11434          │  │
│  └──────┬──────┘    └──────────────────┘  │
│         │  volumes: memhub-data,          │
│         │           ollama-models         │
└─────────┼─────────────────────────────────┘
          │  HTTP (port 8000)
          ▼
┌─────────────────────┐
│   Client Machine    │
│   agents/tools.py   │
│   client_example.py │
│   notebooks, etc.   │
└─────────────────────┘
```

### 1. Deploy the server

On the server machine:

```bash
# Clone the repo
git clone <repo-url>
cd memhub

# Start MemHub + Ollama
docker compose up -d

# Pull an LLM model into the Ollama container
docker exec memhub-ollama ollama pull llama3

# Verify
curl http://localhost:8000/v1/health
```

Both services start automatically on reboot (`restart: unless-stopped`).

### 2. Customise with environment variables

Override defaults via a `.env` file next to `docker-compose.yml`:

```bash
# .env
MEMHUB_PORT=8000
OLLAMA_PORT=11434
WORKERS=2
LOG_LEVEL=info
MEMHUB_DISABLE_AUTH=0
MEMHUB_CORS_ORIGINS=*
```

### 3. Enable GPU (NVIDIA)

If the server has an NVIDIA GPU, uncomment the `deploy` block in `docker-compose.yml` under the `ollama` service:

```yaml
ollama:
  ...
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### 4. Run the client remotely

On the **client machine**, you only need `requests` (no Docker required):

```bash
pip install requests
```

Set the `MEMHUB_SERVER` environment variable to point at the server:

```bash
export MEMHUB_SERVER=http://<server-ip>:8000
```

#### Option A: Use the example client

```bash
python client_example.py
```

This script demonstrates health checks, storing memories, and retrieving them.

#### Option B: Use the agent tools directly

```python
import os
os.environ["MEMHUB_BASE_URL"] = "http://<server-ip>:8000/v1"

from agents.tools import store_memory, query_team_memory, check_memhub_health

check_memhub_health()

store_memory(
    agent_id="my-agent",
    content="The experiment results show a 15% improvement.",
    namespace="shared",
    tags=["experiment"],
)

results = query_team_memory(
    agent_id="my-agent",
    query="experiment results",
    include_shared=True,
)
print(results)
```

#### Option C: Plain curl

```bash
# Store
curl -X POST http://<server-ip>:8000/v1/store \
  -H "Content-Type: application/json" \
  -H "X-Agent-ID: agent-01" \
  -d '{"agent_id": "agent-01", "content": "Important finding.", "namespace": "shared"}'

# Retrieve
curl -X POST http://<server-ip>:8000/v1/retrieve \
  -H "Content-Type: application/json" \
  -H "X-Agent-ID: agent-01" \
  -d '{"agent_id": "agent-01", "query": "important", "top_k": 5}'
```

### Docker commands reference

```bash
docker compose up -d              # Start all services
docker compose up -d memhub       # Start server only (bring your own LLM)
docker compose logs -f memhub     # Tail server logs
docker compose ps                 # Check running containers
docker compose down               # Stop everything
docker compose down -v            # Stop and delete all data volumes
docker compose build --no-cache   # Rebuild after code changes
```

### Data persistence

Two named Docker volumes are used:

| Volume | Container path | Contents |
|---|---|---|
| `memhub-data` | `/data/` | `memhub.db` (SQLite) + `chroma_db/` (ChromaDB) |
| `ollama-models` | `/root/.ollama/` | Downloaded LLM model weights |

Data survives `docker compose down` and container rebuilds. Use `docker compose down -v` to wipe everything.