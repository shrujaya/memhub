# MemHub — Team Runbook

Step-by-step guide for installing, running, testing, and using MemHub.

---

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python | 3.12+ | `python3 --version` |
| pip | latest | `pip --version` |
| Git | any | `git --version` |
| Docker + Compose | 24+ / v2 | `docker compose version` _(only for Docker path)_ |
| curl | any | `curl --version` _(for testing)_ |

> [!IMPORTANT]
> You need **either** a local Python environment **or** Docker — not both. Pick the path that matches your setup.

---

## Path A: Local Development Setup

### Step 1 — Clone and enter the repo

```bash
git clone <repo-url>
cd memhub
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows PowerShell
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Expected output ends with `Successfully installed ...`. If `chromadb` fails on older pip, run `pip install --upgrade pip` first.

### Step 4 — Install Ollama (local LLM)

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Verify
ollama --version
```

### Step 5 — Pull a model

```bash
ollama pull llama3
```

This downloads ~4 GB. Wait for it to finish.

### Step 6 — Start Ollama (if not already running)

```bash
ollama serve &
```

Verify: `curl http://localhost:11434/api/tags` should return a JSON list with `llama3`.

### Step 7 — Start the MemHub server

```bash
bash scripts/run_server.sh
```

You should see:

```
[run_server] Starting MemHub on http://0.0.0.0:8000 ...
[run_server] Docs: http://localhost:8000/docs
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 8 — Verify

```bash
curl http://localhost:8000/v1/health
```

Expected:

```json
{"status":"ok","tier1_connected":true,"tier2_connected":true,"version":"0.1.0"}
```

> [!TIP]
> Open `http://localhost:8000/docs` in a browser to explore the full Swagger UI interactively.

---

## Path B: Docker Setup

### Step 1 — Clone and enter the repo

```bash
git clone <repo-url>
cd memhub
```

### Step 2 — Build and start

```bash
docker compose up -d --build
```

Wait for both containers to be healthy:

```bash
docker compose ps
```

Expected:

```
NAME              STATUS
memhub-server     Up (healthy)
memhub-ollama     Up
```

### Step 3 — Pull a model into the Ollama container

```bash
docker exec memhub-ollama ollama pull llama3
```

### Step 4 — Verify

```bash
curl http://localhost:8000/v1/health
```

### GPU support (optional)

If the machine has an NVIDIA GPU with the [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed, edit `docker-compose.yml` and uncomment the `deploy` block under the `ollama` service, then restart:

```bash
docker compose down && docker compose up -d
```

---

## Testing the Application

### Quick smoke test (30 seconds)

Run these three curl commands in order:

```bash
# 1. Health check
curl -s http://localhost:8000/v1/health | python3 -m json.tool

# 2. Store a memory
curl -s -X POST http://localhost:8000/v1/store \
  -H "Content-Type: application/json" \
  -H "X-Agent-ID: test-agent" \
  -d '{
    "agent_id": "test-agent",
    "content": "ACME Corp Q2 revenue grew 12% YoY to $4.2 billion.",
    "namespace": "shared",
    "metadata": {"tags": ["finance", "Q2"], "source": "research"}
  }' | python3 -m json.tool

# 3. Retrieve it
curl -s -X POST http://localhost:8000/v1/retrieve \
  -H "Content-Type: application/json" \
  -H "X-Agent-ID: test-agent" \
  -d '{
    "agent_id": "test-agent",
    "query": "ACME revenue",
    "top_k": 5,
    "include_shared": true
  }' | python3 -m json.tool
```

> [!NOTE]
> If all three return valid JSON with no errors, the installation is working correctly.

### Automated test script

Run the included test script for a more thorough check:

```bash
python tests/test_smoke.py
```

This script tests:
- Health check (Tier-1 + Tier-2 connectivity)
- Store a private memory
- Store a shared memory
- Retrieve by keyword query
- Retrieve by exact memory ID
- Cross-agent shared namespace access
- Policy sweep (eviction/demotion)
- Error handling (missing header, bad payload)

All tests print `✓ PASS` or `✗ FAIL` with details.

### Run the benchmark suite

```bash
bash scripts/run_evals.sh
```

This runs seven benchmark tasks (in-memory, no server needed) and generates charts in `eval/charts/`:

| Chart | What it shows |
|-------|---------------|
| `latency_percentiles.png` | p50/p90/p99 retrieve latency |
| `throughput.png` | Operations per second |
| `token_compression.png` | Before/after token counts from demotion |
| `tier_hit_rates.png` | Tier-1 vs Tier-2 hit distribution |
| `policy_comparison.png` | LRU vs FIFO vs LFU sweep cost |

### Comparative Benchmarking (MemHub vs Baseline)

To see the real-world impact of MemHub policies compared to a standard "flat" memory system, run the comparative suite:

```bash
export PYTHONPATH=$PYTHONPATH:.
python eval/run_comparison.py
```

This generates:
- `comparison_tokens.png`: Shows how MemHub prevents context overflow.
- `comparison_latency.png`: Compares retrieval speeds.
- `comparison_throughput.png`: Measures system overhead.

---

## Using MemHub

### From Python (recommended)

```python
import os
os.environ["MEMHUB_BASE_URL"] = "http://localhost:8000/v1"

from agents.tools import store_memory, query_team_memory

# Store
store_memory(
    agent_id="my-agent",
    content="The experiment yielded a 15% accuracy improvement.",
    namespace="shared",
    tags=["experiment", "results"],
)

# Retrieve
results = query_team_memory(
    agent_id="my-agent",
    query="experiment accuracy improvement",
    include_shared=True,
)
for r in results:
    print(f"  [{r['tier']}] score={r['score']:.3f}  {r['content'][:80]}")
```

### From a remote machine

```bash
export MEMHUB_SERVER=http://<server-ip>:8000
python client_example.py
```

Only `requests` is needed on the client: `pip install requests`.

### With AutoGen agents

```python
from agents.team_config import build_team
from agents.interceptor import attach_interceptors_to_team

# Build the team
team = build_team(task="Analyse ACME Corp Q2 earnings")

# Attach MemHub context injection to all agents
attach_interceptors_to_team(team, {
    "Orchestrator": "agent-orchestrator",
    "Researcher":   "agent-researcher",
    "Analyst":      "agent-analyst",
    "Critic":       "agent-critic",
})

# Start the conversation
team["orchestrator"].initiate_chat(
    team["manager"],
    message="Analyse ACME Corp Q2 earnings and identify key risks.",
)
```

Each agent will:
1. Automatically receive relevant memories injected into its context
2. Store findings by calling `store_memory` / `store_shared_finding` tools
3. Auto-persist `[REMEMBER: <fact>]` annotations from its responses
4. Have its working memory compacted by the PolicyEngine when it gets too large

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Connection refused` on :8000 | Server not running | `bash scripts/run_server.sh` or `docker compose up -d` |
| `tier2_connected: false` in health | ChromaDB init failed | Check `db/chroma_db/` permissions; restart server |
| `401 Missing X-Agent-ID` | No header in request | Add `-H "X-Agent-ID: <agent>"` to curl |
| `403 Forbidden` | Agent lacks namespace access | Auto-registration handles this; check `agent_acl` table |
| Summarisation returns original text | Ollama not running | `ollama serve &` or `docker exec memhub-ollama ollama list` |
| Summarisation fails silently | Incorrect model name | Ensure `core/summarization.py` is set to `llama3` (matches pulled model) |
| `ModuleNotFoundError: tiktoken` | Deps not installed | `pip install -r requirements.txt` |
| Docker build fails on `chromadb` | Needs gcc | Already handled in Dockerfile; try `docker compose build --no-cache` |
| Benchmarks fail on `demotion_compression` | No LLM available | Expected if Ollama is off; other benchmarks still pass |

---

## Stopping the Application

**Local:**

```bash
# Stop the server (Ctrl+C in the terminal, or)
kill $(lsof -ti:8000)

# Stop Ollama
killall ollama
```

**Docker:**

```bash
docker compose down          # Stop containers (data preserved)
docker compose down -v       # Stop and DELETE all data
```

---

## Team Contacts

| Role | Responsibility |
|------|----------------|
| Infrastructure Lead | `api/` — server, routes, auth, deployment |
| Memory Operations Lead | `core/` — policies, summarisation, retrieval |
| Multi-Agent Orchestration Lead | `agents/` — AutoGen team, tools, interceptor |
| Systems Performance Lead | `eval/` — benchmarks, metrics, charts |
