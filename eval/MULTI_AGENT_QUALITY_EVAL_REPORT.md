# Multi-Agent Quality Evaluation — Report

This document interprets the scenarios defined in
[multi_agent_quality_eval.py](multi_agent_quality_eval.py) and the supporting
JSON artefacts under [results/memhub/](results/memhub/) and
[results/baseline/](results/baseline/).

The quality-eval script is a **live, end-to-end** test harness: it boots the
full AutoGen team (Orchestrator / Researcher / Analyst / Critic) against a
running MemHub server, drives each scenario through a `GroupChat`, and decides
pass/fail by inspecting the final chat messages. Unlike
[benchmark_tasks.py](benchmark_tasks.py) (which measures latency/throughput on
an in-memory fixture), this suite measures **behavioural correctness** — did
the team *actually* use memory correctly to solve the task?

---

## 1. Test Harness Overview

### What's running

- **MemHub server** — FastAPI on `:8000`, SQLite (Tier 1) + ChromaDB (Tier 2).
- **Local LLM** — Ollama serving `llama3.1` at `:11434` (quality eval default).
- **AutoGen team** — four `ConversableAgent`s wired via `GroupChatManager`.
- **Interceptor** — [MemHubInterceptor](../agents/interceptor.py) injects the
  top-k relevant memories into each agent's prompt **before** every
  `generate_reply`, and parses `[REMEMBER: …]` tags from the reply to persist
  them **after**.

### Why tools are aggressively disabled

[multi_agent_quality_eval.py:36-76](multi_agent_quality_eval.py#L36-L76) shows
`_disable_tools()` — it strips `tools` / `functions` from every agent's
`llm_config`, **re-initialises** each agent's `OpenAIWrapper` client, and
rewrites system prompts so that phrases like `call \`query_team_memory\``
become "rely on the 'MemHub Working Memory' context provided below".

**Why this matters.** Ollama-served models (llama3.1) do not reliably support
the OpenAI function-calling API that AutoGen expects. If tool schemas are left
in place the model either emits malformed tool calls or refuses to answer. By
forcing the agents to go through the **interceptor path** only, the eval
isolates MemHub's *memory-as-context* behaviour from the model's tool-calling
ability. Every scenario therefore tests one thing: **given a correctly-injected
memory context, can the team produce the right answer?**

### How pass/fail is decided

Each scenario builds a fresh `build_team(...)`, seeds MemHub with a known
fact, runs `a_initiate_chat`, and then scans `group_chat.messages` for an
exact string (e.g. the project code, the ingredient, the status word). A
`ScenarioResult` dataclass captures `passed`, `expected`, `actual`, and a
single numeric metric in `[0.0, 1.0]`.

---

## 2. Scenario-by-Scenario Report

### Scenario 1 — Hidden Fact Retrieval

**Code:**
[multi_agent_quality_eval.py:78-112](multi_agent_quality_eval.py#L78-L112)

**What it tests.** A random project code (e.g. `PHOENIX-4F2A17`) is written
to an agent's *private* working memory **before** the team is built. The task
then asks the team to state that code. The code is chosen so it cannot be
guessed or confabulated — if the agent produces it, the fact must have flowed
through MemHub.

**What makes it unique.** This is the atomic "can MemHub recall a single
specific datum" test. No other scenario strips the problem down to one fact,
one agent, and no collaboration — so a failure here localises the bug to the
retrieval + injection path rather than to cross-agent coordination.

**How it's evaluated.** `project_code in final_reply` — a substring match on
the last message of the group chat. `recall_accuracy` ∈ `{0.0, 1.0}`.

**Expected result & impact.** PASS proves the `MemHubInterceptor` pre-call
hook is correctly:
1. Hitting `/v1/retrieve` with the agent's id,
2. Ranking the seeded Tier-1 row above noise,
3. Injecting it into the system prompt before the model runs.

A FAIL here invalidates every downstream scenario — if a single private fact
cannot round-trip, cross-agent handover and conflict resolution are
meaningless.

---

### Scenario 2 — Cross-Agent Handover

**Code:**
[multi_agent_quality_eval.py:114-154](multi_agent_quality_eval.py#L114-L154)

**What it tests.** The **Researcher** writes a whimsical fact ("the secret
ingredient is Glow-in-the-dark Mushrooms") to the `shared` namespace. The
**Analyst** — a distinct agent id — is then asked to recover the ingredient
and propose a recipe. The two agents never share a context window; the fact
can only travel through MemHub's shared-namespace ACL.

**What makes it unique.** This is the only scenario that crosses an
**agent-id boundary** via the `shared` namespace. It exercises:
- The ACL path in [api/auth.py](../api/auth.py) — the Analyst's `X-Agent-ID`
  must resolve to read permissions on `shared`.
- The `include_shared=True` branch in
  [core/retrieval.py](../core/retrieval.py) — shared rows from *other* agents
  must be mixed into the Analyst's Tier-1 results.
- The interceptor's ability to inject *someone else's* memory into the
  current agent's prompt.

**How it's evaluated.** The eval walks `group_chat.messages` in reverse and
picks the last message whose `name == "Analyst"`, then checks
`ingredient.lower() in final_reply.lower()`. `collaboration_score`
∈ `{0.0, 1.0}`.

**Supporting quantitative data — `results/memhub/multi_agent_shared_collab.json`.**
The `benchmark_tasks.py` suite includes an identically-shaped but purely
quantitative version of this scenario. Its output is the concrete proof that
shared-namespace reads work at scale:

| Metric                    | MemHub  | Baseline (no policies) |
|---------------------------|---------|------------------------|
| Cross-agent reads         | 16      | 16                     |
| `passed`                  | true    | true                   |
| `total_operations`        | 8       | 8                      |
| `tier1_hit_rate`          | 1.00    | 1.00                   |
| Store latency p50 (ms)    | 0.015   | 0.016                  |
| Retrieve latency p50 (ms) | 0.367   | 0.463                  |
| Retrieve latency p90 (ms) | 0.565   | 0.675                  |
| Ops/sec                   | 4753.18 | 4126.59                |

Both configurations achieve 16 cross-agent reads (the expected floor is 12:
4 agents × 3 cross-reads each). MemHub is **~21% faster at retrieval p50 and
~16% faster at p90** than the policy-disabled baseline, because its
Tier-1-first hybrid ranker short-circuits the empty Tier-2 scan when all
content is hot. No demotion fires here (`compression_pct = 0`) because the
working set is tiny — the win is purely from retrieval-path efficiency.

**Impact.** PASS confirms that MemHub delivers its headline feature: agents
can collaborate through a shared memory without seeing each other's
conversation histories. This is the single most-requested capability in
multi-agent systems and is what separates MemHub from per-agent local
caches.

---

### Scenario 3 — Policy Pressure (Demotion)

**Code:**
[multi_agent_quality_eval.py:156-191](multi_agent_quality_eval.py#L156-L191)

**What it tests.** Fifteen **verbose** observations (`"Observation N: …" × 5`
each) are written to one agent before the key fact — "The target deployment
date is October 24th." — is appended last. Each `store_memory` call triggers
`PolicyEngine.run_all()`, so by the time the task starts the oldest 70% of
memories should already have been summarised into Tier 2 via
[core/policies.py::DemotionPolicy](../core/policies.py) and
[core/summarization.py](../core/summarization.py). The team is then asked to
name the deployment date.

**What makes it unique.** This is the only scenario that deliberately
triggers LLM-backed **summarisation** during seeding. It verifies that the
key fact — written *after* the pressure — survives the demotion pipeline,
and that a compressed summary of the noise does **not** hallucinate a
different date. A naive FIFO eviction would delete the key fact; a
demotion-only pipeline must retain it in Tier 1 because it's within the
"newest 30%" carve-out.

**How it's evaluated.** Substring match: `"October 24th" in final_reply`.
`robustness_score` ∈ `{0.0, 1.0}`.

**Expected result & impact.** PASS proves three things simultaneously:
1. The 70/30 split in `summarize_working_memory` protects freshly-written
   facts.
2. The Ollama-served summariser did not drop or alter the salient date
   string when compressing the surrounding noise.
3. The interceptor's **token-ceiling guard** (1500 tokens) correctly triggers
   an on-demand sweep if the injected context grows too large before the
   agent replies.

FAIL usually indicates either (a) the summariser hallucinated, (b) the key
fact got pulled into the oldest-70% bucket because the minimum-age guard was
bypassed, or (c) Ollama was unreachable and `summarize_content` fell back to
returning the original text.

---

### Scenario 4 — Namespace Isolation

**Code:**
[multi_agent_quality_eval.py:193-230](multi_agent_quality_eval.py#L193-L230)

**What it tests.** A single agent writes **two** memories in one
transaction:
- `private`: `"My personal password is 'admin123'."`
- `shared`:  `"The public API key is 'PUBLIC_KEY_2024'."`

The Analyst (a *different* agent id) is then asked to retrieve API keys
from shared memory and explicitly **not** leak private passwords. Pass
requires `PUBLIC_KEY_2024` to appear in the reply **and** `admin123` to be
absent.

**What makes it unique.** Every other scenario treats MemHub as a
cooperative store. This one treats it as a **security boundary**. It is
the only test that fails if MemHub is too *permissive* — the usual failure
mode of memory systems. It directly exercises the negative path of
`require_read_access` in [api/auth.py](../api/auth.py).

**How it's evaluated.** A compound predicate:
`passed = found_shared and not leaked`. The metric `security_score`
∈ `{0.0, 1.0}` is binary and deliberately asymmetric: leaking even when
the shared fact is also reported is still a FAIL.

**Expected result & impact.** PASS means:
1. Tier-1 `LIKE` scans in [core/retrieval.py](../core/retrieval.py) filter by
   `(agent_id = :caller OR namespace = 'shared')` correctly.
2. Tier-2 ChromaDB `where` filters use the same predicate.
3. The interceptor does not accidentally pull cross-agent private rows when
   it calls `/v1/retrieve` on behalf of the Analyst.

Impact: this is the scenario a compliance/security reviewer cares about. A
single FAIL here is a release blocker regardless of how many other scenarios
pass, because it means the `private` namespace is only a label, not a
guarantee.

---

### Scenario 5 — Conflict Resolution

**Code:**
[multi_agent_quality_eval.py:232-264](multi_agent_quality_eval.py#L232-L264)

**What it tests.** The same agent writes two *contradictory* status
records one second apart:
- t=0: `"Current Status: PENDING"`
- t=1: `"Current Status: APPROVED (Updated 1 minute ago)"`

The team is then asked for the "current status…based on the most recent
memory". Pass requires `"APPROVED"` in the reply **and** `"PENDING"`
absent.

**What makes it unique.** The other scenarios test whether data flows
through MemHub at all. This scenario tests whether MemHub's ranker produces
a **temporally correct** ordering when two equally-relevant rows conflict.
It is the only scenario whose correctness depends on the recency-decay term
in [core/retrieval.py](../core/retrieval.py)'s hybrid scoring formula
(`keyword_hits + recency_decay + log(access_count)`). A naive pure-keyword
ranker would return both records with equal weight and let the LLM pick
arbitrarily.

**How it's evaluated.** A compound predicate:
`"APPROVED" in final_reply and "PENDING" not in final_reply`.
`recency_score` ∈ `{0.0, 1.0}`.

**Expected result & impact.** PASS proves that the recency-decay weight is
tuned strongly enough to push a one-second-newer record above its older
twin in the merged top-k — and that the LLM, given both rows in the
injected context, chooses the correct one when instructed. FAIL typically
means one of:
- The two rows both landed in the top-k with near-equal scores and the LLM
  picked the older one (model error masquerading as ranker error).
- The newer row didn't make the top-k at all (ranker error).
- The test was flaky due to the one-second `asyncio.sleep` not actually
  producing distinct `created_at` timestamps under clock skew.

Impact: this scenario justifies the recency term's existence. Without it,
MemHub's hybrid ranker would collapse to "best keyword match wins", and
every stale fact would be a landmine.

---

## 3. Summary Table

| # | Scenario              | Metric name           | Probes                           | MemHub feature validated                     |
|---|-----------------------|-----------------------|----------------------------------|----------------------------------------------|
| 1 | Hidden Fact Retrieval | `recall_accuracy`     | Tier-1 keyword scan + injection  | Interceptor pre-call hook                    |
| 2 | Cross-Agent Handover  | `collaboration_score` | Shared-namespace read across ids | ACL + `include_shared` path                  |
| 3 | Policy Pressure       | `robustness_score`    | LLM summarisation under budget   | `DemotionPolicy` + 70/30 split               |
| 4 | Namespace Isolation   | `security_score`      | Negative-path ACL                | `private` vs `shared` enforcement            |
| 5 | Conflict Resolution   | `recency_score`       | Temporal ordering in the ranker  | Hybrid score's recency-decay term            |

The overall `passed` flag for the suite is the **logical AND** of all five
— any single FAIL means MemHub has shipped a regression.

---

## 4. Supporting Benchmark Artefacts

The quality eval itself does not write JSON — it calls `print_summary()` to
stdout. The closest quantitative artefact is the benchmark-suite run of the
multi-agent scenario:

- [results/memhub/multi_agent_shared_collab.json](results/memhub/multi_agent_shared_collab.json)
  — MemHub with policies enabled.
- [results/baseline/multi_agent_shared_collab.json](results/baseline/multi_agent_shared_collab.json)
  — same task with the `PolicyEngine` disabled.

Both files record 16 successful cross-agent reads (threshold: 12), confirming
the collaboration path covered by Scenario 2 above. The **~21% retrieval p50
speedup** of MemHub over the baseline is attributable to the Tier-1-first
hybrid ranker — not to the policies themselves, which do not fire in this
short task.

---

## 5. How to Re-Run

```bash
# 1. Start MemHub + Ollama
docker compose up -d
docker exec memhub-ollama ollama pull llama3.1

# 2. Confirm health
curl http://localhost:8000/v1/health

# 3. Run the quality eval (prints a pass/fail table)
export PYTHONPATH=$PYTHONPATH:.
python eval/multi_agent_quality_eval.py

# 4. Run the quantitative multi-agent benchmark (writes JSON)
python eval/run_comparison.py
```

Expected stdout from the quality eval:

```
==================================================
SCENARIO                       RESULT
==================================================
Hidden Fact Retrieval          ✓ PASSED
Cross-Agent Handover           ✓ PASSED
Policy Pressure (Demotion)     ✓ PASSED
Namespace Isolation            ✓ PASSED
Conflict Resolution            ✓ PASSED
==================================================
```

Any `✗ FAILED` row should be cross-referenced with the "Impact" section of
the corresponding scenario above to localise the regression.
