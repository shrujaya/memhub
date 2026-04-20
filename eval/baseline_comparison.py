import asyncio
import time
import uuid
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from core.summarization import count_tokens, summarize_content
from core.policies import DemotionPolicy, WORKING_MEMORY_TOKEN_BUDGET
from core.retrieval import MemoryRetriever, MemoryTier
import chromadb

# ── Baseline Implementation (No MemHub) ───────────────────────────────────────

class VanillaMemory:
    """A simple single-tier memory system that caps at 2000 tokens (simulating a fixed context limit)."""
    def __init__(self, cap=2000):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE memory (id TEXT, content TEXT, created_at REAL)")
        self.cap = cap
        
    async def store(self, content: str):
        # Insert the new memory
        self.db.execute("INSERT INTO memory VALUES (?, ?, ?)", (str(uuid.uuid4()), content, time.time()))
        self.db.commit()

        # Enforce the context cap by dropping the oldest memories (Simulating loss of context)
        while self.get_total_tokens() > self.cap:
            self.db.execute("DELETE FROM memory WHERE rowid = (SELECT MIN(rowid) FROM memory)")
            self.db.commit()

    async def retrieve(self, query: str):
        # Simple keyword search
        term = f"%{query}%"
        rows = self.db.execute("SELECT content FROM memory WHERE content LIKE ?", (term,)).fetchall()
        return rows

    def get_total_tokens(self):
        rows = self.db.execute("SELECT content FROM memory").fetchall()
        return sum(count_tokens(r[0]) for r in rows)

# ── MemHub Implementation ───────────────────────────────────────────────────

class MemHubWrapper:
    """Wrapper to use the actual MemHub logic in the benchmark."""
    def __init__(self):
        from api.main import _SCHEMA_SQL
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.executescript(_SCHEMA_SQL)
        self.chroma = chromadb.EphemeralClient()
        self.retriever = MemoryRetriever(self.db, self.chroma)
        self.demotion = DemotionPolicy(self.db, self.chroma, min_age_secs=0)
        self.agent_id = "comparison-agent"

    async def store(self, content: str):
        # Store in Tier 1
        row_id = str(uuid.uuid4())
        now = time.time()
        self.db.execute(
            "INSERT INTO working_memory (id, agent_id, content, created_at, last_accessed, namespace) VALUES (?, ?, ?, ?, ?, ?)",
            (row_id, self.agent_id, content, now, now, "private")
        )
        self.db.commit()
        # Run demotion if needed
        await self.demotion.run(self.agent_id)

    async def retrieve(self, query: str):
        return await self.retriever.retrieve(self.agent_id, query)

    def get_total_tokens(self):
        # Tier 1 tokens
        t1_rows = self.db.execute("SELECT content FROM working_memory WHERE agent_id = ?", (self.agent_id,)).fetchall()
        t1_tokens = sum(count_tokens(r[0]) for r in t1_rows)
        
        # Tier 2 tokens (approximate from Chroma)
        try:
            coll = self.chroma.get_collection("memhub_longterm")
            t2_rows = coll.get()["documents"]
            t2_tokens = sum(count_tokens(d) for d in t2_rows)
        except:
            t2_tokens = 0
            
        return t1_tokens + t2_tokens

# ── Benchmark Execution ──────────────────────────────────────────────────────

async def run_comparison(n_steps=50):
    vanilla = VanillaMemory()
    memhub = MemHubWrapper()
    
    results = []
    
    print(f"Running comparison benchmark for {n_steps} steps...")
    
    for i in range(n_steps):
        content = (
            f"Event {i}: The agent discussed {['finance', 'tech', 'legal', 'health'][i%4]} "
            f"with specific details about {uuid.uuid4().hex[:10]}. "
            "This is a long memory entry designed to fill up the token budget quickly "
            "and trigger the lifecycle policies of MemHub."
        )
        
        # Vanilla
        t0 = time.perf_counter()
        await vanilla.store(content)
        vanilla_latency = (time.perf_counter() - t0) * 1000
        
        # MemHub
        t0 = time.perf_counter()
        await memhub.store(content)
        memhub_latency = (time.perf_counter() - t0) * 1000
        
        results.append({
            "step": i,
            "vanilla_tokens": vanilla.get_total_tokens(),
            "memhub_tokens": memhub.get_total_tokens(),
            "vanilla_latency": vanilla_latency,
            "memhub_latency": memhub_latency
        })
        
        if i % 10 == 0:
            print(f"  Step {i}/{n_steps} complete...")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("eval/results/comparison_data.csv", index=False)
    
    # Generate Charts
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 6))
    
    # Chart 1: Token Usage
    plt.subplot(1, 2, 1)
    plt.plot(df['step'], df['vanilla_tokens'], label='No MemHub (Linear Growth)')
    plt.plot(df['step'], df['memhub_tokens'], label='With MemHub (Compressed)')
    plt.axhline(y=WORKING_MEMORY_TOKEN_BUDGET, color='r', linestyle='--', label='Tier-1 Budget')
    plt.title("Memory Usage Over Time (Tokens)")
    plt.xlabel("Interaction Step")
    plt.ylabel("Total Tokens")
    plt.legend()
    
    # Chart 2: Store Latency
    plt.subplot(1, 2, 2)
    plt.plot(df['step'], df['vanilla_latency'], label='No MemHub')
    plt.plot(df['step'], df['memhub_latency'], label='With MemHub (includes Policy)')
    plt.title("Store Latency (ms)")
    plt.xlabel("Interaction Step")
    plt.ylabel("Latency (ms)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("eval/charts/memhub_vs_baseline.png")
    print("\nBenchmark complete. Chart saved to eval/charts/memhub_vs_baseline.png")

if __name__ == "__main__":
    asyncio.run(run_comparison(100))
