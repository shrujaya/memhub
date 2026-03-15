#!/usr/bin/env python3

import sqlite3
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings

app = FastAPI(title="MemHub", description="Shared memory service for LLM agents")

# --- Constants & Configuration ---
SQLITE_DB_PATH = "memhub_working_memory.db"
CHROMA_DB_PATH = "./chroma_db"

# --- Database Initialization (SQLite) ---
def init_sqlite():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    # Table for agent namespaces and working memory
    # agent_id: Unique identifier for the agent
    # namespace: Private namespace identifier
    # content: Current working memory content (fast-access)
    # authorized_spaces: JSON-serialized list of shared spaces this agent can access
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            agent_id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            working_memory_content TEXT,
            authorized_spaces TEXT,
            last_updated TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_sqlite()

# --- Database Initialization (ChromaDB) ---
# Initialize persistent ChromaDB client for long-term memory tier
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
# We'll use a single collection for simplicity, partitioned by agent_id/namespace in metadata
long_term_collection = chroma_client.get_or_create_collection(name="long_term_memory")

# --- Models ---
class StoreRequest(BaseModel):
    agent_id: str
    text_content: str
    is_shared: bool = False

class RetrieveRequest(BaseModel):
    agent_id: str
    query: str
    top_k: int = 5

class MemoryResponse(BaseModel):
    agent_id: str
    working_memory: Optional[str]
    long_term_memory: List[Dict[str, Any]]

# --- Endpoints ---

@app.post("/store", response_model=Dict[str, str])
async def store_memory(request: StoreRequest = Body(...)):
    """
    Stores text content into either SQLite working memory or ChromaDB long-term memory.
    """
    agent_id = request.agent_id
    content = request.text_content
    is_shared = request.is_shared
    
    # Placeholder Heuristic for Promotion/Demotion:
    # In a real system, we might use token counts, recency, or explicit "importance" scores.
    # Here, we'll implement a simple length-based logic as a placeholder.
    # Short messages stay in working memory; longer ones or those marked 'shared' go to long-term.
    
    # Implementation Detail:
    # 1. Check if agent exists in SQLite, if not, create namespace.
    # 2. Decide where to store based on heuristic.
    
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    # Ensure agent exists
    cursor.execute("SELECT agent_id FROM agent_memory WHERE agent_id = ?", (agent_id,))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO agent_memory (agent_id, namespace, working_memory_content, authorized_spaces, last_updated) VALUES (?, ?, ?, ?, ?)",
            (agent_id, f"ns_{agent_id}", "", "[]", datetime.now())
        )
    
    destination = "working_memory"
    
    # Promotion/Demotion Policy Implementation Point:
    # TODO: Implement more sophisticated logic here (e.g., summarization before demotion).
    if len(content) > 500 or is_shared:
        # Route to ChromaDB (Long-term)
        long_term_collection.add(
            documents=[content],
            metadatas=[{"agent_id": agent_id, "is_shared": is_shared, "timestamp": str(datetime.now())}],
            ids=[str(uuid.uuid4())]
        )
        destination = "long_term_memory"
    else:
        # Route to SQLite (Working memory)
        # For simplicity, we append to the existing content
        cursor.execute("SELECT working_memory_content FROM agent_memory WHERE agent_id = ?", (agent_id,))
        current_content = cursor.fetchone()[0] or ""
        updated_content = (current_content + "\n" + content).strip()
        
        cursor.execute(
            "UPDATE agent_memory SET working_memory_content = ?, last_updated = ? WHERE agent_id = ?",
            (updated_content, datetime.now(), agent_id)
        )
    
    conn.commit()
    conn.close()
    
    return {"status": "success", "stored_in": destination, "agent_id": agent_id}

@app.post("/retrieve", response_model=MemoryResponse)
async def retrieve_memory(request: RetrieveRequest = Body(...)):
    """
    Fetches immediate context from SQLite and performs top-k retrieval from ChromaDB.
    """
    agent_id = request.agent_id
    query = request.query
    top_k = request.top_k
    
    # 1. Fetch from SQLite working memory (Access Control Check)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT working_memory_content, authorized_spaces FROM agent_memory WHERE agent_id = ?", (agent_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Agent not found")
    
    working_content, authorized_spaces_json = row
    conn.close()
    
    # 2. Top-k Retrieval from ChromaDB
    # We filter by agent_id OR shared spaces (Access Control)
    # Note: In a production system, 'authorized_spaces' would be used to build a complex filter.
    results = long_term_collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"agent_id": agent_id} # Placeholder: simplified access control
    )
    
    # Format long-term results
    long_term_results = []
    if results['documents']:
        for i in range(len(results['documents'][0])):
            long_term_results.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })

    # Summarization Policy Implementation Point:
    # TODO: Implement summarization of long-term results or working memory context if too large.

    return MemoryResponse(
        agent_id=agent_id,
        working_memory=working_content,
        long_term_memory=long_term_results
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
