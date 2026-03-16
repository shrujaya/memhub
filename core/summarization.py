import logging
from typing import List, Dict, Any, Optional

import aiohttp
import tiktoken

# ── Ollama for local LLM summarization ─────────────────────────────────────────
try:
    import ollama as ollama_client  # type: ignore
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Preferred Ollama models, tried in order
OLLAMA_MODEL_PREFERENCE = ["llama3", "mistral"]

# Configure logger
logger = logging.getLogger(__name__)

# LLM Endpoint configuration (assuming OpenAI-compatible API like vLLM or Ollama)
LLM_ENDPOINT = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "llama2" # Configure with your specific local model name

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Returns the token count for a given string using tiktoken.
    
    Args:
        text (str): The text to count tokens for.
        model (str): The model encoding to use (default: gpt-3.5-turbo).
        
    Returns:
        int: The number of tokens in the text.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to standard encoding if model is not found, e.g. for local models
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

async def _fetch_summary_from_llm(prompt: str) -> str:
    """
    Makes an async HTTP POST request to the local LLM endpoint to get the summary.
    
    Args:
        prompt (str): The summarization prompt.
        
    Returns:
        str: The generated summary text.
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful assistant that summarizes agent working memory. Extract key decisions, facts, and entities concisely."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.3, # Low temperature for more factual, deterministic summaries
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Assuming OpenAI compatible response format
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    logger.error(f"Unexpected API response format: {data}")
                    return "Error generating summary: Unexpected response format."
                    
    except aiohttp.ClientError as e:
        logger.error(f"Network error communicating with LLM endpoint: {e}")
        return f"Error generating summary: Network error ({e})"
    except Exception as e:
        logger.error(f"Unexpected error during LLM summarization: {e}")
        return f"Error generating summary: Unexpected error ({e})"

async def summarize_working_memory(agent_id: str, memories: List[str], threshold: int = 2000) -> Dict[str, Any]:
    """
    Analyzes an agent's working memory (Tier 1) and compresses it if it exceeds the token threshold.
    
    How this hooks into broader policies:
    When working memory (stored in SQLite) gets too large, this function is called by the 
    eviction policy (core/policies.py). It identifies the oldest 70% of memories,
    summarizes them using a local LLM, and preserves the most recent 30% untouched. 
    
    The calling policy can then take the `demoted_memories` and move them into long-term 
    vector storage (Tier 2, ChromaDB), while keeping the `retained_memories` and appending 
    the `new_summary` to the active working memory (SQLite) for immediate context.
    
    Args:
        agent_id (str): The unique identifier of the agent.
        memories (list[str]): The chronological list of working memories (oldest to newest).
        threshold (int): The maximum allowed tokens before summarization triggers.
        
    Returns:
        dict: A dictionary containing:
            - needs_update (bool): Whether summarization was performed (threshold exceeded).
            - new_summary (str | None): The summarized text (None if no update needed).
            - retained_memories (list[str]): The recent 30% of memories left untouched.
            - demoted_memories (list[str]): The old 70% of memories that were summarized.
    """
    # 1. Calculate total tokens across all memories
    total_tokens = sum(count_tokens(m) for m in memories)
    
    # 2. Check if we exceed the threshold
    if total_tokens <= threshold:
        logger.debug(f"Agent {agent_id} memory ({total_tokens} tokens) is within threshold ({threshold}).")
        return {
            "needs_update": False,
            "new_summary": None,
            "retained_memories": memories,
            "demoted_memories": []
        }
        
    logger.info(f"Agent {agent_id} memory ({total_tokens} tokens) exceeds threshold ({threshold}). Triggering summarization.")
    
    # 3. Split memories: oldest 70% to demote/summarize, newest 30% to retain
    total_memories = len(memories)
    if total_memories == 0:
         return {
            "needs_update": False,
            "new_summary": None,
            "retained_memories": [],
            "demoted_memories": []
        }
        
    # Calculate split index (70% demoted)
    split_index = int(total_memories * 0.70)
    
    # Ensure at least one memory is retained if there are multiple memories
    if split_index == total_memories and total_memories > 1:
        split_index -= 1
        
    # Ensure at least one memory is summarized if we are exceeding the token limit
    # (e.g. edge case where a single huge memory exceeds the threshold itself)
    if split_index == 0 and total_memories > 0:
        split_index = 1
        
    demoted_memories = memories[:split_index]
    retained_memories = memories[split_index:]
    
    # 4. Construct summarization prompt
    combined_old_events = "\n---\n".join(demoted_memories)
    prompt = f"""
Please concisely summarize the following sequence of chronological past events and memories for Agent {agent_id}.
Focus on extracting key decisions, facts, relationships, and entities. 
Discard redundant conversational filler. Keep the summary dense and factual.

MEMORIES TO SUMMARIZE:
{combined_old_events}

SUMMARY:
"""

    # 5. Call local LLM endpoint asynchronously
    new_summary = await _fetch_summary_from_llm(prompt)
    
    return {
        "needs_update": True,
        "new_summary": new_summary,
        "retained_memories": retained_memories,
        "demoted_memories": demoted_memories
    }

def _pick_ollama_model() -> Optional[str]:
    """Return the first available Ollama model from our preference list."""
    if not OLLAMA_AVAILABLE:
        return None
    try:
        available = {m["name"].split(":")[0] for m in ollama_client.list()["models"]}
        for preferred in OLLAMA_MODEL_PREFERENCE:
            if preferred in available:
                return preferred
    except Exception as exc:
        logger.warning("Could not query Ollama model list: %s", exc)
    return None

def summarize_content(text: str) -> str:
    """
    Automatic Summarization — The Policy Engine
    ─────────────────────────────────────────────
    Compresses `text` into a short paragraph using a locally-running LLM via
    Ollama. This is a background policy operation triggered when an agent's
    working memory scratchpad exceeds WORKING_MEMORY_SUMMARY_THRESHOLD chars.
    """
    model = _pick_ollama_model()
    if model is None:
        logger.warning(
            "Ollama not available or no supported model found; "
            "skipping summarization."
        )
        return text

    prompt = (
        "You are a concise assistant. Summarize the following agent memory log "
        "into a single, dense paragraph. Preserve all key facts, decisions, and "
        "unresolved tasks. Do not add any commentary — only output the summary.\n\n"
        f"MEMORY LOG:\n{text}\n\nSUMMARY:"
    )

    try:
        logger.info("Summarizing working memory using model '%s' …", model)
        response = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response["message"]["content"].strip()
        logger.info(
            "Summarization complete: %d chars → %d chars", len(text), len(summary)
        )
        return summary
    except Exception as exc:
        logger.error("Summarization failed: %s", exc)
        # Graceful fallback: preserve original content, do not crash the /store call
        return f"[SUMMARY FAILED — ORIGINAL PRESERVED]\n{text}"
