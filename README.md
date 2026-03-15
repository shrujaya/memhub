# memhub
Python 3.14.3

```
memhub/
├── .gitignore
├── requirements.txt
├── README.md
│
├── db/                       # Local storage (Ignored in Git)
│   ├── chroma_db/            # Embedded ChromaDB files (Long-term memory)
│   └── memhub.db             # SQLite database (Working memory & ACLs)
│
├── api/                      # Owned by: Infrastructure Lead
│   ├── __init__.py
│   ├── main.py               # FastAPI application entry point
│   ├── routes.py             # /store and /retrieve endpoints
│   ├── models.py             # Pydantic data models (MemoryPayload, RetrieveRequest)
│   └── auth.py               # Logic for enforcing private and shared namespaces [cite: 58]
│
├── core/                     # Owned by: Memory Operations Lead
│   ├── __init__.py
│   ├── policies.py           # Eviction, promotion, and demotion logic [cite: 13, 46]
│   ├── summarization.py      # LLM calls to compress old history [cite: 44, 46]
│   └── retrieval.py          # Top-k vector search and working memory fetching [cite: 43, 60]
│
├── agents/                   # Owned by: Multi-Agent Orchestration Lead
│   ├── __init__.py
│   ├── team_config.py        # AutoGen role definitions and setups 
│   ├── tools.py              # Callable functions (e.g., query_team_memory())
│   └── interceptor.py        # Logic overriding default context to call MemHub API
│
├── eval/                     # Owned by: Systems Performance Lead
│   ├── __init__.py
│   ├── benchmark_tasks.py    # Multi-step collaborative agent tasks [cite: 51]
│   ├── metrics.py            # Track latency, throughput, and token usage [cite: 51]
│   └── visualize.py          # Scripts to generate charts comparing policies (e.g., LRU vs FIFO) [cite: 48, 49]
│
└── scripts/                  # Utilities
    ├── run_server.sh         # zsh script to start the FastAPI service
    └── run_evals.sh          # zsh script to trigger the AutoGen agents and log results
```