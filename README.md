## Production RAG Agent (LangGraph)

This repository contains a production-oriented RAG agent built on LangGraph with:
- Query planning + multi-query rewrite (3 queries per selected source)
- Hybrid retrieval per source (vector + BM25-style FULL_TEXT)
- Deterministic RRF ranking
- Deterministic rule-based context sufficiency check with bounded retries
- Grounded answer generation (LLM uses only ranked context)
- Built-in graph visualization export (Mermaid / optional PNG)

## Architecture

```mermaid
flowchart TD
    A[User Question] --> B[Query Analysis and Multi Query Rewrite]
    B --> C[Retrieval Using Hybrid Tools]

    C --> D1[ETQ Hybrid Retriever 3 Queries]
    C --> D2[ITEX Hybrid Retriever 3 Queries]
    C --> D3[ServiceNow KB Hybrid Retriever 3 Queries]
    C --> D4[ServiceNow Ticket Hybrid Retriever 3 Queries]

    D1 --> E[Merge Retrieved Results]
    D2 --> E
    D3 --> E
    D4 --> E

    E --> F[RRF Ranking]
    F --> G{Context Sufficient}
    G -- No --> B
    G -- Yes --> H[Answer Generation]
    H --> I[Final Answer]
```

## Configuration

- Copy `.env.example` to `.env` and fill in your values.
- The agent reads configuration from environment variables via `src/rag_agent/config/settings.py`.

## Graph Visualization (No Inference Required)

This writes deterministic artifacts to `artifacts/graphs/`:
- `rag_workflow.mmd` (Mermaid)
- `rag_workflow.md` (Markdown Mermaid block)

PNG export is optional and fails fast if it cannot render.

## Running the Agent

The agent entrypoint is `src/rag_agent/main.py`. A convenience wrapper exists at repo root: `main.py`.

Common usage patterns:
- Export graph: `python main.py --export-graph`
- Export graph (PNG): `python main.py --export-graph-png`
- Run from Python: `from rag_agent.main import run; print(run("your question"))`

## Testing

Minimal smoke tests exist under `tests/` and focus on:
- Settings validation (fail-fast)
- Graph compilation + Mermaid export (no inference required)

Run tests with:
- `python -m unittest -v`
