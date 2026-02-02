## What I found (repo reality)
- The repo currently has no `src/` directory, and `main.py` assumes `src/` exists.
- LangChain deps are present, but `langgraph` is not yet declared in `pyproject.toml`.

## Update based on your feedback: config-first, controllable parameters
I will make the agent layer config-driven from day one by introducing a typed `Settings` object in `config/settings.py` and ensuring every implemented module accepts `settings` explicitly (dependency injection). Later nodes/tools will follow the same pattern.

## Goals for this first task (unchanged)
- Create the required directory structure under `src/rag_agent/`.
- Define a single shared LangGraph state object.
- Implement the merged “Query Analysis + Multi-Query Rewrite” node (no retrieval).

## Proposed implementation (files + responsibilities)
### 1) Create directory structure (exactly as specified)
Create:
- `src/rag_agent/config/`
- `src/rag_agent/prompts/`
- `src/rag_agent/tools/retrievers/` (empty placeholder)
- `src/rag_agent/nodes/`
- `src/rag_agent/graph/`
- `src/rag_agent/utils/`
- `src/rag_agent/main.py` (thin placeholder)

### 2) Central settings: all controllable parameters live here
Add `src/rag_agent/config/settings.py` with a small, explicit, typed settings object (stdlib dataclasses) and grouped sections so future work naturally “follows through”:
- `LLMSettings`
  - `model_name` (string)
  - `temperature` (float)
  - `max_tokens` (int | None)
  - `timeout_s` (float)
- `QueryPlanningSettings`
  - `queries_per_source` (int; validated to equal 3 to enforce your invariant)
  - `max_selected_sources` (int; safety cap, default = number of sources)
  - `allow_source_fallback` (bool; for ambiguous questions, default True)
- `RankingSettings` (placeholder for later)
  - `rrf_k` (int)
  - `top_k_context` (int)
- `ContextCheckSettings` (placeholder for later)
  - `min_docs` (int)
  - `min_distinct_sources` (int)
  - `max_loop_count` (int; should be 1 per your rules)

Even though retrieval/ranking/sufficiency aren’t implemented yet, defining these now prevents “magic numbers” later and ensures every future node/tool can use the same settings surface.

### 3) Shared state definition (LangGraph-compatible)
Add `src/rag_agent/graph/state.py` defining one shared `AgentState` (`TypedDict` with `NotRequired`).
Keys for now:
- `user_question: str`
- `query_planning_attempts: int`
- `selected_sources: list[str]`
- `queries_by_source: dict[str, list[str]]`

I’ll keep names stable and generic so later nodes can add:
- `retrieved_docs_by_source_query`
- `ranked_docs`
- `final_context_docs`
- `answer`

### 4) Prompts: file-based and configurable
Add under `src/rag_agent/prompts/`:
- `query_planner_system.txt`
- `query_planner_user.txt`

Prompts will:
- Provide a concise “source catalog” with semantics (ETQ, ITEX, SN_KB, SN_TICKETS)
- Require strict JSON output with exactly 3 queries per selected source

### 5) Utilities: prompt loading + robust JSON extraction
Add:
- `src/rag_agent/utils/prompt_loader.py` (loads prompt text from `prompts/`)
- `src/rag_agent/utils/json_extraction.py` (extract first JSON object; avoids brittle parsing)

### 6) Node: Query analysis + multi-query rewrite (config-driven)
Add `src/rag_agent/nodes/query_planner.py`:
- Export `make_query_planner_node(llm, settings)` returning a callable `(state) -> state`.
- The node will:
  - Read `state["user_question"]`
  - Bind LLM runtime parameters from `settings.llm` where supported (e.g., temperature)
  - Call LLM once
  - Parse and validate JSON
  - Enforce `settings.query_planning.queries_per_source == 3` and output exactly 3 queries per selected source
  - Write `selected_sources` and `queries_by_source` back into state

No retrieval imports, no graph wiring.

## Dependency note (LangGraph)
When you confirm execution, I will add `langgraph` to `pyproject.toml` so later graph wiring uses the latest stable LangGraph APIs. This first task’s state/node code will already be written in the “LangGraph node style” (accept/return shared state).

## Verification (safe and local)
- Import-check new modules.
- Run a local-only sanity check that feeds a mocked LLM output into the node’s parsing/validation path (no external calls).

If you confirm this refined plan, I will implement it exactly as described (directory structure + settings + state + query planner node only).