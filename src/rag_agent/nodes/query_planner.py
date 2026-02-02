from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from rag_agent.config.settings import Settings
from rag_agent.graph.state import AgentState
from rag_agent.utils.json_extraction import extract_first_json_object
from rag_agent.prompts.query_planning import build_query_planning_prompt


def make_query_planner_node(
    *, llm: BaseChatModel, settings: Settings
) -> Callable[[AgentState], AgentState]:
    prompt = build_query_planning_prompt(
        queries_per_source=settings.query_planning.queries_per_source
    )

    source_catalog = "\n".join(
        f"{spec.source_id}: {spec.description}" for spec in settings.sources
    )
    allowed_sources = {spec.source_id for spec in settings.sources}

    def node(state: AgentState) -> AgentState:
        user_question = (state.get("user_query") or "").strip()
        if not user_question:
            raise ValueError("state.user_query is required")

        messages = prompt.format_messages(
            user_query=user_question,
            source_catalog=source_catalog,
        )

        response = llm.invoke(messages)
        content = getattr(response, "content", response)
        parsed = extract_first_json_object(str(content))

        selected_sources, queries_by_source = _validate_plan(
            parsed=parsed,
            allowed_sources=allowed_sources,
            queries_per_source=settings.query_planning.queries_per_source,
            max_selected_sources=settings.query_planning.max_selected_sources,
        )

        new_state: AgentState = dict(state)
        new_state["queries_by_source"] = queries_by_source
        return new_state

    return node


def _validate_plan(
    *,
    parsed: dict,
    allowed_sources: set[str],
    queries_per_source: int,
    max_selected_sources: int,
) -> tuple[list[str], dict[str, list[str]]]:
    raw_sources = parsed.get("selected_sources")
    if not isinstance(raw_sources, list):
        raise ValueError("selected_sources must be a list")

    selected_sources: list[str] = []
    for item in raw_sources:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value:
            continue
        if value not in allowed_sources:
            raise ValueError(f"Unknown source_id: {value}")
        if value not in selected_sources:
            selected_sources.append(value)

    if not selected_sources:
        raise ValueError("selected_sources must be non-empty")

    if len(selected_sources) > max_selected_sources:
        raise ValueError("selected_sources exceeds max_selected_sources")

    raw_queries_by_source = parsed.get("queries_by_source")
    if not isinstance(raw_queries_by_source, dict):
        raise ValueError("queries_by_source must be an object")

    queries_by_source: dict[str, list[str]] = {}
    for source_id in selected_sources:
        raw_queries = raw_queries_by_source.get(source_id)
        if not isinstance(raw_queries, list):
            raise ValueError(f"queries_by_source.{source_id} must be an array")
        normalized_queries: list[str] = []
        for q in raw_queries:
            if not isinstance(q, str):
                continue
            value = q.strip()
            if value:
                normalized_queries.append(value)
        if len(normalized_queries) != queries_per_source:
            raise ValueError(
                f"queries_by_source.{source_id} must have exactly {queries_per_source} queries"
            )
        queries_by_source[source_id] = normalized_queries

    return selected_sources, queries_by_source
