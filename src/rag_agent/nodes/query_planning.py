from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from rag_agent.config.settings import Settings
from rag_agent.config.logging import get_logger
from rag_agent.graph.state import AgentState
from rag_agent.prompts.query_planning import build_query_planning_prompt
from rag_agent.utils.json_extraction import extract_first_json_object


def make_query_planning_node(
    *, llm: BaseChatModel, settings: Settings
) -> Callable[[AgentState], AgentState]:
    """Analyze the user query, select sources, and produce exactly three queries per source.

    This node performs no retrieval. It only populates `state["queries_by_source"]`.
    """

    prompt = build_query_planning_prompt(
        queries_per_source=settings.query_planning.queries_per_source
    )
    source_catalog = "\n".join(
        f"{spec.source_id}: {spec.description}" for spec in settings.sources
    )
    allowed_sources = {spec.source_id for spec in settings.sources}
    logger = get_logger(__name__, node="query_planning")

    def node(state: AgentState) -> AgentState:
        logger.info(
            "Planning queries for user question",
            extra={"action": "plan_queries"},
        )
        user_query = (state.get("user_query") or "").strip()
        if not user_query:
            raise ValueError("state.user_query is required")

        messages = prompt.format_messages(
            user_query=user_query,
            source_catalog=source_catalog,
        )

        response = llm.invoke(messages)
        content = getattr(response, "content", response)
        parsed = extract_first_json_object(str(content))

        _, queries_by_source = _validate_plan(
            parsed=parsed,
            allowed_sources=allowed_sources,
            queries_per_source=settings.query_planning.queries_per_source,
            max_selected_sources=settings.query_planning.max_selected_sources,
        )
        logger.info(
            "Query planning produced source/query plan",
            extra={
                "action": "plan_done",
            },
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
