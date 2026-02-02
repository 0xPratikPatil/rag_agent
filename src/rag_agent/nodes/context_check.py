from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from rag_agent.config.logging import get_logger
from rag_agent.config.settings import Settings
from rag_agent.graph.state import AgentState


def make_context_check_node(*, settings: Settings) -> Callable[[AgentState], AgentState]:
    """Assemble final context deterministically and perform rule-based sufficiency checks."""

    logger = get_logger(__name__, node="context_check")

    def node(state: AgentState) -> AgentState:
        logger.info("Checking context sufficiency", extra={"action": "check"})
        ranked_chunks = state.get("ranked_chunks")
        if not isinstance(ranked_chunks, list):
            raise ValueError("state.ranked_chunks is required")

        deduped = _dedupe_documents(ranked_chunks)
        limited = deduped[: settings.ranking.top_k_context]

        is_sufficient, reason = _check_sufficiency(
            docs=limited,
            min_docs=settings.context_check.min_docs,
            min_distinct_sources=settings.context_check.min_distinct_sources,
            authoritative_sources=set(settings.context_check.authoritative_sources),
            min_authoritative_docs=settings.context_check.min_authoritative_docs,
        )

        new_state: AgentState = dict(state)
        new_state["ranked_chunks"] = limited
        new_state["context_is_sufficient"] = is_sufficient
        if not is_sufficient:
            previous_loop_count = int(state.get("loop_count", 0))
            new_state["loop_count"] = previous_loop_count + 1
            new_state["context_insufficiency_reason"] = reason
            logger.warning(
                "Context insufficient",
                extra={"action": "insufficient"},
            )
        else:
            new_state.pop("context_insufficiency_reason", None)
            logger.info(
                "Context sufficient",
                extra={"action": "sufficient"},
            )
        return new_state

    return node


def _dedupe_documents(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    deduped: list[Document] = []

    for doc in docs:
        doc_id = _doc_identity(doc)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        deduped.append(doc)

    return deduped


def _doc_identity(doc: Document) -> str:
    metadata: dict[str, Any] = doc.metadata or {}
    for key in ("id", "doc_id", "document_id", "_id", "chunk_id", "pk"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)

    source_system = metadata.get("source_system")
    query_index = metadata.get("query_index")
    return f"{source_system}|{query_index}|{doc.page_content}"


def _check_sufficiency(
    *,
    docs: list[Document],
    min_docs: int,
    min_distinct_sources: int,
    authoritative_sources: set[str],
    min_authoritative_docs: int,
) -> tuple[bool, str]:
    if len(docs) < min_docs:
        return False, "Insufficient number of ranked chunks"

    distinct_sources: set[str] = set()
    authoritative_count = 0

    for doc in docs:
        source = (doc.metadata or {}).get("source_system")
        if isinstance(source, str) and source.strip():
            distinct_sources.add(source)
            if source in authoritative_sources:
                authoritative_count += 1

    if len(distinct_sources) < min_distinct_sources:
        return False, "Insufficient number of distinct sources in context"

    if authoritative_count < min_authoritative_docs:
        return False, "Missing required authoritative chunks in context"

    return True, ""
