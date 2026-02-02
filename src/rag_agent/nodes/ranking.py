from __future__ import annotations

from collections.abc import Callable

from langchain_core.documents import Document

from rag_agent.config.logging import get_logger
from rag_agent.config.settings import Settings
from rag_agent.graph.state import AgentState
from rag_agent.utils.rrf import reciprocal_rank_fusion


def make_ranking_node(*, settings: Settings) -> Callable[[AgentState], AgentState]:
    """Rank retrieved documents deterministically using Reciprocal Rank Fusion (RRF)."""

    logger = get_logger(__name__, node="ranking")

    def node(state: AgentState) -> AgentState:
        logger.info("Ranking retrieved chunks", extra={"action": "rank"})
        retrieved_chunks = state.get("retrieved_chunks")
        if not isinstance(retrieved_chunks, list) or not retrieved_chunks:
            raise ValueError("state.retrieved_chunks is required")

        ranked_lists = _group_ranked_lists(retrieved_chunks)
        rrf_ranked = reciprocal_rank_fusion(
            ranked_lists=ranked_lists, rrf_k=settings.ranking.rrf_k
        )

        top_k = settings.ranking.top_k_context
        top_docs: list[Document] = []
        for item in rrf_ranked[:top_k]:
            metadata = dict(item.document.metadata or {})
            metadata["rrf_score"] = item.score
            top_docs.append(Document(page_content=item.document.page_content, metadata=metadata))

        new_state: AgentState = dict(state)
        new_state["ranked_chunks"] = top_docs
        logger.info(
            "Ranking completed",
            extra={
                "action": "rank_done",
            },
        )
        return new_state

    return node


def _group_ranked_lists(retrieved: list[Document]) -> list[list[Document]]:
    grouped: dict[tuple[str, int], list[Document]] = {}
    for doc in retrieved:
        metadata = doc.metadata or {}
        source_system = metadata.get("source_system")
        query_index = metadata.get("query_index")
        if not isinstance(source_system, str) or not isinstance(query_index, int):
            continue
        key = (source_system, query_index)
        grouped.setdefault(key, []).append(doc)

    ordered_keys = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))
    ranked_lists: list[list[Document]] = []
    for key in ordered_keys:
        ranked_lists.append(grouped[key])
    return ranked_lists
