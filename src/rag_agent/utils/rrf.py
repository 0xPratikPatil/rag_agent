from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from langchain_core.documents import Document


@dataclass(frozen=True, slots=True)
class RankedDocument:
    document: Document
    score: float


def reciprocal_rank_fusion(
    *, ranked_lists: list[list[Document]], rrf_k: int
) -> list[RankedDocument]:
    """Compute Reciprocal Rank Fusion (RRF) over multiple ranked document lists.

    This is a deterministic, pure-Python implementation intended for production use.

    Args:
        ranked_lists: Each inner list is ordered best-to-worst.
        rrf_k: RRF constant (commonly 60). Must be > 0.

    Returns:
        RankedDocument entries sorted by descending score with deterministic tie-breaks.
    """

    if rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")

    score_by_id: dict[str, float] = {}
    doc_by_id: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = _document_id(doc)
            doc_by_id.setdefault(doc_id, doc)
            score_by_id[doc_id] = score_by_id.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    ranked: list[RankedDocument] = []
    for doc_id, score in score_by_id.items():
        ranked.append(RankedDocument(document=doc_by_id[doc_id], score=score))

    ranked.sort(key=lambda item: (-item.score, _document_id(item.document)))
    return ranked


def _document_id(doc: Document) -> str:
    metadata: dict[str, Any] = doc.metadata or {}

    for key in ("id", "doc_id", "document_id", "_id", "chunk_id", "pk"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)

    source_system = metadata.get("source_system")
    query_index = metadata.get("query_index")
    stable_prefix = f"{source_system}|{query_index}|".encode("utf-8", errors="ignore")
    content = (doc.page_content or "").encode("utf-8", errors="ignore")
    digest = sha256(stable_prefix + content).hexdigest()
    return digest

