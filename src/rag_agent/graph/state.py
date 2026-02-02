from __future__ import annotations

from typing import NotRequired, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    user_query: str
    queries_by_source: NotRequired[dict[str, list[str]]]
    retrieved_chunks: NotRequired[list[Document]]
    ranked_chunks: NotRequired[list[Document]]
    loop_count: NotRequired[int]
    context_is_sufficient: NotRequired[bool]
    context_insufficiency_reason: NotRequired[str]
    final_answer: NotRequired[str]
