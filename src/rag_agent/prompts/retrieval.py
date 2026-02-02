from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate


def build_retrieval_tool_call_prompt(*, queries_by_source: dict[str, list[str]]) -> ChatPromptTemplate:
    system = (
        "You are a retrieval orchestration node.\n"
        "You must call tools to retrieve documents.\n"
        "You must not change queries, invent sources, or skip queries.\n"
        "For each source, you must call the corresponding retrieval tool once per query.\n"
        "Use the query strings exactly as provided.\n"
        "Return tool calls only."
    )

    user = (
        "Planned queries by source (JSON):\n"
        f"{json.dumps(queries_by_source, ensure_ascii=False)}\n"
        "\n"
        "Call the retrieval tools now."
    )

    return ChatPromptTemplate.from_messages([("system", system), ("human", user)])

