from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from rag_agent.config.logging import get_logger
from rag_agent.config.settings import Settings
from rag_agent.graph.state import AgentState
from rag_agent.prompts.retrieval import build_retrieval_tool_call_prompt
from rag_agent.tools.retrievers.databricks_ensemble import build_source_tools


def make_retrieval_node(*, llm: BaseChatModel, settings: Settings) -> Callable[[AgentState], AgentState]:
    """Execute retrieval by tool-calling based on planned queries already in state."""

    tools = build_source_tools(settings=settings)
    tool_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)
    logger = get_logger(__name__, node="retrieval")

    async def node(state: AgentState) -> AgentState:
        logger.info("Starting retrieval tool-calling", extra={"action": "retrieve"})
        queries_by_source = state.get("queries_by_source")
        if not isinstance(queries_by_source, dict) or not queries_by_source:
            raise ValueError("state.queries_by_source is required")

        expected_calls = _expected_tool_calls_count(
            queries_by_source=queries_by_source,
            expected_queries_per_source=settings.query_planning.queries_per_source,
        )

        prompt = build_retrieval_tool_call_prompt(queries_by_source=queries_by_source)
        messages: list[BaseMessage] = prompt.format_messages(queries_by_source=queries_by_source)

        assistant = await llm_with_tools.ainvoke(messages)
        tool_calls = list(getattr(assistant, "tool_calls", []) or [])
        if len(tool_calls) != expected_calls:
            raise ValueError("Model did not produce the required retrieval tool calls")

        validated_calls = _validate_tool_calls(
            tool_calls=tool_calls,
            tool_by_name=tool_by_name,
            queries_by_source=queries_by_source,
            expected_queries_per_source=settings.query_planning.queries_per_source,
        )

        async def _call_one(call: dict[str, Any]) -> tuple[dict[str, Any], list[Document]]:
            tool = tool_by_name[call["name"]]
            args = call.get("args") or {}
            query = args.get("query")
            if not isinstance(query, str):
                raise ValueError("Tool call args must include a string 'query'")
            docs = await tool.ainvoke(query)
            return call, docs

        results = await asyncio.gather(*[_call_one(c) for c in validated_calls])

        retrieved: list[Document] = []
        usage_cursors: dict[str, int] = {source_id: 0 for source_id in queries_by_source.keys()}

        for call, docs in results:
            tool_name = call["name"]
            source_id = _source_id_from_tool_name(tool_name)
            query = call["args"]["query"]
            query_index = _assign_query_index(
                source_id=source_id,
                query=query,
                queries_by_source=queries_by_source,
                cursor=usage_cursors,
            )
            retrieved.extend(
                _tag_docs(
                    docs,
                    source_id=source_id,
                    query_index=query_index,
                    tool_name=tool_name,
                )
            )

        new_state: AgentState = dict(state)
        new_state["retrieved_chunks"] = retrieved
        logger.info(
            "Retrieval completed",
            extra={
                "action": "retrieve_done",
            },
        )
        return new_state

    return node


def _expected_tool_calls_count(
    *, queries_by_source: dict[str, list[str]], expected_queries_per_source: int
) -> int:
    total = 0
    for queries in queries_by_source.values():
        if not isinstance(queries, list) or len(queries) != expected_queries_per_source:
            raise ValueError(
                "queries_by_source must contain the configured number of queries per source"
            )
        total += expected_queries_per_source
    return total


def _source_id_from_tool_name(tool_name: str) -> str:
    if not tool_name.startswith("retrieve_"):
        raise ValueError("Unexpected tool name")
    return tool_name.removeprefix("retrieve_").upper()


def _validate_tool_calls(
    *,
    tool_calls: list[dict[str, Any]],
    tool_by_name: dict[str, Any],
    queries_by_source: dict[str, list[str]],
    expected_queries_per_source: int,
) -> list[dict[str, Any]]:
    source_to_queries = {k: list(v) for k, v in queries_by_source.items()}
    counters = {source_id: 0 for source_id in source_to_queries.keys()}
    validated: list[dict[str, Any]] = []

    for call in tool_calls:
        name = call.get("name")
        args = call.get("args")
        if not isinstance(name, str) or name not in tool_by_name:
            raise ValueError("Model selected an unknown tool")
        if not isinstance(args, dict) or "query" not in args:
            raise ValueError("Tool call args must include 'query'")
        query = args["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Tool call query must be a non-empty string")

        source_id = _source_id_from_tool_name(name)
        if source_id not in source_to_queries:
            raise ValueError("Model called a tool for a source that was not planned")
        if query not in source_to_queries[source_id]:
            raise ValueError("Model changed the planned query")

        counters[source_id] += 1
        validated.append({"name": name, "args": {"query": query}})

    for source_id, count in counters.items():
        if count != expected_queries_per_source:
            raise ValueError(
                "Model did not call tools the configured number of times per source"
            )

    return validated


def _assign_query_index(
    *,
    source_id: str,
    query: str,
    queries_by_source: dict[str, list[str]],
    cursor: dict[str, int],
) -> int:
    queries = queries_by_source[source_id]
    start = cursor[source_id]
    for idx in range(start, len(queries)):
        if queries[idx] == query:
            cursor[source_id] = idx + 1
            return idx
    for idx, q in enumerate(queries):
        if q == query:
            cursor[source_id] = max(cursor[source_id], idx + 1)
            return idx
    raise ValueError("Could not assign query_index for retrieved documents")


def _tag_docs(
    docs: list[Document], *, source_id: str, query_index: int, tool_name: str
) -> list[Document]:
    tagged: list[Document] = []
    for doc in docs:
        metadata = dict(doc.metadata or {})
        metadata["source_system"] = source_id
        metadata["query_index"] = query_index
        metadata["retrieval_tool"] = tool_name
        tagged.append(Document(page_content=doc.page_content, metadata=metadata))
    return tagged
