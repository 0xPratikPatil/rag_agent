# Databricks notebook source
# MAGIC %pip install -q databricks-sdk databricks-vectorsearch langchain langchain-core langchain-community langchain-classic langgraph kairos-llm pydantic
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from databricks.vector_search.client import VectorSearchClient
from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from kairos_llm.models import ModelType, create_embedding_mode, create_langchain_llm

# COMMAND ----------
RAG_AGENT_DATABRICKS_HOST = ""
RAG_AGENT_DATABRICKS_TOKEN = ""
RAG_AGENT_DATABRICKS_VECTOR_SEARCH_ENDPOINT = ""

RAG_AGENT_ETQ_INDEX = ""
RAG_AGENT_ITEX_INDEX = ""
RAG_AGENT_SN_KB_INDEX = ""
RAG_AGENT_SN_TICKETS_INDEX = ""

RAG_AGENT_LLM_PROVIDER = "kairos"
RAG_AGENT_LLM_MODEL_TYPE = "Default"
RAG_AGENT_LLM_TEMPERATURE = 0.2
RAG_AGENT_LLM_TIMEOUT_S = 60.0

RAG_AGENT_EMBEDDINGS_PROVIDER = "kairos"
RAG_AGENT_EMBEDDINGS_MODEL_TYPE = "Default"

RAG_AGENT_QUERIES_PER_SOURCE = 3
RAG_AGENT_MAX_SELECTED_SOURCES = 4
RAG_AGENT_ALLOW_SOURCE_FALLBACK = True

RAG_AGENT_RETRIEVAL_TOP_K = 8
RAG_AGENT_RETRIEVAL_VECTOR_WEIGHT = 0.6
RAG_AGENT_RETRIEVAL_BM25_WEIGHT = 0.4
RAG_AGENT_RETRIEVAL_VECTOR_QUERY_TYPE = "ann"
RAG_AGENT_RETRIEVAL_BM25_QUERY_TYPE = "FULL_TEXT"
RAG_AGENT_RETRIEVAL_TEXT_COLUMN = "text"
RAG_AGENT_RETRIEVAL_RETURN_COLUMNS = ("text",)

RAG_AGENT_TOP_K_CONTEXT = 8

RAG_AGENT_MIN_DOCS = 4
RAG_AGENT_MIN_DISTINCT_SOURCES = 1
RAG_AGENT_MAX_LOOP_COUNT = 1

RAG_AGENT_GRAPH_RECURSION_LIMIT = 100

# COMMAND ----------
@dataclass(frozen=True, slots=True)
class LLMSettings:
    provider: str
    model_type: str
    temperature: float
    timeout_s: float


@dataclass(frozen=True, slots=True)
class EmbeddingsSettings:
    provider: str
    model_type: str


@dataclass(frozen=True, slots=True)
class QueryPlanningSettings:
    queries_per_source: int
    max_selected_sources: int
    allow_source_fallback: bool


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    top_k: int
    vector_weight: float
    bm25_weight: float
    vector_query_type: str
    bm25_query_type: str
    text_column: str
    return_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankingSettings:
    top_k_context: int


@dataclass(frozen=True, slots=True)
class ContextCheckSettings:
    min_docs: int
    min_distinct_sources: int
    max_loop_count: int


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    graph_recursion_limit: int


@dataclass(frozen=True, slots=True)
class DatabricksSettings:
    host: str
    token: str | None
    vector_search_endpoint: str


@dataclass(frozen=True, slots=True)
class SourceSpec:
    source_id: str
    description: str
    index: str


@dataclass(frozen=True, slots=True)
class Settings:
    llm: LLMSettings
    embeddings: EmbeddingsSettings
    query_planning: QueryPlanningSettings
    retrieval: RetrievalSettings
    ranking: RankingSettings
    context_check: ContextCheckSettings
    runtime: RuntimeSettings
    databricks: DatabricksSettings
    sources: tuple[SourceSpec, ...]


def load_settings() -> Settings:
    llm = LLMSettings(
        provider=RAG_AGENT_LLM_PROVIDER,
        model_type=RAG_AGENT_LLM_MODEL_TYPE,
        temperature=RAG_AGENT_LLM_TEMPERATURE,
        timeout_s=RAG_AGENT_LLM_TIMEOUT_S,
    )
    embeddings = EmbeddingsSettings(
        provider=RAG_AGENT_EMBEDDINGS_PROVIDER,
        model_type=RAG_AGENT_EMBEDDINGS_MODEL_TYPE,
    )
    query_planning = QueryPlanningSettings(
        queries_per_source=RAG_AGENT_QUERIES_PER_SOURCE,
        max_selected_sources=RAG_AGENT_MAX_SELECTED_SOURCES,
        allow_source_fallback=RAG_AGENT_ALLOW_SOURCE_FALLBACK,
    )
    retrieval = RetrievalSettings(
        top_k=RAG_AGENT_RETRIEVAL_TOP_K,
        vector_weight=RAG_AGENT_RETRIEVAL_VECTOR_WEIGHT,
        bm25_weight=RAG_AGENT_RETRIEVAL_BM25_WEIGHT,
        vector_query_type=RAG_AGENT_RETRIEVAL_VECTOR_QUERY_TYPE,
        bm25_query_type=RAG_AGENT_RETRIEVAL_BM25_QUERY_TYPE,
        text_column=RAG_AGENT_RETRIEVAL_TEXT_COLUMN,
        return_columns=RAG_AGENT_RETRIEVAL_RETURN_COLUMNS,
    )
    ranking = RankingSettings(top_k_context=RAG_AGENT_TOP_K_CONTEXT)
    context_check = ContextCheckSettings(
        min_docs=RAG_AGENT_MIN_DOCS,
        min_distinct_sources=RAG_AGENT_MIN_DISTINCT_SOURCES,
        max_loop_count=RAG_AGENT_MAX_LOOP_COUNT,
    )
    runtime = RuntimeSettings(graph_recursion_limit=RAG_AGENT_GRAPH_RECURSION_LIMIT)
    databricks = DatabricksSettings(
        host=RAG_AGENT_DATABRICKS_HOST,
        token=RAG_AGENT_DATABRICKS_TOKEN or None,
        vector_search_endpoint=RAG_AGENT_DATABRICKS_VECTOR_SEARCH_ENDPOINT,
    )
    sources = (
        SourceSpec(
            source_id="ETQ",
            description="ETQ quality processes, deviations, CAPA, SOPs, compliance documentation.",
            index=RAG_AGENT_ETQ_INDEX,
        ),
        SourceSpec(
            source_id="ITEX",
            description="ITEX technical procedures, engineering documentation, internal how-tos.",
            index=RAG_AGENT_ITEX_INDEX,
        ),
        SourceSpec(
            source_id="SN_KB",
            description="ServiceNow Knowledge Base articles: official troubleshooting and guidance.",
            index=RAG_AGENT_SN_KB_INDEX,
        ),
        SourceSpec(
            source_id="SN_TICKETS",
            description="ServiceNow incident/request tickets: historical issues, resolutions, timelines.",
            index=RAG_AGENT_SN_TICKETS_INDEX,
        ),
    )
    return Settings(
        llm=llm,
        embeddings=embeddings,
        query_planning=query_planning,
        retrieval=retrieval,
        ranking=ranking,
        context_check=context_check,
        runtime=runtime,
        databricks=databricks,
        sources=sources,
    )


def validate_settings(settings: Settings) -> None:
    missing: list[str] = []
    if (settings.llm.provider or "").strip().lower() != "kairos":
        missing.append("RAG_AGENT_LLM_PROVIDER")
    if (settings.embeddings.provider or "").strip().lower() != "kairos":
        missing.append("RAG_AGENT_EMBEDDINGS_PROVIDER")

    if not settings.databricks.vector_search_endpoint.strip():
        missing.append("RAG_AGENT_DATABRICKS_VECTOR_SEARCH_ENDPOINT")
    if not settings.databricks.host.strip():
        missing.append("RAG_AGENT_DATABRICKS_HOST")
    if not settings.databricks.token:
        missing.append("RAG_AGENT_DATABRICKS_TOKEN")

    for spec in settings.sources:
        if not spec.index.strip():
            missing.append(f"RAG_AGENT_{spec.source_id}_INDEX")
    if missing:
        raise ValueError("Missing configuration values: " + ", ".join(sorted(set(missing))))


# COMMAND ----------
def build_llm(settings: Settings) -> BaseChatModel:
    model_type = getattr(ModelType, settings.llm.model_type, ModelType.Default)
    return create_langchain_llm(
        model_type=model_type,
        temperature=settings.llm.temperature,
        timeout=settings.llm.timeout_s,
    )


def build_embeddings(settings: Settings):
    model_type = getattr(ModelType, settings.embeddings.model_type, ModelType.Default)
    return create_embedding_mode(model_type=model_type)


# COMMAND ----------
def extract_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unterminated JSON object")


# COMMAND ----------
@dataclass(frozen=True, slots=True)
class _DatabricksIndexConfig:
    index_name: str
    retrieval_channel: str
    query_type: str
    top_k: int
    text_column: str
    return_columns: tuple[str, ...]


class _DatabricksVectorSearchRetriever(BaseRetriever):
    client: VectorSearchClient
    endpoint_name: str
    index_config: _DatabricksIndexConfig

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        index = self.client.get_index(
            endpoint_name=self.endpoint_name, index_name=self.index_config.index_name
        )
        result = index.similarity_search(
            query_text=query,
            columns=list(self.index_config.return_columns),
            num_results=self.index_config.top_k,
            query_type=self.index_config.query_type,
        )
        return _results_to_documents(
            result,
            text_column=self.index_config.text_column,
            query=query,
            endpoint_name=self.endpoint_name,
            index_name=self.index_config.index_name,
            retrieval_channel=self.index_config.retrieval_channel,
            query_type=self.index_config.query_type,
        )

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return await asyncio.to_thread(
            self._get_relevant_documents, query, run_manager=run_manager
        )


def _results_to_documents(
    result: Any,
    *,
    text_column: str,
    query: str,
    endpoint_name: str,
    index_name: str,
    retrieval_channel: str,
    query_type: str,
) -> list[Document]:
    if not isinstance(result, dict):
        return []
    manifest_cols = result.get("manifest", {}).get("columns")
    if isinstance(manifest_cols, list):
        columns = [c.get("name") for c in manifest_cols if isinstance(c, dict)]
        columns = [c for c in columns if isinstance(c, str)]
    else:
        columns = []
    data_array = None
    maybe_result = result.get("result")
    if isinstance(maybe_result, dict):
        data_array = maybe_result.get("data_array") or maybe_result.get("data")
        if not columns:
            cols2 = maybe_result.get("columns")
            if isinstance(cols2, list) and all(isinstance(x, str) for x in cols2):
                columns = cols2
    if data_array is None:
        data_array = result.get("data_array") or result.get("data")
    if not isinstance(data_array, list):
        return []

    docs: list[Document] = []
    for row in data_array:
        if not isinstance(row, list):
            continue
        metadata: dict[str, Any] = {
            "retrieval_query": query,
            "retrieval_channel": retrieval_channel,
            "retrieval_query_type": query_type,
            "databricks_vector_search_endpoint": endpoint_name,
            "databricks_vector_search_index": index_name,
        }
        for idx, value in enumerate(row):
            if idx < len(columns) and columns[idx]:
                metadata[columns[idx]] = value
            else:
                metadata[f"col_{idx}"] = value
        page_content = ""
        if text_column in metadata and isinstance(metadata[text_column], str):
            page_content = metadata[text_column]
        elif "text" in metadata and isinstance(metadata["text"], str):
            page_content = metadata["text"]
        elif "content" in metadata and isinstance(metadata["content"], str):
            page_content = metadata["content"]
        else:
            page_content = str(row[0]) if row else ""
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def _build_vector_search_client(settings: Settings) -> VectorSearchClient:
    if settings.databricks.host and settings.databricks.token:
        return VectorSearchClient(
            workspace_url=settings.databricks.host,
            personal_access_token=settings.databricks.token,
        )
    return VectorSearchClient()


def build_source_tools(*, settings: Settings, llm: BaseChatModel) -> list[BaseTool]:
    class _RetrieverInput(BaseModel):
        query: str = Field(..., description="Search query string")

    tools: list[BaseTool] = []
    for spec in settings.sources:
        tool_name = f"retrieve_{spec.source_id.lower()}"
        tool_description = f"Retrieve documents from {spec.source_id}. {spec.description}"

        def _run(query: str, *, _spec: SourceSpec = spec) -> list[Document]:
            client = _build_vector_search_client(settings)
            endpoint = settings.databricks.vector_search_endpoint
            index_name = _spec.index

            vector_retriever = _DatabricksVectorSearchRetriever(
                client=client,
                endpoint_name=endpoint,
                index_config=_DatabricksIndexConfig(
                    index_name=index_name,
                    retrieval_channel="vector",
                    query_type=settings.retrieval.vector_query_type,
                    top_k=settings.retrieval.top_k,
                    text_column=settings.retrieval.text_column,
                    return_columns=settings.retrieval.return_columns,
                ),
            )

            bm25_retriever = _DatabricksVectorSearchRetriever(
                client=client,
                endpoint_name=endpoint,
                index_config=_DatabricksIndexConfig(
                    index_name=index_name,
                    retrieval_channel="bm25",
                    query_type=settings.retrieval.bm25_query_type,
                    top_k=settings.retrieval.top_k,
                    text_column=settings.retrieval.text_column,
                    return_columns=settings.retrieval.return_columns,
                ),
            )

            ensemble = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[settings.retrieval.vector_weight, settings.retrieval.bm25_weight],
            )
            multi = MultiQueryRetriever.from_llm(
                retriever=ensemble,
                llm=llm,
                include_original=True,
            )
            docs = multi.invoke(query)
            return list(docs)[: settings.retrieval.top_k]

        async def _arun(query: str, *, _spec: SourceSpec = spec) -> list[Document]:
            return await asyncio.to_thread(_run, query, _spec=_spec)

        tools.append(
            StructuredTool.from_function(
                name=tool_name,
                description=tool_description,
                func=_run,
                coroutine=_arun,
                args_schema=_RetrieverInput,
            )
        )
    return tools


# COMMAND ----------
def build_query_planning_prompt(*, queries_per_source: int) -> ChatPromptTemplate:
    placeholders = ", ".join([f'"<q{i}>"' for i in range(1, queries_per_source + 1)])
    system = (
        "You are the Query Planning node for an enterprise RAG system.\n"
        "You do not retrieve documents and you do not call tools.\n"
        "Your task is to: (1) analyze the user's question, (2) decide which sources are required, "
        f"(3) generate exactly {queries_per_source} search queries per selected source.\n"
        "\n"
        "Rules:\n"
        "- Choose only from the provided source IDs.\n"
        "- If the question is ambiguous, select all plausible sources.\n"
        f"- For each selected source, output exactly {queries_per_source} queries.\n"
        "- Output must be valid JSON only. No markdown, no explanations.\n"
        "\n"
        "Output schema (JSON):\n"
        "{\n"
        '  "selected_sources": ["ETQ", "ITEX"],\n'
        '  "queries_by_source": {\n'
        f'    "ETQ": [{placeholders}],\n'
        f'    "ITEX": [{placeholders}]\n'
        "  }\n"
        "}\n"
    )
    user = (
        "User question:\n"
        "{user_query}\n"
        "\n"
        "Available sources (ID: description):\n"
        "{source_catalog}\n"
        "\n"
        "Return JSON only."
    )
    return ChatPromptTemplate.from_messages([("system", system), ("human", user)])


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


def build_answer_generation_prompt() -> ChatPromptTemplate:
    system = (
        "You are an enterprise assistant answering user questions using ONLY the provided context.\n"
        "Do not use external knowledge.\n"
        "If the context does not contain enough information to answer, respond exactly:\n"
        "information not found in provided sources\n"
        "Be concise and factual.\n"
        "When possible, reference the source_system in your answer."
    )
    user = (
        "User question:\n"
        "{user_query}\n"
        "\n"
        "Context chunks:\n"
        "{context}\n"
        "\n"
        "Answer:"
    )
    return ChatPromptTemplate.from_messages([("system", system), ("human", user)])


# COMMAND ----------
class AgentState(TypedDict, total=False):
    user_query: str
    queries_by_source: dict[str, list[str]]
    retrieved_chunks: list[Document]
    ranked_chunks: list[Document]
    loop_count: int
    context_is_sufficient: bool
    final_answer: str


def make_query_planning_node(*, llm: BaseChatModel, settings: Settings):
    prompt = build_query_planning_prompt(queries_per_source=settings.query_planning.queries_per_source)
    source_catalog = "\n".join(f"{spec.source_id}: {spec.description}" for spec in settings.sources)
    allowed_sources = {spec.source_id for spec in settings.sources}

    def _validate_plan(parsed: dict[str, Any]) -> dict[str, list[str]]:
        raw_sources = parsed.get("selected_sources")
        if not isinstance(raw_sources, list):
            raise ValueError("selected_sources must be a list")
        selected_sources: list[str] = []
        for item in raw_sources:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if value and value not in selected_sources:
                if value not in allowed_sources:
                    raise ValueError(f"Unknown source_id: {value}")
                selected_sources.append(value)
        if not selected_sources:
            raise ValueError("selected_sources must be non-empty")
        if len(selected_sources) > settings.query_planning.max_selected_sources:
            raise ValueError("selected_sources exceeds max_selected_sources")

        raw_queries_by_source = parsed.get("queries_by_source")
        if not isinstance(raw_queries_by_source, dict):
            raise ValueError("queries_by_source must be an object")
        queries_by_source: dict[str, list[str]] = {}
        for source_id in selected_sources:
            raw_queries = raw_queries_by_source.get(source_id)
            if not isinstance(raw_queries, list):
                raise ValueError(f"queries_by_source.{source_id} must be an array")
            normalized: list[str] = []
            for q in raw_queries:
                if isinstance(q, str) and q.strip():
                    normalized.append(q.strip())
            if len(normalized) != settings.query_planning.queries_per_source:
                raise ValueError("Wrong number of queries for source: " + source_id)
            queries_by_source[source_id] = normalized
        return queries_by_source

    def node(state: AgentState) -> AgentState:
        user_query = (state.get("user_query") or "").strip()
        if not user_query:
            raise ValueError("state.user_query is required")
        messages = prompt.format_messages(user_query=user_query, source_catalog=source_catalog)
        response = llm.invoke(messages)
        content = getattr(response, "content", response)
        parsed = extract_first_json_object(str(content))
        queries_by_source = _validate_plan(parsed)
        new_state: AgentState = dict(state)
        new_state["queries_by_source"] = queries_by_source
        return new_state

    return node


def make_retrieval_node(*, llm: BaseChatModel, settings: Settings):
    tools = build_source_tools(settings=settings, llm=llm)
    tool_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    def _expected_tool_calls_count(queries_by_source: dict[str, list[str]]) -> int:
        total = 0
        for queries in queries_by_source.values():
            if len(queries) != settings.query_planning.queries_per_source:
                raise ValueError("queries_by_source has wrong number of queries")
            total += len(queries)
        return total

    def _validate_tool_calls(
        tool_calls: list[dict[str, Any]],
        *,
        queries_by_source: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        allowed_tool_names = {f"retrieve_{k.lower()}" for k in queries_by_source.keys()}
        expected_queries = {q for qs in queries_by_source.values() for q in qs}
        validated: list[dict[str, Any]] = []
        for call in tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            query = args.get("query")
            if name not in allowed_tool_names:
                continue
            if not isinstance(query, str) or query not in expected_queries:
                continue
            if name not in tool_by_name:
                continue
            validated.append(call)
        if len(validated) != _expected_tool_calls_count(queries_by_source):
            raise ValueError("Model did not produce the required retrieval tool calls")
        return validated

    async def node(state: AgentState) -> AgentState:
        queries_by_source = state.get("queries_by_source")
        if not isinstance(queries_by_source, dict) or not queries_by_source:
            raise ValueError("state.queries_by_source is required")

        prompt = build_retrieval_tool_call_prompt(queries_by_source=queries_by_source)
        messages: list[BaseMessage] = prompt.format_messages(queries_by_source=queries_by_source)

        assistant = await llm_with_tools.ainvoke(messages)
        tool_calls = list(getattr(assistant, "tool_calls", []) or [])
        validated_calls = _validate_tool_calls(tool_calls, queries_by_source=queries_by_source)

        async def _call_one(call: dict[str, Any]) -> list[Document]:
            tool = tool_by_name[call["name"]]
            args = call.get("args") or {}
            query = args.get("query")
            return await tool.ainvoke(query)

        results = await asyncio.gather(*[_call_one(c) for c in validated_calls])
        retrieved: list[Document] = []
        for docs in results:
            retrieved.extend(list(docs))

        new_state: AgentState = dict(state)
        new_state["retrieved_chunks"] = retrieved
        return new_state

    return node


def make_ranking_node(*, settings: Settings):
    def node(state: AgentState) -> AgentState:
        chunks = state.get("retrieved_chunks") or []
        if not isinstance(chunks, list):
            chunks = []
        ranked = chunks[:]
        new_state: AgentState = dict(state)
        new_state["ranked_chunks"] = ranked[: settings.ranking.top_k_context]
        return new_state

    return node


def make_context_check_node(*, settings: Settings):
    def node(state: AgentState) -> AgentState:
        ranked = state.get("ranked_chunks") or []
        distinct_sources = set()
        for doc in ranked:
            if isinstance(doc, Document):
                distinct_sources.add((doc.metadata or {}).get("source_system", "UNKNOWN"))
        is_sufficient = len(ranked) >= settings.context_check.min_docs and len(distinct_sources) >= settings.context_check.min_distinct_sources
        new_state: AgentState = dict(state)
        new_state["context_is_sufficient"] = is_sufficient
        new_state["loop_count"] = int(state.get("loop_count", 0)) + 1
        return new_state

    return node


def make_generation_node(*, llm: BaseChatModel):
    prompt = build_answer_generation_prompt()

    def node(state: AgentState) -> AgentState:
        user_query = (state.get("user_query") or "").strip()
        ranked_chunks = state.get("ranked_chunks") or []
        if not user_query:
            raise ValueError("state.user_query is required")
        if not ranked_chunks:
            new_state: AgentState = dict(state)
            new_state["final_answer"] = "information not found in provided sources"
            return new_state

        context_lines: list[str] = []
        for i, doc in enumerate(ranked_chunks, start=1):
            metadata = doc.metadata or {}
            source_system = metadata.get("source_system", "UNKNOWN")
            context_lines.append(f"[{i}] source={source_system}")
            context_lines.append(doc.page_content)
            context_lines.append("")
        context = "\n".join(context_lines).strip()
        messages = prompt.format_messages(user_query=user_query, context=context)
        response = llm.invoke(messages)
        answer = getattr(response, "content", response)
        answer_text = str(answer).strip() or "information not found in provided sources"
        new_state: AgentState = dict(state)
        new_state["final_answer"] = answer_text
        return new_state

    return node


# COMMAND ----------
def build_graph(*, llm: BaseChatModel, settings: Settings):
    graph = StateGraph(AgentState)
    graph.add_node("query_planning", make_query_planning_node(llm=llm, settings=settings))
    graph.add_node("retrieval", make_retrieval_node(llm=llm, settings=settings))
    graph.add_node("ranking", make_ranking_node(settings=settings))
    graph.add_node("context_check", make_context_check_node(settings=settings))
    graph.add_node("generation", make_generation_node(llm=llm))

    graph.add_edge(START, "query_planning")
    graph.add_edge("query_planning", "retrieval")
    graph.add_edge("retrieval", "ranking")
    graph.add_edge("ranking", "context_check")

    def route_after_context_check(state: AgentState) -> Literal["query_planning", "generation"]:
        is_sufficient = bool(state.get("context_is_sufficient", False))
        if is_sufficient:
            return "generation"
        loop_count = int(state.get("loop_count", 0))
        if loop_count < settings.context_check.max_loop_count:
            return "query_planning"
        return "generation"

    graph.add_conditional_edges("context_check", route_after_context_check)
    graph.add_edge("generation", END)
    return graph.compile()


async def run_agent(query: str) -> str:
    settings = load_settings()
    validate_settings(settings)
    llm = build_llm(settings)
    embeddings = build_embeddings(settings)
    graph = build_graph(llm=llm, settings=settings)
    initial_state: AgentState = {"user_query": query, "loop_count": 0}
    result = await graph.ainvoke(initial_state, {"recursion_limit": settings.runtime.graph_recursion_limit})
    answer = result.get("final_answer") or ""
    return str(answer).strip() or "information not found in provided sources"


# COMMAND ----------
settings = load_settings()
validate_settings(settings)
llm = build_llm(settings)
embeddings = build_embeddings(settings)
source_catalog = "\n".join(f"{s.source_id}: {s.index}" for s in settings.sources)
print("Settings OK")
print("Vector search endpoint:", settings.databricks.vector_search_endpoint)
print(source_catalog)

# COMMAND ----------
query = "Explain the root cause and resolution steps for the incident described"
answer = asyncio.run(run_agent(query))
print(answer)
