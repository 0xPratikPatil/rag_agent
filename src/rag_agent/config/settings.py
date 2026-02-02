from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class LLMSettings:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int | None = None
    timeout_s: float = 60.0


@dataclass(frozen=True, slots=True)
class EmbeddingsSettings:
    provider: str = "openai"
    model_name: str = "text-embedding-3-small"
    databricks_model_serving_endpoint: str | None = None


@dataclass(frozen=True, slots=True)
class QueryPlanningSettings:
    queries_per_source: int = 3
    max_selected_sources: int = 4
    allow_source_fallback: bool = True


@dataclass(frozen=True, slots=True)
class RankingSettings:
    rrf_k: int = 60
    top_k_context: int = 8


@dataclass(frozen=True, slots=True)
class ContextCheckSettings:
    min_docs: int = 4
    min_distinct_sources: int = 1
    max_loop_count: int = 1
    min_authoritative_docs: int = 1
    authoritative_sources: tuple[str, ...] = ("ETQ", "SN_KB")


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    top_k: int = 8
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    vector_query_type: str = "ann"
    bm25_query_type: str = "FULL_TEXT"
    text_column: str = "text"
    return_columns: tuple[str, ...] = ("text",)


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    graph_recursion_limit: int = 100


@dataclass(frozen=True, slots=True)
class DatabricksSettings:
    host: str
    token: str | None
    vector_search_endpoint: str


@dataclass(frozen=True, slots=True)
class SourceSpec:
    source_id: str
    description: str
    vector_index: str
    bm25_index: str


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


def default_sources() -> tuple[SourceSpec, ...]:
    return (
        SourceSpec(
            source_id="ETQ",
            description="ETQ quality processes, deviations, CAPA, SOPs, compliance documentation.",
            vector_index="ETQ_VECTOR_INDEX",
            bm25_index="ETQ_BM25_INDEX",
        ),
        SourceSpec(
            source_id="ITEX",
            description="ITEX technical procedures, engineering documentation, internal how-tos.",
            vector_index="ITEX_VECTOR_INDEX",
            bm25_index="ITEX_BM25_INDEX",
        ),
        SourceSpec(
            source_id="SN_KB",
            description="ServiceNow Knowledge Base articles: official troubleshooting and guidance.",
            vector_index="SN_KB_VECTOR_INDEX",
            bm25_index="SN_KB_BM25_INDEX",
        ),
        SourceSpec(
            source_id="SN_TICKETS",
            description="ServiceNow incident/request tickets: historical issues, resolutions, timelines.",
            vector_index="SN_TICKETS_VECTOR_INDEX",
            bm25_index="SN_TICKETS_BM25_INDEX",
        ),
    )


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_optional_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value


def _env_first_str(names: tuple[str, ...], default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is None or value.strip() == "":
            continue
        return value
    return default



def load_settings() -> Settings:
    llm = LLMSettings(
        model_name=_env_str("RAG_AGENT_LLM_MODEL", LLMSettings.model_name),
        temperature=_env_float("RAG_AGENT_LLM_TEMPERATURE", LLMSettings.temperature),
        timeout_s=_env_float("RAG_AGENT_LLM_TIMEOUT_S", LLMSettings.timeout_s),
    )

    embeddings = EmbeddingsSettings(
        provider=_env_str("RAG_AGENT_EMBEDDINGS_PROVIDER", EmbeddingsSettings.provider),
        model_name=_env_str("RAG_AGENT_EMBEDDINGS_MODEL", EmbeddingsSettings.model_name),
        databricks_model_serving_endpoint=_env_optional_str(
            "RAG_AGENT_DATABRICKS_EMBEDDINGS_ENDPOINT"
        ),
    )

    query_planning = QueryPlanningSettings(
        queries_per_source=_env_int(
            "RAG_AGENT_QUERIES_PER_SOURCE", QueryPlanningSettings.queries_per_source
        ),
        max_selected_sources=_env_int(
            "RAG_AGENT_MAX_SELECTED_SOURCES", QueryPlanningSettings.max_selected_sources
        ),
        allow_source_fallback=_env_bool(
            "RAG_AGENT_ALLOW_SOURCE_FALLBACK",
            QueryPlanningSettings.allow_source_fallback,
        ),
    )

    retrieval = RetrievalSettings(
        top_k=_env_int("RAG_AGENT_RETRIEVAL_TOP_K", RetrievalSettings.top_k),
        vector_weight=_env_float(
            "RAG_AGENT_RETRIEVAL_VECTOR_WEIGHT", RetrievalSettings.vector_weight
        ),
        bm25_weight=_env_float(
            "RAG_AGENT_RETRIEVAL_BM25_WEIGHT", RetrievalSettings.bm25_weight
        ),
        vector_query_type=_env_str(
            "RAG_AGENT_RETRIEVAL_VECTOR_QUERY_TYPE", RetrievalSettings.vector_query_type
        ),
        bm25_query_type=_env_str(
            "RAG_AGENT_RETRIEVAL_BM25_QUERY_TYPE", RetrievalSettings.bm25_query_type
        ),
        text_column=_env_str("RAG_AGENT_RETRIEVAL_TEXT_COLUMN", RetrievalSettings.text_column),
        return_columns=tuple(
            c.strip()
            for c in _env_str("RAG_AGENT_RETRIEVAL_RETURN_COLUMNS", "text").split(",")
            if c.strip()
        ),
    )

    ranking = RankingSettings(
        rrf_k=_env_int("RAG_AGENT_RRF_K", RankingSettings.rrf_k),
        top_k_context=_env_int("RAG_AGENT_TOP_K_CONTEXT", RankingSettings.top_k_context),
    )

    context_check = ContextCheckSettings(
        min_docs=_env_int("RAG_AGENT_MIN_DOCS", ContextCheckSettings.min_docs),
        min_distinct_sources=_env_int(
            "RAG_AGENT_MIN_DISTINCT_SOURCES", ContextCheckSettings.min_distinct_sources
        ),
        max_loop_count=_env_int(
            "RAG_AGENT_MAX_LOOP_COUNT", ContextCheckSettings.max_loop_count
        ),
        min_authoritative_docs=_env_int(
            "RAG_AGENT_MIN_AUTHORITATIVE_DOCS", ContextCheckSettings.min_authoritative_docs
        ),
        authoritative_sources=tuple(
            s.strip()
            for s in _env_str("RAG_AGENT_AUTHORITATIVE_SOURCES", "ETQ,SN_KB").split(",")
            if s.strip()
        ),
    )

    runtime = RuntimeSettings(
        graph_recursion_limit=_env_int(
            "RAG_AGENT_GRAPH_RECURSION_LIMIT", RuntimeSettings.graph_recursion_limit
        )
    )

    databricks = DatabricksSettings(
        host=_env_first_str(("RAG_AGENT_DATABRICKS_HOST", "DATABRICKS_HOST"), ""),
        token=_env_optional_str("RAG_AGENT_DATABRICKS_TOKEN")
        or _env_optional_str("DATABRICKS_TOKEN"),
        vector_search_endpoint=_env_str("RAG_AGENT_DATABRICKS_VECTOR_SEARCH_ENDPOINT", ""),
    )

    sources = []
    for spec in default_sources():
        sources.append(
            SourceSpec(
                source_id=spec.source_id,
                description=spec.description,
                vector_index=_env_str(f"RAG_AGENT_{spec.source_id}_VECTOR_INDEX", spec.vector_index),
                bm25_index=_env_str(f"RAG_AGENT_{spec.source_id}_BM25_INDEX", spec.bm25_index),
            )
        )

    settings = Settings(
        llm=llm,
        embeddings=embeddings,
        query_planning=query_planning,
        retrieval=retrieval,
        ranking=ranking,
        context_check=context_check,
        runtime=runtime,
        databricks=databricks,
        sources=tuple(sources),
    )

    if settings.query_planning.queries_per_source <= 0:
        raise ValueError("query_planning.queries_per_source must be > 0")
    if settings.query_planning.max_selected_sources <= 0:
        raise ValueError("query_planning.max_selected_sources must be > 0")

    if settings.query_planning.queries_per_source != 3:
        raise ValueError("queries_per_source must be 3 for this agent architecture")

    if settings.ranking.rrf_k <= 0:
        raise ValueError("ranking.rrf_k must be > 0")
    if settings.ranking.top_k_context <= 0:
        raise ValueError("ranking.top_k_context must be > 0")

    if settings.retrieval.top_k <= 0:
        raise ValueError("retrieval.top_k must be > 0")
    if settings.retrieval.vector_weight < 0 or settings.retrieval.bm25_weight < 0:
        raise ValueError("retrieval weights must be >= 0")
    if settings.retrieval.vector_weight == 0 and settings.retrieval.bm25_weight == 0:
        raise ValueError("At least one retrieval weight must be > 0")
    weight_sum = settings.retrieval.vector_weight + settings.retrieval.bm25_weight
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError("retrieval weights must sum to 1.0")
    if not settings.retrieval.vector_query_type.strip():
        raise ValueError("retrieval.vector_query_type must be non-empty")
    if not settings.retrieval.bm25_query_type.strip():
        raise ValueError("retrieval.bm25_query_type must be non-empty")
    if not settings.retrieval.text_column.strip():
        raise ValueError("retrieval.text_column must be non-empty")
    if not settings.retrieval.return_columns:
        raise ValueError("retrieval.return_columns must be non-empty")

    if settings.context_check.min_docs <= 0:
        raise ValueError("context_check.min_docs must be > 0")
    if settings.context_check.min_distinct_sources <= 0:
        raise ValueError("context_check.min_distinct_sources must be > 0")
    if settings.context_check.max_loop_count < 0:
        raise ValueError("context_check.max_loop_count must be >= 0")
    if settings.context_check.min_authoritative_docs < 0:
        raise ValueError("context_check.min_authoritative_docs must be >= 0")
    if (
        settings.context_check.min_authoritative_docs > 0
        and not settings.context_check.authoritative_sources
    ):
        raise ValueError("authoritative_sources must be set when min_authoritative_docs > 0")

    if settings.runtime.graph_recursion_limit <= 0:
        raise ValueError("runtime.graph_recursion_limit must be > 0")

    if not settings.databricks.vector_search_endpoint.strip():
        raise ValueError("databricks.vector_search_endpoint must be set")

    for spec in settings.sources:
        if not spec.source_id.strip():
            raise ValueError("sources[].source_id must be non-empty")
        if not spec.vector_index.strip():
            raise ValueError(f"sources[{spec.source_id}].vector_index must be non-empty")
        if not spec.bm25_index.strip():
            raise ValueError(f"sources[{spec.source_id}].bm25_index must be non-empty")

    return settings
