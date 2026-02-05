from __future__ import annotations

import argparse
import asyncio
import sys
import os

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

from rag_agent.config.logging import configure_logging
from rag_agent.config.logging import get_logger
from rag_agent.config.settings import Settings, load_settings
from rag_agent.graph.builder import build_graph, export_built_graph_visualization
from rag_agent.graph.state import AgentState
from kairos_llm.model import create_langchain_llm,create_embedding_model

def _build_llm(settings: Settings) -> BaseChatModel:
    return create_langchain_llm(
        model_type=settings.llm.model_type,
        temperature=settings.llm.temperature,
    )


def _build_embeddings(settings: Settings) -> BaseEmbeddings:
    return create_embedding_mode(
        name=settings.embeddings.model_type,
    )



def run(query: str) -> str:
    load_dotenv()
    configure_logging()
    logger = get_logger(__name__, node="main")
    settings = load_settings()
    _validate_execution_settings(settings)
    logger.info("Settings loaded", extra={"action": "settings"})
    llm = _build_llm(settings)
    graph = build_graph(llm=llm, settings=settings)

    initial_state: AgentState = {"user_query": query, "loop_count": 0}
    result = asyncio.run(
        graph.ainvoke(
            initial_state, {"recursion_limit": settings.runtime.graph_recursion_limit}
        )
    )

    answer = result.get("final_answer")
    if not isinstance(answer, str) or not answer.strip():
        return "information not found in provided sources"
    return answer.strip()


def export_graph_visualization() -> None:
    load_dotenv()
    configure_logging()
    logger = get_logger(__name__, node="main")
    settings = load_settings()
    logger.info("Exporting graph visualization (mermaid)", extra={"action": "export_graph"})
    export_built_graph_visualization(
        llm=_GraphOnlyStubModel(), settings=settings, render_png=False
    )


def export_graph_visualization_png() -> None:
    load_dotenv()
    configure_logging()
    logger = get_logger(__name__, node="main")
    settings = load_settings()
    logger.info("Exporting graph visualization (png)", extra={"action": "export_graph_png"})
    export_built_graph_visualization(
        llm=_GraphOnlyStubModel(), settings=settings, render_png=True
    )


class _GraphOnlyStubModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise NotImplementedError()

    async def ainvoke(self, messages):
        raise NotImplementedError()

def _validate_execution_settings(settings: Settings) -> None:
    missing: list[str] = []

    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")

    if not settings.databricks.vector_search_endpoint.strip():
        missing.append("RAG_AGENT_DATABRICKS_VECTOR_SEARCH_ENDPOINT")

    for spec in settings.sources:
        if not spec.index.strip() or spec.index == f"{spec.source_id}_INDEX":
            missing.append(f"RAG_AGENT_{spec.source_id}_INDEX")

    if missing:
        raise ValueError("Missing/placeholder environment variables: " + ", ".join(sorted(set(missing))))


def check_config() -> None:
    load_dotenv()
    configure_logging()
    logger = get_logger(__name__, node="main")
    settings = load_settings()
    _validate_execution_settings(settings)
    logger.info(
        "Config OK",
        extra={
            "action": "check_config",
        },
    )
    print("Config OK")
    print(f"Databricks endpoint: {settings.databricks.vector_search_endpoint}")
    for spec in settings.sources:
        print(f"- {spec.source_id}: index={spec.index}")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--export-graph", action="store_true")
    parser.add_argument("--export-graph-png", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("query_pos", nargs="*", help="Question to answer using the RAG graph.")
    args = parser.parse_args()

    load_dotenv()
    if args.verbose:
        os.environ["RAG_AGENT_LOG_LEVEL"] = "DEBUG"
    configure_logging()
    logger = get_logger(__name__, node="main")

    try:
        if args.check_config:
            check_config()
            return
        if args.export_graph:
            export_graph_visualization()
            return
        if args.export_graph_png:
            export_graph_visualization_png()
            return

        query = (args.query or " ".join(args.query_pos)).strip()
        if not query:
            parser.print_help()
            return
        logger.info("Running agent", extra={"action": "run"})
        print(run(query))
    except Exception as exc:
        logger.exception("Fatal error", extra={"action": "fatal"})
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
