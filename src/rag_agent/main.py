from __future__ import annotations

import argparse
import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from rag_agent.config.logging import configure_logging
from rag_agent.config.settings import Settings, load_settings
from rag_agent.graph.builder import build_graph, export_built_graph_visualization
from rag_agent.graph.state import AgentState


def _build_llm(settings: Settings) -> BaseChatModel:
    return ChatOpenAI(
        model=settings.llm.model_name,
        temperature=settings.llm.temperature,
        timeout=settings.llm.timeout_s,
    )


def run(query: str) -> str:
    configure_logging()
    load_dotenv()
    settings = load_settings()
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
    configure_logging()
    load_dotenv()
    settings = load_settings()
    export_built_graph_visualization(
        llm=_GraphOnlyStubModel(), settings=settings, render_png=False
    )


def export_graph_visualization_png() -> None:
    configure_logging()
    load_dotenv()
    settings = load_settings()
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


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--export-graph", action="store_true")
    parser.add_argument("--export-graph-png", action="store_true")
    args = parser.parse_args()

    if args.export_graph:
        export_graph_visualization()
        return
    if args.export_graph_png:
        export_graph_visualization_png()
        return

    raise NotImplementedError("Use rag_agent.main.run(query: str) to invoke the graph.")
