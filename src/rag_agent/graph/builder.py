from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from rag_agent.config.settings import Settings
from rag_agent.graph.state import AgentState
from rag_agent.nodes.context_check import make_context_check_node
from rag_agent.nodes.generation import make_generation_node
from rag_agent.nodes.query_planning import make_query_planning_node
from rag_agent.nodes.ranking import make_ranking_node
from rag_agent.nodes.retrieval import make_retrieval_node
from rag_agent.utils.graph_viz import GraphVisualizationPaths, export_graph_visualization


def build_graph(*, llm: BaseChatModel, settings: Settings):
    """Construct and compile the production RAG workflow graph."""

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


def export_built_graph_visualization(
    *,
    llm: BaseChatModel,
    settings: Settings,
    output_dir: str = "artifacts/graphs",
    base_filename: str = "rag_workflow",
    render_png: bool = False,
) -> GraphVisualizationPaths:
    graph = build_graph(llm=llm, settings=settings)
    try:
        return export_graph_visualization(
            graph=graph,
            output_dir=__project_root() / output_dir,
            base_filename=base_filename,
            render_png=render_png,
        )
    except Exception as exc:
        raise RuntimeError(f"Graph export failed: {exc}") from exc


def __project_root():
    return Path(__file__).resolve().parents[3]
