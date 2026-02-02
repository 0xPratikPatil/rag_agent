from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langgraph.pregel import Pregel


@dataclass(frozen=True, slots=True)
class GraphVisualizationPaths:
    mermaid_path: Path
    markdown_path: Path
    png_path: Path | None


def export_graph_visualization(
    *,
    graph: Pregel,
    output_dir: Path,
    base_filename: str = "rag_workflow",
    render_png: bool = False,
) -> GraphVisualizationPaths:
    """Export a compiled LangGraph graph to Mermaid and (optionally) a PNG image.

    This function operates only on the compiled graph structure and does not execute the graph.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    drawable = graph.get_graph()
    mermaid = drawable.draw_mermaid()
    if not isinstance(mermaid, str) or not mermaid.strip():
        raise RuntimeError("LangGraph Mermaid export returned empty output")

    mermaid_path = output_dir / f"{base_filename}.mmd"
    markdown_path = output_dir / f"{base_filename}.md"

    mermaid_path.write_text(mermaid, encoding="utf-8")
    markdown_path.write_text(f"```mermaid\n{mermaid}\n```\n", encoding="utf-8")

    png_path: Path | None = None
    if render_png:
        try:
            png_bytes = drawable.draw_mermaid_png()
            png_path = output_dir / f"{base_filename}.png"
            png_path.write_bytes(png_bytes)
        except Exception:
            raise RuntimeError("Failed to render graph PNG via LangGraph Mermaid export")

    return GraphVisualizationPaths(
        mermaid_path=mermaid_path, markdown_path=markdown_path, png_path=png_path
    )
