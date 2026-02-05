from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from rag_agent.config.logging import get_logger
from rag_agent.graph.state import AgentState
from rag_agent.prompts.answer_generation import build_answer_generation_prompt


def make_generation_node(*, llm: BaseChatModel) -> Callable[[AgentState], AgentState]:
    """Generate the final answer strictly from ranked context already in state."""

    prompt = build_answer_generation_prompt()
    logger = get_logger(__name__, node="generation")

    def node(state: AgentState) -> AgentState:
        logger.info("Generating answer from ranked context", extra={"action": "generate"})
        user_query = (state.get("user_query") or "").strip()
        if not user_query:
            raise ValueError("state.user_query is required")

        ranked_chunks = state.get("ranked_chunks")
        if not isinstance(ranked_chunks, list) or not ranked_chunks:
            new_state: AgentState = dict(state)
            new_state["final_answer"] = "information not found in provided sources"
            logger.warning(
                "No ranked chunks available for generation",
                extra={"action": "no_context"},
            )
            return new_state

        context_lines: list[str] = []
        for i, doc in enumerate(ranked_chunks, start=1):
            metadata = doc.metadata or {}
            source_system = metadata.get("source_system", "UNKNOWN")
            query_index = metadata.get("query_index", "NA")
            context_lines.append(f"[{i}] source={source_system} query_index={query_index}")
            context_lines.append(doc.page_content)
            context_lines.append("")

        context = "\n".join(context_lines).strip()
        messages = prompt.format_messages(
            user_query=user_query,
            context=context,
        )

        response = llm.invoke(messages)
        answer = getattr(response, "content", response)
        answer_text = str(answer).strip()
        if not answer_text:
            answer_text = "information not found in provided sources"

        new_state: AgentState = dict(state)
        new_state["final_answer"] = answer_text
        logger.info("Answer generation completed", extra={"action": "generate_done"})
        return new_state

    return node
