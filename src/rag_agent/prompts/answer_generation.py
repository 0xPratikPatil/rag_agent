from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


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

