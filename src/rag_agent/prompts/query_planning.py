from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_query_planning_prompt(*, queries_per_source: int) -> ChatPromptTemplate:
    placeholders = ", ".join([f"\"<q{i}>\"" for i in range(1, queries_per_source + 1)])
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
        "- Queries should be concise and suitable for hybrid retrieval (keyword + semantic).\n"
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
