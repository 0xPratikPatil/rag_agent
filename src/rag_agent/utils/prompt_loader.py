from __future__ import annotations

from importlib.resources import files


def load_prompt_text(filename: str) -> str:
    return files("rag_agent.prompts").joinpath(filename).read_text(encoding="utf-8")

