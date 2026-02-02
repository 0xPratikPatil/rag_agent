from __future__ import annotations

import json


def extract_first_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    idx = 0
    max_tries = 50
    tries = 0
    while tries < max_tries:
        tries += 1
        brace_idx = text.find("{", idx)
        if brace_idx == -1:
            break
        candidate = text[brace_idx:].lstrip()
        try:
            obj, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            idx = brace_idx + 1
            continue
        if isinstance(obj, dict):
            return obj
        idx = brace_idx + 1
    raise ValueError("No JSON object found in LLM output")

