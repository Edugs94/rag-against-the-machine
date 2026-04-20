# Templates for System Instructions
from src.constants import (
    SYSTEM_ROLE,
    SYSTEM_RULES,
    CHUNKS_FOR_LLM,
    MAX_CHARS_PER_CHUNK,
)


def _truncate(text: str, max_chars: int) -> str:
    """Truncates a chunk to max_chars, cutting on the last whitespace."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " [...]"


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    selected = chunks[:CHUNKS_FOR_LLM]

    context = "\n\n---\n\n".join(
        f"[Source: {c['file_path']}]\n"
        f"{_truncate(c['text'], MAX_CHARS_PER_CHUNK)}"
        for c in selected
    )

    system_content = f"{SYSTEM_ROLE}\n\n{SYSTEM_RULES}"
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    return messages
