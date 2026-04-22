# Templates for System Instructions
from src.constants import (
    SYSTEM_ROLE,
    SYSTEM_RULES,
    CHUNKS_FOR_LLM,
)


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    """
    Builds the messages for the LLM using the full retrieved context.
    """
    selected = chunks[:CHUNKS_FOR_LLM]

    context_parts = []
    for c in selected:
        header = f"[Source: {c['file_path']}]"
        context_parts.append(f"{header}\n{c['text']}")

    context = "\n\n---\n\n".join(context_parts)

    system_content = f"{SYSTEM_ROLE}\n\n{SYSTEM_RULES}"
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    return messages
