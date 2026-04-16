# Templates for System Instructions
from src.constants import SYSTEM_ROLE, SYSTEM_RULES


def build_prompt(query: str, context: str) -> str:

    role = f"[ROLE]:\n{SYSTEM_ROLE}\n\n"
    rules = f"[RULES]:\n{SYSTEM_RULES}\n\n"
    context_str = f"[CONTEXT]:\n{context}\n\n"
    question = f"[QUESTION]:\n{query}\n\n"

    trigger = "[ANSWER]:\n"

    return role + rules + context_str + question + trigger
