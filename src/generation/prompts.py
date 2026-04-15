# Templates for System Instructions
from src.constants import SYSTEM_ROLE, SYSTEM_RULES


def build_prompt(query: str, context: str) -> str:

    role = f"[ROLE]:{SYSTEM_ROLE}\n"
    rules = f"[RULES]:{SYSTEM_RULES}\n"
    context = f"[CONTEXT]:{context}\n"
    question = f"[QUESTION]:{query}\n"

    return role + rules + context + question
