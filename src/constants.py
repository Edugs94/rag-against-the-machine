DEFAULT_INDEX_PATH: str = "data/processed/bm25_index"

DEFAULT_CHUNK_SIZE: int = 2000

DOCS_PER_QUERY: int = 10

SYSTEM_ROLE: str = (
    "You are an expert technical assistant for the vLLM library. "
    "Your objective is to provide accurate and concise answers."
)

SYSTEM_RULES = (
    "Answer using ONLY the information in the [CONTEXT] section. "
    "If the answer is not in the context, you must respond: "
    "'I don't have enough information to answer that.' "
    "Do not invent code or explanations."
)
