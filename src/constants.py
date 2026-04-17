BM25_PATH: str = "data/processed/bm25_index"
CHROMA_DB_PATH: str = "data/processed/chroma_db"

DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

DEFAULT_LLM_MODEL: str = "Qwen/Qwen3-0.6B"

DEFAULT_CHUNK_SIZE: int = 2000
DOCS_PER_QUERY: int = 10

RRF_K: int = 60
CHROMA_DB_BATCH_SIZE = 250

SYSTEM_ROLE: str = (
    "You are an expert technical assistant for the vLLM library. "
    "Your objective is to provide accurate and concise answers."
)

SYSTEM_RULES = (
    "Your ONLY source of truth is the [CONTEXT]. "
    "Never use outside knowledge. "
    "Answer the user's question directly in a complete, grammatical"
    "sentence based ONLY on the text provided. "
    "If the text does not contain the answer, say 'Context insufficient'. "
    "No extra formatting."
)
