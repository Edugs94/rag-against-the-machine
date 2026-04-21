BM25_PATH: str = "data/processed/bm25_index"
CHROMA_DB_PATH: str = "data/processed/chroma_db"

DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

DEFAULT_LLM_MODEL: str = "Qwen/Qwen3-0.6B"
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_CANDIDATES: int = 30
DEFAULT_CHUNK_SIZE: int = 2000
DEFAULT_CHUNK_OVERLAP: int = 150
DOCS_PER_QUERY: int = 10

CHUNKS_FOR_LLM: int = DOCS_PER_QUERY

RRF_K: int = 60
CHROMA_DB_BATCH_SIZE = 250

MAX_CHARS_PER_CHUNK: int = 1000

SYSTEM_ROLE: str = (
    "You answer technical questions about the vLLM repository using the "
    "provided CONTEXT."
)

SYSTEM_RULES = (
    "Answer in 1-2 concise English sentences, using only information "
    "from the CONTEXT. Do not invent facts. If the CONTEXT does not "
    "contain the answer, reply exactly: Context insufficient."
)

CACHE_PATH: str = "data/processed/diskcache"