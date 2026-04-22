# ----------- PATHS -----------

BM25_PATH: str = "data/processed/bm25_index"
CHROMA_DB_PATH: str = "data/processed/chroma_db"
CACHE_PATH: str = "data/processed/diskcache"
DEFAULT_REPO_PATH: str = "data/raw/vllm-0.10.1"

# ---------- MODELS ----------

DEFAULT_LLM_MODEL: str = "Qwen/Qwen3-0.6B"
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

# ------ SYSTEM PROMPTS ------

SYSTEM_ROLE: str = (
    "You answer technical questions about the vLLM repository using the "
    "provided CONTEXT."
)

SYSTEM_RULES = (
    "Answer in 1-2 concise English sentences, using only information "
    "from the CONTEXT. Do not invent facts. If the CONTEXT does not "
    "contain the answer, reply exactly: Context insufficient."
)

# ------- DEFAULT INTS -------

DEFAULT_CHUNK_SIZE: int = 2000
MIN_CHUNK_SIZE: int = 150
DEFAULT_CHUNK_OVERLAP: int = 150

CHUNKS_PER_QUERY: int = 10
CHUNKS_FOR_LLM: int = CHUNKS_PER_QUERY

CHROMA_DB_BATCH_SIZE: int = 250

RERANKER_CANDIDATES: int = 30
RRF_K: int = 60
MIN_OVERLAP_RATIO: float = 0.05
