# RAG against the machine 🤖

**A hybrid Retrieval-Augmented Generation system that answers natural-language questions about any codebase, built from scratch in Python.**


---

## Overview

A full Retrieval-Augmented Generation pipeline built from the ground up — from repository ingestion and type-aware chunking to hybrid retrieval, cross-encoder reranking, and local LLM generation. The reference target is the [vLLM](https://github.com/vllm-project/vllm) codebase, but any repository can be indexed and queried. Every layer is decoupled, every data boundary is Pydantic-validated, and the full pipeline runs locally with `Qwen/Qwen3-0.6B`.

---

## Architecture

The system is split into four decoupled stages — **index → retrieve → rerank → generate** — each exposed through a single Python Fire CLI.

![RAG Architecture Diagram](assets/architecture.svg)

### Indexing — Corpus Preparation

Indexing components own corpus traversal, chunk segmentation, and persistence. They have zero knowledge of queries or generation.

| Class | Responsibility |
|:---|:---|
| `RepositoryReader` | Walks the target repository, filters by allowed extensions, streams `(path, content)` pairs |
| `TextChunker` | Dispatches per file type — Markdown headers, Python class/def boundaries, C/C++/CUDA separators — and preserves exact character offsets |
| `IndexBuilder` | Orchestrates chunking and writes both the BM25 lexical index and the ChromaDB vector store to disk |

### Retrieval — Hybrid Search and Reranking

The retriever combines a lexical and a semantic index, fuses their rankings, and reranks the shortlist with a cross-encoder. All ranking logic lives here; the rest of the pipeline sees only a clean `list[Chunk]`.

| Class | Responsibility |
|:---|:---|
| `Searcher` | Loads both indices, orchestrates hybrid search, applies Reciprocal Rank Fusion, calls the cross-encoder, caches top-k results per query |

### Generation — Prompt Assembly and LLM

Generation is deliberately thin. The retriever does the heavy lifting; the LLM layer only formats context and runs inference.

| Class | Responsibility |
|:---|:---|
| `build_messages` | Formats retrieved snippets into a system+user chat template with explicit source headers |
| `LLM` | Wraps `Qwen/Qwen3-0.6B` via `transformers`, resolves all Qwen-specific stop tokens, supports both blocking and streaming generation |
| `RAGPipeline` | Glues `Searcher` + `LLM` together for the single-query path |

### Evaluation and Interface

| Class | Responsibility |
|:---|:---|
| `recall_at_k` | Character-span overlap metric — a ground-truth source is *found* if any retrieved chunk covers ≥ `MIN_OVERLAP_RATIO`% of its span |
| `RAGCli` | Python Fire entry point — validates arguments, loads resources lazily, surfaces readable errors instead of tracebacks |

---

## Retrieval Pipeline

The retriever is the core of the project. Each stage trades recall for precision as candidates flow through it.

| Stage | Technology | Purpose |
|:---|:---|:---|
| Lexical search | `bm25s` | Exact-match recall — strong on rare identifiers typical of code |
| Dense search | `chromadb` + `BAAI/bge-small-en-v1.5` | Semantic recall — handles paraphrased natural-language queries |
| Fusion | Reciprocal Rank Fusion | Score-scale-agnostic merge of the two rankings |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | High-precision scoring of `(query, chunk)` pairs on the shortlist |
| Caching | In-memory dict keyed by `(query, k)` | Zero-cost repeated queries during batch evaluation |

The candidate pool size is `max(k*3, 50)` for each retriever and `max(k*2, 30)` after RRF — enough headroom for the reranker to recover relevant chunks without paying the full cost on thousands of candidates.

---

## Key Technical Decisions

- **Type-aware chunking**: Markdown is split by H1–H6 headers, Python by `class`/`def` boundaries, C/C++/CUDA by `}`/`;`. A single recursive splitter would break functions mid-body and markdown mid-section — retrieving a complete unit matters far more than uniform size.
- **RRF over weighted sum**: BM25 and cosine similarity live on incomparable scales. Rank-based fusion sidesteps the tuning problem entirely and is robust to corpus changes.
- **Rerank only the shortlist**: Cross-encoders are 100× slower than bi-encoders. Running them on `max(k*2, 30)` candidates captures most of the quality gain at a fraction of the cost.
- **Persistent indices**: BM25 and Chroma are written to `data/processed/` once; every subsequent command is cold-start only for the models, not the corpus.
- **Deterministic generation**: `do_sample=False` plus a careful resolution of Qwen's `<|im_end|>` / `<|endoftext|>` stop tokens — reproducible answers and no trailing garbage.
- **Pydantic at every boundary**: Datasets, search results, answers — every JSON artifact is a typed model. Malformed input fails loudly and early.

---

## Configuration

Chunk size is configurable via `--max_chunk_size` (default `2000`, min `150`). Model names, index paths, RRF constant, overlap ratio and all other hyperparameters live in `src/constants.py` — a single file to tune the entire pipeline.

## Evaluation

Retrieval quality is measured with **recall@k** using character-span overlap: a ground-truth source is considered *found* if any retrieved chunk covers ≥ `MIN_OVERLAP_RATIO`% of its span. Evaluation runs against the provided `AnsweredQuestions` datasets and reports scores at k ∈ {1, 3, 5, 10}.

## CLI Commands

| Command | Action |
|:---|:---|
| `index` | Ingest the repository and persist BM25 + ChromaDB indices |
| `search` | Retrieve the top-`k` chunks for a single query |
| `answer` | Retrieve and generate an answer for a single query |
| `search_dataset` | Batch retrieval over a JSON dataset of questions |
| `answer_dataset` | Generate answers from a previously saved search file |
| `evaluate` | Compute `recall@{1,3,5,10}` against ground truth |
| `answer_streaming` | Stream generation token by token to stdout |

---

## Setup

This project **does not ship the corpus** it indexes. Drop the repository you want to query into `data/raw/` and build the index:

```bash
# Option 1 — reference target: vLLM 0.10.1
mkdir -p data/raw
curl -L https://github.com/vllm-project/vllm/archive/refs/tags/v0.10.1.tar.gz \
  | tar -xz -C data/raw

# Option 2 — any other repository
cp -r /path/to/your/repo data/raw/

# Build the index
uv run python -m src index --max_chunk_size 2000
```

Indices are written to `data/processed/` and reused by every subsequent command.

## Installation & Usage

**Requirements:** Python 3.10+, [`uv`](https://github.com/astral-sh/uv)

**Option 1: Using Make (Recommended)**
```bash
make install
make run
```

**Option 2: Using `uv` manually**
```bash
uv sync
uv run python -m src answer "What are the key capabilities of Ray Serve LLM for vLLM deployment?" --k 10
```

---

## Technical Stack

| Component | Technology |
|:---|:---|
| Language | Python 3.10+ |
| Lexical retrieval | `bm25s` |
| Vector store | `chromadb` |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `Qwen/Qwen3-0.6B` via `transformers` |
| Text splitting | `langchain-text-splitters` |
| Data validation | Pydantic |
| CLI | Python Fire |
| Progress bars | tqdm |
| Linting | flake8 + mypy |
| Package manager | uv |
