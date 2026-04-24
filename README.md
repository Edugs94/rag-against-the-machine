# RAG against the machine 🤖

**A hybrid Retrieval-Augmented Generation system that answers natural-language questions about any codebase, built from scratch in Python.**

---

## Overview

Index any codebase, ask questions in natural language, get grounded answers — fully local. Hybrid BM25 + dense retrieval, cross-encoder reranking, `Qwen/Qwen3-0.6B` for generation. Reference target is the [vLLM](https://github.com/vllm-project/vllm) codebase.

---

## Architecture

Four decoupled stages — **index → retrieve → rerank → generate** — exposed through a single Python Fire CLI.

![RAG Architecture Diagram](assets/architecture.svg)

### Indexing — Corpus Preparation

Walks the repo, chunks per file type, persists both indices. Knows nothing about queries.

| Class | Responsibility |
|:---|:---|
| `RepositoryReader` | Walks the target repo, filters by extension, streams `(path, content)` pairs |
| `TextChunker` | Per-type splitting — Markdown headers, Python `class`/`def`, C/C++/CUDA separators — preserves exact character offsets |
| `IndexBuilder` | Orchestrates chunking, writes BM25 lexical index and ChromaDB vector store |

### Retrieval — Hybrid Search and Reranking

All ranking logic lives here; the rest of the pipeline sees a clean `list[Chunk]`.

| Class | Responsibility |
|:---|:---|
| `Searcher` | Hybrid search (BM25 + dense), Reciprocal Rank Fusion, cross-encoder rerank, in-memory cache |

### Generation — Prompt Assembly and LLM

Deliberately thin. Format context, run inference, done.

| Class | Responsibility |
|:---|:---|
| `build_messages` | Assembles system+user chat template with explicit source headers |
| `LLM` | Wraps `Qwen/Qwen3-0.6B`, resolves Qwen-specific stop tokens, blocking and streaming generation |
| `RAGPipeline` | Glues `Searcher` + `LLM` for the single-query path |

### Evaluation and Interface

| Class | Responsibility |
|:---|:---|
| `recall_at_k` | Character-span overlap metric — found if any retrieved chunk covers ≥ `MIN_OVERLAP_RATIO` of a ground-truth span |
| `RAGCli` | Python Fire entry point — validates arguments, lazy resource loading, readable errors instead of tracebacks |

---

## Retrieval Pipeline

The core of the project. Each stage trades recall for precision.

| Stage | Technology | Purpose |
|:---|:---|:---|
| Lexical search | `bm25s` | Exact match recall on rare identifiers |
| Dense search | `chromadb` + `BAAI/bge-small-en-v1.5` | Semantic recall on paraphrased queries |
| Fusion | Reciprocal Rank Fusion | Score-scale-agnostic merge |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision pass on the shortlist |
| Caching | In-memory dict keyed by `(query, k)` | Free repeated queries during batch eval |

---

## Key Technical Decisions

- **Type-aware chunking** — splitting code mid-function or markdown mid-section destroys retrieval quality.
- **RRF over weighted sum** — BM25 and cosine scores live on incomparable scales; rank fusion sidesteps tuning.
- **Persistent indices on disk** — index once, query forever.
- **Deterministic generation** — `do_sample=False` plus careful Qwen stop-token handling, reproducible answers.
- **Pydantic at every boundary** — malformed inputs fail loudly and early.

---

## CLI

| Command | Action |
|:---|:---|
| `index` | Ingest the repo and persist both indices |
| `search` | Top-`k` chunks for a single query |
| `answer` | Retrieve + generate for a single query |
| `search_dataset` | Batch retrieval over a JSON dataset |
| `answer_dataset` | Generate answers from a saved search file |
| `evaluate` | `recall@{1,3,5,10}` against ground truth |
| `answer_streaming` | Stream tokens to stdout |

All hyperparameters (chunk size, model names, RRF constant, overlap ratio, etc.) live in `src/constants.py`.

---

## Setup

The corpus is **not bundled**. Drop the repo to index into `data/raw/` and build the index:

```bash
# Reference target: vLLM 0.10.1
mkdir -p data/raw
curl -L https://github.com/vllm-project/vllm/archive/refs/tags/v0.10.1.tar.gz \
  | tar -xz -C data/raw

# Build the index (writes to data/processed/)
make
```

## Installation & Usage

**Requirements:** Python 3.10+, [`uv`](https://github.com/astral-sh/uv)

```bash
uv sync
uv run python -m src answer "What are the key capabilities of Ray Serve LLM for vLLM deployment?"
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
| Linting | flake8 + mypy |
| Package manager | uv |