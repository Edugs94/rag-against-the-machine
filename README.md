# RAG-against-the-machine

> 🚧 Work in progress.

RAG pipeline for answering technical questions over the [vLLM](https://github.com/vllm-project/vllm) codebase. Hybrid BM25 + dense retrieval, RRF fusion, cross-encoder rerank, extractive generation with `Qwen/Qwen3-0.6B`.

## Stack

| Layer      | Module                      | What it does                                          |
|------------|-----------------------------|-------------------------------------------------------|
| Indexing   | `src/indexing/`             | Reads the repo, chunks files, writes BM25 + Chroma    |
| Retrieval  | `src/retrieval/searcher.py` | Hybrid search, RRF, cross-encoder rerank              |
| Generation | `src/generation/`           | Builds the prompt and runs Qwen3-0.6B                 |
| Evaluation | `src/evaluation/metrics.py` | Recall@k with character overlap                       |

CLI is [`fire`](https://github.com/google/python-fire). Data contracts live in `src/models.py` (Pydantic). The file allowlist is produced by [`filetype_scanner`](https://github.com/Edugs94/filetype_scanner).

## Install

```bash
make install
```

Uses [`uv`](https://github.com/astral-sh/uv) under the hood. `filetype_scanner` is not installed as a package; the `ALLOWED_EXTENSIONS` set is imported directly.

## Usage

All commands are exposed through `python -m src <command>`.

### Index a repo

```bash
python -m src index --repo_path /path/to/vllm --max_chunk_size 2000
```

Writes `data/processed/bm25_index*` and `data/processed/chroma_db/` (collection `vllm_docs`).

### Single query

```bash
python -m src search --query "How does continuous batching work?" --k 10
python -m src answer --query "What is PagedAttention?" --k 10
```

### Batch

```bash
python -m src search_dataset \
    --dataset_path data/questions.json \
    --save_directory data/out \
    --k 10

python -m src answer_dataset \
    --search_results_path data/out/questions.json \
    --save_directory data/answers

python -m src evaluate \
    --search_results_path data/out/questions.json \
    --dataset_path data/questions.json
```

### Dataset format

`search_dataset` takes a JSON file matching the `RagDataset` schema. Questions can be unanswered (just `question_id` + `question`) or answered (with ground-truth `sources` and `answer`). Unanswered ones are ignored by `evaluate`.

```json
{
  "rag_questions": [
    {
      "question_id": "c2d6f1a8-9a4e-4a91-bd1d-37d9b0b5e111",
      "question": "How does PagedAttention work?"
    },
    {
      "question_id": "7e3b1d24-8c5f-4e2a-9a10-1c2f5b3d9a42",
      "question": "Where is continuous batching implemented?",
      "sources": [
        {
          "file_path": "vllm/core/scheduler.py",
          "first_character_index": 1240,
          "last_character_index": 2310
        }
      ],
      "answer": "The Scheduler class in vllm/core/scheduler.py handles continuous batching."
    }
  ]
}
```

## How it works

**Chunking.** Split strategy depends on the file type:

- `.md` / `.mdx`: header-based split (H1–H6). Sections that already fit stay whole.
- `.py`: cuts on `class` / `def`.
- `.cpp`, `.cu`, `.cuh`, `.h`, `.hpp`: cuts on `}` and `;`.
- anything else: newlines and spaces.

Every chunk stores `file_path`, `first_character_index`, `last_character_index`. Non-UTF-8 files are skipped.

**Retrieval.** For each query:

1. BM25 (`bm25s`) and dense (Chroma + `BAAI/bge-small-en-v1.5`) each return 50 candidates.
2. RRF (K = 60) combines both rankings and keeps 30.
3. Cross-encoder `ms-marco-MiniLM-L-6-v2` reranks and returns top-`k` (default 10).

The `Searcher` caches results in memory by `(query, k)`.

**Generation.** `Qwen/Qwen3-0.6B`, greedy decoding, `max_new_tokens=128`, `enable_thinking=False`. The prompt concatenates the top 10 chunks (each truncated at 1000 chars on a word boundary) after a system message that restricts the model to the context. If the context does not contain the answer the model must reply exactly `Context insufficient`.

**Evaluation.** Recall@k with character-overlap matching: a retrieved source is a hit when the file path matches and the interval overlaps at least 5% of the ground-truth interval. Reported at k ∈ {1, 3, 5, 10}.

## Project structure

```
src/
├── __main__.py            Fire CLI
├── constants.py           hyperparameters
├── models.py              Pydantic schemas
├── pipeline.py            Searcher + LLM
├── indexing/
│   ├── reader.py
│   ├── chunker.py
│   └── builder.py
├── retrieval/
│   └── searcher.py
├── generation/
│   ├── llm.py
│   └── prompts.py
└── evaluation/
    └── metrics.py
```

## Configuration

All constants live in `src/constants.py`.

| Constant                   | Value                                    |
|----------------------------|------------------------------------------|
| `DEFAULT_EMBEDDING_MODEL`  | `BAAI/bge-small-en-v1.5`                 |
| `DEFAULT_LLM_MODEL`        | `Qwen/Qwen3-0.6B`                        |
| `RERANKER_MODEL`           | `cross-encoder/ms-marco-MiniLM-L-6-v2`   |
| `DEFAULT_CHUNK_SIZE`       | 2000                                     |
| `DEFAULT_CHUNK_OVERLAP`    | 150                                      |
| `CHUNKS_PER_QUERY`           | 10                                       |
| `RERANKER_CANDIDATES`      | 30                                       |
| `RRF_K`                    | 60                                       |
| `CHUNKS_FOR_LLM`           | 10                                       |
| `CHROMA_DB_BATCH_SIZE`     | 250                                      |
| `BM25_PATH`                | `data/processed/bm25_index`              |
| `CHROMA_DB_PATH`           | `data/processed/chroma_db`               |
