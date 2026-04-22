import bm25s
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from typing import Any, cast
from src.constants import (
    BM25_PATH,
    CHROMA_DB_PATH,
    DEFAULT_EMBEDDING_MODEL,
    CHUNKS_PER_QUERY,
    RRF_K,
    RERANKER_MODEL,
    RERANKER_CANDIDATES,
)


class Searcher:
    """Loads indices and performs hybrid search with RRF + reranking."""

    def __init__(
        self,
        bm25_path: str = BM25_PATH,
        chroma_path: str = CHROMA_DB_PATH,
    ) -> None:
        self.bm25_path = bm25_path
        self.retriever = bm25s.BM25.load(self.bm25_path, load_corpus=True)
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=DEFAULT_EMBEDDING_MODEL
        )
        self.collection = self.chroma_client.get_collection(
            name="vllm_docs", embedding_function=cast(Any, self.emb_fn)
        )
        self.reranker = CrossEncoder(RERANKER_MODEL)
        self._cache: dict[tuple[str, int], list[dict[str, Any]]] = {}

    def search(
        self, query: str, k: int = CHUNKS_PER_QUERY
    ) -> list[dict[str, Any]]:
        """
        Cached entry point. Returns the top-k chunks for a given query,
        falling back to the full hybrid pipeline on cache miss.
        """
        key = (query, k)
        if key in self._cache:
            return self._cache[key]
        result = self._do_search(query, k)
        self._cache[key] = result
        return result

    def _do_search(
        self, query: str, k: int = CHUNKS_PER_QUERY
    ) -> list[dict[str, Any]]:
        """
        Execute hybrid search with RRF fusion and cross-encoder reranking.
        """
        candidate_pool_size = 50

        bm25_results = self._bm25_search(query, candidate_pool_size)
        chromadb_results = self._chromadb_search(query, candidate_pool_size)

        rrf_candidates = self._apply_rrf(
            bm25_results, chromadb_results, k=RERANKER_CANDIDATES
        )

        return self._rerank(query, rrf_candidates, k)

    def _bm25_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Retrieve top documents using BM25."""
        query_tokens = bm25s.tokenize(query)
        k_val = min(k, len(self.retriever.corpus))
        results, _ = self.retriever.retrieve(query_tokens, k=k_val)
        return cast(list[dict[str, Any]], results[0].tolist())

    def _chromadb_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Retrieve top documents using ChromaDB."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas"],
        )
        dense_chunks: list[dict[str, Any]] = []
        if not results["ids"]:
            return dense_chunks

        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]  # type: ignore
            doc_text = results["documents"][0][i]  # type: ignore
            chunk = {
                "file_path": metadata["file_path"],
                "text": doc_text,
                "first_character_index": metadata["start"],
                "last_character_index": metadata["end"],
            }
            dense_chunks.append(chunk)
        return dense_chunks

    def _apply_rrf(
        self,
        bm25_docs: list[dict[str, Any]],
        dense_docs: list[dict[str, Any]],
        k: int,
    ) -> list[dict[str, Any]]:
        """Fuse rankings using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        for rank, doc in enumerate(bm25_docs):
            doc_id = f"{doc['file_path']}_{doc['first_character_index']}"
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            doc_map[doc_id] = doc

        for rank, doc in enumerate(dense_docs):
            doc_id = f"{doc['file_path']}_{doc['first_character_index']}"
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            doc_map[doc_id] = doc

        sorted_ids = sorted(
            scores.keys(), key=lambda x: scores[x], reverse=True
        )
        return [doc_map[i] for i in sorted_ids[:k]]

    def _rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        k: int,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using a cross-encoder model."""
        if not candidates:
            return []

        pairs = [[query, doc["text"]] for doc in candidates]
        scores = self.reranker.predict(pairs)

        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        return [doc for _, doc in scored[:k]]
