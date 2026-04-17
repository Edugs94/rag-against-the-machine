import bm25s
import chromadb
from chromadb.utils import embedding_functions
from typing import Any, cast

from src.constants import (
    BM25_PATH,
    CHROMA_DB_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DOCS_PER_QUERY,
    RRF_K,
)


class Searcher:
    """Loads indices and performs hybrid search with RRF."""

    def __init__(
        self,
        bm25_path: str = BM25_PATH,
        chroma_path: str = CHROMA_DB_PATH,
    ) -> None:
        """Initialize searcher with BM25 and ChromaDB clients."""
        self.bm25_path = bm25_path
        self.retriever = bm25s.BM25.load(self.bm25_path, load_corpus=True)

        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=DEFAULT_EMBEDDING_MODEL
        )
        self.collection = self.chroma_client.get_collection(
            name="vllm_docs", embedding_function=cast(Any, self.emb_fn)
        )

    def search(
        self, query: str, k: int = DOCS_PER_QUERY
    ) -> list[dict[str, Any]]:
        """Execute hybrid search returning fused top chunks."""
        # Retrieve a larger pool of candidates for better fusion
        candidate_pool_size = 50

        bm25_results = self._bm25_search(query, candidate_pool_size)
        dense_results = self._dense_search(query, candidate_pool_size)

        return self._apply_rrf(bm25_results, dense_results, k)

    def _bm25_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Retrieve top documents using BM25."""
        query_tokens = bm25s.tokenize(query)
        # Prevent errors if corpus is smaller than requested k
        k_val = min(k, len(self.retriever.corpus))

        results, _ = self.retriever.retrieve(query_tokens, k=k_val)
        top_chunks: list[dict[str, Any]] = results[0].tolist()

        return top_chunks

    def _dense_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Retrieve top documents using ChromaDB."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas"],
        )

        dense_chunks: list[dict[str, Any]] = []
        if not results["ids"]:
            return dense_chunks

        # Reconstruct standard chunk format from Chroma output
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

        # Process BM25 rankings
        for rank, doc in enumerate(bm25_docs):
            doc_id = f"{doc['file_path']}_{doc['first_character_index']}"
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            doc_map[doc_id] = doc

        # Process Dense rankings
        for rank, doc in enumerate(dense_docs):
            doc_id = f"{doc['file_path']}_{doc['first_character_index']}"
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            doc_map[doc_id] = doc

        # Sort by highest score descending
        sorted_ids = sorted(
            scores.keys(), key=lambda x: scores[x], reverse=True
        )

        # Keep only the requested top k
        top_ids = sorted_ids[:k]

        return [doc_map[i] for i in top_ids]
