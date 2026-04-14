# Logic to load the BM25 index and return Top-K chunks
import bm25s
import numpy as np
from typing import Any

from src.constants import DEFAULT_INDEX_PATH, DOCS_PER_QUERY


class Searcher:
    """Loads the BM25 index and retrieves relevant chunks."""

    def __init__(self, index_path: str = DEFAULT_INDEX_PATH) -> None:
        """Initialize the searcher by loading the index from disk."""
        self.index_path = index_path
        self.retriever = bm25s.BM25.load(self.index_path, load_corpus=True)

    def search(
        self, query: str, k: int = DOCS_PER_QUERY
    ) -> list[dict[str, Any]]:
        """Search the most relevant chunks for a given query."""
        query_tokens = bm25s.tokenize(query)

        results: np.ndarray
        scores: np.ndarray
        results, scores = self.retriever.retrieve(query_tokens, k=k)

        top_chunks: list[dict[str, Any]] = results[0].tolist()

        return top_chunks
