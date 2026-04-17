import fire
import json
from src.indexing.builder import IndexBuilder
from src.retrieval.searcher import Searcher
from src.pipeline import RAGPipeline
from src.constants import (
    BM25_PATH,
    CHROMA_DB_PATH,
    DEFAULT_CHUNK_SIZE,
    DOCS_PER_QUERY,
)


class RAGCli:
    """Main CLI"""

    def index(
        self,
        repo_path: str,
        bm25_save_path: str = BM25_PATH,
        chroma_path: str = CHROMA_DB_PATH,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Index the repository and save BM25 and Chroma models."""
        index_builder = IndexBuilder(
            folder_path=repo_path,
            bm25_save_path=bm25_save_path,
            chroma_path=chroma_path,
            max_chunk_size=max_chunk_size,
        )
        index_builder.build()
        print(f"Ingestion complete! Indices saved under {repo_path}")

    def search(self, query: str, k: int = DOCS_PER_QUERY) -> None:
        """Search the most relevant chunks in the index for a query."""
        searcher = Searcher()
        results = searcher.search(query, k=k)
        print(json.dumps(results, indent=2))

    def answer(self, query: str, k: int = DOCS_PER_QUERY) -> None:
        """Responds to a query with retrieved chunks by hybrid search."""
        pipeline = RAGPipeline()
        print(pipeline.answer(query, k))


def main() -> None:
    fire.Fire(RAGCli)


if __name__ == "__main__":
    main()