import fire
import json
from src.indexing.builder import IndexBuilder
from src.retrieval.searcher import Searcher
from src.constants import (
    DEFAULT_INDEX_PATH,
    DEFAULT_CHUNK_SIZE,
    DOCS_PER_QUERY,
)


class RAGCli:
    """Main CLI"""

    def index(
        self,
        repo_path: str,
        save_path: str = DEFAULT_INDEX_PATH,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Index the repository and save the BM25 model."""
        index_builder = IndexBuilder(
            folder_path=repo_path,
            save_path=save_path,
            max_chunk_size=max_chunk_size,
        )
        index_builder.build()

    def search(self, query: str, k: int = DOCS_PER_QUERY) -> None:
        """Search the most relevant chunks in the index for a query."""
        searcher = Searcher()
        results = searcher.search(query, k=k)
        print(json.dumps(results, indent=2))


def main() -> None:
    fire.Fire(RAGCli)


if __name__ == "__main__":
    main()
