# Logic to build and save the BM25 index
import bm25s
from typing import Any
from src.indexing.reader import RepositoryReader
from src.indexing.chunker import TextChunker


class IndexBuilder:
    """Orchestrates the reading, chunking, and indexing process."""

    def __init__(self, folder_path: str, save_path: str) -> None:
        """Initialize the builder with necessary paths."""
        self.folder_path = folder_path
        self.save_path = save_path
        self.reader = RepositoryReader(self.folder_path)
        self.chunker = TextChunker()

    def build(self) -> None:
        """Build and save the BM25 index from the repository."""
        corpus: list[dict[str, Any]] = []
        corpus_texts: list[str] = []

        for file_path, content in self.reader.get_files_content():
            for chunk in self.chunker.process_file(file_path, content):
                corpus.append(chunk)
                corpus_texts.append(chunk["text"])

        corpus_tokens = bm25s.tokenize(corpus_texts)

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        retriever.save(self.save_path, corpus=corpus)
