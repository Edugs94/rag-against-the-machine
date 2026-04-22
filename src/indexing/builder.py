'''Logic to build and save the BM25 and chromadb index'''
import bm25s
import chromadb
from typing import Any, cast
from tqdm import tqdm
from chromadb.utils import embedding_functions
from chromadb.api.client import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
from src.indexing.reader import RepositoryReader
from src.indexing.chunker import TextChunker
from src.constants import (
    DEFAULT_CHUNK_SIZE,
    BM25_PATH,
    CHROMA_DB_PATH,
    DEFAULT_EMBEDDING_MODEL,
    CHROMA_DB_BATCH_SIZE,
)


class IndexBuilder:
    """Orchestrates the reading, chunking, and dual indexing process."""

    def __init__(
        self,
        folder_path: str,
        bm25_save_path: str = BM25_PATH,
        chroma_path: str = CHROMA_DB_PATH,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Initialize the builder with lexical and semantic providers."""
        self.folder_path = folder_path
        self.bm25_save_path = bm25_save_path
        self.reader = RepositoryReader(self.folder_path)
        self.chunker = TextChunker(max_size=max_chunk_size)

        self.chroma_client: ClientAPI = chromadb.PersistentClient(
            path=chroma_path
        )
        self.emb_fn: SentenceTransformerEmbeddingFunction = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=DEFAULT_EMBEDDING_MODEL
            )
        )
        self.collection: Collection = (
            self.chroma_client.get_or_create_collection(
                name="vllm_docs", embedding_function=cast(Any, self.emb_fn)
            )
        )

    def build(self) -> None:
        """Build and save both BM25 and ChromaDB indices."""
        bm25_corpus = []
        texts_for_bm25 = []

        chroma_texts = []
        chroma_metadatas = []
        chroma_ids = []

        for file_path, content in self.reader.get_files_content():
            for chunk in self.chunker.process_file(file_path, content):
                bm25_corpus.append(chunk)
                texts_for_bm25.append(chunk["text"])

                chroma_texts.append(chunk["text"])
                chroma_metadatas.append(
                    {
                        "file_path": chunk["file_path"],
                        "start": chunk["first_character_index"],
                        "end": chunk["last_character_index"],
                    }
                )
                chunk_id = (
                    f"{chunk['file_path']}_{chunk['first_character_index']}"
                )
                chroma_ids.append(chunk_id)

        # Save Lexical Index (BM25)
        tokens = bm25s.tokenize(texts_for_bm25)
        indexer = bm25s.BM25()
        indexer.index(tokens)
        indexer.save(self.bm25_save_path, corpus=bm25_corpus)

        # Save Semantic Index (ChromaDB)
        existing = self.collection.get()
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        for i in tqdm(
            range(0, len(chroma_ids), CHROMA_DB_BATCH_SIZE),
            desc="Saving to ChromaDB",
        ):
            end_idx = i + CHROMA_DB_BATCH_SIZE
            self.collection.add(
                documents=chroma_texts[i:end_idx],
                metadatas=cast(Any, chroma_metadatas[i:end_idx]),
                ids=chroma_ids[i:end_idx],
            )

        if not bm25_corpus:
            raise RuntimeError(
                f"No indexable files found in {self.folder_path}. "
                f"Check the path and supported extensions."
            )
