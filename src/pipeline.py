# Pipeline for a single query
from src.retrieval.searcher import Searcher
from src.generation.llm import LLM
from src.generation.prompts import build_messages
from src.constants import CHUNKS_PER_QUERY


class RAGPipeline:
    """Orchestrates the Retrieval-Augmented Generation flow."""

    def __init__(self) -> None:
        self.searcher = Searcher()
        self.llm = LLM()

    def answer(self, query: str, k: int = CHUNKS_PER_QUERY) -> str:
        """Executes the full RAG pipeline for a given query."""
        print("Searching for context...")
        context = self.searcher.search(query, k)

        print("Building the prompt...")
        messages = build_messages(query, context)

        print("Generating answer...")
        response = self.llm.generate(messages)

        return response
