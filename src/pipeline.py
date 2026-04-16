# Pipeline for a single query
from src.retrieval.searcher import Searcher
from src.generation.llm import LLM
from src.generation.prompts import build_prompt


class RAGPipeline:
    """Orchestrates the Retrieval-Augmented Generation flow."""

    def __init__(self) -> None:
        """Initializes the core RAG components."""
        self.searcher = Searcher()
        self.llm = LLM()

    def answer(self, query: str, k) -> str:
        """Executes the full RAG pipeline for a given query."""
        print("Searching for context...")
        context = self.searcher.search(query, k)

        print("Building the prompt...")
        prompt = build_prompt(query, context)

        print("Generating answer...")
        response = self.llm.generate(prompt)

        return response
