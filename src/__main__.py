import os
import sys
import json
import fire
import uuid
from pathlib import Path
from tqdm import tqdm
from src.models import AnsweredQuestion
from src.generation.prompts import build_messages
from src.indexing.builder import IndexBuilder
from src.retrieval.searcher import Searcher
from src.pipeline import RAGPipeline
from src.constants import (
    BM25_PATH,
    CHROMA_DB_PATH,
    DEFAULT_CHUNK_SIZE,
    CHUNKS_PER_QUERY,
)
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
    RagDataset,
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
        if max_chunk_size <= 150:
            print("Chunk size must be greater than 149", file=sys.stderr)
            sys.exit(1)
        index_builder = IndexBuilder(
            folder_path=repo_path,
            bm25_save_path=bm25_save_path,
            chroma_path=chroma_path,
            max_chunk_size=max_chunk_size,
        )
        index_builder.build()
        print(f"Ingestion complete! Indices saved under {repo_path}")

    def search(self, query: str, k: int = CHUNKS_PER_QUERY) -> None:
        """
        Search for a single query and return results in Pydantic JSON format.
        """
        if k <= 1:
            print("Chunks retrieved must be greater than 0", file=sys.stderr)
            sys.exit(1)
        searcher = Searcher()

        raw_results = searcher.search(query, k=k)

        minimal_sources = [
            MinimalSource(
                file_path=res["file_path"],
                first_character_index=res["first_character_index"],
                last_character_index=res["last_character_index"],
            )
            for res in raw_results
        ]

        result_output = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id=str(uuid.uuid4()),
                    question_str=query,
                    retrieved_sources=minimal_sources,
                )
            ],
            k=k,
        )

        print(result_output.model_dump_json(indent=2))

    def answer(self, query: str, k: int = CHUNKS_PER_QUERY) -> None:
        """Answer a single query using retrieved context."""
        if k <= 1:
            print("Chunks retrieved must be greater than 0", file=sys.stderr)
            sys.exit(1)
        pipeline = RAGPipeline()

        answer_text = pipeline.answer(query, k=k)

        raw_results = pipeline.searcher.search(query, k=k)
        minimal_sources = [
            MinimalSource(
                file_path=res["file_path"],
                first_character_index=res["first_character_index"],
                last_character_index=res["last_character_index"],
            )
            for res in raw_results
        ]

        result_output = StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id=str(uuid.uuid4()),
                    question_str=query,
                    retrieved_sources=minimal_sources,
                    answer=answer_text,
                )
            ],
            k=k,
        )

        print(result_output.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = CHUNKS_PER_QUERY,
    ) -> None:
        """
        Process multiple questions from a dataset and
        output search results.
        """
        if k <= 1:
            print("Chunks retrieved must be greater than 0", file=sys.stderr)
            sys.exit(1)
        os.makedirs(save_directory, exist_ok=True)
        filename = Path(dataset_path).name
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                dataset = RagDataset(**raw_data)
        except Exception:
            print("System error", file=sys.stderr)
            sys.exit(1)

        searcher = Searcher()
        results_list = []

        for q in tqdm(dataset.rag_questions, desc="Searching questions"):
            raw_results = searcher.search(q.question, k=k)

            minimal_sources = [
                MinimalSource(
                    file_path=res["file_path"],
                    first_character_index=res["first_character_index"],
                    last_character_index=res["last_character_index"],
                )
                for res in raw_results
            ]

            results_list.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question_str=q.question,
                    retrieved_sources=minimal_sources,
                )
            )

        final_output = StudentSearchResults(search_results=results_list, k=k)
        output_path = os.path.join(save_directory, filename)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_output.model_dump_json(indent=2))
        except Exception:
            print("System error", file=sys.stderr)
            sys.exit(1)

        print(f"Saved student_search_results to {output_path}")

    def answer_dataset(
        self,
        search_results_path: str,
        save_directory: str,
    ) -> None:
        """Generate answers from search results."""
        os.makedirs(save_directory, exist_ok=True)
        filename = Path(search_results_path).name

        try:

            with open(search_results_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                search_results_data = StudentSearchResults(**raw_data)

        except Exception:
            print("System error", file=sys.stderr)
            sys.exit(1)

        total_q = len(search_results_data.search_results)
        print(f"Loaded {total_q} questions from {search_results_path}")

        pipeline = RAGPipeline()
        answers_list = []

        for item in tqdm(search_results_data.search_results, desc="Answering"):
            chunks = []
            for source in item.retrieved_sources:
                try:
                    with open(source.file_path, "r", encoding="utf-8") as f:
                        f.seek(source.first_character_index)
                        length = (
                            source.last_character_index
                            - source.first_character_index
                        )
                        chunk_text = f.read(length)
                        chunks.append(
                            {"file_path": source.file_path, "text": chunk_text}
                        )
                except Exception as e:
                    print(f"Warning: Could not read {source.file_path}: {e}")

            messages = build_messages(item.question_str, chunks)
            answer_text = pipeline.llm.generate(messages)

            answers_list.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question_str=item.question_str,
                    retrieved_sources=item.retrieved_sources,
                    answer=answer_text,
                )
            )

        final_output = StudentSearchResultsAndAnswer(
            search_results=answers_list,
            k=search_results_data.k,
        )
        output_path = os.path.join(save_directory, filename)

        try:

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_output.model_dump_json(indent=2))

        except Exception:
            print("System error", file=sys.stderr)
            sys.exit(1)

        print(f"Processed {len(answers_list)} of {total_q} questions")
        print(f"Saved student_search_results_and_answer to {output_path}")

    def evaluate(
        self,
        search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Evaluate retrieval quality at k=1, 3, 5, 10."""
        from src.evaluation.metrics import recall_at_k

        try:

            with open(search_results_path, "r", encoding="utf-8") as f:
                raw_student = json.load(f)
                student_data = StudentSearchResults(**raw_student)

            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_dataset = json.load(f)
                dataset = RagDataset(**raw_dataset)

        except Exception:
            print("System error", file=sys.stderr)
            sys.exit(1)

        answered = [
            q for q in dataset.rag_questions if isinstance(q, AnsweredQuestion)
        ]

        total = len(dataset.rag_questions)
        n_answered = len(answered)
        n_with_results = sum(
            1
            for q in answered
            if any(
                r.question_id == q.question_id
                for r in student_data.search_results
            )
        )

        print("Student data is valid: True")
        print(f"Total number of questions: {total}")
        print(f"Total number of questions with sources: {n_answered}")
        print(
            f"Total number of questions with "
            f"student sources: {n_with_results}"
        )
        print()
        print("Evaluation Results")
        print("=" * 40)
        print(f"Questions evaluated: {n_answered}")

        for k in (1, 3, 5, 10):
            score = recall_at_k(answered, student_data.search_results, k)
            print(f"Recall@{k}: {score:.3f}")


def main() -> None:
    fire.Fire(RAGCli)


if __name__ == "__main__":
    main()
