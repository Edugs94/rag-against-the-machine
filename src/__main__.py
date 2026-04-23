import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import sys  # noqa: E402: E402
import uuid  # noqa: E402: E402
import fire  # noqa: E402
from pathlib import Path  # noqa: E402
from tqdm import tqdm  # noqa: E402
from src.models import AnsweredQuestion  # noqa: E402
from src.generation.prompts import build_messages  # noqa: E402
from src.indexing.builder import IndexBuilder  # noqa: E402
from src.retrieval.searcher import Searcher  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.utils import (load_json_as_model, write_model_as_json,  # noqa: E402
                       sanitize_query, ensure_directory)
from src.constants import (  # noqa: E402
    BM25_PATH,
    CHROMA_DB_PATH,
    DEFAULT_CHUNK_SIZE,
    CHUNKS_PER_QUERY,
    DEFAULT_REPO_PATH,
    MIN_CHUNK_SIZE,
)
from src.models import (  # noqa: E402
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
    RagDataset,
)


def _load_searcher() -> Searcher:
    """Load the search index with a clear error if it's missing."""
    try:
        return Searcher()
    except FileNotFoundError:
        print(
            "Index not found. Run `python -m src index` first.",
            file=sys.stderr,
        )
        sys.exit(1)
    except OSError as e:
        print(f"Failed to load search index: {e}", file=sys.stderr)
        sys.exit(1)


def _load_pipeline() -> RAGPipeline:
    """Load the full RAG pipeline with a clear error on failure."""
    try:
        return RAGPipeline()
    except FileNotFoundError:
        print(
            "Index not found. Run `python -m src index` first.",
            file=sys.stderr,
        )
        sys.exit(1)
    except OSError as e:
        print(f"Failed to load pipeline resources: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Failed to initialize pipeline: {e}", file=sys.stderr)
        sys.exit(1)


class RAGCli:
    """Main CLI"""

    def index(
        self,
        repo_path: str = DEFAULT_REPO_PATH,
        bm25_save_path: str = BM25_PATH,
        chroma_path: str = CHROMA_DB_PATH,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Index the repository and save BM25 and Chroma models."""
        try:
            max_chunk_size = int(max_chunk_size)
        except ValueError:
            print("Chunks size must be positive integer",
                  file=sys.stderr)
            sys.exit(1)
        if max_chunk_size <= MIN_CHUNK_SIZE:
            print(f"Chunk size must be greater than {MIN_CHUNK_SIZE}",
                  file=sys.stderr)
            sys.exit(1)
        index_builder = IndexBuilder(
            folder_path=repo_path,
            bm25_save_path=bm25_save_path,
            chroma_path=chroma_path,
            max_chunk_size=max_chunk_size,
        )
        try:
            index_builder.build()
        except FileNotFoundError:
            print(f"Repository not found: {repo_path}", file=sys.stderr)
            sys.exit(1)
        except RuntimeError as e:
            print(f"Indexing failed: {e}", file=sys.stderr)
            sys.exit(1)
        except PermissionError as e:
            print(
                f"Permission denied while building index: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        except OSError as e:
            print(f"I/O error while building index: {e}", file=sys.stderr)
            sys.exit(1)
        print("Ingestion complete!")
        print(f"BM25 Indices saved under {bm25_save_path}")
        print(f"Chromadb vectorized database saved under {chroma_path}")

    def search(self, query: str, k: int = CHUNKS_PER_QUERY) -> None:
        """Search for a single query and return results as Pydantic JSON."""
        try:
            k = int(k)
        except ValueError:
            print("Chunks to retrieve must be positive integer",
                  file=sys.stderr)
            sys.exit(1)
        if k < 1:
            print("Chunks retrieved must be greater than 0", file=sys.stderr)
            sys.exit(1)
        query = sanitize_query(query)
        searcher = _load_searcher()

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
        try:
            k = int(k)
        except ValueError:
            print("Chunks to retrieve must be positive integer",
                  file=sys.stderr)
            sys.exit(1)
        if k < 1:
            print("Chunks retrieved must be greater than 0", file=sys.stderr)
            sys.exit(1)
        query = sanitize_query(query)
        pipeline = _load_pipeline()

        raw_results = pipeline.searcher.search(query, k=k)
        messages = build_messages(query, raw_results)
        answer_text = pipeline.llm.generate(messages)

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
        """Process questions from a dataset and output search results."""
        try:
            k = int(k)
        except ValueError:
            print("Chunks to retrieve must be positive integer",
                  file=sys.stderr)
            sys.exit(1)
        if k < 1:
            print("Chunks retrieved must be greater than 0", file=sys.stderr)
            sys.exit(1)
        ensure_directory(save_directory)
        filename = Path(dataset_path).name

        dataset = load_json_as_model(dataset_path, RagDataset, "Dataset")

        searcher = _load_searcher()
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
        write_model_as_json(output_path, final_output, "search results")

        print(f"Saved student_search_results to {output_path}")

    def answer_dataset(
        self,
        search_results_path: str,
        save_directory: str,
    ) -> None:
        """Generate answers from search results."""
        ensure_directory(save_directory)
        filename = Path(search_results_path).name

        search_results_data = load_json_as_model(
            search_results_path, StudentSearchResults, "Search results"
        )

        total_q = len(search_results_data.search_results)
        print(f"Loaded {total_q} questions from {search_results_path}")

        pipeline = _load_pipeline()
        answers_list = []

        for item in tqdm(search_results_data.search_results, desc="Answering"):
            chunks = []
            for source in item.retrieved_sources:
                try:
                    with open(source.file_path, "r", encoding="utf-8") as f:
                        full_text = f.read()
                    chunk_text = full_text[
                        source.first_character_index:source.
                        last_character_index
                    ]
                    chunks.append(
                        {"file_path": source.file_path, "text": chunk_text}
                    )
                except FileNotFoundError:
                    print(
                        f"Warning: source file not found, skipping: "
                        f"{source.file_path}",
                        file=sys.stderr,
                    )
                except UnicodeDecodeError:
                    print(
                        f"Warning: non-UTF8 file, skipping: "
                        f"{source.file_path}",
                        file=sys.stderr,
                    )
                except OSError as e:
                    print(
                        f"Warning: could not read {source.file_path}: {e}",
                        file=sys.stderr,
                    )

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
        write_model_as_json(output_path, final_output, "answers")

        print(f"Processed {len(answers_list)} of {total_q} questions")
        print(f"Saved student_search_results_and_answer to {output_path}")

    def answer_streaming(
        self, query: str, k: int = CHUNKS_PER_QUERY
    ) -> None:
        """Answer a query and stream the response to stdout (bonus command)."""
        try:
            k = int(k)
        except ValueError:
            print("Chunks to retrieve must be positive integer",
                  file=sys.stderr)
            sys.exit(1)
        if k < 1:
            print("Chunks retrieved must be greater than 0", file=sys.stderr)
            sys.exit(1)
        query = sanitize_query(query)
        pipeline = _load_pipeline()
        pipeline.answer_streaming(query, k=k)

    def evaluate(
        self,
        search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Evaluate retrieval quality at k=1, 3, 5, 10."""
        from src.evaluation.metrics import recall_at_k

        student_data = load_json_as_model(
            search_results_path, StudentSearchResults, "Search results"
        )
        dataset = load_json_as_model(dataset_path, RagDataset, "Dataset")

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
    try:
        fire.Fire(RAGCli)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
