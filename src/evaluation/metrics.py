'''Recall@k calculation logic'''
from src.models import (
    MinimalSource,
    AnsweredQuestion,
    MinimalSearchResults,
)
from constants import MIN_OVERLAP_RATIO


def _overlap_length(a: MinimalSource, b: MinimalSource) -> int:
    """
    Number of characters overlapping between two sources of the same file.
    Returns 0 if the files differ or if the intervals don't overlap.
    """
    if a.file_path != b.file_path:
        return 0
    start = max(a.first_character_index, b.first_character_index)
    end = min(a.last_character_index, b.last_character_index)
    return max(0, end - start)


def _source_is_found(
    ground_truth: MinimalSource,
    retrieved: list[MinimalSource],
    min_overlap_ratio: float = MIN_OVERLAP_RATIO,
) -> bool:
    """
    True if any retrieved source covers >=MIN_OVERLAP_RATIO%
    of the ground truth source.
    """
    gt_length = (
        ground_truth.last_character_index
        - ground_truth.first_character_index
    )
    if gt_length <= 0:
        return False

    for r in retrieved:
        overlap = _overlap_length(ground_truth, r)
        if overlap / gt_length >= min_overlap_ratio:
            return True
    return False


def recall_for_question(
    ground_truth_sources: list[MinimalSource],
    retrieved_sources: list[MinimalSource],
    k: int,
) -> float:
    """Recall@k for a single question.

    Returns the fraction of ground truth sources that are 'found' in the
    top-k retrieved sources.
    """
    if not ground_truth_sources:
        return 0.0

    top_k = retrieved_sources[:k]
    found = sum(
        1 for gt in ground_truth_sources if _source_is_found(gt, top_k)
    )
    return found / len(ground_truth_sources)


def recall_at_k(
    answered_questions: list[AnsweredQuestion],
    search_results: list[MinimalSearchResults],
    k: int,
) -> float:
    """Mean recall@k across a whole dataset."""
    results_by_id = {r.question_id: r for r in search_results}

    per_question = []
    for q in answered_questions:
        retrieved = results_by_id.get(q.question_id)
        if retrieved is None:
            per_question.append(0.0)
            continue
        per_question.append(
            recall_for_question(q.sources, retrieved.retrieved_sources, k)
        )

    if not per_question:
        return 0.0
    return sum(per_question) / len(per_question)
