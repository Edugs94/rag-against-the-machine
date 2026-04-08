# Data Layer: Your Pydantic classes

import uuid
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represents a minimal source of information retrieved."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represents a question that has not been answered yet."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents a question answered with sources."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a dataset of RAG questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represents the search results for a specific query."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represents an answer generated from search results."""

    answer: str


class StudentSearchResults(BaseModel):
    """Represents the search results for a dataset of questions."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Represents search results including generated answers."""

    search_results: list[MinimalAnswer]
