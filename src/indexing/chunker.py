# Logic to split Python and Markdown files
from collections.abc import Iterator
from typing import Any


class TextChunker:
    """Splits text into chunks and tracks character indices."""

    def __init__(self, max_size: int = 2000) -> None:
        """Initialize with the maximum chunk size."""
        self.max_size = max_size

    def process_file(
        self, file_path: str, text: str
    ) -> Iterator[dict[str, Any]]:
        """Route to the appropriate chunking strategy."""
        if file_path.endswith(".py"):
            seps = ["\nclass ", "\ndef ", "\n\n", "\n", " "]
        elif file_path.endswith((".cpp", ".cu", ".cuh", ".h", ".hpp")):
            seps = ["}\n", ";\n", "\n\n", "\n", " "]
        else:
            seps = ["\n\n", "\n", " "]

        yield from self._chunk_with_separators(file_path, text, seps)

    def _chunk_with_separators(
        self, file_path: str, text: str, separators: list[str]
    ) -> Iterator[dict[str, Any]]:
        """Create chunks using prioritized separators."""
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.max_size, text_len)

            if end == text_len:
                yield self._format_chunk(file_path, text, start, end)
                break

            split_idx = -1
            for sep in separators:
                found_idx = text.rfind(sep, start, end)
                if found_idx != -1:
                    if sep in ("\nclass ", "\ndef "):
                        split_idx = found_idx + 1
                    else:
                        split_idx = found_idx + len(sep)
                    break

            if split_idx == -1:
                split_idx = end

            yield self._format_chunk(file_path, text, start, split_idx)
            start = split_idx

    def _format_chunk(
        self, file_path: str, text: str, start: int, end: int
    ) -> dict[str, Any]:
        """Format the chunk output to match required data models."""
        return {
            "file_path": file_path,
            "text": text[start:end],
            "first_character_index": start,
            "last_character_index": end,
        }
