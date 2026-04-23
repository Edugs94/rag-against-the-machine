'''Logic to split Python and Markdown files'''
from collections.abc import Iterator
from typing import Any
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from src.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


class TextChunker:
    """Splits text into chunks and tracks character indices."""

    def __init__(
        self,
        max_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """Initialize with the maximum chunk size and overlap."""
        self.max_size = max_size
        self.overlap = overlap
        self._md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
                ("####", "H4"),
                ("#####", "H5"),
                ("######", "H6"),
            ],
            strip_headers=False,
        )

    def process_file(
        self, file_path: str, text: str
    ) -> Iterator[dict[str, Any]]:
        """Route to the appropriate chunking strategy."""

        if file_path.endswith((".md", ".mdx")):
            yield from self._chunk_markdown(file_path, text)
            return

        if file_path.endswith(".py"):
            seps = ["\nclass ", "\ndef ", "\n\n", "\n", " "]
        elif file_path.endswith((".cpp", ".cu", ".cuh", ".h", ".hpp")):
            seps = ["}\n", ";\n", "\n\n", "\n", " "]
        else:
            seps = ["\n\n", "\n", " "]
        yield from self._chunk_with_separators(file_path, text, seps)

    def _chunk_markdown(
        self, file_path: str, text: str
    ) -> Iterator[dict[str, Any]]:
        """Split markdown by headers,
        large sections are subdivided with overlap."""
        try:
            sections = self._md_header_splitter.split_text(text)
        except (ValueError, TypeError, IndexError, AttributeError):
            seps = ["\n# ", "\n## ", "\n### ", "\n#### ",
                    "\n```", "\n\n", "\n", " "]
            yield from self._chunk_with_separators(file_path, text, seps)
            return

        section_offsets = self._locate_sections(text, sections)

        recursive = RecursiveCharacterTextSplitter(
            chunk_size=self.max_size,
            chunk_overlap=self.overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n#####",
                        "\n```", "\n\n", "\n", " ", ""],
            keep_separator=True,
        )

        for section_text, section_start in section_offsets:
            if section_start is None:
                continue

            section_end = section_start + len(section_text)

            if len(section_text) <= self.max_size:
                yield self._format_chunk(
                    file_path, text, section_start, section_end
                )
                continue

            sub_chunks = recursive.split_text(section_text)
            cursor = section_start
            for sub in sub_chunks:
                local_start = text.find(sub, cursor, section_end)
                if local_start == -1:
                    local_start = cursor
                local_end = min(local_start + len(sub), section_end)
                yield self._format_chunk(
                    file_path, text, local_start, local_end
                )
                cursor = max(local_start + 1, local_end - self.overlap)

    def _locate_sections(
        self, text: str, sections: list[Any]
    ) -> list[tuple[str, int | None]]:
        """Locate each section's start offset in the
        original text using its first line as anchor."""
        results: list[tuple[str, int | None]] = []
        search_from = 0
        for section in sections:
            section_text = section.page_content
            if not section_text:
                continue

            anchor = section_text.split("\n", 1)[0].strip()
            if not anchor:
                results.append((section_text, None))
                continue

            idx = text.find(anchor, search_from)
            if idx == -1:
                idx = text.find(anchor)
                if idx == -1:
                    results.append((section_text, None))
                    continue

            results.append((section_text, idx))
            search_from = idx + 1
        return results

    def _chunk_with_separators(
        self, file_path: str, text: str, separators: list[str]
    ) -> Iterator[dict[str, Any]]:
        """Create chunks using prioritized separators."""
        header_seps = {"\nclass ", "\ndef ", "\n# ", "\n## ",
                       "\n### ", "\n#### "}
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
                    if sep in header_seps:
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
