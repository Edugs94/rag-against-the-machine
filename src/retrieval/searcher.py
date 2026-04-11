# Logic to load the BM25 index and return Top-K chunks
from collections.abc import Iterator
from pathlib import Path
from filetype_scanner.allowed_extensions import ALLOWED_EXTENSIONS


class RepositoryReader:
    """Reads specified text files from a given repository directory."""

    def __init__(self, path: str) -> None:
        """Initialize the reader with a path."""
        self.path = Path(path)

    def get_files_content(self) -> Iterator[tuple[str, str]]:
        """Extract content from files matching allowed extensions."""
        for file_path in self.path.rglob("*"):
            if file_path.is_file() and file_path.suffix in ALLOWED_EXTENSIONS:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    yield str(file_path), content
                except (UnicodeDecodeError, OSError):
                    pass
