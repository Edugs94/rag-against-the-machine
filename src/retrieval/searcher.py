# Logic to load the BM25 index and return Top-K chunks
from pathlib import Path


class RepositoryReader:
    """Reads specified text files from a given repository directory."""

    def __init__(
        self,
        path: str,
        extensions: tuple[str, ...] = (
            ".py", ".md", ".txt", ".sh", ".yaml", ".yml",
            ".json", ".toml", ".rst", ".c", ".cpp", ".h"
        )
    ) -> None:
        """Initialize the reader with a path and allowed extensions."""
        self.path = Path(path)
        self.extensions = extensions

    def get_files_content(self) -> dict[str, str]:
        """Extract content from files matching the allowed extensions."""
        content = {}

        for file_path in self.path.rglob("*"):
            if file_path.is_file() and file_path.suffix in self.extensions:
                try:
                    text = file_path.read_text(encoding="utf-8")
                    content[str(file_path)] = text
                except UnicodeDecodeError:
                    pass

        return content
