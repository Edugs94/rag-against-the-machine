"""Utility helpers for the CLI: I/O + pydantic validation with clear errors."""
import sys
import json
from typing import TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def load_json_as_model(path: str, model: type[T], label: str) -> T:
    """Load a JSON file and validate it against a pydantic model."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return model(**raw)
    except FileNotFoundError:
        print(f"{label} not found: {path}", file=sys.stderr)
    except PermissionError:
        print(f"Permission denied reading {label}: {path}", file=sys.stderr)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {label} ({path}): {e}", file=sys.stderr)
    except ValidationError as e:
        print(f"{label} schema invalid ({path}):\n{e}", file=sys.stderr)
    except OSError as e:
        print(f"OS error reading {label} ({path}): {e}", file=sys.stderr)
    sys.exit(1)


def write_model_as_json(path: str, payload: BaseModel, label: str) -> None:
    """Write a pydantic model to disk as indented JSON."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload.model_dump_json(indent=2))
    except PermissionError:
        print(f"Permission denied writing {label}: {path}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Failed to write {label} ({path}): {e}", file=sys.stderr)
        sys.exit(1)


def sanitize_query(query: object) -> str:
    """
    Coerce Fire inputs to str and validate the query is non-empty
    """
    text = str(query).strip()
    if not text:
        print("Query cannot be empty", file=sys.stderr)
        sys.exit(1)
    return text
