#!/usr/bin/env python3
"""List all file paths matching a given extension type."""

import argparse
import sys
from pathlib import Path


def find_files(
    root_dir: str, extension: str, ignore_hidden: bool = True
) -> list[Path]:
    base_path = Path(root_dir)

    if not base_path.is_dir():
        print(f"Error: '{root_dir}' is not a valid directory.")
        sys.exit(1)

    matches: list[Path] = []
    want_no_ext = extension.lower() in ("no_ext", "no-ext", "noext")

    for file_path in sorted(base_path.rglob("*")):
        if not file_path.is_file():
            continue
        if ignore_hidden and any(p.startswith(".") for p in file_path.parts):
            continue

        ext = file_path.suffix.lower() if file_path.suffix else "no_ext"

        if want_no_ext and ext == "no_ext":
            matches.append(file_path)
        elif not want_no_ext and ext == extension.lower():
            matches.append(file_path)

    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="get_types_path",
        description="Print all file paths matching a given extension.",
    )
    parser.add_argument(
        "extension", help="Extension to search (e.g. .py, .json, no_ext)"
    )
    parser.add_argument(
        "path", nargs="?", default=".", help="Directory to scan (default: .)"
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files/folders",
    )

    args = parser.parse_args()

    ext = (
        args.extension
        if args.extension.startswith(".")
        or args.extension.lower() in ("no_ext", "no-ext", "noext")
        else f".{args.extension}"
    )

    results = find_files(args.path, ext, ignore_hidden=not args.include_hidden)

    if not results:
        print(f"No files found with extension '{ext}' in '{args.path}'")
        sys.exit(0)

    print(f"\n  {ext}  —  {len(results)} file(s) found\n")
    for f in results:
        print(f"  {f}")
    print()


if __name__ == "__main__":
    main()
