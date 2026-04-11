#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import List, Tuple


def scan_repository(root_dir: str, ignore_hidden: bool = True) -> Counter[str]:
    """
    Recursively scans a directory with a live progress indicator.
    """
    base_path = Path(root_dir)
    extension_counts: Counter[str] = Counter()
    processed_count = 0

    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: The path '{root_dir}' is not a valid directory.")
        sys.exit(1)

    print("Scanning repository...")

    for file_path in base_path.rglob("*"):
        processed_count += 1

        if processed_count % 100 == 0:
            sys.stdout.write(f"\rFiles analyzed: {processed_count}...")
            sys.stdout.flush()

        if not file_path.is_file():
            continue

        if ignore_hidden and any(
            part.startswith(".") for part in file_path.parts
        ):
            continue

        extension = file_path.suffix.lower() if file_path.suffix else "no_ext"
        extension_counts[extension] += 1

    sys.stdout.write("\r" + " " * 40 + "\r")
    sys.stdout.flush()

    return extension_counts


def generate_extensions_file(stats: Counter[str]) -> None:
    """
    Creates a formatted Python file with extensions in a vertical list.
    """
    extensions = sorted([ext for ext in stats.keys() if ext != "no_ext"])
    has_no_ext = "no_ext" in stats
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / "allowed_extensions.py"

    variable_name = "ALLOWED_EXTENSIONS"

    lines = [f'    "{ext}",' for ext in extensions]
    if has_no_ext:
        lines.append('    "",  # no extension files')
    list_content = "\n".join(lines)
    content = f"{variable_name} = [\n{list_content}\n]\n"

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Output generated at: {output_path}")
    except IOError as e:
        print(f"Error writing to {output_path}: {e}")


def print_formatted_report(stats: Counter[str], target_path: str) -> None:
    """
    Displays a summary table with 'no_ext' always at the bottom.
    """
    total_files = sum(stats.values())
    if total_files == 0:
        print(f"No files found in: {target_path}")
        return

    print(f"\n--- RAG Source Scan: {target_path} ---")
    print(f"{'Extension':<18} | {'Count':<8} | {'Percentage':<10}")
    print("-" * 42)

    other_extensions: List[Tuple[str, int]] = sorted(
        [(ext, count) for ext, count in stats.items() if ext != "no_ext"],
        key=lambda x: x[1],
        reverse=True,
    )

    final_order = other_extensions
    if "no_ext" in stats:
        final_order.append(("no_ext", stats["no_ext"]))

    for ext, count in final_order:
        percentage = (count / total_files) * 100
        print(f"{ext:<18} | {count:<8} | {percentage:>9.2f}%")

    print("-" * 42)
    print(f"{'Total Files':<18} | {total_files:<8} | 100.00%\n")


def main() -> None:
    """
    CLI Entry point for the scanner tool.
    """
    parser = argparse.ArgumentParser(
        description="Identify file types for RAG ingestion."
    )
    parser.add_argument(
        "path", type=str, nargs="?", default=".", help="Directory to scan"
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and folders",
    )
    parser.add_argument(
        "--no-list",
        action="store_true",
        help="Skip generating allowed_extensions.py",
    )

    if len(sys.argv) < 2:
        print("Run python3 filetype_scanner <folder_to_scan>", file=sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    results = scan_repository(args.path, ignore_hidden=not args.include_hidden)

    print_formatted_report(results, args.path)

    if not args.no_list:
        generate_extensions_file(results)


if __name__ == "__main__":
    main()
