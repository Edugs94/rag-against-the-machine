# filetype-inspector

A lightweight CLI toolkit that **scans any repository, catalogues every file extension present, and generates an importable allowlist** — purpose-built for the ingestion phase of RAG (Retrieval-Augmented Generation) pipelines.

Skipping irrelevant files **before** reading them is the cheapest optimisation you can make — no embeddings are computed, no chunks are created, no storage is wasted.

---

## Getting started

Clone the tool **inside the root of your RAG** project:

```bash
git clone https://github.com/Edugs94/filetype-scanner.git && rm -rf filetype-scanner/.git && rm -f filetype-scanner/.gitignore
```

Run it from the project root:

```bash
python3 filetype-scanner <dir>
```

That's it. You'll get a full report of every file type in your repository and an auto-generated `allowed_extensions.py` inside `filetype-scanner/` ready to curate:

```
--- RAG Source Scan: . ---
Extension          | Count    | Percentage
------------------------------------------
.py                | 1749     |     63.97%
.json              | 388      |     14.19%
.md                | 173      |      6.33%
.cu                | 71       |      2.60%
.png               | 52       |      1.90%
...
no_ext             | 6        |      0.22%
------------------------------------------
Total Files        | 2734     | 100.00%
```

---

## How it works

`Scan → Generate → Curate → Import`

1. **Scan** — the tool recursively walks the target directory and counts every file extension it finds.
2. **Generate** — it writes `allowed_extensions.py`, a sorted Python list containing every extension detected.
3. **Curate** — you open that file and remove the extensions you don't want ingested (images, compiled assets, platform stubs, etc.).
4. **Import** — in your ingestion code, use the curated list to filter files:

```python
from filetype_scanner.allowed_extensions import VARIABLE_EXTENSIONS

for file in repo.rglob("*"):
    if file.suffix.lower() in VARIABLE_EXTENSIONS:
        ingest(file)
```

By filtering at the file-collection stage you avoid unnecessary I/O on irrelevant files, wasted embedding API calls, and a polluted vector store that degrades retrieval quality.

---

## Flags

| Flag | Description |
|---|---|
| `dir` | Directory to scan |
| `--no-list` | Skip generating `allowed_extensions.py` |
| `--include-hidden` | Include hidden files and directories |

```bash
# scan a specific directory without generating the allowlist
python3 filetype-scanner <dir> --no-list

# include dotfiles and hidden folders
python3 filetype-scanner <dir> --include-hidden
```

---

## Extra tools

### get_types_path

When the scan report shows extensions with low file counts — `no_ext`, `.patch`, `.tpl`, etc. — you probably want to **inspect those files before deciding** whether to keep or drop them. `get_types_path` gives you every matching file path instantly:

```bash
python3 filetype-scanner/get_types_path.py no_ext <dir>
```

```
  no_ext  —  5 file(s) found

  ./Dockerfile
  ./Makefile
  ./LICENSE
  ./csrc/quantization/LICENCE
  ./scripts/run
```

It also works for any regular extension:

```bash
python3 filetype-scanner/get_types_path.py .cu ./csrc
```

| Argument | Description |
|---|---|
| `extension` | Extension to search (`.py`, `json`, `no_ext`) |
| `path` | Directory to scan (default: `.`) |
| `--include-hidden` | Include hidden files and directories |

---

## Requirements

Python 3.10+ — no external dependencies.

---

## License

MIT
