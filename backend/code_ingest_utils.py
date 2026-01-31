"""Utilities for ingesting codebases: language detection, structured chunking, and document creation.

This module provides helpers to:
- determine file language from extension
- read files while preserving formatting
- split code into logical chunks (functions / classes / top-level sections)
- produce langchain `Document` objects with rich metadata

The splitter is heuristic-based and supports common languages (py, js, ts, jsx, tsx).
For other files (markdown, json, toml) we fall back to paragraph or character splitting.
"""
from __future__ import annotations

import re
from typing import List, Tuple
from pathlib import Path

from langchain.schema import Document


EXT_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".json": "json",
    ".md": "markdown",
    ".toml": "toml",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".cfg": "ini",
    ".ini": "ini",
    ".txt": "text",
}


def detect_language_from_path(path: str) -> str:
    ext = Path(path).suffix.lower()
    return EXT_LANG_MAP.get(ext, "text")


def read_file_preserve(path: str) -> List[str]:
    """Read file and return lines preserving newlines and indentation."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def find_code_block_boundaries(lines: List[str], language: str) -> List[Tuple[int, int]]:
    """Return list of (start_line, end_line) 0-based indexes for logical blocks.

    Heuristics:
    - For Python: split at top-level 'def ' or 'class ' lines
    - For JS/TS: split at top-level 'function ' or `=>` exports or 'class '
    - For others: fallback to chunk by 200 lines
    """
    boundaries: List[Tuple[int, int]] = []
    if language == "python":
        pattern = re.compile(r"^(def |class )")
    elif language in ("javascript", "typescript"):
        pattern = re.compile(r"^(export\s+)?(function |class |const |let |var )")
    else:
        # fallback: fixed-size blocks
        size = 200
        n = len(lines)
        for i in range(0, n, size):
            boundaries.append((i, min(i + size, n)))
        return boundaries

    indices = []
    for i, line in enumerate(lines):
        if pattern.match(line.lstrip()):
            indices.append(i)

    if not indices:
        # no definitions found; return single block
        return [(0, len(lines))]

    # Build ranges from indices
    for i, start in enumerate(indices):
        end = indices[i + 1] if i + 1 < len(indices) else len(lines)
        # include possible decorators/imports above a def/class for python
        if language == "python":
            # walk backwards to include decorators and comments
            j = start - 1
            while j >= 0 and lines[j].lstrip().startswith("@"):
                j -= 1
            start = j + 1
        boundaries.append((start, end))

    # If first block doesn't start at 0, include leading lines as a block
    if boundaries and boundaries[0][0] > 0:
        boundaries.insert(0, (0, boundaries[0][0]))

    return boundaries


def create_documents_from_file(path: str, chunk_max_lines: int = 400) -> List[Document]:
    """Read file and create Documents representing logical chunks.

    Each Document.metadata includes: path, filename, language, chunk_type, line_start, line_end
    """
    language = detect_language_from_path(path)
    lines = read_file_preserve(path)
    boundaries = find_code_block_boundaries(lines, language)
    docs: List[Document] = []

    for start, end in boundaries:
        # further split large blocks into manageable sizes
        cur = start
        while cur < end:
            chunk_end = min(cur + chunk_max_lines, end)
            chunk_lines = lines[cur:chunk_end]
            content = "".join(chunk_lines).rstrip()
            if not content.strip():
                cur = chunk_end
                continue

            metadata = {
                "path": str(path),
                "filename": Path(path).name,
                "language": language,
                "chunk_type": "code" if language not in ("markdown", "text", "json", "toml", "yaml") else "prose",
                "line_start": cur + 1,  # 1-based
                "line_end": chunk_end,
            }
            docs.append(Document(page_content=content, metadata=metadata))
            cur = chunk_end

    return docs
