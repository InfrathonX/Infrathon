#!/usr/bin/env python3
"""Conservative filesystem admission for an SDV2 runtime_vNNN package."""

from __future__ import annotations

import ast
import re
from pathlib import Path


VERSION = re.compile(r"runtime_v([0-9]{3})\Z")
ALLOWED_SUFFIXES = {".py", ".md", ".txt", ".json"}
MAX_FILES = 128
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024


class AdmissionError(RuntimeError):
    pass


def admit_candidate(path: str | Path) -> dict:
    root = Path(path).resolve()
    if not VERSION.fullmatch(root.name) or not root.is_dir() or root.is_symlink():
        raise AdmissionError("candidate must be a regular runtime_vNNN directory")
    files = sorted(item for item in root.rglob("*") if item.is_file() or item.is_symlink())
    if not files or len(files) > MAX_FILES:
        raise AdmissionError("candidate file count is outside the admitted range")
    total = 0
    python_files = 0
    for item in files:
        if item.is_symlink() or not item.is_file():
            raise AdmissionError(f"links and special files are forbidden: {item.relative_to(root)}")
        relative = item.relative_to(root)
        if "__pycache__" in relative.parts or item.suffix not in ALLOWED_SUFFIXES:
            raise AdmissionError(f"inadmissible candidate file: {relative}")
        size = item.stat().st_size
        if size > MAX_FILE_BYTES:
            raise AdmissionError(f"candidate file too large: {relative}")
        total += size
        if item.suffix == ".py":
            python_files += 1
            try:
                ast.parse(item.read_text(encoding="utf-8"), filename=str(relative))
            except (SyntaxError, UnicodeError) as exc:
                raise AdmissionError(f"invalid Python source {relative}: {exc}") from exc
    if total > MAX_TOTAL_BYTES:
        raise AdmissionError("candidate source total exceeds 8 MiB")
    entry = root / "runtime.py"
    if not entry.is_file() or entry.is_symlink():
        raise AdmissionError("runtime.py entrypoint is required")
    package = root / "__init__.py"
    if not package.is_file() or package.is_symlink():
        raise AdmissionError("__init__.py is required for spawn-safe candidate imports")
    tree = ast.parse(entry.read_text(encoding="utf-8"), filename="runtime.py")
    if not any(isinstance(node, ast.ClassDef) and node.name == "Runtime" for node in tree.body):
        raise AdmissionError("runtime.py must define Runtime")
    return {"version": int(root.name[-3:]), "files": len(files), "python_files": python_files, "bytes": total}
