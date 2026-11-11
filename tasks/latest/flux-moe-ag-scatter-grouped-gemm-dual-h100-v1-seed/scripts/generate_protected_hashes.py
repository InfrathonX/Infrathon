#!/usr/bin/env python3
"""Regenerate the Judge's protected-file SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


PROTECTED_FILES = (
    ("judge:candidate_worker.py", "images/judge/candidate_worker.py"),
    ("judge:run_judge.py", "images/judge/run_judge.py"),
    ("judge:source_admission.py", "images/judge/source_admission.py"),
    ("judge:verify_moe_ag.py", "images/judge/verify_moe_ag.py"),
    ("worker:moe_ag_runtime.py", "runtime/moe_ag_runtime.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.task_root.resolve()
    output = root / "images/judge/protected_hashes.txt"
    text = "".join(
        f"{sha256(root / relative)} {label}\n"
        for label, relative in PROTECTED_FILES
    )
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
