#!/usr/bin/env python3
"""Rewrite images/judge/protected_hashes.txt from the trusted Judge files.

    scripts/generate_protected_hashes.py            # rewrite in place
    scripts/generate_protected_hashes.py --check    # report drift, change nothing
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
TARGET = TASK_ROOT / "images/judge/protected_hashes.txt"
FILES = (
    ("judge", "candidate_worker.py", "images/judge/candidate_worker.py"),
    ("judge", "run_judge.py", "images/judge/run_judge.py"),
    ("judge", "source_admission.py", "images/judge/source_admission.py"),
    ("judge", "verify_ssd.py", "images/judge/verify_ssd.py"),
    ("worker", "mamba2_ssd_runtime.py", "runtime/mamba2_ssd_runtime.py"),
)


def build() -> str:
    lines = ["# SHA-256 records for the trusted Judge adapter, verifier, and shared runtime."]
    for kind, name, relative in FILES:
        digest = hashlib.sha256((TASK_ROOT / relative).read_bytes()).hexdigest()
        lines.append(f"{digest} {kind}:{name}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    text = build()
    if "--check" in argv:
        return 0 if TARGET.read_text(encoding="utf-8") == text else 1
    TARGET.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
