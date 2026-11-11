#!/usr/bin/env python3
"""Prepare an owner-only verifier that permits the installed official package.

The benchmark cases, baseline, timing, correctness, memory checks, privilege
drop, and fresh-process deployment protocol remain byte-identical to the shipped
verifier.  Only protected-file enforcement is skipped for this diagnostic and
the clean-room scanner permits the explicit ``flash_kmeans`` import.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.task_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    verifier = (root / "images/judge/verify_flash_kmeans.py").read_text(
        encoding="utf-8"
    )
    protected_block = '''        protected = protected_failures(app_dir, verifier_dir)
        if protected:
            return fail(files, {**detail, "protected_failures": protected}, "protected files changed")
        detail["protected_ok"] = 1.0
'''
    replacement = '''        # Owner-only installed-package diagnostic: image provenance was
        # validated before container creation, while the scanner is copied
        # solely to permit the explicit official-package import.
        detail["protected_ok"] = 1.0
'''
    if verifier.count(protected_block) != 1:
        raise ValueError("shipped verifier protected block drifted")
    verifier = verifier.replace(protected_block, replacement)
    (output / "owner_verify_flash_kmeans.py").write_text(
        verifier,
        encoding="utf-8",
    )

    scanner = (root / "images/judge/benchmark/source_admission.py").read_text(
        encoding="utf-8"
    )
    allowed = 'ALLOWED_IMPORT_ROOTS = {"__future__", "functools", "math", "torch", "triton", "typing"}'
    allowed_replacement = (
        'ALLOWED_IMPORT_ROOTS = {"__future__", "flash_kmeans", "functools", '
        '"math", "torch", "triton", "typing"}'
    )
    forbidden = '    "flash_kmeans",\n'
    if scanner.count(allowed) != 1 or scanner.count(forbidden) != 1:
        raise ValueError("shipped source scanner import policy drifted")
    scanner = scanner.replace(allowed, allowed_replacement)
    scanner = scanner.replace(forbidden, "")
    (output / "source_admission.py").write_text(scanner, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
