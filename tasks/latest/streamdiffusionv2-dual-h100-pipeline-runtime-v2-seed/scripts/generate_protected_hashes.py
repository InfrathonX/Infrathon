#!/usr/bin/env python3
"""Generate the Judge's hash manifest from its exact immutable build inputs.

The goldens are measured on the card, so the manifest is per hardware type:
images/judge/protected_hashes/<hardware>.txt, from this host's build/ goldens.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(root: Path):
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ):
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--hardware", required=True, choices=("h100-94gb-400w", "h100-80gb-700w"),
                        help="the hardware type build/goldens_judge was generated on")
    args = parser.parse_args()
    task = args.task.resolve()
    assets = args.assets.resolve()
    for required in (
        assets / "embeddings" / "hidden",
        assets / "embeddings" / "public",
        assets / "clips",
        task / "build" / "goldens_judge",
    ):
        if not required.is_dir():
            raise SystemExit(f"missing protected input directory: {required}")
    entries: list[tuple[str, Path]] = []
    judge = task / "images" / "judge"
    for name in ("run_judge.py", "verify_sdv2_pipeline.py", "candidate_worker.py", "source_admission.py"):
        entries.append((f"judge:{name}", judge / name))
    runtime = task / "runtime"
    for path in source_files(runtime):
        entries.append((f"worker:{path.relative_to(runtime).as_posix()}", path))
    # The pristine seed is re-measured inside every Judge run and supplies every
    # seed-relative limit, so it is protected exactly like the worker runtime.
    seed = task / "oracle" / "candidates"
    for path in source_files(seed / "runtime_v001"):
        entries.append((f"seed:{path.relative_to(seed).as_posix()}", path))
    entries.append(("private:final_cases.json", task / "build" / "private" / "final_cases.json"))
    for path in source_files(assets / "embeddings" / "hidden"):
        entries.append((f"private:embeddings/hidden/{path.name}", path))
    for path in source_files(assets / "embeddings" / "public"):
        entries.append((f"fixture:embeddings/public/{path.name}", path))
    for path in source_files(assets / "clips"):
        entries.append((f"private:clips/{path.relative_to(assets / 'clips').as_posix()}", path))
    goldens = task / "build" / "goldens_judge"
    for path in source_files(goldens):
        if path.name == "manifest.json":
            entries.append((f"private:goldens/{path.relative_to(goldens).as_posix()}", path))
    missing = [str(path) for _, path in entries if not path.is_file() or path.is_symlink()]
    if missing:
        raise SystemExit("missing or linked protected inputs: " + ", ".join(missing))
    output = judge / "protected_hashes" / f"{args.hardware}.txt"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        "".join(f"{sha256(path)} {label}\n" for label, path in entries),
        encoding="utf-8",
    )
    print(f"wrote {len(entries)} entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
