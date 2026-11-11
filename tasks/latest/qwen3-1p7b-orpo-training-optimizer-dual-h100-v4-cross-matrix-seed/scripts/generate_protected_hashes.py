#!/usr/bin/env python3
"""Generate or verify the immutable Judge protected-file manifest.

The Judge image is assembled from two Docker build contexts: the Work base
image supplies ``/app/workdir/orpo_lab/runtime`` and the Judge context supplies
the private controller.  This tool is deliberately run from the complete task
tree before either image is built, so it can make one deterministic manifest
covering both contexts.  Image construction only accepts an already-current
manifest; writing one is an explicit maintainer action.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
from pathlib import Path


HEADER = "# mlsysforge-protected-files-v1\n# sha256  relative-path\n"
TASK_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = TASK_ROOT / "images/judge/protected_hashes.txt"


def hidden_cases_source(task_root: Path) -> Path:
    """The Judge-only hidden cases: the repository copy, else the fetched asset.

    The public tree does not carry them; ``assets/fetch.sh`` installs them at
    ``$INFRATHON_ASSETS_ROOT/<task directory>/judge/private_data/hidden_cases.json``.
    """

    local = task_root / "images/judge/private_data/hidden_cases.json"
    if local.exists() or local.is_symlink():
        return local
    root = Path(os.environ.get("INFRATHON_ASSETS_ROOT") or "~/.cache/infrathon/assets").expanduser()
    return root / task_root.name / "judge/private_data/hidden_cases.json"


def _regular_files(root: Path) -> list[Path]:
    """Return task-source files which Docker will copy, in canonical order."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"protected source directory is not a real directory: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"protected source must not be a symlink: {path}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"protected source must be a regular file: {path}")
        files.append(path)
    return sorted(files, key=lambda item: item.as_posix())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_sources(task_root: Path) -> list[tuple[str, Path]]:
    """Map every protected container-relative path to its task-tree source."""

    task_root = task_root.resolve()
    runtime = task_root / "images/work/workdir/orpo_lab/runtime"
    verifier = task_root / "images/judge/verifier"
    pairs: list[tuple[str, Path]] = []
    for source in _regular_files(runtime):
        pairs.append((f"app/workdir/orpo_lab/runtime/{source.relative_to(runtime).as_posix()}", source))
    for source in _regular_files(verifier):
        relative = source.relative_to(verifier).as_posix()
        pairs.append((f"opt/orpo_judge/verifier/{relative}", source))
    judge_sources = (
        ("opt/orpo_judge/run_judge.py", task_root / "images/judge/run_judge.py"),
        ("opt/orpo_judge/private_data/hidden_cases.json", hidden_cases_source(task_root)),
        # These are root-owned mirrors that the non-root candidate launcher
        # executes.  Hashing both locations prevents a mirror substitution.
        ("opt/orpo_candidate_worker/candidate_worker.py", verifier / "candidate_worker.py"),
        ("opt/orpo_candidate_worker/e2e_rank_bootstrap.py", verifier / "e2e_rank_bootstrap.py"),
    )
    for container_path, source in judge_sources:
        info = source.lstat() if source.exists() else None
        if info is None or source.is_symlink() or not source.is_file() or not stat.S_ISREG(info.st_mode):
            raise ValueError(f"protected source must be a regular file: {source}")
        pairs.append((container_path, source))
    if len({path for path, _ in pairs}) != len(pairs):
        raise AssertionError("internal error: duplicate protected container path")
    return sorted(pairs, key=lambda item: item[0])


def render_manifest(task_root: Path) -> str:
    lines = [HEADER.rstrip("\n")]
    for container_path, source in protected_sources(task_root):
        lines.append(f"{_sha256(source)}  {container_path}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=TASK_ROOT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="explicitly replace the manifest with canonical content")
    action.add_argument("--check", action="store_true", help="fail unless the existing manifest is canonical")
    args = parser.parse_args()
    manifest_path = args.task_root / "images/judge/protected_hashes.txt"
    expected = render_manifest(args.task_root)
    if args.write:
        # --write is intentionally the only code path which replaces this
        # security-sensitive file.  The image build always uses --check.
        manifest_path.write_text(expected, encoding="utf-8")
        return 0
    try:
        actual = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"protected manifest unavailable: {exc}", file=sys.stderr)
        return 1
    if actual != expected:
        print(
            "protected manifest is stale or non-canonical; run "
            "scripts/generate_protected_hashes.py --write after reviewing the diff",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
