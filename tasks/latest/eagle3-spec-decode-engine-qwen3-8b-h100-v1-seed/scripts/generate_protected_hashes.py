#!/usr/bin/env python3
"""Generate or verify the complete trusted Judge/runtime hash manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    """Hash one protected file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def private_data_dir(task_root: Path) -> Path:
    """The Judge-only holdout directory (labelled judge:private_data/...).

    It is a dataset asset: $EAGLE3_PRIVATE_DATA when set (scripts/lib.sh exports
    it), else <INFRATHON_ASSETS_ROOT>/<task directory>/judge/private_data when
    INFRATHON_ASSETS_ROOT is set, else the in-repository copy images/judge/private_data
    when present, else the default asset root (assets/fetch.sh).
    """
    explicit = os.environ.get("EAGLE3_PRIVATE_DATA")
    if explicit:
        return Path(explicit)
    assets = os.environ.get("INFRATHON_ASSETS_ROOT")
    local = task_root / "images/judge/private_data"
    if not assets and local.is_dir():
        return local
    root = Path(assets) if assets else Path.home() / ".cache/infrathon/assets"
    return root / task_root.name / "judge/private_data"


def protected_files(task_root: Path, private_data: Path | None = None) -> dict[str, Path]:
    """Return the canonical label-to-file mapping."""
    judge = task_root / "images/judge"
    private = private_data if private_data is not None else private_data_dir(task_root)
    if not private.is_dir():
        raise FileNotFoundError(
            f"Judge-only private_data is missing: {private} (run assets/fetch.sh)"
        )
    spec = task_root / "images/work/workdir/spec_lab"
    records: dict[str, Path] = {
        "judge:run_judge.py": judge / "run_judge.py",
        "judge:verify_eagle3.py": judge / "verify_eagle3.py",
    }
    for path in sorted((judge / "verifier").glob("*.py")):
        records[f"judge:verifier/{path.name}"] = path
    for path in sorted((spec / "runtime").glob("*")):
        if path.is_file() and not path.is_symlink():
            records[f"worker:runtime/{path.name}"] = path
    # The Judge's own seed copy: engine sources come from the Work layer at image
    # build time, the package marker is committed beside the Judge Dockerfile.
    records["seed:solutions/__init__.py"] = (
        judge / "seed_reference/solutions/__init__.py"
    )
    for path in sorted((spec / "solutions/engine_v001").glob("*")):
        if path.is_file() and not path.is_symlink():
            records[f"seed:solutions/engine_v001/{path.name}"] = path
    for path in sorted(private.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(private).as_posix()
            records[f"judge:private_data/{relative}"] = path
    return records


def render(task_root: Path, private_data: Path | None = None) -> str:
    """Render a deterministic complete manifest."""
    records = protected_files(task_root, private_data)
    missing = [label for label, path in records.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"protected files are missing: {missing}")
    return "".join(
        f"{sha256(path)}  {label}\n"
        for label, path in sorted(records.items())
    )


def atomic_replace(path: Path, content: str) -> None:
    """Replace the manifest atomically in its own directory."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse generation or check mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--task-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--private-data",
        type=Path,
        help="Judge-only private_data directory (default: see private_data_dir)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Check or atomically update protected_hashes.txt."""
    args = parse_args(argv)
    task_root = args.task_root.resolve(strict=True)
    target = task_root / "images/judge/protected_hashes.txt"
    try:
        expected = render(task_root, args.private_data)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    if args.check:
        if target.is_symlink() or not target.is_file():
            raise SystemExit("protected hash manifest is missing or unsafe")
        if target.read_text(encoding="utf-8") != expected:
            raise SystemExit("protected hashes are stale")
        return 0
    atomic_replace(target, expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
