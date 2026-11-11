#!/usr/bin/env python3
"""Unprivileged entry point for one private TileLang Mamba-2 SSD suite."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


RUNTIME_ROOT = Path("/opt/tilelang_mamba2_ssd_worker")
JUDGE_ROOT = Path(__file__).resolve().parent
SUITE_ID = "tilelang-ssd-final-v3"
MANIFEST_SCHEMA_VERSION = 3
# Captured layers of the real cases, copied into the Judge holdout layer by
# images/judge/Dockerfile.private and checked by the runtime before the
# candidate is imported.
REAL_LAYER_ROOT = RUNTIME_ROOT / "real_layers"
# Exit status for a failure the runtime attributes to one named gate.  Only the
# TileLang-only operator record uses it; every other failure stays a plain
# non-zero exit, exactly as before.
GATE_FAILURE_EXIT = 3


def _load_trusted_run_suite() -> Callable[..., dict[str, Any]]:
    """Load the trusted runtime, then hide its import route from the candidate."""
    sys.path.insert(0, str(RUNTIME_ROOT))
    try:
        from mamba2_ssd_runtime import run_suite
    finally:
        sys.path = [
            entry
            for entry in sys.path
            if Path(entry or ".").resolve() not in {RUNTIME_ROOT, JUDGE_ROOT}
        ]
        sys.modules.pop("mamba2_ssd_runtime", None)
        os.environ.pop("PYTHONPATH", None)
    return run_suite


def _strict_stdin_json() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    if not raw or len(raw) > 1024 * 1024:
        raise ValueError("private manifest stdin is empty or oversized")
    parsed = json.loads(
        raw.decode("utf-8", errors="strict"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    sys.stdin.close()
    if not isinstance(parsed, dict):
        raise ValueError("manifest must be a JSON object")
    return parsed


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    args = parser.parse_args(argv)

    manifest = _strict_stdin_json()
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or manifest.get("suite_id") != SUITE_ID:
        raise ValueError("manifest identity differs")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest must contain a cases list")
    run_suite = _load_trusted_run_suite()
    globals().pop("_load_trusted_run_suite", None)
    import torch

    try:
        result = run_suite(
            torch,
            args.candidate.resolve(),
            cases,
            suite_id=str(manifest["suite_id"]),
            trials=args.trials,
            warmups=args.warmups,
            real_layer_root=REAL_LAYER_ROOT,
        )
    except Exception as error:
        if getattr(error, "gate", None) == "tilelang_only_ok":
            _atomic_json(
                args.output.resolve(),
                {"gate_failure": {"gate": "tilelang_only_ok", "detail": str(error)[:2000]}},
            )
            return GATE_FAILURE_EXIT
        raise
    if not isinstance(result, dict):
        raise RuntimeError("runtime result must be an object")
    _atomic_json(args.output.resolve(), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
