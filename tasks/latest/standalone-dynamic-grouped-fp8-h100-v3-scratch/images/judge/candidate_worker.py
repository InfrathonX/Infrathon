#!/usr/bin/env python3
"""Unprivileged entry point for one private standalone grouped-FP8 v4 run.

``--mode suite`` (default) runs the edge cases and then the scored cases and
writes a schema-4 result, including gate-attributed failures.  ``--mode
sanitize`` runs the sanitizer subset once per case with plain allocations;
the Judge launches it under compute-sanitizer memcheck.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


RUNTIME_ROOT = Path("/opt/standalone_grouped_fp8_worker")
JUDGE_ROOT = Path(__file__).resolve().parent
FINAL_SUITE_ID = "final-holdout-v4"
EDGE_SUITE_ID = "edge-holdout-v4"
SCHEMA_VERSION = 4


def _load_trusted_runtime() -> Any:
    """Load the trusted runtime, then hide its import route from the candidate."""
    sys.path.insert(0, str(RUNTIME_ROOT))
    try:
        import grouped_fp8_runtime as runtime
    finally:
        sys.path = [
            entry
            for entry in sys.path
            if Path(entry or ".").resolve() not in {RUNTIME_ROOT, JUDGE_ROOT}
        ]
        sys.modules.pop("grouped_fp8_runtime", None)
        os.environ.pop("PYTHONPATH", None)
    return runtime


def _strict_stdin_json() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(4 * 1024 * 1024 + 1)
    if not raw or len(raw) > 4 * 1024 * 1024:
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
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("suite", "sanitize"), default="suite")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--rounds", type=int)
    args = parser.parse_args(argv)

    manifest = _strict_stdin_json()
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("mode") != args.mode:
        raise ValueError("manifest identity differs")
    if manifest.get("edge_suite_id") != EDGE_SUITE_ID:
        raise ValueError("edge suite identity differs")
    edge_cases = manifest.get("edge_cases")
    if not isinstance(edge_cases, list):
        raise ValueError("manifest must contain an edge_cases list")
    runtime = _load_trusted_runtime()
    globals().pop("_load_trusted_runtime", None)
    if args.mode == "sanitize":
        result = runtime.run_sanitize_cases(args.candidate.resolve(), edge_cases)
    else:
        if manifest.get("suite_id") != FINAL_SUITE_ID:
            raise ValueError("scored suite identity differs")
        cases = manifest.get("cases")
        if not isinstance(cases, list) or args.trials is None or args.warmups is None or args.rounds is None:
            raise ValueError("suite mode needs cases, trials, warm-ups and sequence rounds")
        result = runtime.run_suite(
            args.candidate.resolve(),
            cases,
            edge_cases,
            trials=args.trials,
            warmups=args.warmups,
            rounds=args.rounds,
            require_hardware=True,
            stop_on_edge_failure=True,
        )
    if not isinstance(result, dict):
        raise RuntimeError("runtime result must be an object")
    _atomic_json(args.output.resolve(), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
