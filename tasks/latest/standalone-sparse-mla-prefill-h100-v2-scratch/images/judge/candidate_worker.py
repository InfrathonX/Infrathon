#!/usr/bin/env python3
"""Unprivileged entry point for one private standalone sparse-MLA prefill v2 run.

``--mode suite`` (default) runs the native edge cases (plus any sanitizer-only
case the Judge promotes) and then the scored suite, and writes a schema-3
result, including gate-attributed failures.  ``--mode sanitize`` runs the
sanitizer subset once per case with plain allocations, in one process, and
stops at the first fatal CUDA error; the Judge launches it under
compute-sanitizer memcheck and reads the per-case progress lines it prints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


RUNTIME_ROOT = Path("/opt/standalone_sparse_mla_prefill_worker")
JUDGE_ROOT = Path(__file__).resolve().parent
FINAL_SUITE_ID = "final-holdout-v4"
EDGE_SUITE_ID = "edge-holdout-v3"
SCORED_KEYS = {"case_id", "group", "family", "h_q", "s_q", "s_kv", "sink_mode", "gated", "seed"}
# The one extra field each group carries (the runtime validates its content).
SCORED_GROUP_KEYS = {
    "anchor": set(), "sweep": set(), "heads": set(), "length": {"length_divisor", "ragged"},
    "multi": {"request_lengths"}, "trace": {"calls"}, "replay": {"calls"}, "pattern": {"index_pattern"},
}
EXPECTED_CASE_COUNT = 997


def _load_trusted_runtime() -> Any:
    """Load the trusted runtime, then hide its import route from the candidate."""
    sys.path.insert(0, str(RUNTIME_ROOT))
    try:
        import sparse_mla_prefill_runtime as runtime
    finally:
        sys.path = [
            entry
            for entry in sys.path
            if Path(entry or ".").resolve() not in {RUNTIME_ROOT, JUDGE_ROOT}
        ]
        sys.modules.pop("sparse_mla_prefill_runtime", None)
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


def _finite_json(value: Any) -> Any:
    """Spell non-finite floats as strings so a failing result still reaches the Judge.

    A failure detail may carry an overflowing error metric; the Judge's numeric
    checks reject a string wherever a passing result needs a number.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_finite_json(payload), indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _progress_line(line: str) -> None:
    """One flushed line per case boundary, so the Judge can name the case in
    flight even when a fatal CUDA error takes the process down."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("suite", "sanitize"), default="suite")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--warmups", type=int)
    args = parser.parse_args(argv)

    manifest = _strict_stdin_json()
    if manifest.get("schema_version") != 3 or manifest.get("mode") != args.mode:
        raise ValueError("manifest identity differs")
    if manifest.get("edge_suite_id") != EDGE_SUITE_ID:
        raise ValueError("edge suite identity differs")
    edge_cases = manifest.get("edge_cases")
    if not isinstance(edge_cases, list) or any(not isinstance(case, dict) for case in edge_cases):
        raise ValueError("manifest must contain an edge_cases list of objects")
    promoted: list[str] = []
    if args.mode == "sanitize":
        if any(case.get("sanitize") is not True for case in edge_cases):
            raise ValueError("sanitize mode receives only sanitizer-flagged edge cases")
        if "promoted_sanitize_only" in manifest:
            raise ValueError("sanitize mode takes no promoted cases")
    else:
        raw_promoted = manifest.get("promoted_sanitize_only", [])
        if not isinstance(raw_promoted, list) or any(not isinstance(item, str) for item in raw_promoted):
            raise ValueError("promoted_sanitize_only must be a list of case IDs")
        promoted = list(raw_promoted)
        if manifest.get("suite_id") != FINAL_SUITE_ID:
            raise ValueError("scored suite identity differs")
        cases = manifest.get("cases")
        if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
            raise ValueError(f"manifest must contain exactly {EXPECTED_CASE_COUNT} scored cases")
        if any(
            not isinstance(case, dict) or case.get("group") not in SCORED_GROUP_KEYS
            or set(case) != SCORED_KEYS | SCORED_GROUP_KEYS[case["group"]]
            for case in cases
        ):
            raise ValueError("manifest case fields differ from the controller schema")
        if args.trials is None or args.warmups is None:
            raise ValueError("suite mode needs trials and warm-ups")
    runtime = _load_trusted_runtime()
    globals().pop("_load_trusted_runtime", None)
    if args.mode == "sanitize":
        result = runtime.run_sanitize_cases(
            args.candidate.resolve(), edge_cases, progress=_progress_line
        )
    else:
        result = runtime.run_suite(
            args.candidate.resolve(),
            cases,
            edge_cases,
            trials=args.trials,
            warmups=args.warmups,
            require_hardware=True,
            promoted_sanitize_only=promoted,
            run_scored_after_edge_failure=False,
        )
    if not isinstance(result, dict):
        raise RuntimeError("runtime result must be an object")
    _atomic_json(args.output.resolve(), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
