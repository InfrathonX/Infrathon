#!/usr/bin/env python3
"""Unprivileged worker for one private Attn-QAT training-attention suite.

The Judge runs this file as a uid-dropped subprocess in its own session.  It reads the
suite manifest from stdin, imports the trusted runtime from the worker root, runs the
suite and writes the result atomically.  The runtime root is removed from `sys.path` and
the module dropped from `sys.modules` immediately after the import, so an imported
candidate cannot reach either of them by name.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

RUNTIME_ROOT = Path("/opt/attn_qat_worker")
JUDGE_ROOT = Path(__file__).resolve().parent
CASE_KEYS = {
    "case_id", "family", "tier", "b", "h", "h_kv", "d", "s_q", "s_kv", "causal", "seed",
    "gated", "layer",
}
EXPECTED_CASE_COUNT = 287
SUITE_IDS = {"final-holdout-v3"}
MAX_MANIFEST_BYTES = 1 << 20


def _load_run_suite() -> Callable[..., dict[str, Any]]:
    """Import `run_suite` from the worker root, then make both roots unreachable."""
    sys.path.insert(0, str(RUNTIME_ROOT))
    try:
        from attn_qat_runtime import run_suite
    finally:
        sys.path = [
            entry
            for entry in sys.path
            if Path(entry or ".").resolve() not in {RUNTIME_ROOT, JUDGE_ROOT}
        ]
        sys.modules.pop("attn_qat_runtime", None)
        os.environ.pop("PYTHONPATH", None)
    return run_suite


def _read_manifest() -> dict[str, Any]:
    """Read and validate the suite manifest handed in on stdin."""
    raw = sys.stdin.buffer.read(MAX_MANIFEST_BYTES + 1)
    manifest = json.loads(
        raw.decode(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest identity differs")
    if manifest.get("suite_id") not in SUITE_IDS:
        raise ValueError("manifest identity differs")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError("manifest cases differ")
    for case in cases:
        if not isinstance(case, dict) or set(case) != CASE_KEYS:
            raise ValueError("manifest cases differ")
    return manifest


def _parse_args(argv: Any) -> argparse.Namespace:
    """Parse the Judge's invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Any = None) -> int:
    """Run one private suite and write its result atomically."""
    args = _parse_args(argv)
    manifest = _read_manifest()
    run_suite = _load_run_suite()
    result = run_suite(
        args.candidate.resolve(),
        manifest["cases"],
        trials=args.trials,
        warmups=args.warmups,
        require_hardware=True,
    )
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
