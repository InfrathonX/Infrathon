#!/usr/bin/env python3
"""Unprivileged entry point for one private LMCache KV-transfer suite.

Compiles the admitted submission with the frozen command, audits the
compiled library's undefined symbols *before* loading it, then runs the
requested suite through the trusted runtime.  Everything here runs as the
unprivileged agent account inside a per-run private root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


RUNTIME_ROOT = Path("/opt/lmcache_kv_transfer_worker")
JUDGE_ROOT = Path(__file__).resolve().parent


def _load_trusted_runtime():
    """Load the trusted runtime, then hide its import route."""
    sys.path.insert(0, str(RUNTIME_ROOT))
    try:
        import kv_transfer_runtime
    finally:
        sys.path = [
            entry
            for entry in sys.path
            if Path(entry or ".").resolve() not in {RUNTIME_ROOT, JUDGE_ROOT}
        ]
        sys.modules.pop("kv_transfer_runtime", None)
        os.environ.pop("PYTHONPATH", None)
    return kv_transfer_runtime


def _strict_stdin_json() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(32 * 1024 * 1024 + 1)
    if not raw or len(raw) > 32 * 1024 * 1024:
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
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = _strict_stdin_json()
    runtime = _load_trusted_runtime()
    if manifest.get("schema_version") != 2 or manifest.get("suite_id") != runtime.FINAL_SUITE_ID:
        raise ValueError("manifest identity differs")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("manifest must contain a cases list")
    cases = [runtime.validate_case(case, scored=True) for case in raw_cases]

    import torch

    # Compile with the frozen command for the allocated GPU's native
    # architecture, then audit the binary before loading.
    library, compile_log = runtime.compile_candidate(
        args.candidate.resolve(),
        args.build_dir.resolve(),
        RUNTIME_ROOT,
        arch=runtime.native_arch(torch),
    )
    binary_report = runtime.inspect_binary(library)
    if not binary_report["ok"]:
        raise RuntimeError(
            "compiled submission failed the undefined-symbol audit: "
            + "; ".join(binary_report["problems"][:8])
        )

    result = runtime.run_suite(
        torch,
        library,
        cases,
        suite_id=str(manifest["suite_id"]),
    )
    if not isinstance(result, dict):
        raise RuntimeError("runtime result must be an object")
    result["source_sha256"] = runtime.sha256_file(args.candidate.resolve())
    result["binary_report"] = binary_report
    result["compile_log_tail"] = compile_log[-2000:]
    _atomic_json(args.output.resolve(), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
