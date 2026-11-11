#!/usr/bin/env python3
"""Unprivileged worker for one private quantized-attention suite."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
from typing import Any,Callable

RUNTIME_ROOT = Path("/opt/sageattention_quant_attention_worker")
JUDGE_ROOT = Path(__file__).resolve().parent
CASE_KEYS = {"case_id", "family", "tier", "model", "b", "h_q", "h_kv", "d", "l",
             "is_causal", "regime", "seed", "gated", "layer"}
EXPECTED_CASE_COUNT = 774


def _load() -> Callable[...,dict[str,Any]]:
    sys.path.insert(0, str(RUNTIME_ROOT))
    try:
        from quant_attention_runtime import run_suite
    finally:
        sys.path = [
            x
            for x in sys.path
            if Path(x or ".").resolve() not in {RUNTIME_ROOT, JUDGE_ROOT}
        ]
        sys.modules.pop("quant_attention_runtime", None)
        os.environ.pop("PYTHONPATH", None)
    return run_suite


FAILURE_STAGES = ("setup", "edge", "real", "scored")


def failure_stage(message: str) -> str:
    if message.startswith("edge case "):
        return "edge"
    if message.startswith("real layer "):
        return "real"
    if message.startswith("case "):
        return "scored"
    return "setup"


def _write_failure_stage(output: Path, stage: str) -> None:
    target = output.with_name("failure.json")
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps({"schema_version": 1, "stage": stage}) + "\n")
    os.replace(tmp, target)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trials", type=int, required=True)
    p.add_argument("--warmups", type=int, required=True)
    a = p.parse_args(argv)
    raw = sys.stdin.buffer.read((4 << 20) + 1)
    if len(raw) > 4 << 20:
        raise ValueError("manifest exceeds 4 MiB")
    manifest = json.loads(
        raw.decode(),
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
    )
    if manifest.get("schema_version") != 1 or manifest.get("suite_id") != "final-holdout-v3":
        raise ValueError("manifest identity differs")
    cases = manifest.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != EXPECTED_CASE_COUNT
        or any(not isinstance(x, dict) or set(x) != CASE_KEYS for x in cases)
    ):
        raise ValueError("manifest cases differ")
    try:
        result = _load()(
            a.candidate.resolve(),
            cases,
            trials=a.trials,
            warmups=a.warmups,
            require_hardware=True,
            measure_bf16_reference=True,
        )
    except BaseException as exc:
        # Record only which stage failed, so the verifier can attribute the
        # failing gate; a failure record can never turn a run valid.
        _write_failure_stage(a.output, failure_stage(str(exc)))
        raise
    tmp = a.output.with_name(f".{a.output.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
