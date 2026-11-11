#!/usr/bin/env python3
"""Revalidate the already-measured v24 public pair after the warmup fix.

The original raw legs are retained byte-for-byte inside a new evidence file.
Only the outer validator result is recomputed. This avoids rerunning GPU work
when the old validator rejected nonzero ``pre_measurement_steps`` before it
understood the v24 300-second unscored interval.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAIR_RUNNER = (
    ROOT
    / "runtime/frozen/harbor_run_scripts/"
    "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/"
    "run-gpu67-v2.1-pair.py"
)
PUBLIC_NONCE_SHA256 = (
    "4902baae73726c1a2f372e35e9e104f6bb3db7dc7593822359548fba60e1b00c"
)
PUBLIC_PROFILE = "public-swe192-distinct"
PUBLIC_CONTROLLER_SHA256 = (
    "14e6817031e6a31b6c01835b2c31d490735e66517f6e248bbfaf3111ba2bd66e"
)
TOKEN_PATTERN = re.compile(r"(?<![0-9a-f])([0-9a-f]{64})(?![0-9a-f])")


def load_pair_runner() -> Any:
    spec = importlib.util.spec_from_file_location("kvbench_v24_pair", PAIR_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("v24 pair validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAIR = load_pair_runner()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_nonce(source: Path) -> str:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError("nonce source is unavailable or unsafe")
    matches: set[str] = set()
    with source.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            for token in TOKEN_PATTERN.findall(line):
                if hashlib.sha256(token.encode()).hexdigest() == PUBLIC_NONCE_SHA256:
                    matches.add(token)
    if len(matches) != 1:
        raise RuntimeError("nonce source does not contain exactly one public value")
    return next(iter(matches))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nonce-source", required=True, type=Path)
    args = parser.parse_args()

    source_argument = args.source.expanduser()
    if source_argument.is_symlink():
        raise SystemExit("--source must be a real evidence file")
    source = source_argument.resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if not source.is_file():
        raise SystemExit("--source must be a real evidence file")
    if output.exists() or output.is_symlink():
        raise SystemExit("refusing to overwrite revalidated evidence")
    report = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise SystemExit("source evidence must be a JSON object")
    old_evaluation = report.get("evaluation")
    if any(
        (
            report.get("schema_version") != 1,
            report.get("status") != "complete",
            report.get("suite") != "public",
            report.get("profile") != PUBLIC_PROFILE,
            report.get("workload_nonce_sha256") != PUBLIC_NONCE_SHA256,
            report.get("errors") != [],
            not isinstance(old_evaluation, dict),
            old_evaluation.get("baseline_invalid_reasons")
            != ["step_partition_invalid", "valid_step_ratio_below_floor"],
            old_evaluation.get("candidate_invalid_reasons")
            != ["step_partition_invalid"],
            old_evaluation.get("pair_invalid_reasons") != [],
        )
    ):
        raise SystemExit("source is not the exact warmup-validator rejection")
    legs = report.get("legs")
    baseline = legs.get("baseline") if isinstance(legs, dict) else None
    candidate = legs.get("candidate") if isinstance(legs, dict) else None
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise SystemExit("source pair legs are missing")
    if any(
        leg.get("controller_config_sha256") != PUBLIC_CONTROLLER_SHA256
        for leg in (baseline, candidate)
    ):
        raise SystemExit("source pair used the wrong public controller")

    baseline_reasons = PAIR.leg_validation_reasons(
        baseline, suite="public", profile=PUBLIC_PROFILE, mode="baseline"
    )
    candidate_reasons = PAIR.leg_validation_reasons(
        candidate, suite="public", profile=PUBLIC_PROFILE, mode="candidate"
    )
    pair_reasons = PAIR.pair_validation_reasons(
        baseline, candidate, workload_nonce=public_nonce(args.nonce_source)
    )
    baseline_provenance = PAIR.canonical_sha256(baseline.get("model_provenance"))
    candidate_provenance = PAIR.canonical_sha256(candidate.get("model_provenance"))
    provenance_match = (
        baseline_provenance is not None
        and baseline_provenance == candidate_provenance
    )
    baseline_rate = float(baseline["valid_steps_per_min"])
    candidate_rate = float(candidate["valid_steps_per_min"])
    evaluation = {
        "baseline_valid_steps_per_min": baseline_rate,
        "candidate_valid_steps_per_min": candidate_rate,
        "speedup": candidate_rate / baseline_rate,
        "change_percent": (candidate_rate / baseline_rate - 1.0) * 100.0,
        "baseline_request_success_rate": baseline.get("request_success_rate"),
        "candidate_request_success_rate": candidate.get("request_success_rate"),
        "baseline_valid": not baseline_reasons,
        "candidate_valid": not candidate_reasons,
        "baseline_invalid_reasons": baseline_reasons,
        "candidate_invalid_reasons": candidate_reasons,
        "pair_invalid_reasons": pair_reasons,
        "baseline_model_provenance_sha256": baseline_provenance,
        "candidate_model_provenance_sha256": candidate_provenance,
        "model_provenance_match": provenance_match,
        "performance_improved": candidate_rate > baseline_rate,
        "valid": (
            not baseline_reasons
            and not candidate_reasons
            and not pair_reasons
            and provenance_match
        ),
    }
    if evaluation["valid"] is not True or not all(
        math.isfinite(value) for value in (baseline_rate, candidate_rate)
    ):
        raise SystemExit(f"corrected validator still rejects pair: {evaluation}")
    report["evaluation"] = evaluation
    report["revalidation"] = {
        "schema_version": 1,
        "revalidated_at_utc": PAIR.utc_now(),
        "source_file_sha256": file_sha256(source),
        "reason": "v24 accepts positive pre_measurement_steps for 300s unscored warmup",
        "raw_legs_modified": False,
    }
    PAIR.atomic_write(output, report, initial=True)
    print(
        json.dumps(
            {
                "valid": True,
                "baseline_valid_steps_per_min": baseline_rate,
                "candidate_valid_steps_per_min": candidate_rate,
                "output_sha256": file_sha256(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
