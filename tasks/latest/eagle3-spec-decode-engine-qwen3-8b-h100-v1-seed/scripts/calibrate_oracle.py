#!/usr/bin/env python3
"""Run every ABI-compatible host reference through the private verifier."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

EXPECTED_VERDICTS = {
    "seed": "PASS",
    "greedy_kv_control": "PASS",
    "ladder_l1_device_tree": "PASS",
    "ladder_l2_cuda_graphs": "PASS",
    "ladder_l3_sdpa_tree": "PASS",
    "lossy_control": "FAIL",
    "fp8_target_control": "FAIL",
}


def candidate_sources(task_root: Path) -> dict[str, Path]:
    """Return all controller-compatible candidate sources."""
    host = task_root / "host_reference"
    return {
        "seed": task_root / "images/work/workdir/spec_lab",
        "greedy_kv_control": host / "greedy_kv_control",
        "ladder_l1_device_tree": host / "ladder/engine_l1_device_tree",
        "ladder_l2_cuda_graphs": host / "ladder/engine_l2_cuda_graphs",
        "ladder_l3_sdpa_tree": host / "ladder/engine_l3_sdpa_tree",
        "lossy_control": host / "lossy_control",
        "fp8_target_control": host / "fp8_target_control",
    }


def is_workspace(path: Path) -> bool:
    """Return whether a path already follows the submission layout."""
    return (path / "FINAL_VERSION").is_file() and (path / "solutions").is_dir()


def materialize_workspace(source: Path, destination: Path) -> Path:
    """Wrap a bare candidate module directory in an engine_v001 workspace."""
    if is_workspace(source):
        return source
    version = destination / "solutions/engine_v001"
    version.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, version)
    else:
        raise ValueError(f"candidate source is not a directory: {source}")
    (destination / "FINAL_VERSION").write_text("1\n", encoding="ascii")
    (destination / "notes.md").write_text(
        "Host-only oracle calibration workspace.\n",
        encoding="utf-8",
    )
    return destination


def run_controller(
    python: Path,
    controller: Path,
    candidate: Path,
    judge_root: Path,
    spec_root: Path,
    profile: str,
    output: Path,
    gpu: str,
) -> tuple[int, dict[str, Any] | None]:
    """Execute one full controller/worker/oracle pass."""
    command = [
        str(python),
        str(controller),
        "--candidate-root",
        str(candidate),
        "--profile",
        profile,
        "--all-anchor",
        "--output",
        str(output),
        "--judge-root",
        str(judge_root),
        "--spec-root",
        str(spec_root),
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    completed = subprocess.run(command, env=environment, check=False)
    report = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else None
    return completed.returncode, report


def compact_report(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep calibration evidence without copying token streams."""
    if report is None:
        return None
    oracle = report.get("oracle", {})
    return {
        "status": report.get("status"),
        "score_tokens_per_s": report.get("score_tokens_per_s"),
        "gates": report.get("gates"),
        "cell_tokens_per_s": report.get("cell_tokens_per_s"),
        "oracle": {
            "positions": oracle.get("positions"),
            "soft_divergences": oracle.get("soft_divergences"),
            "max_margin": oracle.get("max_margin"),
            "soft_rate": oracle.get("soft_rate"),
            "tier_results": oracle.get("tier_results"),
        },
    }


def upstream_record(task_root: Path) -> dict[str, Any]:
    """Record why upstream-direct is calibrated by parity, not candidate import."""
    path = task_root / "host_reference/upstream_direct/upstream_direct.py"
    return {
        "status": "LAB_PARITY_CONTROL",
        "path": str(path),
        "verifier_route": "seed ABI port",
        "reason": (
            "upstream_direct is a host CLI around vendor/EAGLE, not a build/generate "
            "submission. B2 established 64/64 token parity and 0.998-1.005 speed ratio; "
            "the byte-shipping seed is the controller-compatible upstream route."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse calibration options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--profile", choices=("anchor", "final"), default="anchor")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--only", action="append", choices=tuple(EXPECTED_VERDICTS))
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="rebuild the aggregate from existing detailed sidecar reports",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run selected controls and write the table helper D consumes."""
    args = parse_args(argv)
    task_root = Path(__file__).resolve().parents[1]
    judge_root = task_root / "images/judge"
    spec_root = task_root / "images/work/workdir/spec_lab"
    controller = judge_root / "verifier/controller.py"
    selected = set(args.only or EXPECTED_VERDICTS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="eagle3-calibration-") as temporary:
        staging = Path(temporary)
        for name, source in candidate_sources(task_root).items():
            if name not in selected:
                continue
            output = args.out.parent / f"{args.out.stem}_{name}.json"
            if args.reports_only:
                report = (
                    json.loads(output.read_text(encoding="utf-8"))
                    if output.is_file()
                    else None
                )
                returncode = 0 if report and report.get("status") == "PASS" else 1
            else:
                workspace = materialize_workspace(source, staging / name)
                returncode, report = run_controller(
                    args.python,
                    controller,
                    workspace,
                    judge_root,
                    spec_root,
                    args.profile,
                    output,
                    args.gpu,
                )
            observed = report.get("status") if report else "ERROR"
            expected = EXPECTED_VERDICTS[name]
            results[name] = {
                "returncode": returncode,
                "expected": expected,
                "observed": observed,
                "classification_ok": observed == expected,
                "report_path": str(output),
                "summary": compact_report(report),
            }
    results["upstream_direct"] = upstream_record(task_root)
    payload = {
        "schema_version": 1,
        "status": "PASS"
        if all(row.get("classification_ok", True) for row in results.values())
        else "FAIL",
        "profile": args.profile,
        "gpu": args.gpu,
        "policy": json.loads(
            (judge_root / "private_data/policy.json").read_text(encoding="utf-8")
        ),
        "controls": results,
    }
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
