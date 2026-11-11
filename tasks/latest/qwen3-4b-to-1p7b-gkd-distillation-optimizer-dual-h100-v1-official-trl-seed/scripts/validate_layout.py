#!/usr/bin/env python3
"""Dependency-free static validation for the frozen task source tree."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path


TASK_ID = "qwen3_4b_to_1p7b_gkd_distillation_optimizer_dual_h100_v1"
TASK_JSON = f"tasks/{TASK_ID}.json"
EXPECTED_SUBMIT = {"solutions", "FINAL_VERSION", "notes.md"}
HIDDEN_CASES = "images/judge/private_data/hidden_cases.json"
HIDDEN_CASES_ASSET = "judge/private_data/hidden_cases.json"


def hidden_cases_source(task_root: Path) -> Path:
    """The Judge-only cases: the in-repository copy, else the task's asset
    (``$INFRATHON_ASSETS_ROOT/<task directory>/judge/private_data/hidden_cases.json``,
    fetched by assets/fetch.sh)."""

    local = task_root / HIDDEN_CASES
    if local.exists() or local.is_symlink():
        return local
    root = os.environ.get("INFRATHON_ASSETS_ROOT") or str(Path.home() / ".cache/infrathon/assets")
    return Path(root).expanduser() / task_root.resolve().name / HIDDEN_CASES_ASSET


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_at(path: Path, errors: list[str]) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"cannot parse {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path} must contain one JSON object")
        return {}
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    root = parser.parse_args().root.resolve()
    errors: list[str] = []
    required = (
        "README.md",
        "tasks/BENCHMARK.yaml",
        TASK_JSON,
        "images/work/Dockerfile",
        "images/work/AGENT_TASK.md",
        "images/work/codex-config.toml",
        "images/work/public_tests/public_cases.json",
        "images/work/public_tests/public_eval.py",
        "images/work/public_tests/public_train_probe.py",
        "images/work/public_tests/run_public_tests.sh",
        "images/work/workdir/gkd_lab/FINAL_VERSION",
        "images/work/workdir/gkd_lab/UPSTREAM_HF_TRL.md",
        "images/work/workdir/gkd_lab/notes.md",
        "images/work/workdir/gkd_lab/runtime/api.py",
        "images/work/workdir/gkd_lab/runtime/hf_reference.py",
        "images/work/workdir/gkd_lab/runtime/loader.py",
        "images/work/workdir/gkd_lab/runtime/source_policy.py",
        "images/work/workdir/gkd_lab/runtime/train_step.py",
        "images/work/workdir/gkd_lab/runtime/distributed_probe.py",
        "images/work/workdir/gkd_lab/solutions/objective_v001/candidate.py",
        "images/judge/Dockerfile",
        "images/judge/run_judge.py",
        "images/judge/verifier/verify_gkd.py",
        "images/judge/verifier/candidate_worker.py",
        "judge/run_judge.py",
        "scripts/build_images.sh",
        "scripts/preflight.py",
        "scripts/prepare_tldr_gkd_data.py",
        "scripts/sforge_dual_h100_entry.py",
        "scripts/validate_model_artifact.py",
        "scripts/smoke_test.sh",
        "tests/test_official_trl_provenance.py",
    )
    for relative in required:
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing regular nonempty file: {relative}")
    hidden_path = hidden_cases_source(root)
    if hidden_path.is_symlink() or not hidden_path.is_file() or hidden_path.stat().st_size == 0:
        errors.append(f"missing regular nonempty file: {HIDDEN_CASES} (Judge-only: run assets/fetch.sh)")

    task = object_at(root / TASK_JSON, errors)
    if task:
        if task.get("task_id") != TASK_ID:
            errors.append("task_id mismatch")
        if set(task.get("submit_paths", [])) != EXPECTED_SUBMIT:
            errors.append("submit_paths differ from the narrow whitelist")
        if task.get("internet") is not True:
            errors.append("task must allow internet")
        if task.get("cwd") != "/app/workdir/gkd_lab":
            errors.append("unexpected task cwd")
        judge = task.get("judge", {})
        if judge.get("parser") != "structured_json":
            errors.append("Judge parser must be structured_json")
        if judge.get("private_artifact_path") != "/logs/verifier/gkd_private.json":
            errors.append("private artifact path mismatch")
        if judge.get("selection") != "valid_then_score":
            errors.append("Judge must select valid_then_score")

    canonical = root / "judge/run_judge.py"
    mirror = root / "images/judge/run_judge.py"
    if canonical.is_file() and mirror.is_file() and digest(canonical) != digest(mirror):
        errors.append("Judge adapter mirror differs from canonical")

    public = object_at(root / "images/work/public_tests/public_cases.json", errors)
    hidden = object_at(hidden_path, errors)
    if public and hidden:
        public_cases = public.get("correctness", []) + public.get("gpu_objective", [])
        private_cases = hidden.get("cases", [])
        public_ids = {case.get("case_id") for case in public_cases}
        private_ids = {case.get("case_id") for case in private_cases}
        public_seeds = {case.get("seed") for case in public_cases}
        private_seeds = {case.get("seed") for case in private_cases}
        if public_ids & private_ids or public_seeds & private_seeds:
            errors.append("public/private cases overlap")
        activation = (
            hidden.get("formal_ready"),
            hidden.get("formal_score_enabled"),
            hidden.get("formal_thresholds_frozen"),
            hidden.get("workload_plan_status"),
        )
        active = (True, True, True, "FROZEN_FORMAL_CROSS_MATRIX")
        cross_matrix_pending = (
            False,
            False,
            False,
            "PENDING_CROSS_MATRIX_ARTIFACTS_AND_CALIBRATION",
        )
        if activation not in (active, cross_matrix_pending):
            errors.append(
                "hidden formal activation flags must be either a fully frozen "
                "formal plan or the fail-closed official-TRL remediation state"
            )

    source_files = list(root.glob("**/*.py"))
    for path in source_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            errors.append(f"invalid Python source {path.relative_to(root)}: {exc}")

    lab = root / "images/work/workdir/gkd_lab"
    if lab.is_dir():
        sys.path.insert(0, str(lab))
        try:
            from runtime.source_policy import admit

            admit(lab)
        except ModuleNotFoundError as exc:
            # This host-side pre-check needs the GPU stack (torch) only to
            # import the seed sources. The authoritative admission gate is
            # images/judge/verifier/source_admission.py, which re-runs the
            # same admit() inside the Judge container where torch exists —
            # so a host without torch skips the preview instead of freezing
            # every build and launcher behind an unrelated dependency.
            print(
                "WARNING: seed source admission preview skipped "
                f"(missing host dependency: {exc.name}); the Judge container "
                "still enforces admission at evaluation time",
                file=sys.stderr,
            )
        except Exception as exc:
            errors.append(f"seed source admission failed: {exc}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"layout validation passed: {TASK_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
