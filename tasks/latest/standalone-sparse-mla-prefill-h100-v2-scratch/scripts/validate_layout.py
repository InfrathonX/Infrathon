#!/usr/bin/env python3
"""Lightweight static validation for the standalone sparse-MLA prefill v2 task."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


TASK_ID = "standalone_sparse_mla_prefill_h100_scratch_v2"
BENCHMARK = "standalone_sparse_mla_prefill"
TASK_DIRECTORY = "standalone-sparse-mla-prefill-h100-v2-scratch"
WORKSPACE = "/app/workdir/sparse_mla_prefill_lab"
SUBMIT_PATHS = ["candidates", "FINAL_VERSION", "notes.md"]
PRIVATE_ARTIFACT = "/logs/verifier/standalone_sparse_mla_prefill_v2_results.json"
# Suite identities and frozen case counts; scripts/lib.sh carries the same
# values for the image build.
FINAL_SUITE_ID = "final-holdout-v4"
EDGE_SUITE_ID = "edge-holdout-v3"
PUBLIC_EDGE_SUITE_ID = "edge-public-v3"
EXPECTED_FINAL_CASES = 997
EXPECTED_EDGE_CASES = 470
EXPECTED_PUBLIC_EDGE_CASES = 73
MANIFEST_SCHEMA_VERSION = 3
# The scored final manifest is schema 5 (scripts/generate_final_manifest.py).
FINAL_MANIFEST_SCHEMA_VERSION = 5
PUBLIC_EDGE_CASES = "images/work/benchmark/public_edge_cases.json"
# Judge-only evaluation inputs: committed with the task, admitted by no Work
# build context.  The frozen final holdout (63-bit seed per upstream performance
# shape) and the frozen private edge cases are the default build inputs;
# ``validate_private_inputs`` proves the Work build can never see them.
SENSITIVE_JUDGE_INPUTS = ("judge/private/final_cases.json", "judge/private/edge_cases.json")
# A tree without the repository copies (the public tree) reads them as dataset
# assets: assets/fetch.sh places each under <INFRATHON_ASSETS_ROOT>/<task directory>/.
JUDGE_INPUT_ASSETS = {
    "judge/private/final_cases.json": "judge/final_cases.json",
    "judge/private/edge_cases.json": "judge/edge_cases.json",
}
# Everything the Work image build can read: its own context directory plus the
# named build contexts scripts/build_images.sh passes to `docker build`.
WORK_BUILD_CONTEXTS = ("images/work",)


def _json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read JSON {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path} must contain a JSON object")
        return {}
    return value


def _yaml(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        errors.append(f"cannot read YAML {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path} must contain a YAML mapping")
        return {}
    return value


def judge_input_path(root: Path, relative: str) -> Path | None:
    """The repository copy of a Judge-only input, else its fetched dataset asset, else None."""
    path = root / relative
    if path.exists() or path.is_symlink():
        return path
    assets_root = os.environ.get("INFRATHON_ASSETS_ROOT") or "~/.cache/infrathon/assets"
    fetched = Path(assets_root).expanduser() / root.name / JUDGE_INPUT_ASSETS[relative]
    return fetched if fetched.is_file() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_layout(root: Path, errors: list[str]) -> None:
    required_files = (
        ".gitignore",
        "README.md",
        "tasks/BENCHMARK.yaml",
        f"tasks/{TASK_ID}.json",
        "images/work/.dockerignore",
        "images/work/Dockerfile",
        "images/work/AGENT_TASK.md",
        "images/work/codex-config.toml",
        "images/work/benchmark/public_runner.py",
        "images/work/benchmark/sparse_mla_prefill_runtime.py",
        "images/work/benchmark/run_public_tests.sh",
        "images/work/benchmark/source_admission.py",
        "images/work/benchmark/prepare_submit.sh",
        PUBLIC_EDGE_CASES,
        "images/work/workdir/sparse_mla_prefill_lab/README_AGENT.md",
        "images/work/workdir/sparse_mla_prefill_lab/FINAL_VERSION",
        "images/work/workdir/sparse_mla_prefill_lab/notes.md",
        "images/work/workdir/sparse_mla_prefill_lab/candidates/kernel_v001/submission.py",
        "images/work/workdir/sparse_mla_prefill_lab/tools/run_one.sh",
        "images/work/workdir/sparse_mla_prefill_lab/tools/run_public_tests.sh",
        "images/judge/.dockerignore",
        "images/judge/Dockerfile",
        "images/judge/Dockerfile.private",
        "images/judge/run_judge.py",
        "images/judge/verify_sparse_mla.py",
        "images/judge/candidate_worker.py",
        "images/judge/source_admission.py",
        "images/judge/protected_hashes.txt",
        "judge/run_judge.py",
        "tests/test_private_inputs.py",
        "tests/test_edge_suite.py",
        "llm_judge/POLICY.md",
        "runtime/sparse_mla_prefill_runtime.py",
        "oracle/FINAL_VERSION",
        "oracle/ADMISSIBLE_VERSION",
        "oracle/notes.md",
        "oracle/candidates/kernel_v001/submission.py",
        "scripts/lib.sh",
        "scripts/build_images.sh",
        "scripts/generate_final_manifest.py",
        "scripts/generate_edge_manifest.py",
        "scripts/make_oracle_archive.sh",
        "scripts/eval_archive.sh",
        "scripts/smoke_test.sh",
        "scripts/run_fixed_budget.sh",
        "scripts/run_single_gpu_fixed_budget.sh",
        "scripts/single_gpu_runtime_preflight.py",
        "scripts/refresh_portable_manifests.py",
        "scripts/restricted_proxy.py",
        "scripts/status.sh",
        "scripts/validate_layout.py",
        "scripts/validate_formal_artifacts.py",
        "portable/resource-manifest.json",
        "portable/build-resource-manifest.json",
        "portable/public-suite-manifest.json",
        "portable/interactive-suite-manifest.json",
        "portable/final-suite-manifest.json",
        "portable/oracle-protocol.json",
        "portable/selection-protocol.json",
        "portable/software-abi.json",
        "portable/calibration-protocol.json",
        "portable/REBUILD_PROMPT.md",
    )
    for relative in required_files:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing regular non-symlink file: {relative}")
    # Internal per-model run wrappers (scripts/launch_<model>_*_tmux.sh) are optional.
    for path in sorted((root / "scripts").glob("launch_*_tmux.sh")):
        if not path.is_file() or path.is_symlink():
            errors.append(f"not a regular non-symlink file: {path.relative_to(root)}")
    tests = root / "tests"
    if not tests.is_dir() or tests.is_symlink():
        errors.append("missing regular non-symlink directory: tests")
    for forbidden_tree in ("upstream", "vendor", "flash-attention", "FlashMLA"):
        if (root / forbidden_tree).exists():
            errors.append(f"standalone task must not contain a {forbidden_tree} source tree")


def _check_task_metadata(root: Path, errors: list[str]) -> None:
    task_path = root / "tasks" / f"{TASK_ID}.json"
    task = _json(task_path, errors)
    expected = {
        "task_id": TASK_ID,
        "base_image": "pytorch280_cu129",
        "platform": "linux/amd64",
        "cwd": WORKSPACE,
        "submit_paths": SUBMIT_PATHS,
    }
    for key, wanted in expected.items():
        if task.get(key) != wanted:
            errors.append(f"{task_path}: {key} must be {wanted!r}")
    work = task.get("work")
    judge = task.get("judge")
    if not isinstance(work, dict):
        errors.append(f"{task_path}: work must be an object")
    else:
        if work.get("image_tag") != "local":
            errors.append(f"{task_path}: work.image_tag must be 'local'")
        if work.get("specs_dir") != WORKSPACE:
            errors.append(f"{task_path}: work.specs_dir must be {WORKSPACE!r}")
    if not isinstance(judge, dict):
        errors.append(f"{task_path}: judge must be an object")
    else:
        expected_judge = {
            "image_tag": "local",
            "parser": "structured_json",
            "private_artifact_path": PRIVATE_ARTIFACT,
            "score_direction": "maximize",
            "selection": "valid_then_score",
        }
        for key, wanted in expected_judge.items():
            if judge.get(key) != wanted:
                errors.append(f"{task_path}: judge.{key} must be {wanted!r}")
        if "rescale" in judge:
            errors.append(f"{task_path}: raw TFLOPS must not be rescaled")

    benchmark_path = root / "tasks/BENCHMARK.yaml"
    benchmark = _yaml(benchmark_path, errors)
    if benchmark.get("name") != BENCHMARK:
        errors.append(f"{benchmark_path}: name must be {BENCHMARK!r}")
    base_images = benchmark.get("base_images")
    if not isinstance(base_images, dict) or "pytorch280_cu129" not in base_images:
        errors.append(f"{benchmark_path}: missing pytorch280_cu129 base image")


def _check_experiment(root: Path, errors: list[str]) -> None:
    # An internal run configuration (experiment-<model>-max-2h.yaml); optional.
    for path in sorted(root.glob("experiment-*-max-2h.yaml")):
        _check_experiment_file(path, errors)


def _check_experiment_file(path: Path, errors: list[str]) -> None:
    config = _yaml(path, errors)
    defaults = config.get("defaults")
    env = config.get("env")
    tasks = config.get("tasks")
    if not isinstance(defaults, dict):
        errors.append(f"{path}: defaults must be a mapping")
        return
    expected = {
        "backend": "docker",
        "agent": "codex",
        "timeout": 7200,
        "eval_interval": 300,
        "submission_cooldown": 300,
        "disable_stop_hook": False,
        "disable_auto_eval": True,
        "disable_auto_resume": False,
    }
    for key, wanted in expected.items():
        if defaults.get(key) != wanted:
            errors.append(f"{path}: defaults.{key} must be {wanted!r}")
    if not isinstance(defaults.get("model"), str) or not defaults["model"]:
        errors.append(f"{path}: defaults.model must name the agent model")
    if not isinstance(tasks, dict) or set(tasks) != {TASK_ID}:
        errors.append(f"{path}: tasks must contain exactly {TASK_ID!r}")
    if not isinstance(env, dict) or env.get("SFORGE_CODEX_REASONING_EFFORT") != "max":
        errors.append(f"{path}: reasoning effort must be max")


def _read(path: Path, errors: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read {path}: {exc}")
        return ""


def _check_images(root: Path, errors: list[str]) -> None:
    work_docker = _read(root / "images/work/Dockerfile", errors)
    judge_docker = _read(root / "images/judge/Dockerfile", errors)
    work_lower = work_docker.lower()
    judge_lower = judge_docker.lower()
    forbidden_source_installs = (
        "copy --from=flash-attention",
        "copy --from=cutlass",
        "/opt/flash-attention",
        "/opt/flash_mla",
        "pip install flash-mla",
        "pip install flash_mla",
        "pip install flash-attn",
    )
    for label, text in (("Work", work_lower), ("Judge", judge_lower)):
        if any(marker in text for marker in forbidden_source_installs):
            errors.append(f"{label} Dockerfile must not install or copy FlashMLA/CUTLASS")
    if "copy --from=task_runtime" not in judge_lower:
        errors.append("Judge Dockerfile must copy the pinned task runtime")
    if "task_runtime" in work_lower:
        errors.append("Work Dockerfile must use its own public runtime, not the Judge's")
    if "from ${work_image}" in judge_lower or "arg work_image" in judge_lower:
        errors.append("Judge image must be built independently, not FROM Work")
    # The public Judge is publishable; only its host-built holdout layer
    # (Dockerfile.private, FROM the public Judge) receives the final_cases context.
    if "final_cases" in judge_lower or "edge_cases.json" in judge_lower:
        errors.append("public Judge Dockerfile must not receive the private manifests")
    private_lower = _read(root / "images/judge/Dockerfile.private", errors).lower()
    if "from ${judge_public_image}" not in private_lower:
        errors.append("Judge holdout layer must be built FROM the public Judge image")
    if "copy --from=final_cases final_cases.json edge_cases.json" not in private_lower:
        errors.append("Judge holdout layer must copy both private manifests from the final_cases context")
    if "compute-sanitizer" not in judge_lower:
        errors.append("Judge Dockerfile must assert that compute-sanitizer is installed")
    if "copy --from=final_cases" in work_lower:
        errors.append("Work Dockerfile must not receive the private final_cases context")
    if "ln -s" in work_lower and "source" in work_lower:
        errors.append("Work Dockerfile must not create a source symlink")
    hook = "/app/workdir/sparse_mla_prefill_lab/.sforge/prepare_submit.sh"
    if f"COPY benchmark/prepare_submit.sh {hook}" not in work_docker:
        errors.append("Work Dockerfile must install the cooperative pre-submit hook")
    if "chown -R root:root /app/workdir/sparse_mla_prefill_lab/.sforge" not in work_docker:
        errors.append("Work Dockerfile must leave the pre-submit hook root-owned")
    if "portable" in work_lower or "final-suite-manifest" in work_lower:
        errors.append("Work Dockerfile must not reference the host-private portable manifests")

    mirrored = (
        ("judge/run_judge.py", "images/judge/run_judge.py"),
        ("images/work/benchmark/source_admission.py", "images/judge/source_admission.py"),
    )
    for left, right in mirrored:
        left_path, right_path = root / left, root / right
        if left_path.is_file() and right_path.is_file() and _sha256(left_path) != _sha256(right_path):
            errors.append(f"mirrored trust-boundary files differ: {left} != {right}")

    forbidden = {"judge", "oracle", "hidden", "host_reference", "upstream"}
    work_root = root / "images/work"
    if work_root.is_dir():
        for path in work_root.rglob("*"):
            if path.is_symlink():
                errors.append(f"symlink is forbidden in Work context: {path}")
                continue
            parts = {part.lower() for part in path.relative_to(work_root).parts}
            if parts.intersection(forbidden):
                errors.append(f"private/reference path is present in Work context: {path}")
            if path.is_file() and path.suffix.lower() in {".c", ".cc", ".cpp", ".cu", ".cuh", ".so", ".a", ".ptx", ".cubin"}:
                errors.append(f"compiled/upstream implementation material is present in Work: {path}")

    # The Oracle is the host-only positive control: it lives outside every
    # build context, and the image builds must not name it.
    judge_root = root / "images/judge"
    if judge_root.is_dir():
        for path in judge_root.rglob("*"):
            if path.is_symlink():
                errors.append(f"symlink is forbidden in Judge context: {path}")
                continue
            parts = {part.lower() for part in path.relative_to(judge_root).parts}
            if parts.intersection(forbidden - {"judge"}):
                errors.append(f"private/reference path is present in Judge context: {path}")
    build_script = _read(root / "scripts/build_images.sh", errors)
    if "MIGRATION_ROOT}/oracle" in build_script:
        errors.append("image build must not reference the oracle tree")
    for label, text in (("images/work/Dockerfile", work_docker), ("images/judge/Dockerfile", judge_docker)):
        for token in ("task_oracle", "/oracle"):
            if token in text:
                errors.append(f"{label} must not copy from {token}")


def _check_starter(root: Path, errors: list[str]) -> None:
    starter = root / (
        "images/work/workdir/sparse_mla_prefill_lab/"
        "candidates/kernel_v001/submission.py"
    )
    if starter.is_file() and starter.stat().st_size != 0:
        errors.append("Work kernel_v001/submission.py must start as a zero-byte file")
    final_version = root / "images/work/workdir/sparse_mla_prefill_lab/FINAL_VERSION"
    if final_version.is_file() and final_version.read_text(encoding="utf-8").strip() != "1":
        errors.append("Work FINAL_VERSION must initially select version 1")
    # kernel_v001 stays the in-task (admissible) control.  oracle/FINAL_VERSION may
    # select a stronger later candidate (the strongest paper solution, which may be
    # forced through a local-only Judge, oracle/paper_reference/forced_judge/), and
    # oracle/ADMISSIBLE_VERSION names the candidate the normal Judge admits.
    for name in ("FINAL_VERSION", "ADMISSIBLE_VERSION"):
        selector = root / "oracle" / name
        if not selector.is_file():
            continue
        selected = selector.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[1-9][0-9]{0,2}", selected) is None or not (
            root / f"oracle/candidates/kernel_v{int(selected):03d}/submission.py"
        ).is_file():
            errors.append(f"Oracle {name} must select an existing oracle candidate")


def _check_lifecycle(root: Path, errors: list[str]) -> None:
    run_text = _read(root / "scripts/run_fixed_budget.sh", errors)
    required = (
        'TASK_TIMEOUT="${TASK_TIMEOUT:-7200}"',
        'EVAL_INTERVAL="${EVAL_INTERVAL:-300}"',
        "--disable-auto-eval",
        '--submission-cooldown "${EVAL_INTERVAL}"',
        '--interval-seconds "${EVAL_INTERVAL}"',
        'assert_gpu_released "${work_gpu_uuid}"',
        'assert_gpu_released "${judge_gpu_uuid}"',
        "Work container still exists at the final-evaluation barrier",
        'validate_formal_artifacts.py" pre',
        'validate_formal_artifacts.py" post',
        "final_archive_sha256",
        "evaluated_archive_sha256",
        'manual_evaluation_suite: "final-holdout-v4"',
        'evaluation_policy: "full-final-shared-v3"',
        'selection_protocol_id: "sparse-mla-shared-final-v3"',
        'scheduling_mode="dual_h100_shared_final_v3"',
        'scheduling_mode="single_gpu_lease_shared_final_v3_handoff"',
        "edge_cases_sha256",
        'validate_formal_artifacts.py" stage',
        'scheduling: $scheduling_mode',
        '--evaluation-mode "${scheduling_mode}"',
        '--expected-manifest-sha256 "${final_cases_sha256}"',
        '--expected-case-count "${final_cases_count}"',
        # Single-allocation branch: one UUID flock, NVML attestation before
        # the first Docker probe, exact DeviceRequest probes, the protected
        # pre-submit hook and the explicit standalone final handoff.
        'SINGLE_GPU_HANDOFF="${SFORGE_SINGLE_GPU_HANDOFF:-0}"',
        'flock -n 8 || die "single GPU allocation',
        'phase="before_work_create"',
        "assert_portable_gpu_idle",
        "assert_container_accelerator",
        "SFORGE_PRE_SUBMIT_HOOK=/app/workdir/sparse_mla_prefill_lab/.sforge/prepare_submit.sh",
        "--standalone-final-handoff",
        '--handoff-source-run-id "${RUN_ID}"',
        '--judge-url "http://127.0.0.1:${JUDGE_PORT}"',
        "deployment_attestation_json",
        "measurement_profile_sha256",
        "raw_score_comparison_identity_sha256",
    )
    for fragment in required:
        if fragment not in run_text:
            errors.append(f"scripts/run_fixed_budget.sh: missing {fragment!r}")
    if "--disable-auto-eval" not in run_text:
        errors.append("scripts/run_fixed_budget.sh must disable scheduled auto evaluation")
    lock_position = run_text.find('flock -n 8 || die "single GPU allocation')
    attestation_position = run_text.find("NvmlAcceleratorObserver().attest_idle(")
    probe_position = run_text.find("assert_container_accelerator")
    if not 0 <= lock_position < attestation_position < probe_position:
        errors.append(
            "scripts/run_fixed_budget.sh: single-allocation branch must lock, "
            "attest idle, then probe the container DeviceRequest, in that order"
        )
    wrapper_text = _read(root / "scripts/run_single_gpu_fixed_budget.sh", errors)
    for fragment in (
        "SFORGE_SINGLE_GPU_HANDOFF=1",
        "SFORGE_ACCELERATOR_OBSERVER=nvml",
        "canonical_allocation_key",
        "single_gpu_runtime_preflight.py",
        'exec "${SCRIPT_DIR}/run_fixed_budget.sh"',
    ):
        if fragment not in wrapper_text:
            errors.append(f"scripts/run_single_gpu_fixed_budget.sh: missing {fragment!r}")

    build_text = _read(root / "scripts/build_images.sh", errors)
    context = '--build-context "task_runtime=${MIGRATION_ROOT}/runtime"'
    if build_text.count(context) != 1:
        errors.append("build_images.sh must pass task_runtime to the Judge build only")
    if "flash-attention" in build_text.lower() or "flash_mla" in build_text.lower():
        errors.append("build_images.sh must not pass a FlashMLA source context")
    final_context = '--build-context "final_cases=${final_cases_context}"'
    if build_text.count(final_context) != 1:
        errors.append("build_images.sh must pass final_cases to the Judge build only")
    for fragment in (
        'FINAL_CASES_PATH',
        'EDGE_CASES_PATH',
        'final_cases_sha256',
        'edge_cases_sha256',
        '"${FINAL_SUITE_ID}"',
        '"${EDGE_SUITE_ID}"',
        '"${PUBLIC_EDGE_SUITE_ID}"',
    ):
        if fragment not in build_text:
            errors.append(f"build_images.sh: missing {fragment!r}")
    lib_text = _read(root / "scripts/lib.sh", errors)
    for fragment in (
        f'TASK_ID="{TASK_ID}"',
        f'FINAL_SUITE_ID="{FINAL_SUITE_ID}"',
        f'EDGE_SUITE_ID="{EDGE_SUITE_ID}"',
        f'PUBLIC_EDGE_SUITE_ID="{PUBLIC_EDGE_SUITE_ID}"',
        f"FINAL_CASE_COUNT={EXPECTED_FINAL_CASES}",
        f"EDGE_CASE_COUNT={EXPECTED_EDGE_CASES}",
        f"PUBLIC_EDGE_CASE_COUNT={EXPECTED_PUBLIC_EDGE_CASES}",
    ):
        if fragment not in lib_text:
            errors.append(f"scripts/lib.sh: missing {fragment!r}")


def _context_files(base: Path):
    for path in sorted(base.rglob("*")):
        if (
            path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        ):
            yield path


def validate_private_inputs(root: Path) -> list[str]:
    """The frozen final holdout and the frozen private edge cases ship with
    the task and only the Judge can ever see them.

    Checks: both are present on the Judge-only path; the build defaults to
    them and stages them 0600 into the Judge-only build context; the suite
    manifest records their digests; the generators draw from a seed rather
    than a system random source; and no Work build context admits either
    instance - not by path (none lives under a Work context) and not by
    content (no file a Work context admits has the same SHA-256).
    """

    errors: list[str] = []
    sensitive: dict[str, str] = {}
    for relative in SENSITIVE_JUDGE_INPUTS:
        path = judge_input_path(root, relative)
        if path is None:
            continue
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            errors.append(f"missing Judge-only evaluation input: {relative}")
            continue
        sensitive[hashlib.sha256(path.read_bytes()).hexdigest()] = relative
    contexts = [(name, (root / name).resolve()) for name in WORK_BUILD_CONTEXTS]
    for relative in SENSITIVE_JUDGE_INPUTS:
        resolved = (root / relative).resolve()
        for name, context in contexts:
            if resolved.is_relative_to(context):
                errors.append(f"{relative} lives inside the Work build context {name}")
    for name, context in contexts:
        if not context.is_dir():
            errors.append(f"missing Work build context: {name}")
            continue
        for path in _context_files(context):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in sensitive:
                errors.append(
                    f"Work build context {name} admits {path.relative_to(root)}, "
                    f"byte-identical to the Judge-only {sensitive[digest]}"
                )
    work_dockerfile = root / "images/work/Dockerfile"
    if work_dockerfile.is_file():
        work = work_dockerfile.read_text(encoding="utf-8")
        for token in ("judge/private", "final_cases", "edge_cases.json"):
            if token in work:
                errors.append(f"Work Dockerfile must not reference {token!r}")
    manifest_path = root / "portable/final-suite-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        edge_suite = manifest.get("edge_suite")
        records = (
            ("frozen_instance", manifest.get("frozen_instance"), SENSITIVE_JUDGE_INPUTS[0], True),
            (
                "edge_suite.frozen_instance",
                edge_suite.get("frozen_instance") if isinstance(edge_suite, dict) else None,
                SENSITIVE_JUDGE_INPUTS[1],
                False,
            ),
        )
        for label, frozen, expected_path, needs_provenance in records:
            if not isinstance(frozen, dict):
                errors.append(f"portable/final-suite-manifest.json: missing {label}")
                continue
            if frozen.get("path") != expected_path:
                errors.append(
                    f"portable/final-suite-manifest.json: {label}.path must be {expected_path!r}"
                )
            instance = judge_input_path(root, expected_path)
            if instance is not None and instance.is_file() and frozen.get("sha256") != hashlib.sha256(
                instance.read_bytes()
            ).hexdigest():
                errors.append(
                    f"portable/final-suite-manifest.json: {label}.sha256 does "
                    "not match the committed instance"
                )
            if needs_provenance and not str(frozen.get("provenance", "")).strip():
                errors.append(
                    f"portable/final-suite-manifest.json: {label}.provenance "
                    "must record which draw this is"
                )
    generator_path = root / "scripts/generate_final_manifest.py"
    if generator_path.is_file():
        generator = generator_path.read_text(encoding="utf-8")
        for fragment in ("--seed", "--redraw", "DEFAULT_SEED", "def default_seed("):
            if fragment not in generator:
                errors.append(
                    f"scripts/generate_final_manifest.py: missing {fragment!r}"
                )
        for forbidden in ("secrets.SystemRandom", "import secrets"):
            if forbidden in generator:
                errors.append(
                    f"scripts/generate_final_manifest.py must not use {forbidden!r}: "
                    "the draw is seeded"
                )
    edge_generator_path = root / "scripts/generate_edge_manifest.py"
    if edge_generator_path.is_file():
        edge_generator = edge_generator_path.read_text(encoding="utf-8")
        for fragment in ("random.Random(", "def default_seed(", "--check", "--write"):
            if fragment not in edge_generator:
                errors.append(f"scripts/generate_edge_manifest.py: missing {fragment!r}")
        if "SystemRandom" in edge_generator or "import secrets" in edge_generator:
            errors.append("scripts/generate_edge_manifest.py must draw from a seeded random.Random")
    build_path = root / "scripts/build_images.sh"
    if build_path.is_file():
        build = build_path.read_text(encoding="utf-8")
        for fragment in (
            'stage_private_manifest "${FINAL_CASES_PATH:-${DEFAULT_FINAL_CASES_PATH}}"',
            'stage_private_manifest "${EDGE_CASES_PATH:-${DEFAULT_EDGE_CASES_PATH}}"',
            '"${final_cases_context}/final_cases.json"',
            '"${final_cases_context}/edge_cases.json"',
            'install -m 0600 "${source}" "${staged}"',
            "[[ \"$(stat -c '%a' \"${staged}\")\" == \"600\" ]]",
            "[[ \"$(stat -c '%u' \"${staged}\")\" == \"$(id -u)\" ]]",
        ):
            if fragment not in build:
                errors.append(f"scripts/build_images.sh: missing {fragment!r}")
        if '--output "${final_cases_source}"' in build:
            errors.append(
                "scripts/build_images.sh must not draw a fresh holdout: the frozen "
                "instance is the default build input"
            )
    lib_path = root / "scripts/lib.sh"
    if lib_path.is_file():
        lib_text = lib_path.read_text(encoding="utf-8")
        for variable, relative in (
            ("DEFAULT_FINAL_CASES_PATH", SENSITIVE_JUDGE_INPUTS[0]),
            ("DEFAULT_EDGE_CASES_PATH", SENSITIVE_JUDGE_INPUTS[1]),
        ):
            frozen_default = f'{variable}="${{MIGRATION_ROOT}}/{relative}"'
            if frozen_default not in lib_text:
                errors.append(f"scripts/lib.sh: missing {frozen_default!r}")
    return errors


def _load_final_generator(root: Path):
    path = root / "scripts/generate_final_manifest.py"
    spec = importlib.util.spec_from_file_location("layout_final_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity(
    path: Path, suite_id: str, count: int, errors: list[str], schema_version: int = MANIFEST_SCHEMA_VERSION
) -> list[dict[str, Any]]:
    payload = _json(path, errors) if path.is_file() else {}
    if not payload:
        return []
    if set(payload) != {"schema_version", "suite_id", "cases"}:
        errors.append(f"{path}: fields must be exactly schema_version, suite_id, cases")
    if payload.get("schema_version") != schema_version:
        errors.append(f"{path}: schema_version must be {schema_version}")
    if payload.get("suite_id") != suite_id:
        errors.append(f"{path}: suite_id must be {suite_id!r}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != count:
        errors.append(f"{path}: must contain exactly {count} cases")
        return []
    return [case for case in cases if isinstance(case, dict)]


def validate_frozen_instances(root: Path) -> list[str]:
    """Identity of the three frozen manifests and agreement with their generators.

    The final holdout must cover every upstream performance shape (family,
    s_kv) once with a 63-bit seed of at least 2**32; the private and
    public edge instances must carry their suite identities and counts, the
    public instance must be recorded in the public suite manifest, and both
    must be byte-for-byte what ``scripts/generate_edge_manifest.py`` draws.
    """

    errors: list[str] = []
    final_path = judge_input_path(root, SENSITIVE_JUDGE_INPUTS[0]) or root / SENSITIVE_JUDGE_INPUTS[0]
    final = _identity(final_path, FINAL_SUITE_ID, EXPECTED_FINAL_CASES, errors, FINAL_MANIFEST_SCHEMA_VERSION)
    if final and (root / "scripts/generate_final_manifest.py").is_file():
        generator = _load_final_generator(root)
        low, high = generator.SEED_RANGE
        seen: set[str] = set()
        for case in final:
            group, family = case.get("group"), case.get("family")
            if (
                group not in generator.GROUPS
                or family not in generator.FAMILIES
                or not set(generator.BASE_KEYS) <= set(case)
            ):
                errors.append(f"final holdout case is malformed: {case.get('case_id')!r}")
                continue
            seen.add(str(family))
            seed = case.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or not low <= seed < high:
                errors.append(f"final holdout {case['case_id']}: seed outside [2**32, 2**63)")
        if seen != set(generator.FAMILIES):
            errors.append("final holdout must cover every family")
    edge_path = judge_input_path(root, SENSITIVE_JUDGE_INPUTS[1]) or root / SENSITIVE_JUDGE_INPUTS[1]
    _identity(edge_path, EDGE_SUITE_ID, EXPECTED_EDGE_CASES, errors)
    public = _identity(root / PUBLIC_EDGE_CASES, PUBLIC_EDGE_SUITE_ID, EXPECTED_PUBLIC_EDGE_CASES, errors)
    if public and any(case.get("native") is not True for case in public):
        errors.append(f"{PUBLIC_EDGE_CASES}: every public edge case must run natively")
    public_manifest = root / "portable/public-suite-manifest.json"
    if public_manifest.is_file() and (root / PUBLIC_EDGE_CASES).is_file():
        edge_suite = _json(public_manifest, errors).get("edge_suite")
        digest = _sha256(root / PUBLIC_EDGE_CASES)
        if (
            not isinstance(edge_suite, dict)
            or edge_suite.get("path") != PUBLIC_EDGE_CASES
            or edge_suite.get("sha256") != digest
            or edge_suite.get("suite_id") != PUBLIC_EDGE_SUITE_ID
        ):
            errors.append(
                "portable/public-suite-manifest.json: edge_suite must record the "
                f"committed {PUBLIC_EDGE_CASES}"
            )
    edge_generator = root / "scripts/generate_edge_manifest.py"
    if edge_generator.is_file():
        # Regenerates both edge instances (the generator validates every case
        # with the trusted runtime) and compares them byte for byte.
        completed = subprocess.run(
            [sys.executable, "-B", str(edge_generator), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stdout + completed.stderr).strip().splitlines()[-3:]
            errors.append(
                "scripts/generate_edge_manifest.py --check: the frozen edge instances "
                f"differ from the seeded generator ({' | '.join(detail)})"
            )
    return errors


def validate_layout(root: Path) -> list[str]:
    root = root.expanduser().resolve()
    errors: list[str] = []
    if root.name != TASK_DIRECTORY:
        errors.append(f"task directory must be named {TASK_DIRECTORY!r}")
    _require_layout(root, errors)
    _check_task_metadata(root, errors)
    _check_experiment(root, errors)
    _check_images(root, errors)
    _check_starter(root, errors)
    _check_lifecycle(root, errors)
    errors.extend(validate_private_inputs(root))
    errors.extend(validate_frozen_instances(root))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args(argv)
    errors = validate_layout(args.root)
    if errors:
        print(f"layout validation failed ({len(errors)} error(s)):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"layout validation passed: {args.root.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
