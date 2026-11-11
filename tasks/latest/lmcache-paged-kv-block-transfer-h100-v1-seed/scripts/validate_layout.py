#!/usr/bin/env python3
"""Lightweight static validation for the LMCache paged-KV block transfer v1 task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


TASK_ID = "lmcache_paged_kv_block_transfer_h100_seed_v1"
BENCHMARK = "lmcache_kv_transfer"
TASK_DIRECTORY = "lmcache-paged-kv-block-transfer-h100-v1-seed"
# Judge-only evaluation inputs: committed with the task, admitted by no Work
# build context.  The frozen qualification draw of the final holdout is the
# default build input; ``validate_private_inputs`` proves the Work build can
# never see it.
SENSITIVE_JUDGE_INPUTS = ("judge/private/final_cases.json",)
# A tree without the repository copy (the public tree) reads it as a dataset
# asset: assets/fetch.sh places it under <INFRATHON_ASSETS_ROOT>/<task directory>/.
JUDGE_INPUT_ASSETS = {"judge/private/final_cases.json": "judge/final_cases.json"}
# Everything the Work image build can read: its own context directory plus the
# named build contexts scripts/build_images.sh passes to `docker build`.
WORK_BUILD_CONTEXTS = ("images/work", "runtime")
WORKSPACE = "/app/workdir/kv_transfer_lab"
SUBMIT_PATHS = ["candidates", "FINAL_VERSION", "notes.md"]
PRIVATE_ARTIFACT = "/logs/verifier/lmcache_kv_transfer_v2_results.json"


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
        "images/work/benchmark/kv_transfer_runtime.py",
        "images/work/benchmark/transfer_abi.h",
        "images/work/benchmark/run_public_tests.sh",
        "images/work/benchmark/source_admission.py",
        "images/work/benchmark/prepare_submit.sh",
        "images/work/workdir/kv_transfer_lab/README_AGENT.md",
        "images/work/workdir/kv_transfer_lab/FINAL_VERSION",
        "images/work/workdir/kv_transfer_lab/notes.md",
        "images/work/workdir/kv_transfer_lab/candidates/kernel_v001/submission.cu",
        "images/work/workdir/kv_transfer_lab/tools/run_one.sh",
        "images/work/workdir/kv_transfer_lab/tools/run_public_tests.sh",
        "images/judge/.dockerignore",
        "images/judge/Dockerfile",
        "images/judge/Dockerfile.private",
        "images/judge/run_judge.py",
        "images/judge/verify_kv_transfer.py",
        "images/judge/candidate_worker.py",
        "images/judge/source_admission.py",
        "images/judge/protected_hashes.txt",
        "judge/run_judge.py",
        "tests/test_private_inputs.py",
        "runtime/kv_transfer_runtime.py",
        "oracle/FINAL_VERSION",
        "oracle/notes.md",
        "oracle/candidates/kernel_v001/submission.cu",
        "scripts/lib.sh",
        "scripts/build_images.sh",
        "scripts/generate_final_manifest.py",
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
    for forbidden_tree in ("upstream", "vendor", "flashmla", "FlashMLA"):
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
        "copy --from=flashmla",
        "copy --from=cutlass",
        "/opt/flashmla",
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
        errors.append("Work Dockerfile must use its own runtime, not the Judge's")
    if "from ${work_image}" in judge_lower or "arg work_image" in judge_lower:
        errors.append("Judge image must be built independently, not FROM Work")
    # The public Judge is publishable; only its host-built holdout layer
    # (Dockerfile.private, FROM the public Judge) receives the final_cases context.
    if "final_cases" in judge_lower or "edge_cases.json" in judge_lower:
        errors.append("public Judge Dockerfile must not receive the private manifests")
    private_lower = _read(root / "images/judge/Dockerfile.private", errors).lower()
    if "from ${judge_public_image}" not in private_lower:
        errors.append("Judge holdout layer must be built FROM the public Judge image")
    if "copy --from=final_cases final_cases.json" not in private_lower:
        errors.append("Judge holdout layer must copy the private manifests from the final_cases context")
    if "copy --from=final_cases" in work_lower:
        errors.append("Work Dockerfile must not receive the private final_cases context")
    if "ln -s" in work_lower and "source" in work_lower:
        errors.append("Work Dockerfile must not create a source symlink")
    hook = "/app/workdir/kv_transfer_lab/.sforge/prepare_submit.sh"
    if f"COPY benchmark/prepare_submit.sh {hook}" not in work_docker:
        errors.append("Work Dockerfile must install the cooperative pre-submit hook")
    if "chown -R root:root /app/workdir/kv_transfer_lab/.sforge" not in work_docker:
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

    work_root = root / "images/work"
    if work_root.is_dir():
        forbidden = {"judge", "oracle", "hidden", "host_reference", "upstream"}
        for path in work_root.rglob("*"):
            if path.is_symlink():
                errors.append(f"symlink is forbidden in Work context: {path}")
                continue
            parts = {part.lower() for part in path.relative_to(work_root).parts}
            if parts.intersection(forbidden):
                errors.append(f"private/reference path is present in Work context: {path}")


def _check_starter(root: Path, errors: list[str]) -> None:
    starter = root / (
        "images/work/workdir/kv_transfer_lab/"
        "candidates/kernel_v001/submission.cu"
    )
    # This is a *seeded* task, not a scratch one: kernel_v001 is a supplied,
    # correct, deliberately unoptimised implementation, and an empty file here
    # would mean the seed failed to ship.
    if starter.is_file():
        text = starter.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            errors.append("Work kernel_v001/submission.cu must ship the seed, not an empty file")
        elif "lmcache_paged_kv_block_transfer" not in text:
            errors.append("Work kernel_v001/submission.py must define lmcache_paged_kv_block_transfer")
    final_version = root / "images/work/workdir/kv_transfer_lab/FINAL_VERSION"
    if final_version.is_file() and final_version.read_text(encoding="utf-8").strip() != "1":
        errors.append("Work FINAL_VERSION must initially select version 1")
    oracle_version = root / "oracle/FINAL_VERSION"
    if oracle_version.is_file() and oracle_version.read_text(encoding="utf-8").strip() != "1":
        errors.append("Oracle FINAL_VERSION must select version 1")


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
        'manual_evaluation_suite: "lmcache-kv-transfer-final-holdout-v2"',
        'scheduling_mode="dual_h100_shared_final_v2"',
        'scheduling_mode="single_gpu_lease_shared_final_v2_handoff"',
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
        "SFORGE_PRE_SUBMIT_HOOK=/app/workdir/kv_transfer_lab/.sforge/prepare_submit.sh",
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
    if "flashmla" in build_text.lower() or "flash_mla" in build_text.lower():
        errors.append("build_images.sh must not pass a FlashMLA source context")
    final_context = '--build-context "final_cases=${final_cases_context}"'
    if build_text.count(final_context) != 1:
        errors.append("build_images.sh must pass final_cases to the Judge build only")
    for fragment in (
        'FINAL_CASES_PATH',
        'final_cases_sha256',
        '"lmcache-kv-transfer-final-holdout-v2"',
    ):
        if fragment not in build_text:
            errors.append(f"build_images.sh: missing {fragment!r}")


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
    """The frozen holdout instance ships with the task and only the Judge can
    ever see it.

    Checks: it is present on the Judge-only path (or, in a tree without the
    repository copy, as a fetched dataset asset; with neither, the content
    checks are skipped); the build defaults to it and
    stages it 0600 into the Judge-only build context; the suite manifest
    records its digest; the generator draws from a seed rather than a system
    random source; and no Work build context admits the instance - not by path
    (none lives under a Work context) and not by content (no file a Work
    context admits has the same SHA-256).
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
        for token in ("judge/private", "final_cases"):
            if token in work:
                errors.append(f"Work Dockerfile must not reference {token!r}")
    manifest_path = root / "portable/final-suite-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen = manifest.get("frozen_instance")
        if not isinstance(frozen, dict):
            errors.append("portable/final-suite-manifest.json: missing frozen_instance")
        else:
            if frozen.get("path") != SENSITIVE_JUDGE_INPUTS[0]:
                errors.append(
                    "portable/final-suite-manifest.json: frozen_instance.path must be "
                    f"{SENSITIVE_JUDGE_INPUTS[0]!r}"
                )
            instance = judge_input_path(root, SENSITIVE_JUDGE_INPUTS[0])
            if instance is not None and instance.is_file() and frozen.get("sha256") != hashlib.sha256(
                instance.read_bytes()
            ).hexdigest():
                errors.append(
                    "portable/final-suite-manifest.json: frozen_instance.sha256 does "
                    "not match the committed instance"
                )
            if not str(frozen.get("provenance", "")).strip():
                errors.append(
                    "portable/final-suite-manifest.json: frozen_instance.provenance "
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
    build_path = root / "scripts/build_images.sh"
    if build_path.is_file():
        build = build_path.read_text(encoding="utf-8")
        for fragment in (
            'final_cases_source="${FINAL_CASES_PATH:-${DEFAULT_FINAL_CASES_PATH}}"',
            'install -m 0600 "${final_cases_source}" '
            '"${final_cases_context}/final_cases.json"',
            "\"$(stat -c '%a' \"${final_cases_context}/final_cases.json\")\" == \"600\"",
            "\"$(stat -c '%u' \"${final_cases_context}/final_cases.json\")\" "
            '== "$(id -u)"',
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
        frozen_default = (
            'DEFAULT_FINAL_CASES_PATH="${MIGRATION_ROOT}/'
            + SENSITIVE_JUDGE_INPUTS[0]
            + '"'
        )
        if frozen_default not in lib_path.read_text(encoding="utf-8"):
            errors.append(f"scripts/lib.sh: missing {frozen_default!r}")
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
