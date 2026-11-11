#!/usr/bin/env python3
"""Lightweight static validation for the Attn-QAT NVFP4 training attention BF16 H100 v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


TASK_ID = "attn_qat_nvfp4_training_attention_h100_seed_v1"
BENCHMARK = "attn_qat_training_attention"
TASK_DIRECTORY = "attn-qat-nvfp4-training-attention-h100-v1-seed"
# Judge-only evaluation inputs: committed with the task, admitted by no Work
# build context.  The frozen qualification draw of the final holdout is the
# default build input (the holdout is not drawn per build);
# ``validate_private_inputs`` proves the Work build can never see it.
SENSITIVE_JUDGE_INPUTS = ("judge/private/final_cases.json",)
# A tree without the repository copy (the public tree) reads it as a dataset
# asset: assets/fetch.sh places it under <INFRATHON_ASSETS_ROOT>/<task directory>/.
JUDGE_INPUT_ASSETS = {"judge/private/final_cases.json": "judge/final_cases.json"}
# Everything the Work image build can read: its own context directory plus the
# named build contexts scripts/build_images.sh passes to `docker build`.
WORK_BUILD_CONTEXTS = ("images/work",)
WORKSPACE = "/app/workdir/attn_qat_lab"
SUBMIT_PATHS = ["candidates", "FINAL_VERSION", "notes.md"]
PRIVATE_ARTIFACT = "/logs/verifier/attn_qat_v1_results.json"
EXPECTED_FINAL_CASES = 287
FIXTURE_NAME = "e7_wan13b_grad_b29_t1_s8192.pt"
FIXTURE_MANIFEST_NAME = "e7_wan13b_grad_b29_t1_s8192.manifest.json"
FIXTURE_RECIPE_ID = "e7-wan13b-qad-stage1-grad-b29-t1-s8192-v1"
FIXTURE_IDENTITY = "per-deployment-capture"
RECIPE_DIR = "fixtures/recipe"
# Generated, git-ignored trees (run logs, build context, host artifacts) are evidence,
# not task source, so the recursive scans below skip them instead of reporting the
# bytecode and archives they legitimately contain.
GENERATED_DIRS = {"logs", "artifacts", "build", ".pytest_cache"}
# Markers of the scaffold's lineage that must not survive the port.  Split so the
# validator itself does not match them.
LEGACY_MARKERS = (
    "attention" + "_jvp",
    "jvp" + "_family",
    "sage" + "attention",
    "sla" + "_forward",
    "sla" + "_backward",
)
# `is_causal` used to be a marker too; causal attention is now part of the contract (the LLM
# domain), and the real-layer capture calls SDPA with it.
PROVENANCE_EXEMPT = ("docs/SOURCE_PROVENANCE.md", "docs/PORT_LEDGER.md")
# The name of the package the images must NOT contain has to appear in the absence
# assertions, in the test that checks them, and in this validator's own forbidden
# install list.  Those four files may carry that one marker and no other.
ABSENCE_ASSERT_FILES = {
    "images/work/Dockerfile",
    "images/judge/Dockerfile",
    "images/judge/source_admission.py",
    "images/work/benchmark/source_admission.py",
    "tests/test_contract.py",
    "tests/test_source_admission.py",
    "scripts/validate_layout.py",
    # The portable software-ABI manifest records the same forbidden-package list
    # as the image absence assertions.
    "portable/software-abi.json",
    # The task README states which attention packages the images do not contain.
    "README.md",
}
ABSENCE_ASSERT_MARKER = "sage" + "attention"


def _is_generated(root: Path, path: Path) -> bool:
    """True when `path` lies under a generated, git-ignored directory."""
    return bool(set(path.relative_to(root).parts[:-1]) & GENERATED_DIRS)


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


def _require_layout(root: Path, errors: list[str], fixture: Path | None = None,
                    fixture_manifest: Path | None = None) -> None:
    required_files = (
        ".gitignore",
        "README.md",
        "tasks/BENCHMARK.yaml",
        f"tasks/{TASK_ID}.json",
        "images/work/.dockerignore",
        "images/work/Dockerfile",
        "images/work/AGENT_TASK.md",
        "images/work/codex-config.toml",
        "images/work/benchmark/attn_qat_runtime.py",
        "images/work/benchmark/public_runner.py",
        "images/work/benchmark/run_public_tests.sh",
        "images/work/benchmark/source_admission.py",
        "images/work/benchmark/prepare_submit.sh",
        "images/work/workdir/attn_qat_lab/README_AGENT.md",
        "images/work/workdir/attn_qat_lab/FINAL_VERSION",
        "images/work/workdir/attn_qat_lab/notes.md",
        "images/work/workdir/attn_qat_lab/candidates/kernel_v001/submission.py",
        "images/work/workdir/attn_qat_lab/tools/run_one.sh",
        "images/work/workdir/attn_qat_lab/tools/run_public_tests.sh",
        "images/judge/.dockerignore",
        "images/judge/Dockerfile",
        "images/judge/Dockerfile.private",
        "images/judge/run_judge.py",
        "images/judge/verify_attn_qat.py",
        "images/judge/candidate_worker.py",
        "images/judge/source_admission.py",
        "images/judge/protected_hashes.txt",
        "judge/run_judge.py",
        "tests/test_private_inputs.py",
        "runtime/attn_qat_runtime.py",
        "oracle/FINAL_VERSION",
        "oracle/notes.md",
        "oracle/candidates/kernel_v001/submission.py",
        "scripts/lib.sh",
        "scripts/build_images.sh",
        "scripts/generate_final_manifest.py",
        "scripts/generate_protected_hashes.py",
        f"{RECIPE_DIR}/RECIPE.json",
        f"{RECIPE_DIR}/capture_e7_wan_grad.py",
        f"{RECIPE_DIR}/build_fixture.sh",
        "scripts/make_oracle_archive.sh",
        "scripts/eval_archive.sh",
        "scripts/smoke_test.sh",
        "scripts/run_fixed_budget.sh",
        "scripts/run_single_gpu_fixed_budget.sh",
        "scripts/single_gpu_runtime_preflight.py",
        "scripts/refresh_portable_manifests.py",
        "scripts/launch_fixed_budget_tmux.sh",
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
    tests = root / "tests"
    if not tests.is_dir() or tests.is_symlink():
        errors.append("missing regular non-symlink directory: tests")
    _check_fixture_recipe(root, errors, fixture=fixture,
                          fixture_manifest=fixture_manifest)
    for forbidden_tree in ("upstream", "vendor", "flash" + "-attention", "FlashMLA"):
        if (root / forbidden_tree).exists():
            errors.append(f"task must not contain a {forbidden_tree} source tree")
    for path in root.rglob("*"):
        if _is_generated(root, path):
            continue
        if path.name == "__pycache__" or (path.is_file() and path.suffix in {".pyc", ".pyo"}):
            errors.append(
                "generated Python bytecode/cache is forbidden in packaged task roots: "
                f"{path.relative_to(root)}"
            )


def _check_build_order(root: Path, errors: list[str]) -> None:
    """`build_images.sh` must validate the layout on both sides of the capture block.

    The first pass runs on a tree that may legitimately carry no capture; the second
    runs once one is in place, and it is the one that judges the pair about to be baked
    into both images.  The supplied-capture branch must be judged before it is staged,
    so the `--fixture` pass has to precede the `install` that copies it in.
    """
    text = _read(root / "scripts/build_images.sh", errors)
    if not text:
        return
    passes = [
        index for index in range(len(text))
        if text.startswith('"${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_layout.py"', index)
    ]
    if len(passes) != 3:
        errors.append(
            "build_images.sh must validate the layout three times: before the capture "
            "block, on a supplied capture before staging it, and after the capture is "
            "in place"
        )
        return
    capture = text.find('"${recipe_dir}/build_fixture.sh"')
    staged = text.find('install -m 0644 "${FIXTURE_PATH}"')
    supplied_pass = text.find(
        '--fixture "${FIXTURE_PATH}" --fixture-manifest "${FIXTURE_MANIFEST_PATH}"'
    )
    build = text.find("docker build")
    if not 0 <= passes[0] < supplied_pass < staged:
        errors.append(
            "build_images.sh: a supplied capture must be validated before it is staged"
        )
    if not 0 <= capture < passes[2] < build:
        errors.append(
            "build_images.sh: the layout must be validated again after the capture "
            "exists and before the images are built"
        )


def _check_fixture_recipe(root: Path, errors: list[str], fixture: Path | None = None,
                          fixture_manifest: Path | None = None) -> None:
    """The e7 capture is recipe-bound, so validate the recipe, not a byte digest.

    With no explicit pair, this judges `fixtures/`: a fresh checkout carries no capture
    at all -- `scripts/build_images.sh` takes one from `fixtures/recipe/` before it
    builds -- so an absent capture is not an error there, but a capture that *is*
    present must be accompanied by the manifest that binds it to the shipped
    `RECIPE.json`.

    `fixture` and `fixture_manifest` name a pair that has not been staged into the task
    tree: the reuse path of `build_images.sh` passes the `FIXTURE_PATH` /
    `FIXTURE_MANIFEST_PATH` it was handed, so a pair that does not match the recipe is
    rejected by this one rule before anything is copied anywhere.  An explicit pair is
    asserted to exist, so absence is an error.
    """
    recipe_path = root / RECIPE_DIR / "RECIPE.json"
    if not recipe_path.is_file() or recipe_path.is_symlink():
        return
    recipe = _json(recipe_path, errors)
    if recipe.get("recipe_id") != FIXTURE_RECIPE_ID:
        errors.append(f"{RECIPE_DIR}/RECIPE.json: recipe_id must be {FIXTURE_RECIPE_ID!r}")
    if recipe.get("fixture_name") != FIXTURE_NAME:
        errors.append(f"{RECIPE_DIR}/RECIPE.json: fixture_name must be {FIXTURE_NAME!r}")
    if recipe.get("identity") != FIXTURE_IDENTITY:
        errors.append(f"{RECIPE_DIR}/RECIPE.json: identity must be {FIXTURE_IDENTITY!r}")
    bands = recipe.get("sanity_bands")
    tensors = recipe.get("capture", {}).get("tensors")
    if not isinstance(tensors, list) or not tensors:
        errors.append(f"{RECIPE_DIR}/RECIPE.json: capture.tensors must be a non-empty list")
    elif not isinstance(bands, dict) or any(
        not isinstance(bands.get(name), dict) for name in tensors
    ):
        errors.append(
            f"{RECIPE_DIR}/RECIPE.json: every captured tensor needs a sanity band"
        )
    else:
        for name in tensors:
            missing = {
                "rms_min", "rms_max", "abs_max_min", "abs_max_max", "finite_fraction_min",
            } - set(bands[name])
            if missing:
                errors.append(
                    f"{RECIPE_DIR}/RECIPE.json: sanity band {name} is missing "
                    f"{sorted(missing)}"
                )

    # The recipe must reach the runtime, and the capture manifest must reach both
    # images next to the capture it describes.
    build_text = _read(root / "scripts/build_images.sh", errors)
    recipe_context = '--build-context "recipe=${MIGRATION_ROOT}/fixtures/recipe"'
    if build_text.count(recipe_context) != 1:
        errors.append("build_images.sh must pass the recipe context to the Judge build only")
    for relative, runtime_directory in (
        ("images/judge/Dockerfile", "/opt/attn_qat_worker"),
    ):
        text = _read(root / relative, errors)
        if f"COPY --from=recipe RECIPE.json {runtime_directory}/RECIPE.json" not in text:
            errors.append(f"{relative} must ship RECIPE.json beside the task runtime")
        if f"COPY --from=fixtures {FIXTURE_MANIFEST_NAME}" not in text:
            errors.append(f"{relative} must ship the capture manifest beside the capture")
        fixture_line = next(
            (line for line in text.splitlines()
             if line.startswith(f"COPY --from=fixtures {FIXTURE_NAME}")),
            "",
        )
        manifest_line = next(
            (line for line in text.splitlines()
             if line.startswith(f"COPY --from=fixtures {FIXTURE_MANIFEST_NAME}")),
            "",
        )
        if fixture_line and manifest_line:
            fixture_directory = fixture_line.rsplit(" ", 1)[-1].rsplit("/", 1)[0]
            manifest_directory = manifest_line.rsplit(" ", 1)[-1].rsplit("/", 1)[0]
            if fixture_directory != manifest_directory:
                errors.append(
                    f"{relative}: the capture manifest must land in the same directory "
                    "as the capture"
                )

    external = fixture is not None or fixture_manifest is not None
    if external and (fixture is None or fixture_manifest is None):
        errors.append("a capture and its manifest must be named together")
        return
    if not external:
        fixture = root / "fixtures" / FIXTURE_NAME
        fixture_manifest = root / "fixtures" / FIXTURE_MANIFEST_NAME
    manifest_path = fixture_manifest
    if fixture.is_symlink() or manifest_path.is_symlink():
        errors.append("the e7 capture and its manifest must be regular files")
        return
    if not fixture.is_file():
        if external:
            errors.append(f"the named e7 capture does not exist: {fixture}")
        elif manifest_path.is_file():
            # Half a pair is worse than none: the manifest describes bytes that are
            # not there, and nothing downstream would say so.
            errors.append(
                "the e7 capture manifest is present without the capture it describes"
            )
        # Otherwise: a fresh checkout carries no capture; build_images.sh takes one.
        return
    if not manifest_path.is_file():
        errors.append(
            "the e7 capture is present without its manifest; recapture it with "
            f"FIXTURE_GPU=<uuid> {RECIPE_DIR}/build_fixture.sh"
        )
        return
    manifest = _json(manifest_path, errors)
    if manifest.get("identity") != FIXTURE_IDENTITY:
        errors.append("e7 capture manifest declares a different identity")
    fixture_block = manifest.get("fixture")
    recipe_block = manifest.get("recipe")
    if not isinstance(fixture_block, dict) or not isinstance(recipe_block, dict):
        errors.append("e7 capture manifest is malformed")
        return
    if fixture_block.get("name") != FIXTURE_NAME:
        errors.append("e7 capture manifest names a different fixture")
    if fixture_block.get("sha256") != _sha256(fixture):
        errors.append("e7 capture does not match its manifest")
    if recipe_block.get("id") != FIXTURE_RECIPE_ID:
        errors.append("e7 capture manifest names a different recipe")
    if recipe_block.get("sha256") != _sha256(recipe_path):
        errors.append("e7 capture was not produced by the shipped RECIPE.json")


def _reject_legacy_markers(root: Path, errors: list[str]) -> None:
    """Fail if any scaffold-lineage identifier survived the port.

    The two provenance documents are exempt: they are where the lineage is
    deliberately named.
    """
    for path in sorted(root.rglob("*")):
        if _is_generated(root, path) or not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in PROVENANCE_EXEMPT or path.suffix in {".pt", ".pyc", ".pyo"}:
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except (OSError, UnicodeError):
            continue
        for marker in LEGACY_MARKERS:
            if marker == ABSENCE_ASSERT_MARKER and relative in ABSENCE_ASSERT_FILES:
                continue
            if marker.lower() in text:
                errors.append(f"legacy scaffold marker {marker!r} in {relative}")


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
        "copy --from=flash" + "-attention",
        "copy --from=cutlass",
        "/opt/flash" + "-attention",
        "/opt/flash_mla",
        "pip install flash-mla",
        "pip install flash_mla",
        "pip install flash-attn",
        "pip install sageattention",
    )
    for label, text in (("Work", work_lower), ("Judge", judge_lower)):
        if any(marker in text for marker in forbidden_source_installs):
            errors.append(f"{label} Dockerfile must not install or copy forbidden attention sources")
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
    if "copy --from=final_cases final_cases.json" not in private_lower:
        errors.append("Judge holdout layer must copy the private manifests from the final_cases context")
    if "copy --from=final_cases" in work_lower:
        errors.append("Work Dockerfile must not receive the private final_cases context")
    if "ln -s" in work_lower and "source" in work_lower:
        errors.append("Work Dockerfile must not create a source symlink")
    hook = "/app/workdir/attn_qat_lab/.sforge/prepare_submit.sh"
    if f"COPY benchmark/prepare_submit.sh {hook}" not in work_docker:
        errors.append("Work Dockerfile must install the cooperative pre-submit hook")
    if "chown -R root:root /app/workdir/attn_qat_lab/.sforge" not in work_docker:
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
            if _is_generated(root, path):
                continue
            if path.is_symlink():
                errors.append(f"symlink is forbidden in Work context: {path}")
                continue
            parts = {part.lower() for part in path.relative_to(work_root).parts}
            if parts.intersection(forbidden):
                errors.append(f"private/reference path is present in Work context: {path}")
            if path.is_file() and path.suffix.lower() in {".c", ".cc", ".cpp", ".cu", ".cuh", ".so", ".a", ".ptx", ".cubin"}:
                errors.append(f"compiled/upstream implementation material is present in Work: {path}")


def _check_starter(root: Path, errors: list[str]) -> None:
    starter = root / (
        "images/work/workdir/attn_qat_lab/"
        "candidates/kernel_v001/submission.py"
    )
    oracle = root / "oracle/candidates/kernel_v001/submission.py"
    if starter.is_file() and oracle.is_file() and _sha256(starter) != _sha256(oracle):
        errors.append("Work and Oracle seed sources must be byte-identical")
    final_version = root / "images/work/workdir/attn_qat_lab/FINAL_VERSION"
    if final_version.is_file() and final_version.read_text(encoding="utf-8").strip() != "1":
        errors.append("Work FINAL_VERSION must initially select version 1")
    # kernel_v001 stays the seed control; the oracle may select a stronger
    # later candidate (e.g. the strongest admissible upstream/paper solution).
    oracle_version = root / "oracle/FINAL_VERSION"
    if oracle_version.is_file():
        selected = oracle_version.read_text(encoding="utf-8").strip()
        if not (selected.isdigit() and 1 <= int(selected) <= 999) or not (
            root / f"oracle/candidates/kernel_v{int(selected):03d}/submission.py"
        ).is_file():
            errors.append("Oracle FINAL_VERSION must select an existing oracle candidate")


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
        'manual_evaluation_suite: "final-holdout-v3"',
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
        "SFORGE_PRE_SUBMIT_HOOK=/app/workdir/attn_qat_lab/.sforge/prepare_submit.sh",
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
    _check_build_order(root, errors)
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
    if "flash" + "-attention" in build_text.lower() or "flash_mla" in build_text.lower():
        errors.append("build_images.sh must not pass a FlashMLA source context")
    final_context = '--build-context "final_cases=${final_cases_context}"'
    if build_text.count(final_context) != 1:
        errors.append("build_images.sh must pass final_cases to the Judge build only")
    fixture_context = '--build-context "fixtures=${MIGRATION_ROOT}/fixtures"'
    if build_text.count(fixture_context) != 1:
        errors.append("build_images.sh must pass fixtures to the Judge build only")
    for fragment in (
        'FINAL_CASES_PATH',
        'final_cases_sha256',
        '"final-holdout-v3"',
        f'"${{final_cases_count}}" -eq {EXPECTED_FINAL_CASES}',
        'FIXTURE_PATH',
        'FIXTURE_MANIFEST_PATH',
        '"${recipe_dir}/build_fixture.sh"',
        'e7 capture was not produced by the shipped RECIPE.json',
        '--fixture "${FIXTURE_PATH}" --fixture-manifest "${FIXTURE_MANIFEST_PATH}"',
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


def validate_layout(root: Path, fixture: Path | None = None,
                    fixture_manifest: Path | None = None) -> list[str]:
    root = root.expanduser().resolve()
    errors: list[str] = []
    if root.name != TASK_DIRECTORY:
        errors.append(f"task directory must be named {TASK_DIRECTORY!r}")
    _require_layout(root, errors, fixture=fixture, fixture_manifest=fixture_manifest)
    _check_task_metadata(root, errors)
    _check_experiment(root, errors)
    _check_images(root, errors)
    _check_starter(root, errors)
    _check_lifecycle(root, errors)
    errors.extend(validate_private_inputs(root))
    _reject_legacy_markers(root, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help="judge this e7 capture instead of the one in fixtures/ (used by "
             "build_images.sh to reject a supplied capture before staging it)",
    )
    parser.add_argument(
        "--fixture-manifest",
        type=Path,
        help="the capture manifest that accompanies --fixture",
    )
    args = parser.parse_args(argv)
    errors = validate_layout(args.root, fixture=args.fixture,
                             fixture_manifest=args.fixture_manifest)
    if errors:
        print(f"layout validation failed ({len(errors)} error(s)):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"layout validation passed: {args.root.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
