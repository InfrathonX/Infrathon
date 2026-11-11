#!/usr/bin/env python3
"""Lightweight static validation for quantized attention BF16 H100 v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


TASK_ID = "sageattention_quantized_attention_bf16_h100_seed_v2"
BENCHMARK = "sageattention_quantized_attention"
TASK_DIRECTORY = "sageattention-quantized-attention-bf16-h100-v2-seed"
# Judge-only evaluation inputs: admitted by no Work build context.  The frozen
# qualification draw of the final holdout is the default build input (the
# holdout is not drawn per build); ``validate_private_inputs`` proves the Work
# build can never see it.
SENSITIVE_JUDGE_INPUTS = ("judge/private/final_cases.json",)
# A tree without the repository copy (the public tree) reads it as a dataset
# asset: assets/fetch.sh places it under <INFRATHON_ASSETS_ROOT>/<task directory>/.
JUDGE_INPUT_ASSETS = {"judge/private/final_cases.json": "judge/final_cases.json"}
# Everything the Work image build can read: its own context directory plus the
# named build contexts scripts/build_images.sh passes to `docker build`.
WORK_BUILD_CONTEXTS = ("images/work",)
WORKSPACE = "/app/workdir/quant_attention_lab"
SUBMIT_PATHS = ["candidates", "FINAL_VERSION", "notes.md"]
PRIVATE_ARTIFACT = "/logs/verifier/sageattention_quant_attention_v2_results.json"
EXPECTED_FINAL_CASES = 774
# The e8 edge input is recipe-bound, not byte-bound: every deployment captures
# its own copy from fixtures/recipe/RECIPE.json and ships the capture manifest
# next to it.  There is deliberately no expected fixture SHA-256 here.
FIXTURE_NAME = "e8_wan13b_p0_b15_s00_cond_8192.pt"
FIXTURE_MANIFEST_NAME = "e8_wan13b_p0_b15_s00_cond_8192.manifest.json"
RECIPE_RELATIVE = "fixtures/recipe/RECIPE.json"
RECIPE_ID = "e8-wan21-t2v-1.3b-p0-b15-s00-cond-8192-v1"
# The reference e8 capture is a public asset (assets/manifest.json).
FIXTURE_ASSETS = {FIXTURE_NAME: f"public/{FIXTURE_NAME}", FIXTURE_MANIFEST_NAME: f"public/{FIXTURE_MANIFEST_NAME}"}


def _task_assets(root: Path) -> Path:
    assets_root = os.environ.get("INFRATHON_ASSETS_ROOT") or "~/.cache/infrathon/assets"
    return Path(assets_root).expanduser() / root.name


def judge_input_path(root: Path, relative: str) -> Path | None:
    """The repository copy of a Judge-only input, else its fetched dataset asset, else None."""
    path = root / relative
    if path.exists() or path.is_symlink():
        return path
    fetched = _task_assets(root) / JUDGE_INPUT_ASSETS[relative]
    return fetched if fetched.is_file() else None


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_layout(
    root: Path,
    errors: list[str],
    fixture: Path | None = None,
    manifest: Path | None = None,
) -> None:
    required_files = (
        ".gitignore",
        "assets/manifest.json",
        "assets/fetch.sh",
        "tasks/BENCHMARK.yaml",
        f"tasks/{TASK_ID}.json",
        "images/work/.dockerignore",
        "images/work/Dockerfile",
        "images/work/Dockerfile.local",
        "images/work/AGENT_TASK.md",
        "images/work/codex-config.toml",
        "images/work/benchmark/quant_attention_runtime.py",
        "images/work/benchmark/public_runner.py",
        "images/work/benchmark/run_public_tests.sh",
        "images/work/benchmark/source_admission.py",
        "images/work/benchmark/prepare_submit.sh",
        "images/work/workdir/quant_attention_lab/README_AGENT.md",
        "images/work/workdir/quant_attention_lab/FINAL_VERSION",
        "images/work/workdir/quant_attention_lab/notes.md",
        "images/work/workdir/quant_attention_lab/candidates/kernel_v001/submission.py",
        "images/work/workdir/quant_attention_lab/tools/run_one.sh",
        "images/work/workdir/quant_attention_lab/tools/run_public_tests.sh",
        "images/judge/.dockerignore",
        "images/judge/Dockerfile",
        "images/judge/Dockerfile.private",
        "images/judge/run_judge.py",
        "images/judge/verify_quant_attention.py",
        "images/judge/candidate_worker.py",
        "images/judge/source_admission.py",
        "images/judge/protected_hashes.txt",
        "judge/run_judge.py",
        # The test that proves the Judge-only evaluation input stays there.
        "tests/test_private_inputs.py",
        "runtime/quant_attention_runtime.py",
        "oracle/FINAL_VERSION",
        "oracle/ADMISSIBLE_VERSION",
        "oracle/notes.md",
        "oracle/candidates/kernel_v001/submission.py",
        "scripts/lib.sh",
        "scripts/build_images.sh",
        "scripts/generate_final_manifest.py",
        "fixtures/recipe/RECIPE.json",
        "fixtures/recipe/capture_e8_wan_qkv.py",
        "fixtures/recipe/build_fixture.sh",
        "fixtures/recipe/real_layers/RECIPE_cogvideox2b.json",
        "fixtures/recipe/real_layers/RECIPE_llama31_8b.json",
        "fixtures/recipe/real_layers/build_real_layers.sh",
        "fixtures/recipe/real_layers/capture_real_layers.py",
        "fixtures/recipe/real_layers/Dockerfile.builder",
        "fixtures/recipe/public_layers/capture_public_layers.py",
        "fixtures/recipe/public_layers/build_public_layers.sh",
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
    _check_fixture_recipe(root, errors, fixture, manifest)
    for forbidden_tree in ("upstream", "vendor", "flash" + "-attention", "FlashMLA"):
        if (root / forbidden_tree).exists():
            errors.append(f"task must not contain a {forbidden_tree} source tree")
    for path in root.rglob("*"):
        if path.name == "__pycache__" or (path.is_file() and path.suffix in {".pyc", ".pyo"}):
            errors.append(
                "generated Python bytecode/cache is forbidden in packaged task roots: "
                f"{path.relative_to(root)}"
            )


def _check_fixture_recipe(
    root: Path,
    errors: list[str],
    fixture: Path | None = None,
    manifest: Path | None = None,
) -> None:
    """The e8 edge input is bound by its recipe, never by a pinned SHA-256.

    A checkout carries ``fixtures/recipe/`` but no capture.  The reference
    capture is a public asset (``assets/fetch.sh`` places it under the asset
    root); a capture made on this host by ``fixtures/recipe/build_fixture.sh``
    lands in ``fixtures/`` and takes precedence.  With neither present there is
    nothing to bind here (``scripts/build_images.sh`` refuses to build).

    ``fixture``/``manifest`` name a capture pair held elsewhere, the way
    ``FIXTURE_PATH``/``FIXTURE_MANIFEST_PATH`` let one deployment reuse
    another's capture.  ``scripts/build_images.sh`` passes whichever pair it is
    about to bake, so the layout gate always checks the bytes actually used.
    """

    recipe_path = root / RECIPE_RELATIVE
    recipe: dict[str, Any] = {}
    if recipe_path.is_file() and not recipe_path.is_symlink():
        recipe = _json(recipe_path, errors)
        if recipe.get("recipe_id") != RECIPE_ID:
            errors.append(f"{RECIPE_RELATIVE}: recipe_id must be {RECIPE_ID!r}")
        if recipe.get("identity") != "per-deployment-capture":
            errors.append(f"{RECIPE_RELATIVE}: identity must be per-deployment-capture")
        fixture_block = recipe.get("fixture")
        if not isinstance(fixture_block, dict) or (
            fixture_block.get("name") != FIXTURE_NAME
            or fixture_block.get("manifest") != FIXTURE_MANIFEST_NAME
        ):
            errors.append(f"{RECIPE_RELATIVE}: fixture block does not name the e8 capture")
        bands = recipe.get("sanity_bands")
        if not isinstance(bands, dict) or set(bands.get("tensors", {})) != {"q", "k", "v"}:
            errors.append(f"{RECIPE_RELATIVE}: sanity bands must cover q, k and v")

    explicit = fixture is not None
    if not explicit:
        fixture = root / "fixtures" / FIXTURE_NAME
        manifest = root / "fixtures" / FIXTURE_MANIFEST_NAME
        if not (fixture.exists() or fixture.is_symlink() or manifest.exists() or manifest.is_symlink()):
            fixture = _task_assets(root) / FIXTURE_ASSETS[FIXTURE_NAME]
            manifest = _task_assets(root) / FIXTURE_ASSETS[FIXTURE_MANIFEST_NAME]
            if not (fixture.exists() or manifest.exists()):
                return  # no capture on this host yet: nothing to bind
    assert fixture is not None and manifest is not None
    for path, label in ((fixture, "e8 capture"), (manifest, "e8 capture manifest")):
        if path.is_symlink():
            errors.append(f"{label} must be a regular file, not a symlink")
            return
    if not fixture.is_file() or not manifest.is_file():
        errors.append(
            "e8 capture is incomplete; run assets/fetch.sh for the reference capture, "
            "capture this deployment's copy with FIXTURE_GPU=<uuid> "
            "fixtures/recipe/build_fixture.sh, or point FIXTURE_PATH/FIXTURE_MANIFEST_PATH "
            "at another deployment's capture"
        )
        return
    declared = _json(manifest, errors)
    if declared.get("fixture") != FIXTURE_NAME:
        errors.append("e8 capture manifest names a different fixture")
    if declared.get("recipe_id") != RECIPE_ID:
        errors.append("e8 capture manifest was produced by a different recipe id")
    if _sha256(fixture) != declared.get("sha256"):
        errors.append("e8 capture does not match its manifest SHA-256")
    if recipe and declared.get("recipe_sha256") != _sha256(recipe_path):
        errors.append(f"e8 capture manifest does not match {RECIPE_RELATIVE}")


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
        errors.append(f"{path}: defaults.model must name a model")
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
    if "task_runtime" in work_lower or "--from=fixtures" in work_lower:
        errors.append("Work Dockerfile must use its own public runtime and data, not the Judge's")
    # The recipe travels in the runtime build context (next to the trusted
    # runtime module) and the capture manifest travels next to the capture, so
    # the runtime can bind file -> manifest -> recipe inside the Judge image.
    for label, docker_text in (("Judge", judge_docker),):
        runtime_copy = next(
            (
                line
                for line in docker_text.splitlines()
                if line.startswith("COPY --from=task_runtime")
                and "quant_attention_runtime.py" in line
            ),
            "",
        )
        runtime_directory = (
            runtime_copy.rsplit(" ", 1)[-1].rsplit("/", 1)[0] if runtime_copy else ""
        )
        if f"COPY --from=fixtures recipe/RECIPE.json {runtime_directory}/RECIPE.json" not in docker_text:
            errors.append(
                f"{label} Dockerfile must copy fixtures/recipe/RECIPE.json next to "
                "the trusted runtime"
            )
        fixture_copy = next(
            (
                line
                for line in docker_text.splitlines()
                if line.startswith("COPY --from=fixtures") and line.endswith(FIXTURE_NAME)
            ),
            "",
        )
        fixture_directory = (
            fixture_copy.rsplit(" ", 1)[-1].rsplit("/", 1)[0] if fixture_copy else ""
        )
        if not fixture_directory:
            errors.append(f"{label} Dockerfile must copy the e8 capture")
        elif (
            f"COPY --from=fixtures {FIXTURE_MANIFEST_NAME} "
            f"{fixture_directory}/{FIXTURE_MANIFEST_NAME}" not in docker_text
        ):
            errors.append(
                f"{label} Dockerfile must copy the e8 capture manifest next to the capture"
            )
    if "from ${work_image}" in judge_lower or "arg work_image" in judge_lower:
        errors.append("Judge image must be built independently, not FROM Work")
    # The publishable images carry neither the final holdout nor a real-layer
    # capture; the host layers add them (Dockerfile.local, Dockerfile.private).
    work_local = _read(root / "images/work/Dockerfile.local", errors)
    judge_private = _read(root / "images/judge/Dockerfile.private", errors)
    for label, text in (("Work", work_lower), ("public Judge", judge_lower)):
        if "copy --from=final_cases" in text:
            errors.append(f"{label} Dockerfile must not receive the private final_cases context")
        if "--from=real_layers" in text:
            errors.append(f"{label} Dockerfile is published and must not copy a real-layer capture")
    if "copy --from=final_cases final_cases.json" not in judge_private.lower():
        errors.append("Judge Dockerfile.private must copy the external final_cases context")
    if "FROM ${JUDGE_PUBLIC_IMAGE}" not in judge_private:
        errors.append("Judge Dockerfile.private must build on the public Judge")
    # v3: the scored real layers are mounted read-only at run time
    # (scripts/sforge_judge_entry.py), never baked into an image.
    if "--from=real_layers" in judge_private:
        errors.append("Judge Dockerfile.private must not copy a real-layer capture; they are mounted at run time")
    judge_entry = _read(root / "scripts/sforge_judge_entry.py", errors)
    if '"/opt/sageattention_quant_attention_judge/fixtures/real_layers"' not in judge_entry:
        errors.append("scripts/sforge_judge_entry.py must mount the real layers where the runtime reads them")
    if "FROM ${WORK_PUBLIC_IMAGE}" not in work_local:
        errors.append("Work Dockerfile.local must build on the published Work image")
    if "COPY --from=real_layers_public --chown=0:0 . /workspace/benchmark/fixtures/\n" not in work_local:
        errors.append("Work Dockerfile.local must copy the two public real layers")
    for token in ("final_cases", "judge/private", "--from=real_layers "):
        if token in work_local:
            errors.append(f"Work Dockerfile.local must not reference {token!r}")
    if "ln -s" in work_lower and "source" in work_lower:
        errors.append("Work Dockerfile must not create a source symlink")
    hook = "/app/workdir/quant_attention_lab/.sforge/prepare_submit.sh"
    if f"COPY benchmark/prepare_submit.sh {hook}" not in work_docker:
        errors.append("Work Dockerfile must install the cooperative pre-submit hook")
    if "chown -R root:root /app/workdir/quant_attention_lab/.sforge" not in work_docker:
        errors.append("Work Dockerfile must leave the pre-submit hook root-owned")
    if "portable" in work_lower or "final-suite-manifest" in work_lower:
        errors.append("Work Dockerfile must not reference the host-private portable manifests")
    hook = "/app/workdir/quant_attention_lab/.sforge/prepare_submit.sh"
    if f"COPY benchmark/prepare_submit.sh {hook}" not in work_docker:
        errors.append("Work Dockerfile must install the cooperative pre-submit hook")
    if "chown -R root:root /app/workdir/quant_attention_lab/.sforge" not in work_docker:
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
            if path.is_file() and path.suffix.lower() in {".c", ".cc", ".cpp", ".cu", ".cuh", ".so", ".a", ".ptx", ".cubin"}:
                errors.append(f"compiled/upstream implementation material is present in Work: {path}")


def _check_starter(root: Path, errors: list[str]) -> None:
    starter = root / (
        "images/work/workdir/quant_attention_lab/"
        "candidates/kernel_v001/submission.py"
    )
    oracle = root / "oracle/candidates/kernel_v001/submission.py"
    if starter.is_file() and oracle.is_file() and _sha256(starter) != _sha256(oracle):
        errors.append("Work and Oracle seed sources must be byte-identical")
    final_version = root / "images/work/workdir/quant_attention_lab/FINAL_VERSION"
    if final_version.is_file() and final_version.read_text(encoding="utf-8").strip() != "1":
        errors.append("Work FINAL_VERSION must initially select version 1")
    # kernel_v001 stays the seed control; the oracle may select a stronger
    # later candidate (e.g. the strongest admissible upstream/paper solution).
    # FINAL_VERSION may select a forced (non-admissible) paper candidate;
    # ADMISSIBLE_VERSION selects the best admissible one for the normal Judge.
    for selector in ("FINAL_VERSION", "ADMISSIBLE_VERSION"):
        oracle_version = root / "oracle" / selector
        if oracle_version.is_file():
            selected = oracle_version.read_text(encoding="utf-8").strip()
            if not (selected.isdigit() and 1 <= int(selected) <= 999) or not (
                root / f"oracle/candidates/kernel_v{int(selected):03d}/submission.py"
            ).is_file():
                errors.append(f"Oracle {selector} must select an existing oracle candidate")


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
        # full-final-shared-v3: every scored submission runs the frozen final suite.
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
        "SFORGE_PRE_SUBMIT_HOOK=/app/workdir/quant_attention_lab/.sforge/prepare_submit.sh",
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
    if "flash" + "-attention" in build_text.lower() or "flash_mla" in build_text.lower():
        errors.append("build_images.sh must not pass a FlashMLA source context")
    final_context = '--build-context "final_cases=${final_cases_context}"'
    if build_text.count(final_context) != 1:
        errors.append("build_images.sh must pass final_cases to the Judge build only")
    fixture_context = '--build-context "fixtures=${fixtures_context}"'
    if build_text.count(fixture_context) != 1:
        errors.append("build_images.sh must pass fixtures to the Judge build only")
    for fragment in (
        'FINAL_CASES_PATH',
        'final_cases_sha256',
        '"final-holdout-v3"',
        f'"${{final_cases_count}}" -eq {EXPECTED_FINAL_CASES}',
        # The e8 capture is drawn per deployment, exactly like the holdout.
        'FIXTURE_PATH',
        'FIXTURE_MANIFEST_PATH',
        'FIXTURE_GPU',
        'fixtures/recipe/build_fixture.sh',
        'recipe_sha256',
        'fixture_sha256',
    ):
        if fragment not in build_text:
            errors.append(f"build_images.sh: missing {fragment!r}")

    capture_text = _read(root / "fixtures/recipe/build_fixture.sh", errors)
    for fragment in ("FIXTURE_GPU", "capture_e8_wan_qkv.py", "hf", "recipe_sha256"):
        if fragment not in capture_text:
            errors.append(f"fixtures/recipe/build_fixture.sh: missing {fragment!r}")
    script_text = _read(root / "fixtures/recipe/capture_e8_wan_qkv.py", errors)
    for forbidden in ("iteration_docs", "edgebench_format_task", "/home/", "/opt/wan2.1\"", "sys.path.append(\"/"):
        if forbidden in script_text:
            errors.append(
                f"fixtures/recipe/capture_e8_wan_qkv.py must not reference {forbidden!r}"
            )


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

    Checks: it is present on the Judge-only path; the build defaults to it and
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


def validate_layout(
    root: Path,
    fixture: Path | None = None,
    manifest: Path | None = None,
) -> list[str]:
    root = root.expanduser().resolve()
    errors: list[str] = []
    if root.name != TASK_DIRECTORY:
        errors.append(f"task directory must be named {TASK_DIRECTORY!r}")
    _require_layout(root, errors, fixture, manifest)
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
    # The e8 capture is a per-deployment input, so it may legitimately live
    # outside the task tree (FIXTURE_PATH / FIXTURE_MANIFEST_PATH).
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--fixture-manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    if (args.fixture is None) != (args.fixture_manifest is None):
        parser.error("--fixture and --fixture-manifest must be given together")
    errors = validate_layout(args.root, args.fixture, args.fixture_manifest)
    if errors:
        print(f"layout validation failed ({len(errors)} error(s)):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"layout validation passed: {args.root.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
