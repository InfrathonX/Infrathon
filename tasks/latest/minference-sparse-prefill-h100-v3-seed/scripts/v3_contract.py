"""Shared, CPU-only resource identities for the real-input v3 task bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

TASK_ID = "minference_sparse_prefill_h100_seed_v3"
PROTOCOL = "minference-kernel-only-prefill-v3"
TASKS = ("niah_single_2", "niah_multikey_1", "vt", "fwe")
BUDGETS = (32768, 65536, 131072)
LAYERS = (1, 4, 15, 27)
PUBLIC_TASKS = ("niah_multikey_2", "cwe", "fwe")
PUBLIC_BUDGETS = (8192, 16384, 24576)
PUBLIC_LAYERS = (8, 20)
ON_DEMAND_PROTOCOL = "minference-single-real-prompt-qkv-v1"
QUALITY_PROTOCOL = "minference-ruler-complete-reference-v1"
ASSET_FILE_COUNTS = {"private-performance": 2, "model": 10}
ON_DEMAND_RUNTIME_FILES = tuple("runtime/" + name + ".py" for name in (
    "single_prompt_archive_v3", "prepare_single_prompt_v3", "measure_single_prompt_v3",
    "run_single_prompt_worker_v3", "stream_parent_v3", "on_demand_suite_v3"))
FULL_QUALITY_FILES = ()
SHA = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")
STALE_SCRIPTS = ("calibrate_gate.py", "generate_final_manifest.py", "stage_activations.py", "verify_local_revision.py")
SCIENCE_FILES = (
    "runtime/minference_prefill_runtime.py", "runtime/activation_archive_v3.py",
    "images/work/AGENT_TASK.md", "images/work/workdir/minference_prefill_lab/README_AGENT.md",
    "images/work/benchmark/public_runner.py", "images/work/benchmark/public_kernel.py",
    "images/work/benchmark/source_admission.py",
    "images/work/benchmark/submission_loader.py", "images/judge/source_admission.py",
    "images/judge/submission_loader.py", "images/judge/verify_minference_prefill.py",
    "images/judge/candidate_worker.py", "images/judge/run_judge.py", "images/judge/protected_hashes.txt",
    "judge/run_judge.py", "oracle/candidates/kernel_v001/submission.py", "oracle/FINAL_VERSION",
    "images/work/workdir/minference_prefill_lab/candidates/kernel_v001/submission.py",
    "activations/recipe/RULER_V3_RECIPE.json", "activations/recipe/ruler_v3.py",
    "activations/recipe/RECIPE.json",
    "activations/recipe/ruler_reference.py", "activations/recipe/build_ruler_v3.py",
    "activations/recipe/build_ruler_prompts.py", "activations/recipe/quality_v3.py",
    "activations/recipe/candidate_attention.py", "activations/recipe/capture_activations.py",

    "scripts/external_assets_on_demand_v3.py",
    "scripts/controller_asset_bindings_v3.py",
    "qualification/external-assets-contract.json",

    "scripts/check_seed_partial_blocks.py", "llm_judge/POLICY.md",
    "qualification/numerical_policy.json",
    "references/flash_attention_3/BUILD.json",
)
# These literal measured proofs bind the clean software epoch and its inherited
# file equivalence. They contain metadata only, never model/text/QKV payloads.
SOFTWARE_PARENT_PROOF_FILES = tuple("references/software-parent/" + name for name in (
    "masked-build-receipt.json", "derivative-equivalence-certificate.json",
    "original-file-equivalence-receipt.json"))
# Public submission transport and source-only diagnostics are delivery inputs;
# the controller still installs the current generated commands at launch.
PUBLIC_WORK_DELIVERY_FILES = (
    "images/work/workdir/minference_prefill_lab/tools/check_source.sh",
    "images/work/commands/sforge-submit", "images/work/commands/sforge-save-version",
    "images/work/commands/source-manifest.json",
)
SCIENCE_FILES += (("runtime/kernel_fidelity_v3.py",) + ON_DEMAND_RUNTIME_FILES + FULL_QUALITY_FILES
                  + SOFTWARE_PARENT_PROOF_FILES)
BUILD_HELPERS = (
    "tasks/BENCHMARK.yaml", "scripts/build_images.sh", "scripts/generate_protected_hashes.py",
    "scripts/built_image_policy.py", "scripts/validate_built_image_policy.py", "scripts/stage_thin_images_v3.py", "scripts/lib.sh",
    # Historical persistent-QKV retention/build helpers are not active delivery inputs.
    "scripts/validate_layout.py", "scripts/validate_formal_artifacts.py", "scripts/v3_contract.py",
    "scripts/refresh_portable_manifests.py", "scripts/make_oracle_archive.sh", "portable/REBUILD_PROMPT.md",
    "scripts/generate_public_prompts.py",
    "scripts/evaluate_local_v3.py", "scripts/evaluate_submit_cli_local_v3.py", "scripts/eval_archive.sh", "scripts/watch_gpu_lease.py",

    "scripts/build_integrity_battery_v3.py", "scripts/calibrate_kernel_fidelity_v3.py", "scripts/capture_integrity_inputs_v3.py",
    "scripts/review_integrity_battery_v3.py", "scripts/run_on_demand_integrity_battery_v3.py",
)
BUILD_HELPERS += PUBLIC_WORK_DELIVERY_FILES + ("references/software-parent/README.md",)


def read_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key}")
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"resource is not a regular non-symlink file: {path}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def normalized_sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("expected lowercase SHA-256")
    return value.removeprefix("sha256:")


def safe_path(root, relative):
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or ".." in Path(relative).parts or "." in Path(relative).parts):
        raise ValueError("resource path must be task-relative without traversal")
    path = Path(root) / relative
    current = Path(root)
    for component in Path(relative).parts:
        current /= component
        if current.is_symlink():
            raise ValueError("resource paths cannot traverse symlinks")
    return path


def task_core_paths(root):
    root = Path(root)
    paths = set(SCIENCE_FILES)
    paths.update(str(p.relative_to(root)) for p in (root / "portable").glob("*.json")
                 if p.name not in ("resource-manifest.json", "build-resource-manifest.json"))
    return sorted(paths)


def build_resource_paths(root):
    root = Path(root)
    paths = set(task_core_paths(root)) | set(BUILD_HELPERS)
    # This is the reviewed Docker ignore allowlist, not an unrestricted tree.
    work = root / "images/work"
    paths.update("images/work/" + name for name in ("Dockerfile", ".dockerignore", "AGENT_TASK.md", "codex-config.toml"))
    for folder in (work / "benchmark", work / "workdir"):
        paths.update(str(p.relative_to(root)) for p in folder.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    judge = root / "images/judge"
    paths.update("images/judge/" + name for name in ("Dockerfile", ".dockerignore", "run_judge.py",
        "verify_minference_prefill.py", "candidate_worker.py", "source_admission.py", "submission_loader.py", "protected_hashes.txt"))
    paths.update(str(p.relative_to(root)) for p in (root / "runtime").glob("*.py"))
    paths.update(str(p.relative_to(root)) for p in (root / "activations/recipe").rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    return sorted(paths)


def resource_document(root, *, delivery=False):
    paths = build_resource_paths(root) if delivery else task_core_paths(root)
    return {"schema_version": 1, "contract_protocol": PROTOCOL,
        "boundary": {"kind": "portable_image_delivery_inputs" if delivery else "scientific_task_resources",
                     "excludes": ["digest-binding task JSON", "this manifest", "deployment qualification evidence",
                                  "external model/text/answer bytes bound through scientific-artifacts.json; transient on-demand QKV is never an asset"],
                     "derivation": "reviewed v3 source path set in scripts/v3_contract.py; image context allowlists included for delivery"},
        "resources": [{"path": p, "sha256": "sha256:" + digest(safe_path(root, p))} for p in paths]}


def serialize(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def grid(partition):
    return [{"case_id": f"{partition}-{task}-n{budget}-L{layer}", "task": task,
             "context_budget": budget, "layer": layer, "trials": 3,
             "actual_n": "per-trial real input length, 2048 <= N < context_budget",
             "warmup_layer": 2, "weight": {"numerator": 1, "denominator": 48}}
            for task in TASKS for budget in BUDGETS for layer in LAYERS]


def public_grid():
    return [{"case_id": f"public-{task}-{budget}-r0-L{layer}", "task": task,
             "context_budget": budget, "layer": layer, "trials": 3,
             "actual_n": "real input length of the prompt", "warmup_layer": 5,
             "weight": {"numerator": 1, "denominator": 18}}
            for task in PUBLIC_TASKS for budget in PUBLIC_BUDGETS for layer in PUBLIC_LAYERS]
