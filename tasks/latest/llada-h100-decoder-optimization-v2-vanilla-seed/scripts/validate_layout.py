#!/usr/bin/env python3
"""Dependency-free validation of the stratified-520 dual-GPU task layout."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import ast
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


TASK_ID = "llada_h100_decoder_optimization_v2"
TASK_JSON = f"tasks/{TASK_ID}.json"
PROFILE = "fast_dllm_v1_llada_12row_stratified520_batch1"
GENERATION_COUNT = 520
REQUESTED_SLOTS = 216_320
# Capability envelope; mirrored in scripts/lib.sh, scripts/h100_policy_contract.py
# and images/judge/verifier/verify_decoder.py.
MIN_COMPUTE_CAPABILITY = "8.0"
MIN_DEVICE_MEMORY_MIB = 34816
EXPECTED_SUBMIT = {"solutions", "FINAL_VERSION", "notes.md"}
EXPECTED_ROWS = {
    "table1_gsm8k_5shot_g256": 39,
    "table1_gsm8k_5shot_g512": 39,
    "table1_math_4shot_g256": 65,
    "table1_math_4shot_g512": 65,
    "table1_humaneval_0shot_g256": 65,
    "table1_humaneval_0shot_g512": 65,
    "table1_mbpp_3shot_g256": 65,
    "table1_mbpp_3shot_g512": 65,
    "table4_gsm8k_5shot_g1024": 13,
    "table5_gsm8k_8shot_g256": 13,
    "table5_gsm8k_8shot_g512": 13,
    "table5_gsm8k_8shot_g1024": 13,
}
CATEGORY_DEFINITION = {
    "kind": "dataset",
    "field": "benchmark",
    "categories": ["gsm8k", "math", "humaneval", "mbpp"],
}
CATEGORY_COUNTS = {category: 130 for category in CATEGORY_DEFINITION["categories"]}
INTERACTIVE_SELECTION = "all_frozen_final_v1"
INTERACTIVE_TOTAL_CASES = 520
INTERACTIVE_CATEGORY_BASE_QUOTA = 130
INTERACTIVE_BONUS_CATEGORIES = 0
INTERACTIVE_MINIMUM_PER_ROW = 1
# Single-allocation (one whole GPU shared serially by Work and Judge) contract.
HOOK = "/app/workdir/decoder_lab/.sforge/prepare_submit.sh"
SCRIPTS = (
    "scripts/run_single_gpu_fixed_budget.sh",
    "scripts/single_gpu_runtime_preflight.py",
    "scripts/refresh_portable_manifests.py",
)
PORTABLE_FILES = (
    "portable/resource-manifest.json",
    "portable/build-resource-manifest.json",
    "portable/public-suite-manifest.json",
    "portable/interactive-suite-manifest.json",
    "portable/final-suite-manifest.json",
    "portable/oracle-protocol.json",
    "portable/selection-protocol.json",
    "portable/software-abi.json",
    "portable/calibration-protocol.json",
    "images/work/public_tests/prepare_submit.sh",
)
# Maintainer docs under docs/ are checked only when present (the public tree
# ships docs/INTERFACE.md and the hash-bound manifests, not these notes).
DOCS = (
    "docs/SINGLE_GPU_MIGRATION.md",
    "portable/REBUILD_PROMPT.md",
)
PRIVATE_DATA_FILES = (
    "hidden_cases.json.gz",
    "reference_quality.json",
    "workload_policy.json",
    "memory_policy.json",
    "quality_controls.json",
)


def private_data_dir(root: Path) -> Path:
    """The Judge-only private_data/ directory (hidden cases, references, policies).

    The in-repository images/judge/private_data/ when it is present, else the
    task's dataset asset that assets/fetch.sh places at
    <INFRATHON_ASSETS_ROOT>/<task directory>/judge/private_data (default asset root:
    ~/.cache/infrathon/assets).  Mirrors scripts/h100_policy_contract.py.
    """
    in_repository = root / "images/judge/private_data"
    if in_repository.is_dir() and any(in_repository.iterdir()):
        return in_repository
    assets_root = os.environ.get("INFRATHON_ASSETS_ROOT") or str(
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        / "infrathon/assets"
    )
    return Path(assets_root).expanduser() / root.name / "judge/private_data"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_sforge_runtime(root: Path) -> dict[str, Any]:
    repo = next(
        (candidate for candidate in (root, *root.parents) if (candidate / "sforge").is_dir()),
        None,
    )
    if repo is None:
        return {}
    paths = sorted(
        (path.resolve() for path in (repo / "sforge").rglob("*.py")),
        key=lambda path: str(path),
    )
    files = [path.relative_to(repo).as_posix() for path in paths]
    aggregate = "".join(
        f"{sha256(path)}  {name}\n" for path, name in zip(paths, files, strict=True)
    ).encode()
    return {
        "scope": "all_sforge_python",
        "sha256": hashlib.sha256(aggregate).hexdigest(),
        "files": files,
        "file_count": len(files),
        "file_set_sha256": hashlib.sha256(
            ("\n".join(files) + "\n").encode()
        ).hexdigest(),
    }


def load_json(path: Path, errors: list[str], expected: type = dict) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"cannot parse {path}: {exc}")
        return expected()
    if not isinstance(value, expected):
        errors.append(f"{path} must contain a JSON {expected.__name__}")
        return expected()
    return value


def validate_workload(root: Path, errors: list[str]) -> tuple[Path, Path, bytes]:
    public_path = root / "images/work/public_tests/public_cases.json"
    hidden_path = private_data_dir(root) / "hidden_cases.json.gz"
    public = load_json(public_path, errors, list)
    try:
        if hidden_path.is_symlink() or not hidden_path.is_file():
            raise ValueError("compressed hidden workload is missing or unsafe")
        hidden_bytes = gzip.decompress(hidden_path.read_bytes())
        if len(hidden_bytes) > 128 * 1024 * 1024:
            raise ValueError("decompressed hidden workload exceeds its bound")
        hidden = json.loads(hidden_bytes)
        if not isinstance(hidden, list):
            raise ValueError("hidden workload must be a list")
    except Exception as exc:
        errors.append(f"cannot validate compressed hidden workload: {exc}")
        hidden_bytes = b"[]\n"
        hidden = []
    counts = Counter(str(case.get("row_id")) for case in hidden if isinstance(case, dict))
    if dict(counts) != EXPECTED_ROWS:
        errors.append(f"private workload rows differ: {dict(counts)}")
    if len(hidden) != GENERATION_COUNT:
        errors.append(f"private workload must contain {GENERATION_COUNT} generations")
    slots = 0
    case_ids: set[str] = set()
    row_records: dict[str, set[str]] = defaultdict(set)
    row_benchmarks: dict[str, str] = {}
    dataset_records: dict[str, set[str]] = defaultdict(set)
    for case in hidden:
        if not isinstance(case, dict):
            errors.append("hidden workload contains a non-object")
            continue
        try:
            row_id = str(case["row_id"])
            benchmark = str(case["benchmark"])
            record_id = str(case["record_id"])
            case_id = str(case["case_id"])
            gen_length = case["gen_length"]
            prompt_tokens = case["prompt_token_count"]
            if (
                isinstance(gen_length, bool)
                or not isinstance(gen_length, int)
                or isinstance(prompt_tokens, bool)
                or not isinstance(prompt_tokens, int)
                or prompt_tokens <= 0
                or prompt_tokens + gen_length > 4096
            ):
                raise ValueError("invalid exact context attestation")
            slots += gen_length
            if case_id in case_ids:
                raise ValueError(f"duplicate case_id {case_id}")
            case_ids.add(case_id)
            if record_id in row_records[row_id]:
                raise ValueError(f"row-local duplicate record {row_id}/{record_id}")
            if record_id in dataset_records[benchmark]:
                raise ValueError(
                    f"target record repeats across {benchmark} score rows: {record_id}"
                )
            row_records[row_id].add(record_id)
            dataset_records[benchmark].add(record_id)
            previous = row_benchmarks.setdefault(row_id, benchmark)
            if previous != benchmark:
                raise ValueError(f"row {row_id} mixes benchmarks")
        except Exception as exc:
            errors.append(f"malformed hidden case: {exc}")
    if slots != REQUESTED_SLOTS:
        errors.append(f"requested-slot count differs: {slots}")
    if {key: len(value) for key, value in dataset_records.items()} != CATEGORY_COUNTS:
        errors.append("private workload must contain 130 unique targets per dataset")
    public_targets = set()
    for case in public:
        if not isinstance(case, dict):
            errors.append("public workload contains a non-object")
            continue
        if case.get("source_split") == "test":
            errors.append("public fixtures may not use an official test record")
        public_targets.add((case.get("benchmark"), case.get("record_id")))
    hidden_targets = {
        (case.get("benchmark"), case.get("record_id"))
        for case in hidden
        if isinstance(case, dict)
    }
    if public_targets & hidden_targets:
        errors.append("public fixture records overlap Judge-only standard-test records")
    return public_path, hidden_path, hidden_bytes


def profile_key(profile: dict[str, str]) -> str:
    """Hash one measurement/device profile exactly as the Judge does."""
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


REFERENCE_PROFILE_FIELDS = ("compute_capability", "gpu_name")


def reference_profile_key(profile: dict[str, str]) -> str:
    """Hash the GPU model + compute capability half (the selection key)."""
    return profile_key({key: profile[key] for key in REFERENCE_PROFILE_FIELDS})


def scoring_equivalent_verifiers(root: Path) -> frozenset[str]:
    """Read the verifier's declared scoring-equivalent predecessor hashes."""
    tree = ast.parse(
        (root / "images/judge/verifier/verify_decoder.py").read_text(encoding="utf-8")
    )
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "SCORING_EQUIVALENT_VERIFY_DECODER_SHA256"
            and isinstance(node.value, ast.Call)
            and node.value.args
        ):
            return frozenset(ast.literal_eval(node.value.args[0]))
    return frozenset()


def validate_reference_cache(root: Path, errors: list[str]) -> None:
    """Each cache entry is filed under its own reference profile key."""
    private = private_data_dir(root)
    installed = private / "reference_quality.json"
    entries = {}
    for path in sorted((private / "reference_cache").glob("*.json")):
        if path.is_symlink() or not path.is_file():
            errors.append(f"reference cache entry is unsafe: {path.name}")
            continue
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))["measurement_profile"]
            key = reference_profile_key(profile)
        except (KeyError, TypeError, ValueError):
            errors.append(f"reference cache entry lacks a measurement profile: {path.name}")
            continue
        if path.stem != key:
            errors.append(
                f"reference cache entry {path.name} is not filed under its "
                f"reference profile key {key}"
            )
        entries[key] = path
    try:
        reference = json.loads(installed.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if reference.get("calibrated") is True:
        try:
            key = reference_profile_key(reference["measurement_profile"])
        except (KeyError, TypeError):
            return
        cached = entries.get(key)
        if cached is None or cached.read_bytes() != installed.read_bytes():
            errors.append(
                "installed reference_quality.json is not byte-identical to its "
                "reference_cache entry"
            )


def validate_reference(
    root: Path, hidden_bytes: bytes, errors: list[str]
) -> None:
    path = private_data_dir(root) / "reference_quality.json"
    ref = load_json(path, errors)
    allow_pilot_drift = os.environ.get("LLADA_PILOT") == "1"
    # Reproducibility checks compare the recorded calibration against the
    # CURRENT tree and sforge runtime. They answer "could this calibration be
    # reproduced byte-for-byte today?" — a recalibration signal, not a build
    # or run precondition: everything that actually gates scoring is
    # re-validated inside the Judge container at evaluation time. Treating
    # them as hard errors froze builds on every unrelated sforge change (and
    # permanently, once an attestation predated the sforge_runtime field).
    # They warn by default; LLADA_STRICT_REFERENCE=1 restores hard failure
    # for formal calibration acceptance.
    strict_reference = os.environ.get("LLADA_STRICT_REFERENCE") == "1"

    def reproducibility_finding(message: str) -> None:
        if strict_reference:
            errors.append(message)
        else:
            print(
                f"WARNING: {message} (stale-calibration signal; set "
                "LLADA_STRICT_REFERENCE=1 to enforce)",
                file=sys.stderr,
            )
    if (
        ref.get("schema_version") != 4
        or ref.get("task_id") != TASK_ID
        or ref.get("workload_profile") != PROFILE
        or not isinstance(ref.get("hardware"), str)
        or not ref.get("hardware")
        or ref.get("generation_count") != GENERATION_COUNT
        or ref.get("requested_slots") != REQUESTED_SLOTS
        or ref.get("dataset_manifest_sha256") != hashlib.sha256(hidden_bytes).hexdigest()
    ):
        errors.append("reference identity/provenance differs")
    historical = ref.get("historical_a6000_reference")
    if (
        not isinstance(historical, dict)
        or historical.get("applicability") != "NONE"
        or historical.get("status") != "UNCALIBRATED_FOR_H100_STRATIFIED520"
        or historical.get("provenance_preserved_in_source_task") is not True
    ):
        errors.append("historical A6000 evidence is not explicitly non-applicable")
    if ref.get("calibrated") is False:
        if (
            ref.get("status") != "PENDING_STRATIFIED520_CALIBRATION"
            or ref.get("rows") != {}
            or ref.get("case_records") != {}
            or ref.get("calibration_provenance") is not None
        ):
            errors.append("pending reference must be an explicit empty placeholder")
        return
    if ref.get("calibrated") is not True or ref.get("status") != "CALIBRATED_STRATIFIED520":
        errors.append("reference calibration state is invalid")
        return
    profile = ref.get("measurement_profile")
    if (
        not isinstance(profile, dict)
        or sorted(profile) != [
            "compute_capability",
            "driver_version",
            "gpu_name",
            "judge_image_id",
            "work_image_id",
        ]
        or any(not isinstance(value, str) or not value for value in profile.values())
        or ref.get("measurement_profile_key")
        != profile_key({key: profile[key] for key in sorted(profile)})
        or ref.get("device_profile_key")
        != profile_key(
            {
                key: profile[key]
                for key in ("compute_capability", "driver_version", "gpu_name")
            }
        )
    ):
        errors.append("calibrated reference lacks a well-formed measurement profile")
        return
    rows = ref.get("rows")
    records = ref.get("case_records")
    provenance = ref.get("calibration_provenance")
    if (
        not isinstance(rows, dict)
        or set(rows) != set(EXPECTED_ROWS)
        or not isinstance(records, dict)
        or len(records) != GENERATION_COUNT
        or not isinstance(provenance, dict)
        or provenance.get("schema_version") != 1
        or provenance.get("task_id") != TASK_ID
        or provenance.get("workload_profile") != PROFILE
        or not isinstance(provenance.get("hardware"), str)
        or not provenance.get("hardware")
        or provenance.get("canonical_candidate_archive_sha256")
        != provenance.get("candidate_archive_sha256")
    ):
        errors.append("calibrated reference shape/provenance is incomplete")
        return
    hex_pattern = re.compile(r"[0-9a-f]{64}\Z")
    required_hashes = (
        "candidate_archive_sha256",
        "canonical_candidate_archive_sha256",
        "candidate_source_sha256",
        "evaluation_validator_sha256",
        "calibration_tool_sha256",
        "image_manifest_sha256",
        "source_digest",
        "judge_runtime_tree_sha256",
        "model_validation_sha256",
        "workload_policy_sha256",
        "memory_policy_sha256",
        "quality_controls_sha256",
        "evaluation_attestation_sha256",
        "private_artifact_sha256",
        "sforge_report_sha256",
        "private_payload_sha256",
        "runtime_provenance_sha256",
        "sforge_runtime_sha256",
        "sforge_runtime_file_set_sha256",
        "verify_decoder_sha256",
        "candidate_worker_sha256",
        "cohort_sha256",
        "quality_sha256",
        "source_admission_sha256",
    )
    if any(hex_pattern.fullmatch(str(provenance.get(key, ""))) is None for key in required_hashes):
        errors.append("calibrated reference contains an invalid provenance digest")
    sforge = provenance.get("sforge_runtime")
    files = sforge.get("files") if isinstance(sforge, dict) else None
    core = provenance.get("judge_invariant_core")
    expected_core = {
        "verify_decoder_sha256": sha256(
            root / "images/judge/verifier/verify_decoder.py"
        ),
        "candidate_worker_sha256": sha256(
            root / "images/judge/verifier/candidate_worker.py"
        ),
        "cohort_sha256": sha256(root / "images/judge/verifier/cohort.py"),
        "quality_sha256": sha256(root / "images/judge/verifier/quality.py"),
        "source_admission_sha256": sha256(
            root / "images/judge/verifier/source_admission.py"
        ),
    }
    if not allow_pilot_drift and (
        not isinstance(sforge, dict)
        or sforge.get("scope") != "all_sforge_python"
        or sforge.get("sha256") != provenance.get("sforge_runtime_sha256")
        or sforge.get("file_set_sha256")
        != provenance.get("sforge_runtime_file_set_sha256")
        or not isinstance(files, list)
        or not files
        or files != sorted(set(files))
        or sforge.get("file_count") != len(files)
        or sforge.get("file_set_sha256")
        != hashlib.sha256(("\n".join(files) + "\n").encode()).hexdigest()
        or any(
            not isinstance(item, str)
            or re.fullmatch(
                r"sforge/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py", item
            )
            is None
            for item in files
        )
        or not isinstance(core, dict)
        or set(core) != set(expected_core)
        or any(
            core.get(field) != value
            and not (
                field == "verify_decoder_sha256"
                and core.get(field) in scoring_equivalent_verifiers(root)
            )
            for field, value in expected_core.items()
        )
        or any(provenance.get(field) != core.get(field) for field in expected_core)
        or sforge != current_sforge_runtime(root)
    ):
        reproducibility_finding(
            "calibrated reference complete SForge/Judge provenance is invalid"
        )
    for role in ("work_image", "judge_image"):
        image = provenance.get(role)
        if (
            not isinstance(image, dict)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(image.get("id", ""))) is None
            or not isinstance(image.get("name"), str)
            or not image["name"]
        ):
            errors.append(f"calibrated reference {role} identity is invalid")
    if not allow_pilot_drift and (
        provenance.get("workload_policy_sha256")
        != (
            load_json(root / "workload_policy.json", errors).get("provisional_parent_sha256")
            if load_json(root / "workload_policy.json", errors).get("release_status") == "FROZEN"
            else sha256(root / "workload_policy.json")
        )
        or provenance.get("memory_policy_sha256") != sha256(root / "memory_policy.json")
        or provenance.get("quality_controls_sha256")
        != sha256(root / "quality_controls.json")
        or provenance.get("evaluation_validator_sha256")
        != sha256(root / "scripts/validate_eval_artifacts.py")
        or provenance.get("calibration_tool_sha256")
        != sha256(root / "scripts/calibrate_reference.py")
        or provenance.get("candidate_source_sha256")
        != sha256(root / "host_reference/corrected_decoder/candidate.py")
    ):
        reproducibility_finding(
            "calibrated reference producer files/policies drifted"
        )
    canonical = {
        hashlib.sha256(str(case["case_id"]).encode()).hexdigest(): case["row_id"]
        for case in json.loads(hidden_bytes)
    }
    if set(records) != set(canonical):
        errors.append("calibrated reference case set differs from canonical workload")
        return
    observed = {row_id: Counter() for row_id in EXPECTED_ROWS}
    for case_hash, item in records.items():
        if not isinstance(item, dict) or item.get("row_id") != canonical[case_hash]:
            errors.append("calibrated reference case row provenance differs")
            return
        row_id = item["row_id"]
        if not isinstance(item.get("correct"), bool) or not isinstance(item.get("parsed"), bool):
            errors.append("calibrated reference case quality flags are invalid")
            return
        useful = item.get("useful_tokens")
        if isinstance(useful, bool) or not isinstance(useful, int) or useful < 0:
            errors.append("calibrated reference useful-token value is invalid")
            return
        observed[row_id].update(
            cases=1,
            correct=int(item["correct"]),
            parsed=int(item["parsed"]),
            useful_tokens=useful,
        )
    for row_id, expected_count in EXPECTED_ROWS.items():
        row = rows.get(row_id)
        if (
            not isinstance(row, dict)
            or row.get("cases") != expected_count
            or any(row.get(key) != observed[row_id][key] for key in ("cases", "correct", "parsed", "useful_tokens"))
        ):
            errors.append(f"calibrated reference aggregate differs: {row_id}")


def validate_protected_hashes(root: Path, public_path: Path, errors: list[str]) -> None:
    sources = {
        "/app/workdir/decoder_lab/runtime/api.py": root / "images/work/workdir/decoder_lab/runtime/api.py",
        "/app/workdir/decoder_lab/runtime/configuration_llada.py": root / "images/work/workdir/decoder_lab/runtime/configuration_llada.py",
        "/app/workdir/decoder_lab/runtime/loader.py": root / "images/work/workdir/decoder_lab/runtime/loader.py",
        "/app/workdir/decoder_lab/runtime/modeling_llada.py": root / "images/work/workdir/decoder_lab/runtime/modeling_llada.py",
        "/app/workdir/decoder_lab/runtime/source_policy.py": root / "images/work/workdir/decoder_lab/runtime/source_policy.py",
        "/workspace/public_tests/public_quality.py": root / "images/work/public_tests/public_quality.py",
        "/workspace/public_tests/public_cases.json": public_path,
        "/workspace/public_tests/memory_contract.json": root / "images/work/public_tests/memory_contract.json",
    }
    seen: set[str] = set()
    try:
        for raw in (root / "images/judge/protected_hashes.txt").read_text().splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            expected, runtime_path = raw.split(None, 1)
            source = sources.get(runtime_path)
            if source is None or sha256(source) != expected:
                errors.append(f"protected hash mismatch/unknown path: {runtime_path}")
            seen.add(runtime_path)
        if seen != set(sources):
            errors.append("protected manifest path set is incomplete")
    except Exception as exc:
        errors.append(f"cannot validate protected hashes: {exc}")


def validate_single_gpu_handoff(root: Path, errors: list[str]) -> None:
    """Deployment-only single-allocation contract: one UUID lock, NVML idle
    attestation before the first Docker probe, the root-owned cooperative hook
    inside the task cwd, the adapter forwarding the coordinator allocation and
    scoping the final Judge checkpoint directory, and the explicit standalone
    final handoff on the still-alive coordinator over loopback."""
    for relative in (*SCRIPTS, *PORTABLE_FILES, *DOCS):
        path = root / relative
        if relative.startswith("docs/") and not path.exists() and not path.is_symlink():
            continue
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            errors.append(f"missing regular nonempty file: {relative}")
    run_path = root / "scripts/run_fixed_budget.sh"
    run_text = run_path.read_text(encoding="utf-8") if run_path.is_file() else ""
    for fragment in (
        'SINGLE_GPU_HANDOFF="${SFORGE_SINGLE_GPU_HANDOFF:-0}"',
        'acquire_single_gpu_lock "${work_gpu_uuid}"',
        'phase="before_work_create"',
        "assert_portable_gpu_idle",
        "assert_container_accelerator",
        f"SFORGE_PRE_SUBMIT_HOOK={HOOK}",
        "--standalone-final-handoff",
        '--handoff-source-run-id "${RUN_ID}"',
        '--judge-url "http://127.0.0.1:${JUDGE_PORT}"',
        'serve \\\n  --host "${docker_bridge_gateway}" "${judge_serve_extra_hosts[@]}" --port "${JUDGE_PORT}"',
        "judge_serve_extra_hosts+=(--host 127.0.0.1)",
        'scheduling_mode="single_gpu_lease_manual_judge_then_standalone_final_handoff"',
        'scheduling_mode="dual_gpu_interactive_then_frozen_final"',
        '--evaluation-mode "${scheduling_mode}"',
        'export LLADA_FINAL_RUNTIME_RUN_ID="${final_eval_id}"',
        'export LLADA_FINAL_JUDGE_CHECKPOINT_DIR="${LLADA_JUDGE_CHECKPOINT_DIR}"',
        'export LLADA_EXECUTION_PROFILE_ID="${execution_profile_id}"',
        "SFORGE_PRIVATE_CONTEXT_WORK_IMAGE_ID=${recorded_work_image_id}",
        "deployment_attestation_json",
        "measurement_profile_sha256",
        "raw_score_comparison_identity_sha256",
        "--disable-auto-eval",
    ):
        if fragment not in run_text:
            errors.append(f"scripts/run_fixed_budget.sh: missing {fragment!r}")
    lock = run_text.find('acquire_single_gpu_lock "${work_gpu_uuid}"')
    attest = run_text.find("NvmlAcceleratorObserver().attest_idle(")
    probe = run_text.find("assert_container_accelerator")
    if not 0 <= lock < attest < probe:
        errors.append(
            "scripts/run_fixed_budget.sh: single-allocation branch must lock, "
            "attest idle, then probe the container DeviceRequest, in that order"
        )
    if run_text.count('--evaluation-mode "${scheduling_mode}"') != 2:
        errors.append("scripts/run_fixed_budget.sh: pre and post validation must record the scheduling mode")
    wrapper_path = root / "scripts/run_single_gpu_fixed_budget.sh"
    wrapper = wrapper_path.read_text(encoding="utf-8") if wrapper_path.is_file() else ""
    for fragment in (
        "SFORGE_SINGLE_GPU_HANDOFF=1",
        "SFORGE_ACCELERATOR_OBSERVER=nvml",
        "canonical_allocation_key",
        "lease-llada-",
        "single_gpu_runtime_preflight.py",
        'exec "${SCRIPT_DIR}/run_fixed_budget.sh"',
    ):
        if fragment not in wrapper:
            errors.append(f"scripts/run_single_gpu_fixed_budget.sh: missing {fragment!r}")
    work_path = root / "images/work/Dockerfile"
    work = work_path.read_text(encoding="utf-8") if work_path.is_file() else ""
    if f"COPY public_tests/prepare_submit.sh {HOOK}" not in work:
        errors.append("Work Dockerfile must install the cooperative pre-submit hook")
    if "chown -R root:root /app/workdir/decoder_lab/.sforge" not in work:
        errors.append("Work Dockerfile must leave the pre-submit hook root-owned")
    if "portable" in work.lower() or "final-suite-manifest" in work.lower():
        errors.append("Work Dockerfile must not reference the host-private portable manifests")
    adapter_path = root / "scripts/sforge_h100_entry.py"
    adapter = adapter_path.read_text(encoding="utf-8") if adapter_path.is_file() else ""
    for fragment in (
        "accelerator_allocation: AcceleratorAllocation | None = None",
        "_validated_handoff_binding_labels",
        "list(accelerator_allocation.device_ids) != [requested_uuid]",
        "def _is_final_handoff_judge(",
        "def _final_judge_checkpoint_volume(",
        'SINGLE_EXECUTION_PROFILE_ID = "single-cuda-sm80-34g"',
        "DockerBackend.create_container = role_bound_create_container",
    ):
        if fragment not in adapter:
            errors.append(f"scripts/sforge_h100_entry.py: missing {fragment!r}")
    prompt_path = root / "images/work/AGENT_TASK.md"
    prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else ""
    if "## Submitting on the single-accelerator profile" not in prompt or ".sforge/prepare_submit.sh" not in prompt:
        errors.append("AGENT_TASK.md lacks the single-accelerator submission section")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    required = (
        "README.md",
        "tasks/BENCHMARK.yaml",
        TASK_JSON,
        "images/work/Dockerfile",
        "images/work/AGENT_TASK.md",
        "images/work/public_tests/public_eval.py",
        "images/work/public_tests/public_cases.json",
        "images/work/public_tests/public_quality.py",
        "images/work/public_tests/memory_contract.json",
        "images/work/public_tests/run_public_tests.sh",
        "images/work/workdir/decoder_lab/runtime/api.py",
        "images/work/workdir/decoder_lab/runtime/loader.py",
        "images/work/workdir/decoder_lab/runtime/modeling_llada.py",
        "images/work/workdir/decoder_lab/runtime/source_policy.py",
        "images/work/workdir/decoder_lab/solutions/decoder_v001/candidate.py",
        "images/judge/Dockerfile",
        "images/judge/run_judge.py",
        "images/judge/verifier/verify_decoder.py",
        "images/judge/verifier/candidate_worker.py",
        "images/judge/verifier/cohort.py",
        "images/judge/verifier/row_lifecycle.py",
        "images/judge/verifier/nvml_sampler.py",
        "images/judge/verifier/quality.py",
        "images/judge/verifier/quality_threshold.py",
        "images/judge/verifier/source_admission.py",
        "images/judge/protected_hashes.txt",
        "judge/run_judge.py",
        "docs/model_manifest.json",
        "docs/dataset_manifest.json",
        "scripts/calibrate_reference.py",
        "scripts/eval_archive.sh",
        "scripts/run_fixed_budget.sh",
        "scripts/sforge_h100_entry.py",
        "scripts/validate_model_artifact.py",
        "scripts/validate_eval_artifacts.py",
        "scripts/validate_formal_artifacts.py",
        "scripts/validate_workload_profile.py",
        "scripts/validate_quality_controls.py",
        "scripts/precommit_h100_quality_controls.py",
        "scripts/precommit_h100_memory_fixtures.py",
        "scripts/make_reference_archives.py",
        "workload_policy.json",
        "memory_policy.json",
        "quality_controls.json",
        "memory_qualification_fixtures.json",
    )
    for relative in required:
        path = root / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            errors.append(f"missing regular nonempty file: {relative}")
    # Judge-only data: the in-repository copy or the fetched dataset asset.
    # Without either (a public checkout before assets/fetch.sh) the checks
    # that read it are skipped; image builds and runs still require it.
    private = private_data_dir(root)
    have_private = private.is_dir()
    if have_private:
        for name in PRIVATE_DATA_FILES:
            path = private / name
            if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
                errors.append(f"missing regular nonempty Judge-only file: {path}")
    else:
        print(
            f"NOTE: Judge-only data not found at {private} (run assets/fetch.sh); "
            "skipping the checks that read it",
            file=sys.stderr,
        )

    task = load_json(root / TASK_JSON, errors)
    if (
        task.get("task_id") != TASK_ID
        or set(task.get("submit_paths", [])) != EXPECTED_SUBMIT
        or task.get("internet") is not True
        or task.get("cwd") != "/app/workdir/decoder_lab"
    ):
        errors.append("task identity/internet/submission contract mismatch")
    judge = task.get("judge")
    if (
        not isinstance(judge, dict)
        or judge.get("parser") != "structured_json"
        or judge.get("score_direction") != "maximize"
        or judge.get("private_artifact_path") != "/logs/verifier/decoder_private.json"
        or judge.get("eval_timeout") != 7_200
    ):
        errors.append("Judge task contract mismatch")

    canonical = root / "judge/run_judge.py"
    mirror = root / "images/judge/run_judge.py"
    if canonical.is_file() and mirror.is_file() and sha256(canonical) != sha256(mirror):
        errors.append("Judge adapter mirror differs; run scripts/build_images.sh")
    if have_private:
        public_path, hidden_path, hidden_bytes = validate_workload(root, errors)
    else:
        public_path = root / "images/work/public_tests/public_cases.json"
    manifest = load_json(root / "docs/dataset_manifest.json", errors)
    if (
        manifest.get("schema_version") != 5
        or manifest.get("task_id") != TASK_ID
        or manifest.get("workload_profile") != PROFILE
        or manifest.get("generation_count") != GENERATION_COUNT
        or manifest.get("requested_slots") != REQUESTED_SLOTS
        or manifest.get("hidden_row_counts") != EXPECTED_ROWS
        or manifest.get("public_sha256") != sha256(public_path)
    ):
        errors.append("dataset manifest identity/shape/hash mismatch")
    if have_private and (
        manifest.get("hidden_sha256") != hashlib.sha256(hidden_bytes).hexdigest()
        or manifest.get("hidden_storage", {}).get("compressed_sha256") != sha256(hidden_path)
        or manifest.get("hidden_storage", {}).get("canonical_uncompressed_sha256")
        != hashlib.sha256(hidden_bytes).hexdigest()
    ):
        errors.append("dataset manifest identity/shape/hash mismatch")
    exclusions = manifest.get("context_validation", {}).get("documented_exclusions")
    if exclusions != [{
        "benchmark": "mbpp",
        "reason": "prompt_plus_generation_exceeds_4096",
        "record_id": "test:482/task:493",
    }]:
        errors.append("dataset must contain exactly the documented MBPP exclusion")

    for name in ("workload_policy.json", "memory_policy.json", "quality_controls.json"):
        if have_private and sha256(root / name) != sha256(private / name):
            errors.append(f"public/private {name} bytes differ")
    workload = load_json(root / "workload_policy.json", errors)
    if (
        workload.get("task_id") != TASK_ID
        or workload.get("profile") != PROFILE
        or workload.get("manifest", {}).get("row_counts") != EXPECTED_ROWS
        or workload.get("manifest", {}).get("generation_count") != GENERATION_COUNT
        or workload.get("manifest", {}).get("requested_slots") != REQUESTED_SLOTS
        or workload.get("interactive", {}).get("category_definition")
        != CATEGORY_DEFINITION
        or workload.get("interactive", {}).get("row_atomic") is not True
        or workload.get("interactive", {}).get("timeout_policy") != "same_as_final"
        or workload.get("interactive", {}).get("selection")
        != INTERACTIVE_SELECTION
        or workload.get("interactive", {}).get("total_cases")
        != INTERACTIVE_TOTAL_CASES
        or workload.get("interactive", {}).get("category_base_quota")
        != INTERACTIVE_CATEGORY_BASE_QUOTA
        or workload.get("interactive", {}).get("bonus_categories")
        != INTERACTIVE_BONUS_CATEGORIES
        or workload.get("interactive", {}).get("minimum_per_row")
        != INTERACTIVE_MINIMUM_PER_ROW
        or "per_row_quotas" in workload.get("interactive", {})
        or "per_category_quotas" in workload.get("interactive", {})
        or workload.get("final", {}).get("selection") != "all"
        or workload.get("final", {}).get("max_infrastructure_retries_per_row") != 1
    ):
        errors.append("workload policy contract mismatch")
    memory = load_json(root / "memory_policy.json", errors)
    if (
        memory.get("task_id") != TASK_ID
        or not isinstance(memory.get("qualification_hardware"), str)
        or not memory.get("qualification_hardware")
        or memory.get("min_compute_capability") != MIN_COMPUTE_CAPABILITY
        or memory.get("min_device_memory_mib") != MIN_DEVICE_MEMORY_MIB
    ):
        errors.append("memory policy identity/envelope mismatch")
    if have_private:
        validate_reference(root, hidden_bytes, errors)
        validate_reference_cache(root, errors)
    validate_protected_hashes(root, public_path, errors)

    work_root = root / "images/work"
    for forbidden in ("hidden_cases.json", "reference_quality.json", "verify_decoder.py"):
        if any(path.name == forbidden for path in work_root.rglob("*")):
            errors.append(f"private file leaked into Work: {forbidden}")
    try:
        admission_path = root / "images/judge/verifier/source_admission.py"
        spec = importlib.util.spec_from_file_location("layout_source_admission", admission_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load source admission")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.admit(root / "images/work/workdir/decoder_lab")
    except Exception as exc:
        errors.append(f"provided starter fails Judge source admission: {exc}")
    validate_single_gpu_handoff(root, errors)
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except Exception as exc:
            errors.append(f"Python syntax error in {path.relative_to(root)}: {exc}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"layout validation PASS: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
