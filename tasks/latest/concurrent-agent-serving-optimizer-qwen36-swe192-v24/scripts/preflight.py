#!/usr/bin/env python3
"""Fail-closed host preflight without printing hidden material."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "runtime/frozen/harbor_run_scripts/"
    "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.2-gpu45/"
    "run-gpu67-v2.2-candidate-only.py"
)
STACK = (
    ROOT
    / "runtime/frozen/harbor_run_scripts/"
    "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/"
    "gpu67-v2.1-validation-stack.sh"
)
PROJECT = "kvbench-qwen36-swe192-v2-1-gpu67-validation"
FROZEN_RUNTIME_SOURCES = ROOT / "runtime/frozen"
TRUSTED_EXECUTION_ROOTS = ("scripts", "judge", "tasks", "images", "oracle")
GENERATED_DIRECTORY_NAMES = frozenset({"__pycache__"})
GENERATED_FILE_SUFFIXES = frozenset({".pyc"})
SOURCE_ATTESTATION_ALGORITHM = (
    "sha256-canonical-json-path-mode-size-content-v1"
)
# ``ROOT`` is ``<mlsysforge>/tasks/latest/<task>``.  Bind provenance to the
# repository checkout itself, not to the task catalog directory.
EXPECTED_MLSYSFORGE_ROOT = ROOT.parents[2]
MLSYSFORGE_GENERATED_EXCLUSIONS = (
    "directory-component:__pycache__",
    "file-suffix:.pyc",
)
MLSYSFORGE_REQUIRED_IMPORTS = {
    "sforge": "sforge/__init__.py",
    "sforge.cli": "sforge/cli.py",
    "sforge.harness.run_agent": "sforge/harness/run_agent.py",
    "sforge.harness.backend.docker_backend": (
        "sforge/harness/backend/docker_backend.py"
    ),
    "sforge.harness.judge_server": "sforge/harness/judge_server.py",
    "sforge.harness.run_evaluation": "sforge/harness/run_evaluation.py",
    "sforge.harness.task_spec": "sforge/harness/task_spec.py",
}
# The staged corpus (scripts/stage_swebench_corpus.py): the 192 digest-locked
# SWE-bench Lite images in the local Docker store plus manifest.json here.
CORPUS_ROOT = Path(
    os.environ.get("KVBENCH_CORPUS_ROOT")
    or Path.home() / ".cache/harbor-kv-scheduler-v2/swebench-lite-heterogeneous-192-gpu45"
).expanduser()
EXPECTED_NONCE_COMMITMENT = (
    "c525ae31a2bb1d3ed0232c345f9fc978fd52574d415aaa2777bead17f2fac20a"
)
EXPECTED_PUBLIC_NONCE_COMMITMENT = (
    "4902baae73726c1a2f372e35e9e104f6bb3db7dc7593822359548fba60e1b00c"
)
EXPECTED_ORACLE_TREE = (
    "b8f92b46bcd25529c19d20354be94d620b632e2bce209909f8ae666a71de0e75"
)
PUBLIC_RUNTIME_CONTRACT = ROOT / "images/work/RUNTIME_CONTRACT.json"
MODEL_PROFILE = (
    ROOT
    / "runtime/frozen/harbor_tasks/"
    "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/"
    "environment/vllm-runtime/model-profile-qwen3.6-27b.json"
)
BACKEND_SERVING_CONTRACT = MODEL_PROFILE.parents[1] / "backend-serving-contract.json"
LEGACY_BACKEND_SERVING_CONTRACT = (
    MODEL_PROFILE.parents[1] / "backend-serving-contract-v1.json"
)
PREVIOUS_BACKEND_SERVING_CONTRACT = (
    MODEL_PROFILE.parents[1] / "backend-serving-contract-v2.5.json"
)
BACKEND_DOCKERFILE = ROOT / "images/backend/Dockerfile"
# Revision v2.5 replaced ``allowed_names: ["NVIDIA H100 NVL"]`` with a
# capability predicate.  Revision v3 replaced the runtime image binding: v1 and
# v2.5 pinned the Docker image ID of one host's vLLM build, which no clean
# clone can reproduce; v3 binds ``runtime.runtime_image_recipe`` (the
# digest-pinned official vLLM release image plus ``images/backend/Dockerfile``)
# and ``runtime.runtime_image_id`` is now the sha256 of that recipe, while the
# Docker image ID of the deployment is measurement provenance recorded in the
# preflight report.  v1 and v2.5 stay on disk so Direct anchors measured under
# them remain attributable, and the three revisions must agree on every field
# that determines what is measured except the two declarations each revision
# changed.  ``SERVING_SYSTEM_IDENTITY_KEYS`` is that measured surface, and it
# is a positive allowlist so a future revision cannot smuggle a
# throughput-affecting field past the equivalence check below.
EXPECTED_BACKEND_SERVING_CONTRACT_SHA256 = (
    "1cf6d2e6ff7af8448d6b7735590621d083d6a5071252bc60b40c9756ffe328c4"
)
LEGACY_BACKEND_SERVING_CONTRACT_SHA256 = (
    "44e4a3408b298a5fdc53d7bb84f0739a22f047da1974212754a98b5755777ba8"
)
PREVIOUS_BACKEND_SERVING_CONTRACT_SHA256 = (
    "18d27d97c736f90e1d74822e80ce1f212685e26627b31a08e47569a1457f52c6"
)
ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256 = frozenset(
    {
        EXPECTED_BACKEND_SERVING_CONTRACT_SHA256,
        LEGACY_BACKEND_SERVING_CONTRACT_SHA256,
        PREVIOUS_BACKEND_SERVING_CONTRACT_SHA256,
    }
)
SERVING_SYSTEM_IDENTITY_KEYS = ("behavior", "model", "runtime", "schema_version")
# The runtime-image binding is the one part of ``runtime`` that revision v3
# changed; everything else in the section still has to match across revisions.
RUNTIME_IMAGE_BINDING_KEYS = frozenset(
    {"runtime_image_id", "runtime_image_ref", "runtime_image_recipe"}
)
# Every nested image is built from the frozen environment by
# scripts/build_images.sh; their IDs are deployment provenance recorded in the
# build manifest and in this report.  Until 2026-09-18 ``main`` and
# ``sandbox-worker`` were pinned to the image IDs of one host's builds, which
# no clean clone could reproduce.
PINNED_IMAGES: dict[str, str] = {}
REQUIRED_DYNAMIC_IMAGES = (
    "harbor-kv/main:qwen36-27b-swe192-v2-1-gpu45-v1",
    "harbor-kv/sandbox-worker:qwen36-27b-tp2-public30-r2",
    "harbor-kv/model-api:qwen36-swe192-v24-url-v1",
    "harbor-kv/bench-controller:qwen36-swe192-v24-url-v1",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is unavailable or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return payload


def _oracle_integer_default(
    source: Path,
    *,
    constant_name: str,
    environment_name: str,
) -> int:
    """Read a frozen Oracle's public geometry default without importing it."""

    if not source.is_file() or source.is_symlink():
        raise RuntimeError("Oracle gateway source is unavailable or unsafe")
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise RuntimeError("Oracle gateway source could not be parsed") from exc
    for node in tree.body:
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or node.targets[0].id != constant_name
        ):
            continue
        outer = node.value
        if (
            not isinstance(outer, ast.Call)
            or not isinstance(outer.func, ast.Name)
            or outer.func.id != "int"
            or len(outer.args) != 1
            or outer.keywords
        ):
            break
        getter = outer.args[0]
        if (
            not isinstance(getter, ast.Call)
            or not isinstance(getter.func, ast.Attribute)
            or getter.func.attr != "get"
            or len(getter.args) != 2
            or getter.keywords
            or not isinstance(getter.args[0], ast.Constant)
            or getter.args[0].value != environment_name
            or not isinstance(getter.args[1], ast.Constant)
            or not isinstance(getter.args[1].value, str)
        ):
            break
        try:
            return int(getter.args[1].value)
        except ValueError as exc:
            raise RuntimeError(
                f"Oracle default {constant_name} is not an integer"
            ) from exc
    raise RuntimeError(f"Oracle default {constant_name} changed shape")


def validate_public_runtime_contract() -> str:
    """Bind the Work-visible static contract to frozen runtime/Oracle facts."""

    contract = _load_json_object(
        PUBLIC_RUNTIME_CONTRACT, "public runtime contract"
    )
    profile = _load_json_object(MODEL_PROFILE, "frozen model profile")
    try:
        source = profile["source"]
        config = profile["config_contract"]
        gpu = profile["gpu_contract"]
        runtime = profile["runtime"]
        repository = source["repository"]
        layer_count = config["num_hidden_layers"]
        attention_layers = config["full_attention_layer_count"]
        # The Work-visible contract states the accelerator envelope the boot
        # genuinely needs, never the device model it was qualified on.
        accelerator = {
            "count": gpu["count"],
            "min_compute_capability": gpu["min_compute_capability"],
            "minimum_memory_mib": gpu["minimum_memory_mib"],
        }
        profile_id = profile["profile_id"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("frozen model profile schema changed") from exc
    if (
        not isinstance(accelerator["count"], int)
        or isinstance(accelerator["count"], bool)
        or accelerator["count"] < 1
        or not isinstance(accelerator["min_compute_capability"], str)
        or not isinstance(accelerator["minimum_memory_mib"], int)
        or isinstance(accelerator["minimum_memory_mib"], bool)
        or accelerator["minimum_memory_mib"] < 1
        or not isinstance(profile_id, str)
        or not profile_id
    ):
        raise RuntimeError("frozen accelerator capability envelope changed")

    oracle_gateway = ROOT / "oracle/reference_gateway/gateway.py"
    raw_pages = _oracle_integer_default(
        oracle_gateway,
        constant_name="KV_RAW_GPU_BLOCKS",
        environment_name="THUNDER_KV_RAW_GPU_BLOCKS",
    )
    reserved_pages = _oracle_integer_default(
        oracle_gateway,
        constant_name="KV_RESERVED_NULL_BLOCKS",
        environment_name="THUNDER_KV_RESERVED_NULL_BLOCKS",
    )
    attention_tokens = _oracle_integer_default(
        oracle_gateway,
        constant_name="ATTENTION_PAGE_TOKENS",
        environment_name="THUNDER_ATTENTION_PAGE_TOKENS",
    )
    recurrent_groups = _oracle_integer_default(
        oracle_gateway,
        constant_name="MAMBA_GROUP_COUNT",
        environment_name="THUNDER_MAMBA_GROUP_COUNT",
    )
    resident_pages = _oracle_integer_default(
        oracle_gateway,
        constant_name="MAMBA_RESIDENT_PAGES_PER_GROUP",
        environment_name="THUNDER_MAMBA_RESIDENT_PAGES_PER_GROUP",
    )
    peak_pages = _oracle_integer_default(
        oracle_gateway,
        constant_name="MAMBA_PEAK_PAGES_PER_GROUP",
        environment_name="THUNDER_MAMBA_PEAK_PAGES_PER_GROUP",
    )

    expected = {
        "schema_version": 1,
        "contract_kind": "fixed-public-serving-system",
        "model": {
            "repository": repository,
            "served_name_is_judge_selected": True,
            "text_backbone_layers": layer_count,
            "full_attention_layers": attention_layers,
            "gated_deltanet_layers": layer_count - attention_layers,
        },
        "runtime": {
            "implementation": runtime["model_impl"],
            "version": runtime["vllm_package_version"],
            "tensor_parallel_size": runtime["tensor_parallel_size"],
            "accelerator": accelerator,
            "weight_dtype": runtime["dtype"],
            "kv_cache_dtype": runtime["kv_cache_dtype"],
            "max_running_sequences": runtime["max_num_seqs"],
            "max_batched_tokens_per_iteration": runtime[
                "max_num_batched_tokens"
            ],
            "prefix_caching": runtime["enable_prefix_caching"],
            "chunked_prefill": runtime["enable_chunked_prefill"],
        },
        "hybrid_cache": {
            "decision_unit": "gpu_pages",
            "raw_gpu_pages": raw_pages,
            "reserved_null_pages": reserved_pages,
            "allocatable_pages": raw_pages - reserved_pages,
            "attention_tokens_per_program_page": attention_tokens,
            # Deliberately no device model and no profile id: those are
            # launcher-side provenance, and naming them here would put a
            # qualification fact back into the agent's contract.
            "measured_on_profile": {
                "reported_by": "vllm:cache_config_info",
                "page_geometry_is_a_profile_measurement": True,
                "refreshed_by_launcher_preflight": True,
                "comparable_only_within_one_measurement_profile": True,
            },
            "recurrent_group_count": recurrent_groups,
            "resident_pages_per_recurrent_group": resident_pages,
            "peak_pages_per_recurrent_group_while_request_pending_or_inflight": (
                peak_pages
            ),
            "request_lifetime_peak_reservation_is_conservative": True,
            "model_level_logical_kv_token_capacity_is_cross_program_additive": (
                False
            ),
        },
        "scope": {
            "static_contract_only": True,
            "contains_live_backend_telemetry": False,
            "contains_workload_identity_or_order": False,
            "contains_reference_policy": False,
        },
    }
    if contract != expected:
        raise RuntimeError(
            "public runtime contract does not match frozen runtime facts"
        )
    return sha256_file(PUBLIC_RUNTIME_CONTRACT)


def _secure_file_record(path: Path, logical_path: str) -> dict[str, Any]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"trusted source disappeared: {logical_path}") from exc
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise RuntimeError(f"trusted source is not regular: {logical_path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"trusted source could not be opened: {logical_path}") from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise RuntimeError(f"trusted source changed before read: {logical_path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(opened, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"trusted source changed while read: {logical_path}")
    return {
        "path": logical_path,
        "mode": stat.S_IMODE(after.st_mode),
        "size": after.st_size,
        "content_sha256": digest.hexdigest(),
    }


def _collect_tree_records(root: Path, *, prefix: str = "") -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"trusted source root is unsafe: {prefix or '.'}")
    records: list[dict[str, Any]] = []

    def walk(directory: Path, relative: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise RuntimeError(f"trusted source directory could not be scanned: {relative}") from exc
        for entry in entries:
            entry_path = Path(entry.path)
            entry_relative = relative / entry.name
            logical = entry_relative.as_posix()
            if prefix:
                logical = f"{prefix}/{logical}"
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError(f"trusted source disappeared: {logical}") from exc
            if stat.S_ISLNK(info.st_mode):
                raise RuntimeError(f"trusted source tree contains a symlink: {logical}")
            if stat.S_ISDIR(info.st_mode):
                if entry.name in GENERATED_DIRECTORY_NAMES:
                    continue
                walk(entry_path, entry_relative)
            elif stat.S_ISREG(info.st_mode):
                if entry_path.suffix in GENERATED_FILE_SUFFIXES:
                    continue
                records.append(_secure_file_record(entry_path, logical))
            else:
                raise RuntimeError(f"trusted source tree contains a special file: {logical}")

    walk(root, Path())
    return records


def _attestation_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: record["path"])
    if not ordered or len({record["path"] for record in ordered}) != len(ordered):
        raise RuntimeError("trusted source attestation has no unique regular files")
    canonical = json.dumps(
        ordered,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return {
        "algorithm": SOURCE_ATTESTATION_ALGORITHM,
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "file_count": len(ordered),
        "total_bytes": sum(int(record["size"]) for record in ordered),
    }


def attest_tree(root: Path) -> dict[str, Any]:
    first = _collect_tree_records(root)
    second = _collect_tree_records(root)
    if first != second:
        raise RuntimeError("trusted source tree changed during attestation")
    return _attestation_from_records(first)


def attest_execution_sources(task_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative_root in TRUSTED_EXECUTION_ROOTS:
        records.extend(
            _collect_tree_records(task_root / relative_root, prefix=relative_root)
        )
    second: list[dict[str, Any]] = []
    for relative_root in TRUSTED_EXECUTION_ROOTS:
        second.extend(
            _collect_tree_records(task_root / relative_root, prefix=relative_root)
        )
    if records != second:
        raise RuntimeError("trusted execution sources changed during attestation")
    return _attestation_from_records(records)


def _require_controlled_mlsysforge_root(configured_root: Path) -> Path:
    """Resolve only the fixed, owner-controlled MLSys Forge checkout.

    The formal launcher is allowed to receive an ``MLSYSFORGE_ROOT`` environment
    variable for ordinary local tooling, but a provenance-bearing run must not
    silently select a different checkout.  Reject symlinks in every path
    component so the fixed lexical path cannot be redirected between checks.
    """

    if not configured_root.is_absolute() or configured_root != EXPECTED_MLSYSFORGE_ROOT:
        raise RuntimeError("MLSys Forge root does not match the fixed formal path")
    current = Path(configured_root.anchor)
    for component in configured_root.parts[1:]:
        current /= component
        try:
            info = current.lstat()
        except OSError as exc:
            raise RuntimeError("fixed MLSys Forge root is unavailable") from exc
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError("fixed MLSys Forge root contains a symlink component")
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("fixed MLSys Forge root contains a non-directory component")
    if configured_root.stat().st_uid != os.getuid():
        raise RuntimeError("fixed MLSys Forge root is not owned by the launching user")
    source_root = configured_root / "sforge"
    try:
        source_info = source_root.lstat()
    except OSError as exc:
        raise RuntimeError("MLSys Forge sforge source root is unavailable") from exc
    if stat.S_ISLNK(source_info.st_mode) or not stat.S_ISDIR(source_info.st_mode):
        raise RuntimeError("MLSys Forge sforge source root is unsafe")
    if source_info.st_uid != os.getuid():
        raise RuntimeError("MLSys Forge sforge source root has an unexpected owner")
    return source_root


def attest_mlsysforge_harness(configured_root: Path) -> dict[str, Any]:
    source_root = _require_controlled_mlsysforge_root(configured_root)
    return attest_tree(source_root)


def validate_mlsysforge_import_origins(configured_root: Path) -> dict[str, str]:
    """Prove the fixed interpreter imports the attested source checkout."""

    _require_controlled_mlsysforge_root(configured_root)
    interpreter = configured_root / ".venv/bin/python"
    if not interpreter.exists() or not os.access(interpreter, os.X_OK):
        raise RuntimeError("fixed MLSys Forge Python interpreter is unavailable")
    probe = (
        "import importlib,json,sys;"
        "names=json.loads(sys.argv[1]);"
        "print(json.dumps({name:getattr(importlib.import_module(name),'__file__',None) "
        "for name in names},sort_keys=True))"
    )
    completed = subprocess.run(
        [
            str(interpreter),
            "-I",
            "-c",
            probe,
            json.dumps(sorted(MLSYSFORGE_REQUIRED_IMPORTS)),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("fixed MLSys Forge Python could not import SForge")
    try:
        observed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("SForge import-origin probe returned invalid JSON") from exc
    if not isinstance(observed, dict) or set(observed) != set(
        MLSYSFORGE_REQUIRED_IMPORTS
    ):
        raise RuntimeError("SForge import-origin probe returned an invalid schema")
    for module_name, relative in MLSYSFORGE_REQUIRED_IMPORTS.items():
        origin = observed.get(module_name)
        if not isinstance(origin, str) or not Path(origin).is_absolute():
            raise RuntimeError(f"SForge import origin is invalid: {module_name}")
        try:
            resolved_origin = Path(origin).resolve(strict=True)
            expected_origin = (configured_root / relative).resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                f"SForge import origin is unavailable: {module_name}"
            ) from exc
        if resolved_origin != expected_origin:
            raise RuntimeError(
                f"SForge module was imported outside the attested tree: {module_name}"
            )
    return dict(MLSYSFORGE_REQUIRED_IMPORTS)


def _validate_attestation(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "algorithm",
        "tree_sha256",
        "file_count",
        "total_bytes",
    }:
        raise RuntimeError(f"{label} attestation schema is invalid")
    if value.get("algorithm") != SOURCE_ATTESTATION_ALGORITHM:
        raise RuntimeError(f"{label} attestation algorithm changed")
    digest = value.get("tree_sha256")
    file_count = value.get("file_count")
    total_bytes = value.get("total_bytes")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError(f"{label} attestation digest is invalid")
    if (
        isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count <= 0
        or isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
        or total_bytes <= 0
    ):
        raise RuntimeError(f"{label} attestation counts are invalid")
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    parent = path.parent.resolve(strict=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError("refusing to overwrite preflight report")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = -1
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def verify_trusted_sources(
    preflight_path: Path,
    mlsysforge_root: Path = EXPECTED_MLSYSFORGE_ROOT,
) -> dict[str, Any]:
    if preflight_path.is_symlink() or not preflight_path.is_file():
        raise RuntimeError("initial preflight report is unavailable or unsafe")
    try:
        initial = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("initial preflight report is invalid JSON") from exc
    if not isinstance(initial, dict) or initial.get("status") != "passed":
        raise RuntimeError("initial preflight report is not passed")
    expected_runtime = _validate_attestation(
        initial.get("trusted_runtime_sources"), "runtime source"
    )
    expected_execution = _validate_attestation(
        initial.get("trusted_execution_sources"), "execution source"
    )
    expected = _validate_attestation(
        initial.get("mlsysforge_harness_sources"), "MLSys Forge harness source"
    )
    if initial.get("mlsysforge_root") != str(EXPECTED_MLSYSFORGE_ROOT):
        raise RuntimeError("initial preflight MLSys Forge root is invalid")
    if initial.get("mlsysforge_generated_exclusions") != list(
        MLSYSFORGE_GENERATED_EXCLUSIONS
    ):
        raise RuntimeError("initial preflight MLSys Forge exclusion policy is invalid")
    if initial.get("mlsysforge_import_origins") != MLSYSFORGE_REQUIRED_IMPORTS:
        raise RuntimeError("initial preflight MLSys Forge import origins are invalid")
    observed_runtime = attest_tree(FROZEN_RUNTIME_SOURCES)
    observed_execution = attest_execution_sources(ROOT)
    observed_import_origins = validate_mlsysforge_import_origins(mlsysforge_root)
    observed = attest_mlsysforge_harness(mlsysforge_root)
    if observed_runtime != expected_runtime:
        raise RuntimeError("trusted frozen runtime sources changed after preflight")
    if observed_execution != expected_execution:
        raise RuntimeError("trusted task execution sources changed after preflight")
    if observed != expected:
        raise RuntimeError("trusted MLSys Forge harness sources changed after preflight")
    return {
        "schema_version": 1,
        "status": "passed",
        "attested_preflight_sha256": sha256_file(preflight_path),
        "trusted_runtime_sources": observed_runtime,
        "trusted_execution_sources": observed_execution,
        "mlsysforge_root": str(EXPECTED_MLSYSFORGE_ROOT),
        "mlsysforge_generated_exclusions": list(MLSYSFORGE_GENERATED_EXCLUSIONS),
        "mlsysforge_import_origins": observed_import_origins,
        "mlsysforge_harness_sources": observed,
    }


def run(command: list[str], *, timeout: int = 120) -> str:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"preflight command failed ({completed.returncode}): {command[0]}"
        )
    return completed.stdout


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("kvbench_v22", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_nonce_source(
    path: Path,
    expected_commitment: str = EXPECTED_NONCE_COMMITMENT,
) -> int:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("hidden nonce source is unavailable or unsafe")
    token_pattern = re.compile(r"(?<![0-9a-f])([0-9a-f]{64})(?![0-9a-f])")
    matches: set[str] = set()
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            for token in token_pattern.findall(line):
                if hashlib.sha256(token.encode()).hexdigest() == expected_commitment:
                    matches.add(token)
    if len(matches) != 1:
        raise RuntimeError("nonce source must contain exactly one committed value")
    return len(matches)


def normalize_backend_url(raw: str) -> str:
    if raw != raw.strip():
        raise RuntimeError("vLLM base URL must not contain surrounding whitespace")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("vLLM base URL is invalid") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            "vLLM base URL must be a credential-free HTTP(S) origin"
        )
    if port is not None and not 1 <= port <= 65535:  # pragma: no cover
        raise RuntimeError("vLLM base URL has an invalid port")
    return raw.rstrip("/")


def _backend_get(url: str, path: str) -> bytes:
    request = Request(
        f"{url}{path}",
        headers={"Accept": "application/json", "User-Agent": "kvbench-preflight/1"},
    )
    try:
        # The serving URL is an operator-selected task-local origin. Host-wide
        # HTTP(S)_PROXY settings belong to Codex egress and must never redirect
        # this trust probe through the external proxy.
        with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"backend {path} returned HTTP {response.status}")
            return response.read(4 * 1024 * 1024)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"backend {path} is unreachable") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _serving_system_identity(contract: dict[str, Any], label: str) -> str:
    try:
        projected = {key: contract[key] for key in SERVING_SYSTEM_IDENTITY_KEYS}
    except KeyError as exc:
        raise RuntimeError(f"{label} lacks a serving-system section") from exc
    runtime = projected.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError(f"{label} runtime section is invalid")
    projected["runtime"] = {
        key: value
        for key, value in runtime.items()
        if key not in RUNTIME_IMAGE_BINDING_KEYS
    }
    return _canonical_sha256(projected)


def validate_runtime_image_recipe(contract: dict[str, Any]) -> dict[str, Any]:
    """Bind the backend runtime to its build recipe, not to one host's image.

    ``runtime.runtime_image_recipe`` names the digest-pinned base image and
    the in-tree Dockerfile; ``runtime.runtime_image_id`` must be the sha256 of
    that recipe, and the Dockerfile on disk must hash to the recipe's
    ``dockerfile_sha256``.  A rebuilt backend therefore carries the same
    identity on every host; the Docker image ID it gets is recorded as
    provenance by :func:`observed_runtime_image` and never gated on.
    """

    runtime = contract.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("backend serving contract runtime section is invalid")
    recipe = runtime.get("runtime_image_recipe")
    if not isinstance(recipe, dict) or recipe.get("kind") != "docker-build-recipe-v1":
        raise RuntimeError("backend serving contract lacks a runtime image recipe")
    for key in (
        "dockerfile",
        "dockerfile_sha256",
        "base_image",
        "base_image_digest",
        "vllm_package_version",
        "vllm_engine_version",
    ):
        if not isinstance(recipe.get(key), str) or not recipe[key]:
            raise RuntimeError(f"runtime image recipe lacks {key}")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", recipe["base_image_digest"]) is None:
        raise RuntimeError("runtime image recipe base image is not digest-pinned")
    if (
        recipe["vllm_package_version"] != runtime.get("vllm_package_version")
        or recipe["vllm_engine_version"] != runtime.get("vllm_engine_version")
    ):
        raise RuntimeError("runtime image recipe disagrees with the runtime section")
    if runtime.get("runtime_image_id") != "sha256:" + _canonical_sha256(recipe):
        raise RuntimeError("runtime image identity is not the recipe digest")
    dockerfile = ROOT / recipe["dockerfile"]
    if dockerfile != BACKEND_DOCKERFILE or not dockerfile.is_file() or dockerfile.is_symlink():
        raise RuntimeError("runtime image recipe names an unexpected Dockerfile")
    if hashlib.sha256(dockerfile.read_bytes()).hexdigest() != recipe["dockerfile_sha256"]:
        raise RuntimeError("backend Dockerfile changed; the runtime image recipe is stale")
    if f"FROM {recipe['base_image']}@{recipe['base_image_digest']}" not in dockerfile.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("backend Dockerfile does not build from the recipe's base image")
    return recipe


def observed_runtime_image(contract: dict[str, Any]) -> dict[str, Any]:
    """Record which local image, if any, carries the recipe's tag.

    The backend runs outside the harness, so the launcher cannot prove which
    image serves the endpoint; it records the ID behind the recipe's tag as
    measurement provenance for the operator's report and leaves it out of
    every gate.
    """

    reference = str(contract.get("runtime", {}).get("runtime_image_ref") or "")
    record: dict[str, Any] = {"runtime_image_ref": reference, "observed_image_id": None}
    if not reference:
        return record
    try:
        observed = run(["docker", "image", "inspect", "--format", "{{.Id}}", reference]).strip()
    except Exception:  # noqa: BLE001 - provenance only, never a gate
        return record
    if re.fullmatch(r"sha256:[0-9a-f]{64}", observed):
        record["observed_image_id"] = observed
    return record


def validate_backend_serving_contract_revisions() -> str:
    """Bind both accepted contract revisions to one measured serving system.

    The shipped revision must be the current one, the retained v1 file must
    still hash to the legacy digest, and the two must be identical across every
    field that determines what is measured.  That is what makes it safe for a
    Direct anchor recorded under v1 to pair with a leg measured under v2.5:
    only the accelerator declaration changed, from a device-model whitelist to
    a capability predicate.
    """

    current = _load_json_object(BACKEND_SERVING_CONTRACT, "backend serving contract")
    legacy = _load_json_object(
        LEGACY_BACKEND_SERVING_CONTRACT, "legacy backend serving contract"
    )
    previous = _load_json_object(
        PREVIOUS_BACKEND_SERVING_CONTRACT, "previous backend serving contract"
    )
    current_digest = _canonical_sha256(current)
    if current_digest != EXPECTED_BACKEND_SERVING_CONTRACT_SHA256:
        raise RuntimeError("backend serving contract changed")
    if _canonical_sha256(legacy) != LEGACY_BACKEND_SERVING_CONTRACT_SHA256:
        raise RuntimeError("retained legacy backend serving contract changed")
    if _canonical_sha256(previous) != PREVIOUS_BACKEND_SERVING_CONTRACT_SHA256:
        raise RuntimeError("retained previous backend serving contract changed")
    identity = _serving_system_identity(current, "backend serving contract")
    if identity != _serving_system_identity(legacy, "legacy backend serving contract") or (
        identity != _serving_system_identity(previous, "previous backend serving contract")
    ):
        raise RuntimeError(
            "backend serving contract revisions describe different serving systems"
        )
    if current.get("contract_revision") != "v3-recipe-bound-runtime":
        raise RuntimeError("backend serving contract is not the recipe-bound revision")
    validate_runtime_image_recipe(current)
    capability = current.get("accelerator_capability")
    if (
        not isinstance(capability, dict)
        or "allowed_names" in capability
        or "compute_capability" in capability
        or capability.get("count") != 2
        or capability.get("min_compute_capability") != "8.0"
        or capability.get("minimum_memory_mib") != 95000
    ):
        raise RuntimeError(
            "backend serving contract must state a capability predicate, "
            "not a device-model whitelist"
        )
    return current_digest


def live_page_geometry(raw_url: str) -> dict[str, Any]:
    """Check the frozen page geometry against the live backend, when it says.

    ``raw_gpu_pages`` and ``attention_tokens_per_program_page`` in the public
    runtime contract are measurements of one backend boot, not task constants.
    vLLM publishes the authoritative values on ``vllm:cache_config_info``, so a
    backend that exposes ``/metrics`` is asked directly and any disagreement
    stops the launcher: the measurement profile has to be refreshed before the
    contract handed to the agent can be trusted.  A backend that does not
    expose the endpoint leaves the geometry as unverified profile provenance,
    which the report records rather than silently accepting.
    """

    contract = _load_json_object(PUBLIC_RUNTIME_CONTRACT, "public runtime contract")
    cache = contract.get("hybrid_cache")
    declared_pages = cache.get("raw_gpu_pages") if isinstance(cache, dict) else None
    declared_tokens = _oracle_integer_default(
        ROOT / "oracle/reference_gateway/gateway.py",
        constant_name="KV_CAPACITY_TOKENS",
        environment_name="THUNDER_KV_CAPACITY_TOKENS",
    )
    record: dict[str, Any] = {
        "declared_raw_gpu_pages": declared_pages,
        "declared_logical_kv_token_capacity": declared_tokens,
        "source": "vllm:cache_config_info",
        "verified_against_live_backend": False,
        "observed_num_gpu_blocks": None,
        "observed_kv_cache_size_tokens": None,
    }
    try:
        raw = _backend_get(normalize_backend_url(raw_url), "/metrics")
    except RuntimeError:
        return record
    try:
        body = raw.decode("utf-8")
    except UnicodeDecodeError:
        return record
    labels: dict[str, int] = {}
    for line in body.splitlines():
        if not line.startswith("vllm:cache_config_info"):
            continue
        for name in ("num_gpu_blocks", "kv_cache_size_tokens"):
            match = re.search(rf'{name}="(\d+)"', line)
            if match is not None:
                value = int(match.group(1))
                if labels.setdefault(name, value) != value:
                    raise RuntimeError(
                        "live backend reports inconsistent KV cache geometry"
                    )
    if set(labels) != {"num_gpu_blocks", "kv_cache_size_tokens"}:
        return record
    record["observed_num_gpu_blocks"] = labels["num_gpu_blocks"]
    record["observed_kv_cache_size_tokens"] = labels["kv_cache_size_tokens"]
    if (
        labels["num_gpu_blocks"] != declared_pages
        or labels["kv_cache_size_tokens"] != declared_tokens
    ):
        raise RuntimeError(
            "live backend KV page geometry does not match the measurement "
            "profile; refresh the profile and rebuild the Work contract"
        )
    record["verified_against_live_backend"] = True
    return record


def validate_backend_endpoint(raw_url: str) -> dict[str, Any]:
    url = normalize_backend_url(raw_url)
    _backend_get(url, "/health")
    try:
        models = json.loads(_backend_get(url, "/v1/models"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("backend /v1/models returned invalid JSON") from exc
    rows = models.get("data") if isinstance(models, dict) else None
    identifiers = {
        row.get("id")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    } if isinstance(rows, list) else set()
    if "target-model" not in identifiers:
        raise RuntimeError("backend does not expose the required target-model alias")
    digest = validate_backend_serving_contract_revisions()
    return {
        "backend_serving_contract_sha256": digest,
        "endpoint_sha256": hashlib.sha256(url.encode()).hexdigest(),
        "served_model": "target-model",
    }


def validate_corpus() -> str:
    manifest_path = CORPUS_ROOT / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("frozen corpus manifest is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances = manifest.get("instances")
    if (
        manifest.get("corpus_mode") != "heterogeneous-192-distinct"
        or manifest.get("selected_instance_count") != 192
        or not isinstance(instances, list)
        or len(instances) != 192
    ):
        raise RuntimeError("frozen corpus manifest identity changed")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce-source", type=Path)
    parser.add_argument("--vllm-base-url")
    parser.add_argument("--hidden-baseline", type=Path)
    parser.add_argument("--public-baseline", type=Path)
    parser.add_argument("--verify-trusted-sources", type=Path)
    parser.add_argument(
        "--mlsysforge-root",
        type=Path,
        default=EXPECTED_MLSYSFORGE_ROOT,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if (args.nonce_source is None) == (args.verify_trusted_sources is None):
        parser.error(
            "exactly one of --nonce-source or --verify-trusted-sources is required"
        )
    if args.nonce_source is not None and any(
        value is None
        for value in (
            args.vllm_base_url,
            args.hidden_baseline,
            args.public_baseline,
        )
    ):
        parser.error(
            "formal preflight requires --vllm-base-url, --hidden-baseline, and --public-baseline"
        )
    output = args.output.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.verify_trusted_sources is not None:
        payload = verify_trusted_sources(
            args.verify_trusted_sources,
            args.mlsysforge_root,
        )
        _atomic_write_json(output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 0

    if run(["docker", "info", "--format", "{{.ID}}"]).strip() == "":
        raise RuntimeError("Docker daemon identity is empty")
    existing = run(
        [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={PROJECT}",
        ]
    ).split()
    if existing:
        raise RuntimeError("the dedicated validation Compose project is already present")

    assert args.vllm_base_url is not None
    assert args.hidden_baseline is not None
    assert args.public_baseline is not None
    backend = validate_backend_endpoint(args.vllm_base_url)
    os.environ["KVBENCH_VLLM_BASE_URL"] = normalize_backend_url(
        args.vllm_base_url
    )
    os.environ["KVBENCH_HIDDEN_BASELINE_PATH"] = str(
        args.hidden_baseline.resolve(strict=True)
    )
    os.environ["KVBENCH_PUBLIC_BASELINE_PATH"] = str(
        args.public_baseline.resolve(strict=True)
    )
    v22 = load_runner()
    baseline = v22.load_and_validate_baseline("hidden")
    public_baseline = v22.load_and_validate_baseline("public")
    oracle = ROOT / "oracle/reference_gateway"
    oracle_tree = v22.V21.tree_sha256(oracle)
    if oracle_tree != EXPECTED_ORACLE_TREE:
        raise RuntimeError("Oracle source commitment changed")
    # This is a static equality check only.  It exposes no live allocator,
    # workload, or reference-policy state to Work.
    validate_public_runtime_contract()
    # The page geometry in that contract is a measurement of one backend boot.
    # Ask the live backend to confirm it before the agent plans against it.
    page_geometry = live_page_geometry(args.vllm_base_url)

    assert args.nonce_source is not None
    nonce_count = validate_nonce_source(args.nonce_source.resolve(strict=True))
    public_nonce_count = validate_nonce_source(
        args.nonce_source.resolve(strict=True),
        EXPECTED_PUBLIC_NONCE_COMMITMENT,
    )
    corpus_sha256 = validate_corpus()
    image_ids: dict[str, str] = {}
    for image, expected in PINNED_IMAGES.items():
        observed = run(["docker", "image", "inspect", "--format", "{{.Id}}", image]).strip()
        if observed != expected:
            raise RuntimeError(f"pinned nested image identity changed: {image}")
        image_ids[image] = observed
    for image in REQUIRED_DYNAMIC_IMAGES:
        observed = run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image]
        ).strip()
        if re.fullmatch(r"sha256:[0-9a-f]{64}", observed) is None:
            raise RuntimeError(f"required nested image is unavailable: {image}")
        image_ids[image] = observed

    stack_env = {
        **os.environ,
        "KVBENCH_VLLM_BASE_URL": normalize_backend_url(args.vllm_base_url),
        "KVBENCH_HIDDEN_BASELINE_PATH": str(args.hidden_baseline.resolve(strict=True)),
        "KVBENCH_PUBLIC_BASELINE_PATH": str(args.public_baseline.resolve(strict=True)),
        "KV_AUTHOR_REFERENCE_GATEWAY_DIR": str(oracle),
        "KV_GPU67_V21_OWNER_TOKEN": "mlsysforge-preflight",
    }
    completed = subprocess.run(
        [str(STACK), "check"],
        env=stack_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("frozen validation stack check failed")

    runtime_sources = attest_tree(FROZEN_RUNTIME_SOURCES)
    execution_sources = attest_execution_sources(ROOT)
    mlsysforge_import_origins = validate_mlsysforge_import_origins(
        args.mlsysforge_root
    )
    mlsysforge_sources = attest_mlsysforge_harness(args.mlsysforge_root)
    payload = {
        "schema_version": 1,
        "status": "passed",
        # The identity every later validator pairs against is the revision the
        # selected Direct anchor was measured under, which need not be the
        # revision this checkout ships: v1 named "NVIDIA H100 NVL" explicitly,
        # v2.5 states the same serving system as a capability predicate, and
        # both are accepted.  The live contract is reported separately so an
        # operator can see which revision the backend is actually serving.
        "backend_serving_contract_sha256": baseline[
            "backend_serving_contract_sha256"
        ],
        "live_backend_serving_contract_sha256": backend[
            "backend_serving_contract_sha256"
        ],
        "runtime_image": observed_runtime_image(
            _load_json_object(BACKEND_SERVING_CONTRACT, "backend serving contract")
        ),
        "backend_endpoint_sha256": backend["endpoint_sha256"],
        "backend_served_model": backend["served_model"],
        "backend_page_geometry": page_geometry,
        "baseline_file_sha256": baseline["file_sha256"],
        "baseline_valid_steps_per_min": baseline["direct_valid_steps_per_min"],
        "nonce_commitment_sha256": EXPECTED_NONCE_COMMITMENT,
        "matching_nonce_count": nonce_count,
        "public_baseline_file_sha256": public_baseline["file_sha256"],
        "public_baseline_valid_steps_per_min": public_baseline[
            "direct_valid_steps_per_min"
        ],
        "public_nonce_commitment_sha256": EXPECTED_PUBLIC_NONCE_COMMITMENT,
        "public_matching_nonce_count": public_nonce_count,
        "corpus_manifest_sha256": corpus_sha256,
        "corpus_instances": 192,
        "oracle_tree_sha256": oracle_tree,
        "nested_image_ids": image_ids,
        "validation_project_absent": True,
        "trusted_runtime_sources": runtime_sources,
        "trusted_execution_sources": execution_sources,
        "mlsysforge_root": str(EXPECTED_MLSYSFORGE_ROOT),
        "mlsysforge_generated_exclusions": list(MLSYSFORGE_GENERATED_EXCLUSIONS),
        "mlsysforge_import_origins": mlsysforge_import_origins,
        "mlsysforge_harness_sources": mlsysforge_sources,
    }
    _atomic_write_json(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
