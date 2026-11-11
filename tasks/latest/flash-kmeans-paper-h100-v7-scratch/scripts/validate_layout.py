#!/usr/bin/env python3
"""Statically validate the security- and timing-critical migration layout.

This intentionally uses only the Python standard library so it can run before
the MLSys Forge virtual environment (and PyYAML) is available.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


EXPECTED_TASK_ID = "flash_kmeans_paper_h100_kernel_lab_v7_scratch"
EXPECTED_SUBMIT_PATHS = {"kernels", "FINAL_VERSION", "notes.md"}
EXPECTED_CODEX_CLI_VERSION = "0.148.0"
EXPECTED_EVAL_INTERVAL = 1800
EXPECTED_SUBMISSION_COOLDOWN = 300
EXPECTED_CWD = "/app/workdir/flash_kmeans_lab"
EXPECTED_BASE_IMAGE_KEY = "pytorch271_cu128"
EXPECTED_OFFICIAL_IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel"
PROTECTED_RUNTIME_PREFIXES = {
    "/opt/flash_kmeans_benchmark/": "images/judge/benchmark",
    "/app/workdir/flash_kmeans_lab/": "images/work/workdir/flash_kmeans_lab",
}
PROTECTED_EXACT_SOURCES = {
    "/app/workdir/flash_kmeans_lab/AGENT_TASK.md": "images/work/AGENT_TASK.md",
    "/opt/flash_kmeans_judge/verify_flash_kmeans.py":
        "images/judge/verify_flash_kmeans.py",
    "/opt/flash_kmeans_judge/run_judge.py": "images/judge/run_judge.py",
    "/opt/flash_kmeans_judge/hidden_cases.py": "images/judge/hidden_cases.py",
}
# Dataset assets (assets/manifest.json): the repository copy may be absent; the asset
# store keeps them at <INFRATHON_ASSETS_ROOT>/<task directory>/<asset path>.
DATASET_ASSETS = {"images/judge/hidden_cases.py": "judge/hidden_cases.py"}
REQUIRED_PROTECTED_PATHS = frozenset(
    {
        "/opt/flash_kmeans_benchmark/kernel_bench.py",
        "/opt/flash_kmeans_benchmark/deployment_bench.py",
        "/opt/flash_kmeans_benchmark/benchmark_cases.py",
        "/opt/flash_kmeans_benchmark/source_admission.py",
        "/app/workdir/flash_kmeans_lab/AGENT_TASK.md",
        "/app/workdir/flash_kmeans_lab/README_AGENT.md",
        "/opt/flash_kmeans_judge/verify_flash_kmeans.py",
        "/opt/flash_kmeans_judge/run_judge.py",
        "/opt/flash_kmeans_judge/hidden_cases.py",
    }
)


def _read_json_object(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        errors.append(f"cannot read {path}: {exc}")
        return {}
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON in {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path} must contain a JSON object")
        return {}
    return value


def _dataset_asset_source(project_root: Path, relative: str) -> Path | None:
    """The repository copy of a dataset asset, else the fetched one, else None."""
    local = project_root / relative
    if local.is_file():
        return local
    configured = os.environ.get("INFRATHON_ASSETS_ROOT")
    assets_root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache/infrathon/assets"
    )
    fetched = assets_root / project_root.name / DATASET_ASSETS[relative]
    return fetched if fetched.is_file() else None


def _check_required_assets(project_root: Path, errors: list[str]) -> None:
    required = (
        "images/work/Dockerfile",
        "images/work/public_tests/public_fresh_worker.py",
        "images/work/public_tests/public_test.py",
        "images/work/public_tests/prepare_submit.sh",
        "images/work/public_tests/run_public_tests.sh",
        "images/judge/Dockerfile",
        "images/judge/benchmark/kernel_bench.py",
        "images/judge/benchmark/deployment_bench.py",
        "images/judge/benchmark/benchmark_cases.py",
        "images/judge/benchmark/source_admission.py",
        "images/judge/run_judge.py",
        "images/judge/verify_flash_kmeans.py",
        "images/judge/protected_hashes.txt",
        "judge/run_judge.py",
        "scripts/make_repo_port_archive.sh",
        "scripts/make_host_calibration_archives.sh",
        "scripts/build_torch_fallback_archive.py",
        "scripts/build_official_repo_head_archive.py",
        "scripts/run_fixed_budget.sh",
        "scripts/run_single_gpu_fixed_budget.sh",
        "scripts/restricted_connect_proxy.py",
        "scripts/sforge_isolated_entry.py",
        "host_reference/baseline/torch_fallback.py",
        "host_reference/baseline/generated_task_adapter.py",
        "host_reference/official_oracle/assign_euclid_triton.py",
        "host_reference/official_oracle/centroid_update_triton.py",
        "host_reference/official_oracle/generated_task_adapter.py",
        "host_reference/README.md",
    )
    for relative in required:
        path = project_root / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f"{path}: required regular, non-symlink file is missing")
            continue
        try:
            if path.stat().st_size <= 0:
                errors.append(f"{path}: required file is empty")
        except OSError as exc:
            errors.append(f"cannot stat required file {path}: {exc}")

    canonical_adapter = project_root / "judge/run_judge.py"
    image_adapter = project_root / "images/judge/run_judge.py"
    if canonical_adapter.is_file() and image_adapter.is_file():
        try:
            if _sha256(canonical_adapter) != _sha256(image_adapter):
                errors.append(
                    f"{image_adapter}: judge adapter mirror differs from "
                    f"{canonical_adapter}; run scripts/build_images.sh"
                )
        except OSError as exc:
            errors.append(f"cannot compare judge adapter mirror: {exc}")


def _check_protected_manifest(project_root: Path, errors: list[str]) -> None:
    """Bind the manifest to finalized files in both image build contexts."""
    manifest = project_root / "images/judge/protected_hashes.txt"
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"cannot read protected hash manifest {manifest}: {exc}")
        return

    entries = 0
    paths_seen: set[str] = set()
    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            errors.append(
                f"{manifest}:{lineno}: expected '<sha256> <absolute-runtime-path>'"
            )
            continue
        expected_hash, runtime_path = parts
        entries += 1
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            errors.append(
                f"{manifest}:{lineno}: SHA-256 must be 64 lowercase hex characters"
            )
        if runtime_path in paths_seen:
            errors.append(
                f"{manifest}:{lineno}: duplicate protected path {runtime_path!r}"
            )
        paths_seen.add(runtime_path)

        source_path: Path | None = None
        exact_source = PROTECTED_EXACT_SOURCES.get(runtime_path)
        if exact_source in DATASET_ASSETS:
            source_path = _dataset_asset_source(project_root, exact_source)
            if source_path is None:
                # Not fetched: assets/fetch.sh verifies it against assets/manifest.json.
                continue
        elif exact_source is not None:
            source_path = project_root / exact_source
        else:
            for runtime_prefix, source_root in PROTECTED_RUNTIME_PREFIXES.items():
                if runtime_path.startswith(runtime_prefix):
                    suffix = runtime_path[len(runtime_prefix) :]
                    if (
                        not suffix
                        or suffix.startswith("/")
                        or ".." in Path(suffix).parts
                    ):
                        errors.append(
                            f"{manifest}:{lineno}: unsafe protected path "
                            f"{runtime_path!r}"
                        )
                    else:
                        source_path = project_root / source_root / suffix
                    break
        if source_path is None:
            errors.append(
                f"{manifest}:{lineno}: protected path has no recognized "
                f"build-context source: {runtime_path!r}"
            )
            continue
        if not source_path.is_file() or source_path.is_symlink():
            errors.append(
                f"{manifest}:{lineno}: protected runtime path has no regular "
                f"build-context source: {source_path}"
            )
            continue
        try:
            actual_hash = _sha256(source_path)
        except OSError as exc:
            errors.append(f"cannot hash protected source {source_path}: {exc}")
            continue
        if re.fullmatch(r"[0-9a-f]{64}", expected_hash) and actual_hash != expected_hash:
            errors.append(
                f"{manifest}:{lineno}: hash drift for {source_path}; "
                f"manifest={expected_hash}, actual={actual_hash}"
            )
    missing_paths = sorted(REQUIRED_PROTECTED_PATHS - paths_seen)
    if missing_paths:
        errors.append(
            f"{manifest}: protected hash manifest is missing required paths: "
            f"{missing_paths}"
        )


def _normalise_submission_path(value: str) -> str:
    value = value.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


def _check_task_json(path: Path, errors: list[str]) -> dict[str, Any]:
    task = _read_json_object(path, errors)
    if not task:
        return task

    task_id = task.get("task_id")
    if task_id != EXPECTED_TASK_ID:
        errors.append(
            f"{path}: task_id must be {EXPECTED_TASK_ID!r}, got {task_id!r}"
        )

    raw_paths = task.get("submit_paths")
    if not isinstance(raw_paths, list) or not all(
        isinstance(item, str) for item in raw_paths
    ):
        errors.append(f"{path}: submit_paths must be a list of strings")
    else:
        paths = {_normalise_submission_path(item) for item in raw_paths}
        if paths != EXPECTED_SUBMIT_PATHS or len(raw_paths) != len(paths):
            errors.append(
                f"{path}: submit_paths must be the exact whitelist "
                f"{sorted(EXPECTED_SUBMIT_PATHS)!r}, got {raw_paths!r}"
            )
        for item in paths:
            parts = Path(item).parts
            if (
                not item
                or item in {".", "/"}
                or item.startswith("/")
                or ".." in parts
                or any(char in item for char in "*?[")
            ):
                errors.append(f"{path}: unsafe submit path {item!r}")

    raw_excludes = task.get("submit_exclude")
    if not isinstance(raw_excludes, list) or not all(
        isinstance(item, str) for item in raw_excludes
    ):
        errors.append(f"{path}: submit_exclude must be a list of strings")
    else:
        excludes = {
            _normalise_submission_path(item).lower() for item in raw_excludes
        }
        required_excludes = {".codex", "auth.json", "judge", "tests"}
        missing = sorted(required_excludes - excludes)
        if missing:
            errors.append(
                f"{path}: submit_exclude is missing credential/judge guards: {missing}"
            )

    if task.get("cwd") != EXPECTED_CWD:
        errors.append(
            f"{path}: cwd must stay narrowly scoped to /app/workdir/flash_kmeans_lab"
        )
    if task.get("base_image") != EXPECTED_BASE_IMAGE_KEY:
        errors.append(
            f"{path}: base_image must be {EXPECTED_BASE_IMAGE_KEY!r}, "
            f"got {task.get('base_image')!r}"
        )

    for role in ("work", "judge"):
        section = task.get(role)
        if not isinstance(section, dict):
            errors.append(f"{path}: {role} must be an object")
            continue
        tag = section.get("image_tag")
        if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
            errors.append(f"{path}: {role}.image_tag must be a Docker-safe tag")

    judge = task.get("judge")
    if isinstance(judge, dict):
        if judge.get("parser") != "structured_json":
            errors.append(f"{path}: judge.parser must be structured_json")
        if judge.get("selection") != "valid_then_score":
            errors.append(f"{path}: judge.selection must be valid_then_score")
        if judge.get("score_direction") != "minimize":
            errors.append(f"{path}: judge.score_direction must be minimize")
        if "rescale" in judge or "rescale_calibration" in judge:
            errors.append(
                f"{path}: candidate-only raw latency must not use a baseline "
                "rescale or calibration anchor"
            )
    return task


def _top_level_yaml_scalar(path: Path, key: str, errors: list[str]) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"cannot read {path}: {exc}")
        return None
    pattern = re.compile(rf"^{re.escape(key)}\s*:\s*([^#]+?)\s*(?:#.*)?$")
    for line in lines:
        match = pattern.match(line)
        if match:
            value = match.group(1).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            return value
    errors.append(f"{path}: missing top-level {key}: value")
    return None


def _strip_yaml_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in "'\"":
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == "#" and quote is None:
            return line[:index]
    return line


def _parse_yaml_scalar(text: str) -> Any:
    value = text.strip()
    if value == "{}":
        return {}
    if value == "[]":
        return []
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "~"}:
        return None
    if value and value[0] in "'\"" and value[-1:] == value[0]:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", value):
        return float(value)
    return value


def _parse_simple_yaml_mapping(path: Path, errors: list[str]) -> dict[str, Any]:
    """Parse the mapping-only subset used by the experiment file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"cannot read {path}: {exc}")
        return {}

    root: dict[str, Any] = {}
    # Stack entries are (indent of the mapping's owning key, mapping).
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for lineno, original in enumerate(lines, 1):
        if "\t" in original[: len(original) - len(original.lstrip())]:
            errors.append(f"{path}:{lineno}: tabs are not supported in YAML indentation")
            continue
        uncommented = _strip_yaml_comment(original).rstrip()
        if not uncommented.strip():
            continue
        indent = len(uncommented) - len(uncommented.lstrip(" "))
        content = uncommented.strip()
        if content.startswith("- "):
            errors.append(
                f"{path}:{lineno}: experiment validator expects mappings, not lists"
            )
            continue
        match = re.fullmatch(r"([^:]+):(?:\s*(.*))?", content)
        if not match:
            errors.append(f"{path}:{lineno}: unsupported YAML syntax")
            continue
        key = match.group(1).strip()
        raw_value = (match.group(2) or "").strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            errors.append(f"{path}:{lineno}: inconsistent indentation")
            stack = [(-1, root)]
        parent = stack[-1][1]
        if key in parent:
            errors.append(f"{path}:{lineno}: duplicate YAML key {key!r}")
            continue
        if raw_value:
            parent[key] = _parse_yaml_scalar(raw_value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _effective(defaults: dict[str, Any], task: dict[str, Any], key: str) -> Any:
    return task[key] if key in task else defaults.get(key)


def _check_experiment(
    path: Path,
    task_id: str,
    errors: list[str],
    *,
    expected_timeout: int | None = None,
    expected_model: str | None = None,
    expected_reasoning: str | None = None,
) -> dict[str, Any]:
    """Check an experiment-*.yaml run profile; the model, effort and timeout
    are compared only when given (from the profile's run wrapper)."""
    config = _parse_simple_yaml_mapping(path, errors)
    if not config:
        return config
    defaults = config.get("defaults")
    tasks = config.get("tasks")
    if not isinstance(defaults, dict):
        errors.append(f"{path}: defaults must be a mapping")
        defaults = {}
    if not isinstance(tasks, dict):
        errors.append(f"{path}: tasks must be a non-empty mapping")
        tasks = {}
    if set(tasks) != {task_id}:
        errors.append(
            f"{path}: tasks must contain exactly {task_id!r}; got {sorted(tasks)!r}"
        )
    task_override = tasks.get(task_id)
    if not isinstance(task_override, dict):
        task_override = {}

    expected = {
        "agent": "codex",
        "eval_interval": EXPECTED_EVAL_INTERVAL,
        "max_submissions": None,
        "submission_cooldown": EXPECTED_SUBMISSION_COOLDOWN,
        "disable_stop_hook": False,
        "disable_auto_eval": True,
        "disable_auto_resume": False,
    }
    if expected_timeout is not None:
        expected["timeout"] = expected_timeout
    for key, wanted in expected.items():
        actual = _effective(defaults, task_override, key)
        if actual != wanted:
            errors.append(
                f"{path}: effective {key} must be {wanted!r}, got {actual!r}"
            )

    model = _effective(defaults, task_override, "model")
    if model is None and isinstance(config.get("model"), dict):
        model = config["model"].get("model")
    if expected_model is not None and model != expected_model:
        errors.append(
            f"{path}: effective model must be {expected_model!r}, got {model!r}"
        )

    env = config.get("env")
    effort = env.get("SFORGE_CODEX_REASONING_EFFORT") if isinstance(env, dict) else None
    if expected_reasoning is not None and effort != expected_reasoning:
        errors.append(
            f"{path}: SFORGE_CODEX_REASONING_EFFORT must be "
            f"{expected_reasoning!r}, got {effort!r}"
        )
    cli_version = (
        env.get("SFORGE_CODEX_CLI_VERSION")
        if isinstance(env, dict)
        else None
    )
    if cli_version != EXPECTED_CODEX_CLI_VERSION:
        errors.append(
            f"{path}: SFORGE_CODEX_CLI_VERSION must be "
            f"{EXPECTED_CODEX_CLI_VERSION!r}, got {cli_version!r}"
        )

    # A committed key or credential path would undermine the host-only auth
    # contract even if it happened to be syntactically valid YAML.
    model_section = config.get("model")
    if isinstance(model_section, dict) and model_section.get("api_key"):
        errors.append(f"{path}: model.api_key must not be committed")
    text = path.read_text(encoding="utf-8")
    if re.search(r"(?im)^\s*(?:CODEX_AUTH_JSON_PATH|SFORGE_CODEX_AUTH_JSON_PATH)\s*:", text):
        errors.append(f"{path}: auth JSON paths belong only in the host launcher")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_regular_files(root: Path, errors: list[str]) -> Iterable[Path]:
    if not root.is_dir():
        errors.append(f"work image context is not a directory: {root}")
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            try:
                target = path.resolve(strict=True)
            except OSError as exc:
                errors.append(f"{path}: broken symlink: {exc}")
                continue
            try:
                target.relative_to(root.resolve())
            except ValueError:
                errors.append(f"{path}: symlink escapes the work build context")
                continue
        elif path.is_file():
            files.append(path)
    return files


def _sensitive_hashes(project_root: Path) -> dict[str, Path]:
    candidates = [
        project_root / "images/judge/verify_flash_kmeans.py",
        project_root / "images/judge/protected_hashes.txt",
        project_root / "images/judge/run_judge.py",
        project_root / "judge/run_judge.py",
    ]
    hidden_cases = _dataset_asset_source(project_root, "images/judge/hidden_cases.py")
    if hidden_cases is not None:
        candidates.append(hidden_cases)
    judge_benchmark = project_root / "images/judge/benchmark"
    if judge_benchmark.is_dir():
        candidates.extend(
            path for path in judge_benchmark.rglob("*") if path.is_file()
        )
    host_reference = project_root / "host_reference"
    if host_reference.is_dir():
        candidates.extend(
            path for path in host_reference.rglob("*") if path.is_file()
        )
    hashes: dict[str, Path] = {}
    for path in candidates:
        if path.is_file() and path.stat().st_size:
            hashes[_sha256(path)] = path
    return hashes


def _check_work_context(
    project_root: Path, work_context: Path, errors: list[str]
) -> None:
    sensitive = _sensitive_hashes(project_root)
    forbidden_names = {
        "verify_flash_kmeans.py",
        "protected_hashes.txt",
        "run_judge.py",
        "flash_kmeans_v002.py",
        "torch_fallback.py",
        "credentials.json",
    }
    credential_name = re.compile(r"(?:^|[._-])auth\.json$", re.IGNORECASE)
    credential_content = re.compile(
        rb'(?i)(?:"(?:access_token|refresh_token|api_key)"\s*:|'
        rb'\bsk-[A-Za-z0-9_-]{20,})'
    )

    files = list(_iter_regular_files(work_context, errors))
    legacy_benchmark = work_context / "benchmark"
    if legacy_benchmark.exists():
        errors.append(
            f"{legacy_benchmark}: formal benchmark/reference code must not "
            "exist in the Work build context"
        )
    for path in files:
        relative = path.relative_to(work_context)
        lowered_parts = {part.lower() for part in relative.parts}
        name = path.name.lower()
        if (
            lowered_parts.intersection({".codex", "judge", "oracle", "hidden"})
            or name in forbidden_names
            or name == "auth.json"
            or credential_name.search(name)
        ):
            errors.append(f"{path}: forbidden hidden/judge/solution/auth path in work context")
            continue
        try:
            file_hash = _sha256(path)
        except OSError as exc:
            errors.append(f"cannot hash {path}: {exc}")
            continue
        if file_hash in sensitive:
            errors.append(
                f"{path}: duplicates private asset {sensitive[file_hash]}"
            )
        try:
            if path.stat().st_size <= 2 * 1024 * 1024:
                content = path.read_bytes()
                if credential_content.search(content):
                    errors.append(f"{path}: looks like embedded authentication material")
        except OSError as exc:
            errors.append(f"cannot inspect {path}: {exc}")

    public_test = work_context / "public_tests/public_test.py"
    try:
        public_source = public_test.read_text(encoding="utf-8")
        public_tree = ast.parse(public_source, filename=str(public_test))
    except (OSError, SyntaxError) as exc:
        errors.append(f"cannot parse implementation-free public test: {exc}")
    else:
        allowed_import_roots = {
            "__future__",
            "argparse",
            "importlib",
            "json",
            "math",
            "os",
            "pathlib",
            "re",
            "shutil",
            "signal",
            "statistics",
            "subprocess",
            "sys",
            "tempfile",
            "time",
            "torch",
            "types",
            "typing",
        }
        imported_roots: set[str] = set()
        for node in ast.walk(public_tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        unexpected_imports = sorted(imported_roots - allowed_import_roots)
        if unexpected_imports:
            errors.append(
                f"{public_test}: public tests import non-harness modules "
                f"{unexpected_imports}"
            )
        forbidden_definitions = {
            "assign",
            "update",
            "baseline_assign",
            "baseline_update",
            "reference_assign",
            "reference_update",
            "standard_blockwise_assign",
            "standard_scatter_update",
        }
        defined = {
            node.name
            for node in ast.walk(public_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        leaked = sorted(defined & forbidden_definitions)
        if leaked:
            errors.append(
                f"{public_test}: public tests define candidate/reference "
                f"implementation functions: {leaked}"
            )
        forbidden_calls = {
            "torch.argmin",
            "torch.bincount",
            "torch.bmm",
            "torch.cdist",
            "torch.einsum",
            "torch.matmul",
            "torch.mm",
            "torch.scatter_add",
        }
        seen_calls: set[str] = set()
        seen_attributes: set[str] = set()
        for node in ast.walk(public_tree):
            if isinstance(node, ast.Attribute):
                seen_attributes.add(node.attr)
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            if isinstance(target, ast.Attribute) and isinstance(
                target.value, ast.Name
            ):
                seen_calls.add(f"{target.value.id}.{target.attr}")
        leaked_calls = sorted(seen_calls & forbidden_calls)
        if leaked_calls:
            errors.append(
                f"{public_test}: public tests contain K-Means solution "
                f"primitives instead of golden outputs: {leaked_calls}"
            )
        leaked_methods = sorted(
            seen_attributes
            & {
                "argmin",
                "bincount",
                "cdist",
                "index_add",
                "index_add_",
                "scatter_add",
                "scatter_add_",
                "scatter_reduce",
                "scatter_reduce_",
            }
        )
        if leaked_methods:
            errors.append(
                f"{public_test}: public tests contain K-Means solution "
                f"methods instead of literal golden outputs: {leaked_methods}"
            )
        for fragment in (
            "standard_blockwise_assign",
            "standard_scatter_update",
            "reference_assign",
            "reference_update",
            "paired_timing_reference",
        ):
            if fragment in public_source:
                errors.append(
                    f"{public_test}: public test leaks private implementation "
                    f"fragment {fragment!r}"
                )

    public_worker = work_context / "public_tests/public_fresh_worker.py"
    try:
        worker_source = public_worker.read_text(encoding="utf-8")
        worker_tree = ast.parse(worker_source, filename=str(public_worker))
    except (OSError, SyntaxError) as exc:
        errors.append(f"cannot parse public fresh-process worker: {exc}")
    else:
        allowed_worker_imports = {
            "__future__",
            "argparse",
            "ctypes",
            "hashlib",
            "importlib",
            "json",
            "math",
            "os",
            "pathlib",
            "signal",
            "statistics",
            "time",
            "torch",
            "types",
            "typing",
        }
        imported_roots: set[str] = set()
        for node in ast.walk(worker_tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        unexpected = sorted(imported_roots - allowed_worker_imports)
        if unexpected:
            errors.append(
                f"{public_worker}: worker imports non-harness modules {unexpected}"
            )
        defined = {
            node.name
            for node in ast.walk(worker_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        leaked = sorted(
            defined
            & {
                "assign",
                "update",
                "baseline_assign",
                "baseline_update",
                "standard_blockwise_assign",
                "standard_scatter_update",
            }
        )
        if leaked:
            errors.append(
                f"{public_worker}: worker defines reusable candidate/baseline "
                f"entrypoints {leaked}"
            )
        for fragment in (
            "hidden_deployment_",
            "official_oracle",
            "host_reference",
            "paired_timing_reference",
        ):
            if fragment in worker_source:
                errors.append(
                    f"{public_worker}: worker leaks private/host fragment {fragment!r}"
                )

    dockerfile = work_context / "Dockerfile"
    if dockerfile.is_file():
        text = dockerfile.read_text(encoding="utf-8")
        allowed_copy_sources = {
            "AGENT_TASK.md",
            "codex-config.toml",
            "public_tests/prepare_submit.sh",
            "public_tests/",
            "workdir/",
        }
        for lineno, line in enumerate(text.splitlines(), 1):
            match = re.match(r"(?i)^\s*(?:COPY|ADD)\s+(?:--\S+\s+)*(\S+)", line)
            if not match:
                continue
            source = match.group(1).strip("'\"")
            if source.startswith(("/", "../")) or "/../" in source:
                errors.append(
                    f"{dockerfile}:{lineno}: COPY/ADD source escapes the work context"
                )
            elif source not in allowed_copy_sources:
                errors.append(
                    f"{dockerfile}:{lineno}: unexpected Work COPY/ADD source "
                    f"{source!r}; only task files and implementation-free "
                    "public tests may enter the image"
                )

    codex_config = work_context / "codex-config.toml"
    try:
        codex_text = codex_config.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read clean-room Codex config {codex_config}: {exc}")
    else:
        if not re.search(
            r'(?m)^\s*web_search\s*=\s*["\']live["\']\s*$',
            codex_text,
        ):
            errors.append(
                f"{codex_config}: web_search must be live"
            )
        if dockerfile.is_file() and "codex-config.toml" not in dockerfile.read_text(
            encoding="utf-8"
        ):
            errors.append(
                f"{dockerfile}: clean-room Codex config is not copied into the image"
            )


def _check_dockerignore(project_root: Path, errors: list[str]) -> None:
    required = {
        "images/work/.dockerignore": {
            "**",
            "**/__pycache__/",
            "**/*.pyc",
            "!Dockerfile",
            "!AGENT_TASK.md",
            "!codex-config.toml",
            "!public_tests/**",
            "!workdir/**",
        },
        "images/judge/.dockerignore": {
            "**",
            "**/__pycache__/",
            "**/*.pyc",
            "!Dockerfile",
            "!verify_flash_kmeans.py",
            "!protected_hashes.txt",
            "!run_judge.py",
            "!benchmark/**",
        },
    }
    for relative, expected in required.items():
        path = project_root / relative
        try:
            ordered_patterns = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            patterns = set(ordered_patterns)
        except OSError as exc:
            errors.append(f"cannot read build-context allowlist {path}: {exc}")
            continue
        missing = sorted(expected - patterns)
        if missing:
            errors.append(f"{path}: missing required allowlist patterns {missing}")
            continue
        last_reinclude = max(
            (
                index
                for index, pattern in enumerate(ordered_patterns)
                if pattern.startswith("!")
            ),
            default=-1,
        )
        for bytecode_pattern in ("**/__pycache__/", "**/*.pyc"):
            if ordered_patterns.index(bytecode_pattern) <= last_reinclude:
                errors.append(
                    f"{path}: {bytecode_pattern} must follow every re-include "
                    "so generated bytecode cannot re-enter the build context"
                )


def _shell_assignment(text: str, name: str) -> str | None:
    match = re.search(
        rf"(?m)^\s*{re.escape(name)}=(?P<quote>['\"])(?P<value>.*?)(?P=quote)\s*$",
        text,
    )
    return match.group("value") if match else None


def _expand_known_shell(value: str, variables: dict[str, str]) -> str:
    result = value
    for _ in range(10):
        changed = False
        for name, replacement in variables.items():
            updated = result.replace("${" + name + "}", replacement)
            changed = changed or updated != result
            result = updated
        if not changed:
            break
    return result


def _check_image_names(
    project_root: Path,
    task: dict[str, Any],
    benchmark_path: Path,
    errors: list[str],
) -> None:
    benchmark_name = _top_level_yaml_scalar(benchmark_path, "name", errors)
    task_id = task.get("task_id")
    work = task.get("work")
    judge = task.get("judge")
    if (
        not benchmark_name
        or not isinstance(task_id, str)
        or not isinstance(work, dict)
        or not isinstance(judge, dict)
        or not isinstance(work.get("image_tag"), str)
        or not isinstance(judge.get("image_tag"), str)
    ):
        return
    expected_work = f"{benchmark_name}.work.{task_id}:{work['image_tag']}"
    expected_judge = f"{benchmark_name}.judge.{task_id}:{judge['image_tag']}"

    lib_path = project_root / "scripts/lib.sh"
    try:
        lib_text = lib_path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read {lib_path}: {exc}")
        return
    declared_task = _shell_assignment(lib_text, "TASK_ID")
    if declared_task != task_id:
        errors.append(
            f"{lib_path}: TASK_ID must be {task_id!r}, got {declared_task!r}"
        )
    variables = {"TASK_ID": task_id}
    declared_work = _shell_assignment(lib_text, "WORK_IMAGE")
    declared_judge = _shell_assignment(lib_text, "JUDGE_IMAGE")
    if declared_work is not None:
        declared_work = _expand_known_shell(declared_work, variables)
    if declared_judge is not None:
        declared_judge = _expand_known_shell(declared_judge, variables)
    if declared_work != expected_work:
        errors.append(
            f"{lib_path}: WORK_IMAGE must be {expected_work!r}, got {declared_work!r}"
        )
    if declared_judge != expected_judge:
        errors.append(
            f"{lib_path}: JUDGE_IMAGE must be {expected_judge!r}, got {declared_judge!r}"
        )

    judge_dockerfile = project_root / "images/judge/Dockerfile"
    try:
        judge_text = judge_dockerfile.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read {judge_dockerfile}: {exc}")
        return
    match = re.search(r"(?m)^\s*ARG\s+WORK_IMAGE=(\S+)\s*$", judge_text)
    default_parent = match.group(1) if match else None
    if default_parent != expected_work:
        errors.append(
            f"{judge_dockerfile}: default WORK_IMAGE must be "
            f"{expected_work!r}, got {default_parent!r}"
        )


def _check_base_image(
    project_root: Path, benchmark_path: Path, errors: list[str]
) -> None:
    try:
        benchmark_text = benchmark_path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read {benchmark_path}: {exc}")
        return
    key_pattern = re.compile(
        rf"(?m)^\s{{2}}{re.escape(EXPECTED_BASE_IMAGE_KEY)}:\s*$"
    )
    image_pattern = re.compile(
        rf"(?m)^\s{{4}}official_image:\s*"
        rf"{re.escape(EXPECTED_OFFICIAL_IMAGE)}\s*$"
    )
    if not key_pattern.search(benchmark_text):
        errors.append(
            f"{benchmark_path}: missing base image key "
            f"{EXPECTED_BASE_IMAGE_KEY!r}"
        )
    if not image_pattern.search(benchmark_text):
        errors.append(
            f"{benchmark_path}: official_image must be "
            f"{EXPECTED_OFFICIAL_IMAGE!r}"
        )

    dockerfile = project_root / "images/work/Dockerfile"
    try:
        docker_text = dockerfile.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read {dockerfile}: {exc}")
        return
    from_match = re.search(r"(?im)^\s*FROM\s+(\S+)", docker_text)
    actual = from_match.group(1) if from_match else None
    if actual != EXPECTED_OFFICIAL_IMAGE:
        errors.append(
            f"{dockerfile}: first FROM image must match BENCHMARK.yaml "
            f"({EXPECTED_OFFICIAL_IMAGE!r}), got {actual!r}"
        )


def _shell_default(text: str, name: str) -> str | None:
    value = _shell_assignment(text, name)
    if value is None:
        return None
    match = re.fullmatch(r"\$\{" + re.escape(name) + r":-([^}]+)\}", value)
    return match.group(1) if match else value


# Maintainer run wrappers: scripts/run_<name>.sh pinning one model, effort and
# budget and exec-ing run_fixed_budget.sh, plus their launch_<name>_tmux.sh
# launchers and experiment-<name>.yaml profiles. A released tree may omit them;
# those present must pin exactly what they default to.
RUN_WRAPPER_EXEC = 'exec "${SCRIPT_DIR}/run_fixed_budget.sh"'
RUN_WRAPPER_PINNED = (
    "MODEL",
    "REASONING_EFFORT",
    "CODEX_CLI_VERSION",
    "TASK_TIMEOUT",
    "EVAL_INTERVAL",
    "SUBMISSION_COOLDOWN",
)
RUN_WRAPPER_FIXED = {
    "CODEX_CLI_VERSION": EXPECTED_CODEX_CLI_VERSION,
    "EVAL_INTERVAL": str(EXPECTED_EVAL_INTERVAL),
    "SUBMISSION_COOLDOWN": str(EXPECTED_SUBMISSION_COOLDOWN),
}
DETACHED_RUN_FRAGMENTS = (
    '[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]',
    "launcher_exit=",
    'launch-${run_id}.exit',
    "bash -o pipefail -c",
    "exit_tmp=",
    "mv -f --",
    'echo "final exit:',
)
DETACHED_CHAIN_FRAGMENTS = ("launcher_exit=", "bash -o pipefail -c", "mv -f --")


def run_wrappers(project_root: Path) -> dict[str, str]:
    """The run wrappers present, by file name."""
    wrappers: dict[str, str] = {}
    for path in sorted((project_root / "scripts").glob("run_*.sh")):
        if path.name == "run_fixed_budget.sh" or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if RUN_WRAPPER_EXEC in text and _shell_assignment(text, "MODEL") is not None:
            wrappers[path.name] = text
    return wrappers


def _check_run_wrappers(project_root: Path, errors: list[str]) -> None:
    for name, text in run_wrappers(project_root).items():
        path = project_root / "scripts" / name
        for variable in RUN_WRAPPER_PINNED:
            default = _shell_default(text, variable)
            if default is None:
                errors.append(f"{path}: {variable} default is missing")
                continue
            wanted = RUN_WRAPPER_FIXED.get(variable)
            if wanted is not None and default != wanted:
                errors.append(
                    f"{path}: {variable} default must be {wanted!r}, got {default!r}"
                )
            pin = f'[[ "${{{variable}}}" == "{default}" ]]'
            if pin not in text:
                errors.append(f"{path}: missing required run pin {pin!r}")


def _check_launcher(project_root: Path, errors: list[str]) -> None:
    common_path = project_root / "scripts/run_fixed_budget.sh"
    try:
        common_text = common_path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read {common_path}: {exc}")
        return
    launcher_sources = common_text
    required_fragments = (
        "assert_built_images_current",
        '--submission-cooldown "${SUBMISSION_COOLDOWN}"',
        "sforge_isolated_entry.py",
        "restricted_connect_proxy.py",
        "docker_internal_restricted_connect_proxy",
        "docker network create",
        "--internal",
        "--enable-internet",
        "direct_public_egress: false",
        'sidecar_public_egress="--allow-public-egress"',
        '${sidecar_public_egress:+"${sidecar_public_egress}"}',
        "public_connect_egress: $public_connect_egress",
        'export MLSYSFORGE_WORK_INTERNET="${task_internet}"',
        'SFORGE_ADMIN_SECRET',
        'final_result.json',
        'final_snapshot.json',
        'validate_formal_artifacts.py" pre',
        'validate_formal_artifacts.py" post',
        'formal-precheck.json',
        'formal-postcheck.json',
        'final_archive_sha256',
        'status: "passed"',
        'write_command_json_atomically',
        'final-${final_minutes}m-report.json',
        '--final-eval',
        'FLASH_KMEANS_TARGET=H100',
    )
    for fragment in required_fragments:
        if fragment not in launcher_sources:
            errors.append(f"{common_path}: missing required fixed-run fragment {fragment!r}")
    if "--max-submissions" in launcher_sources:
        errors.append(
            f"{common_path}: fixed runner must leave manual submission count unlimited"
        )
    if "--disable-stop-hook" in launcher_sources:
        errors.append(
            f"{common_path}: Codex stop hook must remain enabled to block early completion"
        )
    if re.search(r"collect_curve[^\n]*final_result\.json", launcher_sources):
        errors.append(f"{common_path}: final_result.json cannot be used for the 120m point")
    if "GEMM_CUDA_ARCH" in launcher_sources:
        errors.append(f"{common_path}: stale GEMM_CUDA_ARCH must not enter Flash-KMeans runs")
    if "FLASH_KMEANS_VERIFY_SUITE=smoke" in common_text:
        errors.append(
            f"{common_path}: formal runner must not force the reduced smoke suite"
        )
    for name, expected in (("WORK_GPU", "1"), ("JUDGE_GPU", "2")):
        # The portable branch deliberately uses scheduler-owned labels rather
        # than local GPU indexes.  Keep validating the historical branch's
        # exact defaults without mistaking its earlier single-mode assignment
        # for a change to the legacy contract.
        if "SFORGE_SINGLE_GPU_HANDOFF" in common_text:
            fragment = f'{name}="${{{name}:-{expected}}}"'
            actual = expected if fragment in common_text else None
        else:
            actual = _shell_default(common_text, name)
        if actual != expected:
            errors.append(
                f"{common_path}: {name} default must be {expected!r}, got {actual!r}"
            )

    smoke_path = project_root / "scripts/run_codex_smoke.sh"
    try:
        smoke_text = smoke_path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read {smoke_path}: {exc}")
    else:
        for fragment in (
            "FLASH_KMEANS_VERIFY_SUITE=smoke",
            "FLASH_KMEANS_VERIFY_WARMUP=1",
            "FLASH_KMEANS_VERIFY_ITERS=2",
            "FLASH_KMEANS_VERIFY_REPEATS=1",
            "flash_kmeans_h100_verifier_adapter",
            "candidate_output_suppressed",
            "result_protocol_records",
            "verifier_returncode",
        ):
            if fragment not in smoke_text:
                errors.append(
                    f"{smoke_path}: missing reduced hidden smoke override {fragment!r}"
                )

    image_smoke_path = project_root / "scripts/smoke_test.sh"
    try:
        image_smoke_text = image_smoke_path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read {image_smoke_path}: {exc}")
    else:
        for fragment in (
            'run_public_candidate "official-oracle"',
            "make_host_calibration_archives.sh",
            "/workspace/public_tests/run_public_tests.sh",
            'result.get("authoritative") is not False',
            '"score" in result',
            "all_public_gates_ok",
            "/opt/flash_kmeans_benchmark/kernel_bench.py",
            'assert_invalid_fixture "invalid-source"',
            'assert_invalid_fixture "incorrect-output"',
            "flash_kmeans_h100_verifier_adapter",
            "candidate_output_suppressed",
            "result_protocol_records",
            "verifier_returncode",
        ):
            if fragment not in image_smoke_text:
                errors.append(
                    f"{image_smoke_path}: missing full/negative smoke gate "
                    f"{fragment!r}"
                )


def _check_tmux_launcher(project_root: Path, errors: list[str]) -> None:
    scripts = project_root / "scripts"
    wrappers = run_wrappers(project_root)
    for path in sorted(scripts.glob("launch_*_tmux.sh")):
        runner = "run_" + path.name[len("launch_"):-len("_tmux.sh")] + ".sh"
        if runner in wrappers:
            fragments = (*DETACHED_RUN_FRAGMENTS, runner)
        elif (scripts / runner).is_file():
            fragments = (*DETACHED_CHAIN_FRAGMENTS, runner)
        else:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue
        for fragment in fragments:
            if fragment not in text:
                errors.append(
                    f"{path}: missing required detached-run status fragment "
                    f"{fragment!r}"
                )


def _check_web_search_contract(project_root: Path, errors: list[str]) -> None:
    isolated_path = project_root / "scripts/sforge_isolated_entry.py"
    config_path = project_root / "images/work/codex-config.toml"
    common_path = project_root / "scripts/run_fixed_budget.sh"
    for path, fragments in (
        (isolated_path, ("DockerBackend.create_container",)),
        (config_path, ('web_search = "live"',)),
        (
            common_path,
            (
                'codex_web_search="live"',
                '--arg codex_web_search "${codex_web_search}"',
                "codex_web_search: $codex_web_search",
                "sforge_isolated_entry.py",
            ),
        ),
    ):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue
        for fragment in fragments:
            if fragment not in text:
                errors.append(f"{path}: missing live web-search contract {fragment!r}")
    # The entry takes the Codex/prompt internet flag from the task JSON (via
    # MLSYSFORGE_WORK_INTERNET) and never hard-codes it.
    try:
        isolated_text = isolated_path.read_text(encoding="utf-8")
    except OSError:
        isolated_text = ""
    for fragment in (
        'WORK_INTERNET_ENV = "MLSYSFORGE_WORK_INTERNET"',
        'kwargs["internet"] = _task_internet(kwargs.get("internet", True))',
        "CodexAgent.format_run_cmd = task_internet_format_run_cmd",
        "run_agent.generate_evolve_prompt = task_internet_generate_prompt",
    ):
        if fragment not in isolated_text:
            errors.append(f"{isolated_path}: missing task internet contract {fragment!r}")
    for forced in ('kwargs["internet"] = True', 'kwargs["internet"] = False'):
        if forced in isolated_text:
            errors.append(f"{isolated_path}: must not hard-code {forced!r}")


def _check_formal_artifact_validator(
    project_root: Path, errors: list[str]
) -> None:
    path = project_root / "scripts/validate_formal_artifacts.py"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"cannot read {path}: {exc}")
        return
    required_fragments = (
        "def validate_pre_final(",
        "def validate_final_eval(",
        '"status": "completed"',
        "member.isdir()",
        "member.isfile()",
        "duplicate member",
        "regular, non-symlink file",
        "symlink path component",
        "MAX_ARCHIVE_MEMBERS",
        "MAX_TOTAL_FILE_BYTES",
        "regular-file ancestor conflict",
        "validated_report",
        "independent final report must have valid=true",
        "independent final report.score must be positive",
        "flash_kmeans_h100_verifier_adapter",
        "candidate_output_suppressed",
        "result_protocol_records",
        "verifier_returncode",
        "did not use the pre-validated archive",
    )
    for fragment in required_fragments:
        if fragment not in text:
            errors.append(
                f"{path}: missing required formal-artifact gate {fragment!r}"
            )


def validate_layout(project_root: Path) -> list[str]:
    project_root = project_root.expanduser().resolve()
    errors: list[str] = []
    task_path = project_root / "tasks/flash_kmeans_paper_h100_kernel_lab_v7_scratch.json"
    benchmark_path = project_root / "tasks/BENCHMARK.yaml"
    work_context = project_root / "images/work"

    _check_required_assets(project_root, errors)
    _check_protected_manifest(project_root, errors)
    task = _check_task_json(task_path, errors)
    task_id = task.get("task_id") if isinstance(task.get("task_id"), str) else EXPECTED_TASK_ID
    # The experiment files are maintainer run configurations; check those
    # present, against their run wrapper's pins when it is present too.
    wrappers = run_wrappers(project_root)
    for path in sorted(project_root.glob("experiment-*.yaml")):
        runner = "run_" + path.stem[len("experiment-"):].replace("-", "_") + ".sh"
        kwargs: dict[str, Any] = {}
        if runner in wrappers:
            text = wrappers[runner]
            timeout = _shell_default(text, "TASK_TIMEOUT") or ""
            kwargs = {
                "expected_model": _shell_default(text, "MODEL"),
                "expected_reasoning": _shell_default(text, "REASONING_EFFORT"),
                "expected_timeout": int(timeout) if timeout.isdigit() else None,
            }
        _check_experiment(path, task_id, errors, **kwargs)
    _check_work_context(project_root, work_context, errors)
    _check_dockerignore(project_root, errors)
    _check_base_image(project_root, benchmark_path, errors)
    _check_image_names(project_root, task, benchmark_path, errors)
    _check_launcher(project_root, errors)
    _check_run_wrappers(project_root, errors)
    _check_tmux_launcher(project_root, errors)
    _check_web_search_contract(project_root, errors)
    _check_formal_artifact_validator(project_root, errors)
    return errors


def _build_parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "project_root",
        nargs="?",
        type=Path,
        default=default_root,
        help=f"migration root (default: {default_root})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    errors = validate_layout(args.project_root)
    if errors:
        print(f"layout validation failed ({len(errors)} error(s)):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"layout validation passed: {args.project_root.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
