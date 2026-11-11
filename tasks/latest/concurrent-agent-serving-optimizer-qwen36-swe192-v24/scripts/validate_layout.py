#!/usr/bin/env python3
"""Validate the relocatable, pre-built MLSys Forge task layout.

The validator is deliberately standard-library-only so image builds can fail
closed before either Docker build context is sent to the daemon.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_ID = "concurrent_agent_serving_optimizer_qwen36_swe192_v24"
BENCHMARK_NAME = "kvbench_qwen36"
WORK_IMAGE = f"{BENCHMARK_NAME}.work.{TASK_ID}:local"
JUDGE_IMAGE = f"{BENCHMARK_NAME}.judge.{TASK_ID}:local"
# Task roots from before the task was made relocatable (split literals so this
# file does not match itself).
FORBIDDEN_ABSOLUTE_ROOTS = (
    "MLSys Forge/local_tasks/" "concurrent-agent-serving-optimizer-qwen36-swe192-v22-mlsysforge",
    "MLSys Forge/local_tasks/" "concurrent-agent-serving-optimizer-qwen36-swe192-v24",
)

RUNTIME_STEM = Path(
    "runtime/frozen/harbor_run_scripts/"
    "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45"
)
RUNTIME_V22_STEM = Path(
    "runtime/frozen/harbor_run_scripts/"
    "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.2-gpu45"
)

REQUIRED_FILES = (
    "README.md",
    "pytest.ini",
    "tasks/BENCHMARK.yaml",
    f"tasks/{TASK_ID}.json",
    "images/work/.dockerignore",
    "images/work/Dockerfile",
    "images/work/AGENT_TASK.md",
    "images/work/RUNTIME_CONTRACT.json",
    "images/work/RUN_BUDGET.json",
    "images/work/codex-config.toml",
    "images/work/submission/ARCHITECTURE.md",
    "images/work/submission/gateway.py",
    "images/work/submission/launch.sh",
    "images/work/protocol_tests/protocol_smoke.py",
    "images/work/protocol_tests/run_protocol_smoke.sh",
    "images/work/public_benchmark/kvbench",
    "images/work/public_benchmark/kvbench.py",
    "images/judge/.dockerignore",
    "images/judge/Dockerfile",
    "judge/run_judge.py",
    "judge/run_public.py",
    "scripts/build_images.sh",
    "scripts/eval_archive.sh",
    "scripts/launch_fixed_budget_tmux.sh",
    "scripts/lib.sh",
    "scripts/preflight.py",
    "scripts/public_api_smoke.sh",
    "scripts/run_public_candidate.py",
    "scripts/run_fixed_budget.sh",
    "scripts/public_api_smoke.sh",
    "scripts/sforge_runtime_entry.py",
    "scripts/smoke_test.sh",
    "scripts/status.sh",
    RUNTIME_STEM / "gpu67-v2.1-validation-stack.sh",
    RUNTIME_STEM / "gpu67-v2.1-validation.compose.yaml",
    RUNTIME_STEM / "run-gpu67-v2.1-pair.py",
    RUNTIME_V22_STEM / "run-gpu67-v2.2-candidate-only.py",
    RUNTIME_V22_STEM / "run-gpu67-v2.2-hidden-candidate-only.py",
    "oracle/reference_gateway/THIRD_PARTY_LICENSE.md",
    "oracle/reference_gateway/gateway.py",
    "oracle/reference_gateway/launch.sh",
    "oracle/reference_gateway/scheduler.py",
)

REQUIRED_DIRECTORIES = (
    "tasks",
    "images/work",
    "images/work/submission",
    "images/work/protocol_tests",
    "images/work/public_benchmark",
    "images/judge",
    "judge",
    "scripts",
    "runtime/frozen",
    "oracle/reference_gateway",
)

EXECUTABLE_FILES = (
    "images/work/submission/launch.sh",
    "images/work/protocol_tests/run_protocol_smoke.sh",
    "images/work/public_benchmark/kvbench",
    "images/work/public_benchmark/kvbench.py",
    "scripts/build_images.sh",
    "scripts/eval_archive.sh",
    "scripts/launch_fixed_budget_tmux.sh",
    "scripts/run_fixed_budget.sh",
    "scripts/run_public_candidate.py",
    RUNTIME_STEM / "gpu67-v2.1-validation-stack.sh",
    RUNTIME_STEM / "run-gpu67-v2.1-pair.py",
    RUNTIME_V22_STEM / "run-gpu67-v2.2-candidate-only.py",
    RUNTIME_V22_STEM / "run-gpu67-v2.2-hidden-candidate-only.py",
    "oracle/reference_gateway/launch.sh",
)

# Run profiles and per-run wrapper scripts, checked only when present (a
# released tree may not carry them).
OPTIONAL_FILE_GLOBS = (
    "experiment-*.yaml",
    "scripts/launch_*_tmux.sh",
    "scripts/run_*.sh",
)

# The Judge-only nonce source: the in-repository copy, else the copy that
# assets/fetch.sh places under the asset root (assets/manifest.json path).
NONCE_SOURCE_IN_TREE = "judge/private/nonce_source.txt"
NONCE_SOURCE_ASSET = "judge/nonce_source.txt"


def _assets_root() -> Path:
    value = os.environ.get("INFRATHON_ASSETS_ROOT")
    if value:
        return Path(value).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "infrathon" / "assets"


def resolve_nonce_source(root: Path) -> Path | None:
    for candidate in (
        root / NONCE_SOURCE_IN_TREE,
        _assets_root() / root.name / NONCE_SOURCE_ASSET,
    ):
        if candidate.exists() or candidate.is_symlink():
            return candidate
    return None


SCAN_ROOTS = (
    "README.md",
    "docs",
    "images",
    "judge",
    "oracle",
    "runtime",
    "scripts",
    "tasks",
    "tests",
)


class _DuplicateKeyError(ValueError):
    pass


def _display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _check_regular(path: Path, root: Path, errors: list[str]) -> None:
    label = _display(path, root)
    try:
        info = path.lstat()
    except FileNotFoundError:
        errors.append(f"missing required file: {label}")
        return
    except OSError as exc:
        errors.append(f"cannot stat required file {label}: {exc}")
        return
    if stat.S_ISLNK(info.st_mode):
        errors.append(f"required file must not be a symlink: {label}")
    elif not stat.S_ISREG(info.st_mode):
        errors.append(f"required path is not a regular file: {label}")


def _check_directory(path: Path, root: Path, errors: list[str]) -> None:
    label = _display(path, root)
    try:
        info = path.lstat()
    except FileNotFoundError:
        errors.append(f"missing required directory: {label}")
        return
    except OSError as exc:
        errors.append(f"cannot stat required directory {label}: {exc}")
        return
    if stat.S_ISLNK(info.st_mode):
        errors.append(f"required directory must not be a symlink: {label}")
    elif not stat.S_ISDIR(info.st_mode):
        errors.append(f"required path is not a directory: {label}")


def _read_text(path: Path, root: Path, errors: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read {_display(path, root)} as UTF-8: {exc}")
        return ""


def _read_json(path: Path, root: Path, errors: list[str]) -> dict[str, Any]:
    text = _read_text(path, root, errors)
    if not text:
        return {}

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateKeyError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    def reject_non_finite(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except (json.JSONDecodeError, _DuplicateKeyError, ValueError) as exc:
        errors.append(f"invalid JSON in {_display(path, root)}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{_display(path, root)} must contain a JSON object")
        return {}
    return value


def _yaml_top_scalar(text: str, key: str) -> str | None:
    match = re.search(
        rf"(?m)^{re.escape(key)}\s*:\s*([^#\r\n]+?)\s*(?:#.*)?$", text
    )
    if not match:
        return None
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def _shell_assignment(text: str, name: str) -> str | None:
    match = re.search(
        rf"(?m)^\s*{re.escape(name)}=(?:\"([^\"]*)\"|'([^']*)'|([^\s#]+))\s*(?:#.*)?$",
        text,
    )
    if not match:
        return None
    return next(value for value in match.groups() if value is not None)


def _check_task_contract(root: Path, errors: list[str]) -> None:
    tasks_dir = root / "tasks"
    task_files = sorted(tasks_dir.glob("*.json")) if tasks_dir.is_dir() else []
    if [path.name for path in task_files] != [f"{TASK_ID}.json"]:
        errors.append(
            "tasks/ must contain exactly one task JSON named "
            f"{TASK_ID}.json; found {[path.name for path in task_files]!r}"
        )
        return

    benchmark_files = (
        sorted(
            path
            for path in tasks_dir.iterdir()
            if path.is_file() and path.name.lower() in {"benchmark.yaml", "benchmark.yml"}
        )
        if tasks_dir.is_dir()
        else []
    )
    if [path.name for path in benchmark_files] != ["BENCHMARK.yaml"]:
        errors.append(
            "tasks/ must contain exactly one benchmark descriptor named "
            f"BENCHMARK.yaml; found {[path.name for path in benchmark_files]!r}"
        )

    task = _read_json(task_files[0], root, errors)
    if task.get("task_id") != TASK_ID:
        errors.append(
            f"tasks/{TASK_ID}.json: task_id must be {TASK_ID!r}, "
            f"got {task.get('task_id')!r}"
        )
    if task.get("name") != (
        "Concurrent Agent Serving Optimizer — Qwen3.6-27B SWE-192 v2.4"
    ):
        errors.append(f"tasks/{TASK_ID}.json: name must identify v2.4")
    if task.get("base_image") != "ubuntu2404":
        errors.append(f"tasks/{TASK_ID}.json: base_image must be 'ubuntu2404'")
    if task.get("cwd") != "/app":
        errors.append(f"tasks/{TASK_ID}.json: cwd must be '/app'")
    if task.get("submit_paths") != ["submission"]:
        errors.append(f"tasks/{TASK_ID}.json: submit_paths must be ['submission']")

    for role in ("work", "judge"):
        section = task.get(role)
        if not isinstance(section, dict):
            errors.append(f"tasks/{TASK_ID}.json: {role} must be an object")
        elif section.get("image_tag") != "local":
            errors.append(
                f"tasks/{TASK_ID}.json: {role}.image_tag must be 'local', "
                f"got {section.get('image_tag')!r}"
            )

    work = task.get("work")
    if isinstance(work, dict):
        query = work.get("agent_query")
        required_query_markers = (
            "/app/AGENT_TASK.md",
            "/app/RUNTIME_CONTRACT.json",
            "/app/RUN_BUDGET.json",
            "/app/submission/ARCHITECTURE.md",
            "workflow-level",
            "online monitoring",
            "harness-reported agent deadline",
        )
        if not isinstance(query, str):
            errors.append(f"tasks/{TASK_ID}.json: work.agent_query must be text")
        else:
            for marker in required_query_markers:
                if marker not in query:
                    errors.append(
                        f"tasks/{TASK_ID}.json: work.agent_query lacks {marker!r}"
                    )
            if "two-hour boundary" in query or "twelve-hour boundary" in query:
                errors.append(
                    f"tasks/{TASK_ID}.json: work.agent_query hard-codes a deadline"
                )

    judge = task.get("judge")
    if isinstance(judge, dict):
        expected = {
            "parser": "structured_json",
            "selection": "valid_then_score",
            "score_direction": "maximize",
            "public_eval_cmd": (
                "python3 /opt/kvbench-mlsysforge/run_public.py "
                "--workspace /app/submission --log-dir /logs/verifier"
            ),
            "public_eval_timeout": 2100,
            "public_private_artifact_path": (
                "/logs/verifier/kvbench-public-private-evidence.json"
            ),
        }
        for key, value in expected.items():
            if judge.get(key) != value:
                errors.append(
                    f"tasks/{TASK_ID}.json: judge.{key} must be {value!r}"
                )

    benchmark_path = root / "tasks/BENCHMARK.yaml"
    benchmark_text = _read_text(benchmark_path, root, errors)
    benchmark = _yaml_top_scalar(benchmark_text, "name")
    if benchmark != BENCHMARK_NAME:
        errors.append(
            f"tasks/BENCHMARK.yaml: name must be {BENCHMARK_NAME!r}, got {benchmark!r}"
        )


def _check_image_contract(root: Path, errors: list[str]) -> None:
    lib_path = root / "scripts/lib.sh"
    lib_text = _read_text(lib_path, root, errors)
    expected_assignments = {
        "TASK_ID": TASK_ID,
        "BENCHMARK_NAME": BENCHMARK_NAME,
        "WORK_IMAGE": "${BENCHMARK_NAME}.work.${TASK_ID}:local",
        "JUDGE_IMAGE": "${BENCHMARK_NAME}.judge.${TASK_ID}:local",
    }
    for key, expected in expected_assignments.items():
        actual = _shell_assignment(lib_text, key)
        if actual != expected:
            errors.append(
                f"scripts/lib.sh: {key} must be {expected!r}, got {actual!r}"
            )

    judge_dockerfile = root / "images/judge/Dockerfile"
    judge_text = _read_text(judge_dockerfile, root, errors)
    match = re.search(r"(?m)^\s*ARG\s+WORK_IMAGE=(\S+)\s*$", judge_text)
    actual_parent = match.group(1) if match else None
    if actual_parent != WORK_IMAGE:
        errors.append(
            "images/judge/Dockerfile must default to the matching v2.4 Work "
            f"image {WORK_IMAGE!r}, got {actual_parent!r}"
        )
    if "FROM ${WORK_IMAGE}" not in judge_text:
        errors.append("images/judge/Dockerfile must inherit FROM ${WORK_IMAGE}")
    for required_copy in (
        "judge/run_public.py",
        "scripts/run_public_candidate.py",
    ):
        if required_copy not in judge_text:
            errors.append(
                f"images/judge/Dockerfile must copy {required_copy}"
            )

    build_text = _read_text(root / "scripts/build_images.sh", root, errors)
    validator_call = '"${MIGRATION_ROOT}/scripts/validate_layout.py"'
    first_build = build_text.find("docker build")
    validation = build_text.find(validator_call)
    if validation < 0 or first_build < 0 or validation > first_build:
        errors.append(
            "scripts/build_images.sh must invoke validate_layout.py before any docker build"
        )
    if '"${MIGRATION_ROOT}/images/work"' not in build_text:
        errors.append("Work docker build context must be ${MIGRATION_ROOT}/images/work")
    for marker in (
        '--build-arg "WORK_IMAGE=${WORK_IMAGE}"',
        '--tag "${JUDGE_IMAGE}"',
        '--tag "${MODEL_API_IMAGE}"',
        '"${NESTED_ENVIRONMENT_ROOT}"',
        '--tag "${CONTROLLER_IMAGE}"',
        '"${NESTED_ENVIRONMENT_ROOT}/bench-controller"',
    ):
        if marker not in build_text:
            errors.append(f"v2.4 image build lacks required build marker: {marker}")


def _check_work_context(root: Path, errors: list[str]) -> None:
    context = root / "images/work"
    forbidden_markers = ("judge", "runtime", "oracle", "nonce", "baseline")
    if context.is_dir() and not context.is_symlink():
        for path in sorted(context.rglob("*")):
            try:
                relative = path.relative_to(context)
            except ValueError:
                errors.append(f"Work build context escapes its root: {path}")
                continue
            lowered = [part.lower() for part in relative.parts]
            if relative.as_posix() == "RUNTIME_CONTRACT.json":
                lowered = []
            hits = [
                marker
                for part in lowered
                for marker in forbidden_markers
                if marker in part
            ]
            if hits:
                errors.append(
                    f"Work build context contains private/reserved path {relative}"
                )
            if path.is_symlink():
                errors.append(f"Work build context must not contain symlinks: {relative}")

    dockerignore = _read_text(root / "images/work/.dockerignore", root, errors)
    required_rules = {
        "**",
        "!Dockerfile",
        "!.dockerignore",
        "!AGENT_TASK.md",
        "!RUNTIME_CONTRACT.json",
        "!RUN_BUDGET.json",
        "!codex-config.toml",
        "!submission/",
        "!submission/**",
        "!protocol_tests/",
        "!protocol_tests/**",
        "!public_benchmark/",
        "!public_benchmark/**",
    }
    rules = {
        line.strip()
        for line in dockerignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(required_rules - rules)
    if missing:
        errors.append(f"images/work/.dockerignore is missing rules: {missing!r}")


def _check_in_tree_inputs(root: Path, errors: list[str]) -> None:
    """The clone-only inputs: corpus definitions, nonce source, backend recipe.

    ``judge/private/nonce_source.txt`` is the Judge-only two-line nonce source;
    the root ``.dockerignore`` is a whitelist that never admits ``judge/private``
    and no Dockerfile may reference it.
    """

    for relative in (
        "corpus/swebench-lite-300-source.json",
        "corpus/swebench-heterogeneous-192-selection.json",
        "scripts/stage_swebench_corpus.py",
        "images/backend/Dockerfile",
    ):
        _check_regular(root / relative, root, errors)
    nonce = resolve_nonce_source(root)
    if nonce is None:
        # Not fetched yet (assets/fetch.sh); the launchers fail closed on it.
        pass
    elif nonce.is_symlink() or not nonce.is_file():
        errors.append("judge/private/nonce_source.txt must be a regular file")
    else:
        lines = nonce.read_text(encoding="utf-8").split("\n")
        if len(lines) != 3 or lines[2] != "" or any(
            re.fullmatch(r"[0-9a-f]{64}", line) is None for line in lines[:2]
        ) or lines[0] == lines[1]:
            errors.append(
                "judge/private/nonce_source.txt must hold exactly two distinct "
                "64-hex lines (hidden, public)"
            )
        # git does not carry file modes, so a fresh clone checks out 0644; the
        # formal launcher copies the values into a dedicated 0600 source
        # (scripts/prepare_nonce_source.py) before anything reads them.
    root_ignore = _read_text(root / ".dockerignore", root, errors)
    rules = [line.strip() for line in root_ignore.splitlines() if line.strip()]
    if not rules or rules[0] != "**" or any(
        rule.startswith("!judge/private") or rule == "!judge/**" for rule in rules
    ):
        errors.append(".dockerignore must be a whitelist that never admits judge/private")
    for dockerfile in sorted(root.glob("images/*/Dockerfile")):
        if "judge/private" in _read_text(dockerfile, root, errors):
            errors.append(f"{dockerfile.relative_to(root)} references judge/private")


def _check_no_old_absolute_path(root: Path, errors: list[str]) -> None:
    starts = [root / entry for entry in SCAN_ROOTS] + sorted(root.glob("experiment-*.yaml"))
    for start in starts:
        paths = [start] if start.is_file() else sorted(start.rglob("*")) if start.is_dir() else []
        for path in paths:
            if not path.is_file() or path.is_symlink():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            try:
                data = path.read_bytes()
            except OSError as exc:
                errors.append(f"cannot scan {_display(path, root)}: {exc}")
                continue
            for forbidden_root in FORBIDDEN_ABSOLUTE_ROOTS:
                if forbidden_root.encode() in data:
                    errors.append(
                        "hardcoded pre-relocation task root in "
                        f"{_display(path, root)}"
                    )


def _check_v24_work_contract(root: Path, errors: list[str]) -> None:
    runtime = _read_json(root / "images/work/RUNTIME_CONTRACT.json", root, errors)
    expected_runtime = {
        ("schema_version",): 1,
        ("runtime", "version"): "0.24.0",
        ("runtime", "tensor_parallel_size"): 2,
        ("runtime", "max_running_sequences"): 192,
        ("runtime", "max_batched_tokens_per_iteration"): 8192,
        ("hybrid_cache", "decision_unit"): "gpu_pages",
        ("hybrid_cache", "raw_gpu_pages"): 1253,
        ("hybrid_cache", "reserved_null_pages"): 1,
        ("hybrid_cache", "allocatable_pages"): 1252,
        ("hybrid_cache", "attention_tokens_per_program_page"): 1568,
        ("hybrid_cache", "recurrent_group_count"): 3,
        ("hybrid_cache", "resident_pages_per_recurrent_group"): 1,
        (
            "hybrid_cache",
            "peak_pages_per_recurrent_group_while_request_pending_or_inflight",
        ): 2,
        ("scope", "static_contract_only"): True,
        ("scope", "contains_live_backend_telemetry"): False,
        ("scope", "contains_workload_identity_or_order"): False,
        ("scope", "contains_reference_policy"): False,
    }
    for keys, expected in expected_runtime.items():
        value: Any = runtime
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if value != expected:
            errors.append(
                "images/work/RUNTIME_CONTRACT.json: "
                f"{'.'.join(keys)} must be {expected!r}, got {value!r}"
            )

    budget = _read_json(root / "images/work/RUN_BUDGET.json", root, errors)
    if budget.get("schema_version") != 1:
        errors.append("images/work/RUN_BUDGET.json: schema_version must be 1")
    if budget.get("expected_hidden_runtime_seconds") != 2400:
        errors.append(
            "images/work/RUN_BUDGET.json: expected hidden runtime must be 2400s"
        )
    if budget.get("agent_timeout_seconds") is not None:
        errors.append(
            "images/work/RUN_BUDGET.json: image placeholder timeout must be null"
        )

    instruction = _read_text(root / "images/work/AGENT_TASK.md", root, errors)
    required_instruction_markers = (
        "workflow-level serving-systems problem",
        "online monitoring and control plane",
        "/app/RUNTIME_CONTRACT.json",
        "/app/RUN_BUDGET.json",
        "/app/submission/ARCHITECTURE.md",
        "REASONING",
        "ACTING",
        "PAUSED",
        "materially different architecture",
        "harness-reported deadline",
    )
    for marker in required_instruction_markers:
        if marker not in instruction:
            errors.append(f"images/work/AGENT_TASK.md lacks {marker!r}")
    for forbidden in (
        "ThunderAgent",
        "THUNDER_",
        "reference_gateway",
        "101.7333",
        "two-hour boundary",
        "twelve-hour boundary",
    ):
        if forbidden in instruction:
            errors.append(
                f"images/work/AGENT_TASK.md leaks or hard-codes {forbidden!r}"
            )

    architecture = _read_text(
        root / "images/work/submission/ARCHITECTURE.md", root, errors
    )
    for marker in (
        "## Resource model",
        "## Program state machine",
        "## Online monitoring",
        "## Admission, scheduling, and restoration policy",
        "## Experiment ledger",
        "## Protected best",
    ):
        if marker not in architecture:
            errors.append(
                f"images/work/submission/ARCHITECTURE.md lacks {marker!r}"
            )


def validate(root: Path) -> dict[str, Any]:
    supplied_root = root.absolute()
    errors: list[str] = []

    if supplied_root.is_symlink():
        errors.append(f"task root must not be a symlink: {supplied_root}")
    root = supplied_root.resolve()
    if not root.is_dir():
        errors.append(f"task root is not a directory: {root}")

    for relative in REQUIRED_DIRECTORIES:
        _check_directory(root / relative, root, errors)
    for pattern in OPTIONAL_FILE_GLOBS:
        for path in sorted(root.glob(pattern)):
            _check_regular(path, root, errors)
    for relative in REQUIRED_FILES:
        _check_regular(root / relative, root, errors)
    _check_in_tree_inputs(root, errors)

    optional_scripts = [
        path.relative_to(root).as_posix()
        for pattern in OPTIONAL_FILE_GLOBS[1:]
        for path in sorted(root.glob(pattern))
    ]
    for relative in (*EXECUTABLE_FILES, *optional_scripts):
        path = root / relative
        if path.is_file() and not path.is_symlink():
            if not (path.stat().st_mode & stat.S_IXUSR):
                errors.append(f"required executable lacks owner execute bit: {relative}")

    _check_task_contract(root, errors)
    _check_image_contract(root, errors)
    _check_work_context(root, errors)
    _check_v24_work_contract(root, errors)
    _check_no_old_absolute_path(root, errors)

    return {
        "schema_version": 1,
        "kind": "mlsysforge-layout-validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "task_id": TASK_ID,
        "benchmark": BENCHMARK_NAME,
        "expected_images": {"work": WORK_IMAGE, "judge": JUDGE_IMAGE},
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
    }


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = validate(args.root)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        _write_atomic(args.output, payload)
    sys.stdout.write(payload)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
