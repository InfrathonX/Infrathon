#!/usr/bin/env python3
"""Root-side artifact, numerical-screening and timing verification for v3."""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import math
import os
import pwd
import re
import secrets
import shutil
import signal
import statistics
import subprocess
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
sys.path.insert(0,str(Path(__file__).resolve().parent / "runtime"))
if not (Path(__file__).resolve().parent/"runtime").is_dir():sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"runtime"))
from source_admission import admit_workspace
from submission_loader import SealedSubmission, seal_approved_submission

PROTOCOL = "minference-real-ruler-prefill-v3"
TRIAL_RANDOMNESS_PROTOCOL = "private-query-row-nonce-v3"
QUERY_HEADS = 32
HEAD_DIM = 128
FINAL_CASE_COUNT = 48
TRIALS_PER_CASE = 3
TASKS = ("niah_single_2", "niah_multikey_1", "vt", "fwe")
FORMAL_LENGTHS = (32768, 65536, 131072)
LAYERS = (1, 4, 15, 27)
WARMUP_LAYER = 2
LIMIT_KEYS = ("nrmse", "head_max_nrmse", "tail_nrmse", "tail_head_max_nrmse")
RESULT_NAME = "minference_prefill_h100_seed_v3_results.json"
FINAL_SUITE_ID = "minference-prefill-final-v3"
PROTECTED_RELATIVE_PATHS = frozenset(['runtime/minference_prefill_runtime.py', 'runtime/activation_archive_v3.py', 'source_admission.py', 'submission_loader.py', 'candidate_worker.py', 'verify_minference_prefill.py', 'run_judge.py', 'scripts/external_assets_on_demand_v3.py', 'activations/recipe/RECIPE.json', 'activations/recipe/RULER_V3_RECIPE.json', 'activations/recipe/candidate_attention.py', 'activations/recipe/quality_v3.py', 'activations/recipe/ruler_v3.py', 'activations/recipe/ruler_reference.py', 'activations/recipe/capture_activations.py', 'runtime/measure_single_prompt_v3.py', 'runtime/prepare_single_prompt_v3.py', 'runtime/stream_parent_v3.py', 'runtime/single_prompt_archive_v3.py', 'runtime/on_demand_suite_v3.py', 'runtime/run_single_prompt_worker_v3.py', 'runtime/kernel_fidelity_v3.py', 'private/final_manifest.json', 'private/numerical_policy.json', 'private/external-assets-contract.json'])
PROTECTED_FILE_COUNT = len(PROTECTED_RELATIVE_PATHS)
RUNTIME_ROOT = Path("/opt/minference_prefill_judge/runtime")
JUDGE_ROOT = Path("/opt/minference_prefill_judge")
ACTIVATION_ROOT = JUDGE_ROOT / "external-assets/private-activations"
WORKER_PATH = JUDGE_ROOT / "candidate_worker.py"
FINAL_MANIFEST_PATH = JUDGE_ROOT / "private/final_manifest.json"
NUMERICAL_POLICY_PATH = JUDGE_ROOT / "private/numerical_policy.json"
RECIPE_ROOT = JUDGE_ROOT / "activations/recipe"
MODEL_ROOT = JUDGE_ROOT / "model"
EVALUATION_CONTEXT_PATH = Path("/tmp/.sforge-evaluation-context.json")
PROTECTED_HASHES_PATH = JUDGE_ROOT / "protected_hashes.txt"
VERSION_RE = re.compile(r"kernel_v([0-9]{3})\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
MIN_COMPUTE_CAPABILITY = (9, 0)
MIN_DEVICE_MEMORY_BYTES = 80_000_000_000
WORKER_FAILURE_CATEGORIES = {
    "dependency", "candidate_import", "candidate_exception", "contract",
    "measurement_isolation", "timing", "input_mutation", "incomplete_output",
    "nondeterminism", "correctness", "stale_output", "timeout", "crash",
}

@dataclass(frozen=True)
class WorkerOutcome:
    completed: bool
    returncode: int | None
    category: str
    payload: dict[str, Any]

def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive")
    return number


def dense_flops(n: Any) -> int:
    """FLOPs of dense causal attention over ``n`` tokens: 8192 * n * (n + 1)."""

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise RuntimeError("case length must be a positive integer")
    return 2 * QUERY_HEADS * HEAD_DIM * n * (n + 1)


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise RuntimeError(f"{label} must be finite and nonnegative")
    return number


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("geometric mean requires values")
    checked = [_finite_positive(value, "geometric mean input") for value in values]
    return _finite_positive(
        math.exp(math.fsum(math.log(value) for value in checked) / len(checked)),
        "geometric mean",
    )


def admit_and_seal_candidate(
    candidate_root: Path,
    sealed_root: Path,
    *,
    owner_uid: int | None = None,
    owner_gid: int | None = None,
) -> SealedSubmission:
    approved = admit_workspace(Path(candidate_root))
    return seal_approved_submission(
        approved,
        Path(sealed_root),
        owner_uid=owner_uid,
        owner_gid=owner_gid,
        directory_mode=0o711 if owner_uid is not None else 0o700,
    )


def _strict_json(path: Path, *, max_bytes: int) -> tuple[dict[str, Any], bytes]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or not 0 < metadata.st_size <= max_bytes:
        raise RuntimeError(f"unsafe JSON artifact: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns):
            raise RuntimeError(f"JSON artifact changed before its read: {path}")
        raw = stream.read(max_bytes + 1)
        closed = os.fstat(stream.fileno())
    after = path.lstat()
    if (not raw or len(raw) != metadata.st_size
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
            or (closed.st_size, closed.st_mtime_ns) != (metadata.st_size, metadata.st_mtime_ns)):
        raise RuntimeError(f"empty or unstable JSON artifact: {path}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result
    parsed = json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if not isinstance(parsed, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return parsed, raw


def _read_root_only_json(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    exact_mode: int,
    expected_uid: int,
) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{label} root-only file is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != expected_uid
        or stat.S_IMODE(metadata.st_mode) != exact_mode
        or not 0 < metadata.st_size <= max_bytes
    ):
        raise RuntimeError(f"{label} does not satisfy the root-only file contract")
    parsed, raw = _strict_json(path, max_bytes=max_bytes)
    after = path.lstat()
    if (
        len(raw) != metadata.st_size
        or (after.st_dev, after.st_ino, after.st_size, after.st_mode)
        != (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mode)
    ):
        raise RuntimeError(f"{label} changed while being read")
    return parsed


def read_evaluation_context(
    path: Path, *, expected_uid: int = 0
) -> dict[str, Any]:
    context = _read_root_only_json(
        Path(path),
        label="evaluation context",
        max_bytes=16 * 1024,
        exact_mode=0o600,
        expected_uid=expected_uid,
    )
    round_id = context.get("round_id")
    if not isinstance(round_id, str) or not round_id:
        raise RuntimeError("evaluation context lacks round_id")
    return context


def _signal_session(session_id: int, requested_signal: signal.Signals) -> None:
    groups: set[int] = set()
    try:
        process_entries = tuple(Path("/proc").iterdir())
    except OSError:
        process_entries = ()
    for entry in process_entries:
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            if os.getsid(pid) == session_id:
                groups.add(os.getpgid(pid))
        except (OSError, ProcessLookupError, ValueError):
            continue
    groups.add(session_id)
    for group in sorted(groups, reverse=True):
        try:
            os.killpg(group, requested_signal)
        except (OSError, ProcessLookupError):
            pass


def _terminate_session(process: subprocess.Popen[bytes]) -> None:
    _signal_session(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    _signal_session(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def run_isolated_worker(
    command: Sequence[str],
    *,
    manifest_bytes: bytes,
    output_path: Path,
    timeout: float,
    environ: dict[str, str] | None = None,
) -> WorkerOutcome:
    """Execute a worker in its own session with a private stdin manifest."""

    if not command or not manifest_bytes or len(manifest_bytes) > 1024 * 1024:
        raise ValueError("worker command and bounded manifest bytes are required")
    output = Path(output_path)
    try:
        if output.exists() or output.is_symlink():
            output.unlink()
    except OSError:
        return WorkerOutcome(False, None, "dependency", {})
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=dict(os.environ if environ is None else environ),
            start_new_session=True,
        )
        try:
            process.communicate(input=manifest_bytes, timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_session(process)
            return WorkerOutcome(False, None, "timeout", {})
        # A successful worker must not leave grandchildren in its session.
        _signal_session(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        if process is not None:
            _terminate_session(process)
        return WorkerOutcome(False, None, "dependency", {})
    payload: dict[str, Any] = {}
    try:
        payload, _raw = _strict_json(output, max_bytes=16 * 1024 * 1024)
    except (OSError, UnicodeError, ValueError, RuntimeError, json.JSONDecodeError):
        payload = {}
    if process.returncode == 0 and payload:
        return WorkerOutcome(True, 0, "pass", payload)
    claimed = payload.get("failure_category")
    category = claimed if claimed in WORKER_FAILURE_CATEGORIES else "crash"
    return WorkerOutcome(True, process.returncode, category, payload)


def verify_installed_protected_hashes(path: Path = PROTECTED_HASHES_PATH) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return ["protected hash manifest is unreadable"]
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(None, 1)
        if (
            len(fields) != 2
            or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None
            or not fields[1].startswith("/")
            or fields[1] in seen
        ):
            errors.append("malformed or duplicate protected hash record")
            continue
        expected, installed = fields
        seen.add(installed)
        target = Path(installed)
        try:
            metadata = target.lstat()
        except OSError:
            errors.append(f"protected file is unavailable: {installed}")
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            errors.append(f"protected file permissions differ: {installed}")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            errors.append(f"protected file hash differs: {installed}")
    expected_paths = {str(JUDGE_ROOT / relative) for relative in PROTECTED_RELATIVE_PATHS}
    if seen != expected_paths:
        errors.append(
            f"protected hash path set must match the exact {PROTECTED_FILE_COUNT} scientific resources"
        )
    return errors


def locate_submission_root(app_dir: Path) -> Path:
    candidates = (
        app_dir / "workdir/minference_prefill_lab",
        app_dir / "environment/workdir/minference_prefill_lab",
        app_dir,
    )
    matches: list[Path] = []
    for candidate in candidates:
        try:
            metadata = candidate.lstat()
        except OSError:
            continue
        if (
            stat.S_ISDIR(metadata.st_mode)
            and not candidate.is_symlink()
            and (candidate / "candidates").is_dir()
            and (candidate / "FINAL_VERSION").is_file()
        ):
            absolute = candidate.absolute()
            if absolute not in matches:
                matches.append(absolute)
    if len(matches) != 1:
        raise RuntimeError(f"expected one submission root, found {len(matches)}")
    return matches[0]


def select_candidate_workspace(
    lab_root: Path, *, version_override: int | None = None
) -> tuple[Path, int, int]:
    candidates_root = lab_root / "candidates"
    versions: dict[int, Path] = {}
    for path in candidates_root.iterdir():
        match = VERSION_RE.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"unexpected candidate path: {path.name}")
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
            raise RuntimeError("candidate versions must be real directories")
        version = int(match.group(1))
        if version == 0 or version in versions:
            raise RuntimeError("candidate version set is invalid")
        versions[version] = path
    if not versions:
        raise RuntimeError("submission contains no candidate versions")
    if version_override is None:
        selector = (lab_root / "FINAL_VERSION").read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[1-9][0-9]{0,2}", selector) is None:
            raise RuntimeError("FINAL_VERSION must select 1-999")
        selected = int(selector)
    else:
        if isinstance(version_override, bool) or not isinstance(version_override, int):
            raise RuntimeError("candidate version override is invalid")
        selected = version_override
    if selected not in versions:
        raise RuntimeError("selected candidate version is absent")
    return versions[selected], selected, len(versions)


def _candidate_environment(cache_root: Path) -> dict[str, str]:
    allowed = (
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
    )
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update(
        {
            "HOME": "/home/agent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(cache_root),
            "TRITON_CACHE_DIR": str(cache_root / "triton"),
        }
    )
    return env


def _atomic_private_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode(
                    "utf-8"
                )
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=float(os.environ.get("MINFERENCE_PREFILL_WORKER_TIMEOUT", "1800")),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.log_dir, 0o700)
    for name in ("reward.json", "reward.txt", RESULT_NAME, "evaluation_nonce.json", "quality_summary.json", "trusted_dense_quality.json", "kernel-worker-result.json", "candidate-quality.json", "paired-quality.json", "full-quality-evidence.json", "quality-controller-result.json", "quality-native-input-verification.json", "quality-native-input-verification.stdout", "quality-native-input-verification.stderr"):
        try:
            (args.log_dir / name).unlink()
        except FileNotFoundError:
            pass
    reward, detail, returncode = evaluate(args)
    _atomic_private_json(args.log_dir / "reward.json", reward)
    _atomic_private_json(args.log_dir / RESULT_NAME, detail)
    reward_text = args.log_dir / "reward.txt"
    descriptor = os.open(
        reward_text,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"{reward['reward']}\n")
    return returncode



def _digest(value, label):
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise RuntimeError(f"invalid {label} digest")
    return value


def validate_private_manifest(manifest):
    """Frozen180 identities and48 groups; actual five-file hashes are per-call."""
    sys.path.insert(0, str(_independent_runtime_root()))
    from single_prompt_archive_v3 import PROTOCOL as ON_DEMAND_PROTOCOL, validate_plan
    if (not isinstance(manifest, dict) or manifest.get("partition") != "final"
            or manifest.get("protocol") != ON_DEMAND_PROTOCOL
            or manifest.get("capture_state") != "on-demand-fixed-input-plan-v1"):
        raise RuntimeError("private on-demand input plan differs")
    validate_plan(manifest)
    if any("sha256" in f or "size" in f for f in manifest["fixtures"].values()):
        raise RuntimeError("a fixed input plan must not inherit historical QKV hashes")
    bindings = manifest.get("input_bindings")
    if not isinstance(bindings, dict) or set(bindings) != {"prompt_manifest_sha256", "prompts_sha256", "recipe_sha256"}:
        raise RuntimeError("private fixed input binding schema differs")
    for name,value in bindings.items(): _digest(value, name)
    _digest(manifest.get("manifest_sha256"), "manifest")
    return manifest


def _independent_runtime_root():
    return RUNTIME_ROOT if RUNTIME_ROOT.is_dir() else Path(__file__).resolve().parents[2] / "runtime"


def load_protected_runtime_constants(runtime_path=None):
    path = _independent_runtime_root() / "minference_prefill_runtime.py" if runtime_path is None else Path(runtime_path)
    wanted = {"PROBE_ROWS", "TAIL_ROWS", "BAD_ROW_THRESHOLD", "TRIAL_RANDOMNESS_PROTOCOL", "MIN_COMPUTE_CAPABILITY", "MIN_DEVICE_MEMORY_BYTES"}
    values = {}
    for node in ast.parse(path.read_bytes()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in wanted:
            name = node.targets[0].id
            if name in values:
                raise RuntimeError("ambiguous protected runtime constants")
            values[name] = ast.literal_eval(node.value)
    if set(values) != wanted or values["TRIAL_RANDOMNESS_PROTOCOL"] != TRIAL_RANDOMNESS_PROTOCOL or values["MIN_COMPUTE_CAPABILITY"] != MIN_COMPUTE_CAPABILITY or values["MIN_DEVICE_MEMORY_BYTES"] != MIN_DEVICE_MEMORY_BYTES:
        raise RuntimeError("protected runtime contract differs")
    if type(values["PROBE_ROWS"]) is not int or values["PROBE_ROWS"] != 512 or type(values["TAIL_ROWS"]) is not int or values["TAIL_ROWS"] != 64 or values["BAD_ROW_THRESHOLD"] != 0.5:
        raise RuntimeError("protected runtime numerical probe contract differs")
    return values


def validate_numerical_policy(policy,constants):
    from kernel_fidelity_v3 import validate_limits
    if policy.get("schema_version")!=1 or policy.get("protocol")!="minference-kernel-energy-fidelity-v1" or policy.get("status")!="frozen":
        raise RuntimeError("frozen kernel-only fidelity policy required")
    if policy.get("uniform_probe_rows")!=512 or policy.get("tail_probe_rows")!=64 or set(policy.get("limits",{}))!={"default"}:
        raise RuntimeError("kernel fidelity sampling/budget differs")
    validate_limits(policy["limits"]["default"]);_digest(policy.get("numerical_policy_sha256"),"numerical policy")
    return policy

def _case_limits(policy, group):
    key = f"{group['task']}:{group['context_budget']}:{group['layer']}"
    return policy["limits"].get(key, policy["limits"]["default"])


def _check_stats(stats, *, n, probes, label):
    if not isinstance(stats,dict) or stats.get("finite") is not True or stats.get("rows")!=QUERY_HEADS*min(n,probes):
        raise RuntimeError("energy probe coverage differs")
    heads=stats.get("head_nrmse")
    if not isinstance(heads,list) or len(heads)!=32:raise RuntimeError("per-head fidelity missing")
    values={k:_finite_nonnegative(stats.get(k),k) for k in ("nrmse","head_max_nrmse")}
    if not math.isclose(max(_finite_nonnegative(x,"head energy") for x in heads),values["head_max_nrmse"],rel_tol=1e-6):
        raise RuntimeError("worst-head energy metadata differs")
    return {**values,"finite":True}

def _equal_number(actual, expected, label):
    checked = _finite_positive(actual, label)
    if not math.isclose(checked, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"{label} differs from root derivation")


def _contains_private_randomness(value, *, depth=0):
    if depth > 64:
        raise RuntimeError("worker result nesting exceeds the bound")
    if isinstance(value, dict):
        return (any(key in value for key in ("nonce", "evaluation_nonce", "trial_nonce"))
                or any(_contains_private_randomness(child, depth=depth + 1)
                       for child in value.values()))
    if isinstance(value, (list, tuple)):
        return any(_contains_private_randomness(child, depth=depth + 1) for child in value)
    return False


def validate_on_demand_preparation(worker, manifest):
    """Bind actual FA2 generation provenance independently of candidate output."""
    from single_prompt_archive_v3 import (PROTOCOL as ON_DEMAND_PROTOCOL,
        on_demand_plan, canonical_sha, validate_bundle)
    value = worker.get("on_demand", {})
    bundles = value.get("preparation_bundles", [])
    if (value.get("protocol") != ON_DEMAND_PROTOCOL
            or value.get("purpose") != "private-judge"
            or value.get("parent_execution_uid") != 0
            or value.get("candidate_execution_uid") != 1000
            or value.get("candidate_privilege_drop") is not True
            or value.get("preparation_timed_in_score") is not False
            or value.get("persistent_qkv_files") != 0
            or not isinstance(bundles, list) or len(bundles) != 36):
        raise RuntimeError("private on-demand preparation/FD/drop coverage differs")
    receipt, _ = _root_private_json_value(JUDGE_ROOT / "private/external-assets-receipt.json", max_bytes=4*1024**2)
    recipe, _ = _root_private_json_value(RECIPE_ROOT / "RULER_V3_RECIPE.json", max_bytes=1024**2)
    source_sha = lambda path: sha256_bytes(Path(path).read_bytes())
    bindings = bundles[0].get("bindings", {})
    if (bindings.get("image_id") != receipt["image_id"]
            or bindings.get("model") != recipe["model"]
            or bindings.get("model_files_verified") is not True
            or bindings.get("prompt_manifest_sha256") != manifest["input_bindings"]["prompt_manifest_sha256"]
            or bindings.get("prompts_sha256") != manifest["input_bindings"]["prompts_sha256"]
            or bindings.get("recipe_protocol_sha256") != manifest["input_bindings"]["recipe_sha256"]
            or bindings.get("attention_backend") != "flash_attention_2"
            or bindings.get("flash_attention_version") != "2.8.3"
            or bindings.get("software") != {"torch":"2.8.0+cu129", "cuda":"12.9", "transformers":"4.57.1"}
            or bindings.get("capture_helper_sha256") != source_sha(RECIPE_ROOT / "capture_activations.py")
            or bindings.get("on_demand_suite_sha256") != source_sha(RUNTIME_ROOT / "on_demand_suite_v3.py")
            or bindings.get("on_demand_sources") != {name:source_sha(RUNTIME_ROOT/name) for name in (
                "single_prompt_archive_v3.py", "prepare_single_prompt_v3.py", "measure_single_prompt_v3.py",
                "run_single_prompt_worker_v3.py", "stream_parent_v3.py")}):
        raise RuntimeError("on-demand image/model/input/software/source binding differs")
    fixed = {key: child for key,child in manifest.items() if key != "manifest_sha256"}
    execution_plan = on_demand_plan(fixed, bindings)
    if value.get("execution_plan_sha256") != canonical_sha(execution_plan):
        raise RuntimeError("on-demand execution plan differs")
    expected = {f["prompt_id"] for f in manifest["fixtures"].values()}
    if {b.get("prompt_id") for b in bundles} != expected:
        raise RuntimeError("on-demand capture prompt coverage differs")
    capture_sha = source_sha(RUNTIME_ROOT / "prepare_single_prompt_v3.py")
    for bundle in bundles:
        validate_bundle(bundle, execution_plan)
        if (bundle.get("purpose") != "private-judge"
                or bundle.get("implementation_sha256") != capture_sha
                or bundle.get("preparation_timed_in_score") is not False):
            raise RuntimeError("on-demand bundle preparation identity differs")
        _finite_positive(bundle.get("preparation_seconds"), "model preparation seconds")
    if (value.get("scratch_bytes_max") != max(b["scratch_bytes"] for b in bundles)
            or type(value.get("scratch_bytes_max")) is not int
            or value["scratch_bytes_max"] > 8*1024**3):
        raise RuntimeError("on-demand scratch cap differs")
    _equal_number(value.get("preparation_seconds_total"), sum(b["preparation_seconds"] for b in bundles), "model preparation total")


def validate_worker_result(worker, *, manifest, numerical_policy, approved_sha256, runtime_constants_loader=None):
    manifest = validate_private_manifest(manifest)
    constants = load_protected_runtime_constants() if runtime_constants_loader is None else runtime_constants_loader()
    policy = validate_numerical_policy(numerical_policy, constants)
    if not isinstance(worker, dict) or worker.get("schema_version") != 3 or worker.get("protocol") != PROTOCOL or worker.get("suite_type") != "final" or worker.get("ok") is not True or worker.get("score_enabled") is not True:
        raise RuntimeError("candidate worker did not return a successful frozen schema-v3 result")
    from kernel_fidelity_v3 import validate_boundary_checks
    boundaries=validate_boundary_checks(worker.get("boundary_checks"),policy["limits"]["default"])
    validate_on_demand_preparation(worker, manifest)
    if _contains_private_randomness(worker):
        raise RuntimeError("worker artifact contains private randomness")
    if worker.get("candidate_sha256") != approved_sha256 or worker.get("manifest_sha256") != manifest["manifest_sha256"] or worker.get("numerical_policy_sha256") != policy["numerical_policy_sha256"]:
        raise RuntimeError("candidate worker artifact identities differ")
    hardware = worker.get("hardware")
    if not isinstance(hardware, dict) or type(hardware.get("count")) is not int or hardware["count"] != 1 or not isinstance(hardware.get("capability"), list) or len(hardware["capability"]) != 2 or any(type(v) is not int for v in hardware["capability"]) or tuple(hardware["capability"]) < MIN_COMPUTE_CAPABILITY or type(hardware.get("total_memory_bytes")) is not int or hardware["total_memory_bytes"] < MIN_DEVICE_MEMORY_BYTES:
        raise RuntimeError("candidate worker hardware violates the v3 capability envelope")
    rows = worker.get("cases")
    if not isinstance(rows, list) or len(rows) != FINAL_CASE_COUNT:
        raise RuntimeError("candidate worker case coverage differs")
    medians, by_budget, by_task, derived_cases = [], {}, {}, []
    for row, group in zip(rows, manifest["case_groups"], strict=True):
        if not isinstance(row, dict) or any(row.get(key) != group[key] for key in ("case_id", "task", "context_budget", "layer")) or row.get("trial_ids") != group["trials"] or row.get("numeric_passed") is not True:
            raise RuntimeError("candidate worker group identity or gate differs")
        trials = row.get("trials")
        arrays = {key: row.get(key) for key in ("trial_ns", "trial_seconds", "trial_host_seconds", "trial_event_seconds")}
        if not isinstance(trials, list) or len(trials) != TRIALS_PER_CASE or any(not isinstance(value, list) or len(value) != TRIALS_PER_CASE for value in arrays.values()):
            raise RuntimeError("candidate worker lacks exactly three measured trial records")
        times, tflops = [], []
        for index, (trial, fid) in enumerate(zip(trials, group["trials"], strict=True)):
            fixture = manifest["fixtures"][fid]
            if not isinstance(trial, dict) or trial.get("fixture_id") != fid or any(trial.get(key) != fixture[key] for key in ("prompt_id", "input_token_sha256", "actual_n")) or type(trial.get("actual_n")) is not int or type(arrays["trial_ns"][index]) is not int or arrays["trial_ns"][index] != fixture["actual_n"]:
                raise RuntimeError("candidate worker trial real-input identity differs")
            for gate in ("numeric_passed", "full_write_ok", "input_immutable_ok"):
                if trial.get(gate) is not True:
                    raise RuntimeError(f"candidate worker trial gate failed: {gate}")
            limits = _case_limits(policy, group)
            reported = trial.get("limits")
            if not isinstance(reported, dict) or set(reported) != set(LIMIT_KEYS) or any(_finite_nonnegative(reported[key], "reported numerical limit") != limits[key] for key in LIMIT_KEYS):
                raise RuntimeError("candidate worker numerical limits differ from protected policy")
            uniform = _check_stats(trial.get("uniform_stats"), n=fixture["actual_n"], probes=constants["PROBE_ROWS"], label="uniform")
            tail = _check_stats(trial.get("tail_stats"), n=fixture["actual_n"], probes=constants["TAIL_ROWS"], label="tail")
            from kernel_fidelity_v3 import passes
            if not passes(uniform,tail,limits):raise RuntimeError("candidate exceeds independent kernel energy-error budget")
            host = _finite_positive(trial.get("host_seconds"), "trial host seconds")
            event = _finite_positive(trial.get("event_seconds"), "trial event seconds")
            elapsed = max(host, event)
            _equal_number(trial.get("seconds"), elapsed, "trial seconds")
            for key, expected in (("trial_host_seconds", host), ("trial_event_seconds", event), ("trial_seconds", elapsed)):
                _equal_number(arrays[key][index], expected, key)
            times.append(elapsed)
            tflops.append(dense_flops(fixture["actual_n"]) / elapsed / 1e12)
        median = statistics.median(tflops)
        _equal_number(row.get("median_tflops"), median, "group median TFLOPS")
        medians.append(median)
        by_budget.setdefault(str(group["context_budget"]), []).append(median)
        by_task.setdefault(group["task"], []).append(median)
        derived_cases.append({**{key: group[key] for key in ("case_id", "task", "context_budget", "layer")}, "trial_ns": list(arrays["trial_ns"]), "trial_seconds": times, "trial_tflops": tflops, "median_tflops": median})
    score = math.exp(statistics.fmean(math.log(value) for value in medians))
    _equal_number(worker.get("score"), score, "aggregate score")
    _equal_number(worker.get("raw_primary_tflops"), score, "raw aggregate TFLOPS")
    return {"case_count": len(rows), "boundary_case_count":len(boundaries), "raw_primary_tflops": score, "score_enabled": True,
            "length_geomean_tflops": {key: _geometric_mean(value) for key, value in by_budget.items()},
            "family_geomean_tflops": {key: _geometric_mean(value) for key, value in by_task.items()},
            "cases": derived_cases, "hardware": hardware, "candidate_sha256": approved_sha256,
            "manifest_sha256": manifest["manifest_sha256"], "numerical_policy_sha256": policy["numerical_policy_sha256"],
            **{key: True for key in ("suite_ok", "source_ok", "hardware_ok", "correctness_ok", "fresh_input_ok", "fresh_output_ok", "probe_fresh_ok", "manifest_ok")}}


def read_final_manifest(path, *, expected_uid=0):
    data = _read_root_only_json(Path(path), label="final manifest", max_bytes=4*1024**2, exact_mode=0o400, expected_uid=expected_uid)
    data = {**data, "manifest_sha256": sha256_bytes(Path(path).read_bytes())}
    return validate_private_manifest(data)


def read_numerical_policy(path, *, expected_uid=0):
    data = _read_root_only_json(Path(path), label="numerical policy", max_bytes=1024**2, exact_mode=0o400, expected_uid=expected_uid)
    data = {**data, "numerical_policy_sha256": sha256_bytes(Path(path).read_bytes())}
    return validate_numerical_policy(data, load_protected_runtime_constants())


def _profile_and_manifest(context, *, final_manifest_expected_uid=0):
    round_id = context["round_id"]
    override = context.get("candidate_version")
    if not isinstance(round_id, str) or not re.fullmatch(r"(?:agent|auto|final)-[1-9][0-9]*|candidate-v[1-9][0-9]*-attempt-[1-9][0-9]*|candidate-impl-v[0-9]{3}-[0-9a-f]{12}", round_id):
        raise RuntimeError("unknown evaluation round profile")
    replay = re.fullmatch(r"candidate-impl-v([0-9]{3})-[0-9a-f]{12}", round_id)
    if replay:
        if type(override) is not int or not 1 <= override <= 999 or override != int(replay.group(1)):
            raise RuntimeError("per-version replay identity differs")
    elif override is not None:
        raise RuntimeError("this evaluation round cannot override a candidate version")
    manifest = read_final_manifest(FINAL_MANIFEST_PATH, expected_uid=final_manifest_expected_uid)
    private = context.get("private")
    expected = private.get("FINAL_MANIFEST_SHA256") if isinstance(private, dict) else None
    if _digest(expected, "context manifest") != manifest["manifest_sha256"]:
        raise RuntimeError("evaluation context does not bind the exact final manifest")
    if private.get("EVALUATION_KIND") != "kernel-only-v3":raise RuntimeError("explicit trusted kernel-only context required")
    phase = private.get("EVALUATION_PHASE")
    if phase not in ("interactive", "final"):
        raise RuntimeError("root evaluation context lacks an admitted phase")
    # Round/request provenance does not select a smaller scoring workload.
    # A missing quality bootstrap still fails closed in _evaluate_quality.
    return "final", manifest, override


def _prepare_evaluation_manifest(manifest, *, numerical_policy_sha256, candidate_sha256, log_dir, round_id):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata = log_dir.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or log_dir.is_symlink():
        raise RuntimeError("private evaluation log directory must be a real directory")
    os.chown(log_dir, os.geteuid(), os.getegid()); os.chmod(log_dir, 0o700)
    nonce = secrets.randbits(128) or 1
    receipt = {"schema_version": 3, "protocol": PROTOCOL, "round_id": round_id,
               "trial_randomness_protocol": TRIAL_RANDOMNESS_PROTOCOL,
               "evaluation_nonce": nonce, "manifest_sha256": manifest["manifest_sha256"],
               "numerical_policy_sha256": numerical_policy_sha256, "candidate_sha256": candidate_sha256}
    _atomic_private_json(log_dir / "evaluation_nonce.json", receipt)
    return {key: value for key, value in receipt.items() if key != "round_id"} | {"suite_type": "final"}


def failure_records(category, message, *, final_version=0, num_candidate_versions=0):
    common = {"schema_version": 3, "protocol": PROTOCOL, "reward": 0.0, "raw_primary_tflops": 0.0,
              "score_enabled": 0, "case_count": 0, "final_version": final_version,
              "num_candidate_versions": num_candidate_versions,
              **{key: 0 for key in ("suite_ok", "source_ok", "hardware_ok", "correctness_ok", "fresh_input_ok", "fresh_output_ok", "probe_fresh_ok", "manifest_ok")}}
    return common, {**common, "failure_category": category, "private_error": message}, 1



def _atomic_private_bytes(path, raw):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path); path.chmod(0o600)
    finally:
        try: temporary.unlink()
        except FileNotFoundError: pass

def _retain_worker_report(source, destination, *, expected_payload=None):
    """Keep the worker's actual bytes, never serialize an invented raw report."""
    parsed, raw = _strict_json(Path(source), max_bytes=16*1024**2)
    if expected_payload is not None and _canonical_json(parsed) != _canonical_json(expected_payload):
        raise RuntimeError("worker report bytes changed after independent validation")
    _atomic_private_bytes(Path(destination), raw)
    return parsed, sha256_bytes(raw)

def build_result_records(derived, *, suite_id=FINAL_SUITE_ID, final_version,
                         num_candidate_versions, evaluation_phase):
    if evaluation_phase not in ("interactive", "final"):
        raise RuntimeError("result requires a root-bound evaluation phase")
    common = {"schema_version": 3, "protocol": PROTOCOL, "reward": derived["raw_primary_tflops"],
              "raw_primary_tflops": derived["raw_primary_tflops"], "score_enabled": 1,
              "case_count": derived["case_count"], "final_version": final_version,
              "num_candidate_versions": num_candidate_versions,
              "evaluation_phase": evaluation_phase,
              "evaluation_kind": "kernel-only-v3",
              "boundary_case_count":derived["boundary_case_count"],
              "provisional": False,
              "length_geomean_tflops": derived["length_geomean_tflops"],
              "family_geomean_tflops": derived["family_geomean_tflops"],
              **{key: 1 for key in ("suite_ok", "source_ok", "hardware_ok", "correctness_ok", "fresh_input_ok", "fresh_output_ok", "probe_fresh_ok", "manifest_ok")}}
    detail = {**common, "suite_id": suite_id, "evaluation_profile": "final",
              "qualification_state": "kernel-only-fidelity-frozen",
              **{key: derived[key] for key in ("manifest_sha256", "numerical_policy_sha256", "candidate_sha256", "hardware", "cases")}}
    return common, detail, 0


def _root_private_json_value(path, *, max_bytes, expected_uid=0):
    """Read a root-only document, including Docker inspect's single-item list."""
    path = Path(path); before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_uid != expected_uid or stat.S_IMODE(before.st_mode) != 0o400
            or not 0 < before.st_size <= max_bytes):
        raise RuntimeError("unsafe root-only quality artifact")
    raw = path.read_bytes(); after = path.lstat()
    if len(raw) != before.st_size or (before.st_dev, before.st_ino, before.st_size, before.st_mode) != (after.st_dev, after.st_ino, after.st_size, after.st_mode):
        raise RuntimeError("quality artifact changed during its read")
    def unique(pairs):
        value = {}
        for key, child in pairs:
            if key in value: raise ValueError("duplicate quality JSON key")
            value[key] = child
        return value
    value = json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite quality JSON")))
    return value, sha256_bytes(raw)



def _external_assets_module():
    """Import only the already protected root helper, before candidate admission."""
    loaded = sys.modules.get("minference_external_assets_v3")
    if loaded is not None:
        return loaded
    path = JUDGE_ROOT / "scripts/external_assets_on_demand_v3.py"
    spec = importlib.util.spec_from_file_location("minference_external_assets_v3", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_external_asset_bootstrap(context):
    return _external_assets_module().verify_runtime_bindings(context)


def evaluate(args):
    selected, version_count = 0, 0
    try:
        errors = verify_installed_protected_hashes()
        if errors:
            raise RuntimeError("; ".join(errors))
        context = read_evaluation_context(EVALUATION_CONTEXT_PATH)
        verify_external_asset_bootstrap(context)
        profile, manifest, override = _profile_and_manifest(context)
        policy = read_numerical_policy(NUMERICAL_POLICY_PATH)
        lab = locate_submission_root(args.app_dir.resolve())
        candidate_root, selected, version_count = select_candidate_workspace(lab, version_override=override)
        account = pwd.getpwnam("agent")
        run_root = Path(tempfile.mkdtemp(prefix="minference-prefill-v3-judge-", dir="/tmp")); run_root.chmod(0o711)
        try:
            sealed = admit_and_seal_candidate(candidate_root, run_root / "approved", owner_uid=account.pw_uid, owner_gid=account.pw_gid)
            worker_root = run_root / "worker"; worker_root.mkdir(mode=0o700); os.chown(worker_root, account.pw_uid, account.pw_gid)
            cache_root = run_root / "cache"; cache_root.mkdir(mode=0o700); os.chown(cache_root, account.pw_uid, account.pw_gid)
            output = worker_root / "result.json"
            transport = _prepare_evaluation_manifest(manifest, numerical_policy_sha256=policy["numerical_policy_sha256"], candidate_sha256=sealed.sha256, log_dir=args.log_dir, round_id=context["round_id"])
            # Metadata-only input assets replace the retired QKV archive. The
            # model is trusted root preparation; candidate imports only afterward.
            kernel_context = run_root / "kernel-context.json"
            external, _ = _root_private_json_value(JUDGE_ROOT / "private/external-assets-receipt.json", max_bytes=4*1024**2)
            image_id = external["image_id"]
            inspection, _ = _root_private_json_value(JUDGE_ROOT / "private/container-inspect.json", max_bytes=4*1024**2)
            inspection = inspection[0] if isinstance(inspection, list) else inspection
            # Docker inspect or the PyroMind platform receipt; the protected
            # external-assets module has already validated whichever it is.
            kernel_uuid = _external_assets_module().judge_gpu_uuid(inspection)
            _atomic_private_json(kernel_context, {"image_id": image_id,
                "gpu_uuid": kernel_uuid, "purpose": "private-judge"})
            kernel_context.chmod(0o400)
            kernel_transport = {key: transport[key] for key in ("candidate_sha256",
                "manifest_sha256", "numerical_policy_sha256", "evaluation_nonce")}
            environment = _candidate_environment(cache_root)
            # Preparation and candidate share the one allocated physical GPU.
            environment["CUDA_VISIBLE_DEVICES"] = kernel_uuid
            scratch_parent = run_root / "qkv-scratch"
            run_output = run_root / "on-demand-metadata"
            outcome = run_isolated_worker([sys.executable, str(RUNTIME_ROOT / "on_demand_suite_v3.py"),
                "--purpose", "private-judge", "--trusted-root", str(JUDGE_ROOT),
                "--candidate", str(sealed.path), "--plan", str(FINAL_MANIFEST_PATH),
                "--numerical-policy", str(NUMERICAL_POLICY_PATH),
                "--prompts", str(JUDGE_ROOT / "external-assets/kernel-prompts"),
                "--model", str(MODEL_ROOT / "d10aef7999a2b5ba950ab3974312feeedbfe0b77"), "--context", str(kernel_context),
                "--scratch-parent", str(scratch_parent), "--run-output", str(run_output),
                "--output", str(output)], manifest_bytes=_canonical_json(kernel_transport),
                output_path=output, timeout=args.worker_timeout, environ=environment)
            if outcome.category != "pass":
                if outcome.payload:
                    _atomic_private_json(args.log_dir / "kernel-worker-result.json",outcome.payload)
                return failure_records(outcome.category, "private worker failed closed: "+str(outcome.payload.get("private_message","")), final_version=selected, num_candidate_versions=version_count)
            _retain_worker_report(output, Path(args.log_dir) / "kernel-worker-result.json", expected_payload=outcome.payload)
            derived = validate_worker_result(outcome.payload, manifest=manifest, numerical_policy=policy, approved_sha256=sealed.sha256)
            if sha256_bytes(sealed.path.read_bytes()) != sealed.sha256:
                raise RuntimeError("sealed candidate changed after evaluation")
            return build_result_records(derived, suite_id=FINAL_SUITE_ID, final_version=selected, num_candidate_versions=version_count, evaluation_phase=profile)
        finally:
            shutil.rmtree(run_root, ignore_errors=True)
    except BaseException as exc:
        return failure_records("judge_failure", f"{type(exc).__name__}: {exc}", final_version=selected, num_candidate_versions=version_count)


if __name__ == "__main__":
    raise SystemExit(main())
