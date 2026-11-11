#!/usr/bin/env python3
"""Private verifier for standalone K-grouped FP8 wgrad v2.

Order of evaluation: protected hashes, evaluation profile, the two root-only
manifests (scored grid and edge cases), source admission, hardware, the
compute-sanitizer memcheck subset in a separate unprivileged process (first,
so an out-of-bounds kernel never reaches the hardware natively), and then the
unprivileged worker (edge cases, then the scored grid).  Every aggregate the
worker reports is recomputed here from its trial records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pwd
import random
import re
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_admission import validate_submission  # noqa: E402


JUDGE_ROOT = Path("/opt/standalone_k_grouped_wgrad_judge")
RUNTIME_ROOT = Path("/opt/standalone_k_grouped_wgrad_worker")
WORKER = JUDGE_ROOT / "candidate_worker.py"
SANITIZER = Path("/usr/local/cuda/compute-sanitizer/compute-sanitizer")
EVALUATION_CONTEXT = Path("/tmp/.sforge-evaluation-context.json")
FINAL_MANIFEST = JUDGE_ROOT / "private/final_cases.json"
EDGE_MANIFEST = JUDGE_ROOT / "private/edge_cases.json"
RESULT_NAME = "standalone_k_grouped_wgrad_v2_results.json"
FINAL_SUITE_ID = "final-holdout-v3"
EDGE_SUITE_ID = "edge-holdout-v3"
EVALUATION_POLICY = "full-final-shared-v4"
SELECTION_PROTOCOL_ID = "k-grouped-fp8-wgrad-shared-final-v4"
SCHEMA_VERSION = 4
PROFILE_CODES = {"auto": 1.0, "final": 2.0}
FAMILIES = ("large", "many_groups", "small_batch", "sequence")
DISTRIBUTIONS = ("uniform", "dirichlet")
WGRAD_ENTRYPOINT = "k_grouped_fp8_gemm_tn_contiguous"
ENTRYPOINTS = {WGRAD_ENTRYPOINT: WGRAD_ENTRYPOINT}
# Keep the three limits identical to the runtime (tests assert the pairs).
ELEMENT_RATIO_LIMIT = 0.05
EXACT_ERROR_LIMIT = 2.0e-7
SOURCE_ERROR_LIMIT = 1.0e-3
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
SCORED_KEYS = {
    "case_id", "family", "model", "g", "m", "n", "m2", "n2", "dist", "total_k", "alpha", "hint_scale", "seed",
}
EDGE_KEYS = {"case_id", "category", "g", "m", "n", "ks", "hint", "seed", "sanitize"}
K_ALIGNMENT = 128
MAX_TOTAL_K = 262_144
MAX_OUTPUT_ELEMENTS = (1 << 31) - 1
MAX_HINT = 1 << 20
SEQUENCE_LAYERS = 58
SEQUENCE_OUT_POOL = 2
SANITIZER_EXIT_CODE = 86
# compute-sanitizer prints this when it cannot attach (for example under a GPU
# virtualization hook); the target then runs uninstrumented, which proves nothing.
SANITIZER_UNAVAILABLE_MARKER = "No attachable process found"
# The deployment, never the candidate, selects the sanitizer mode (Judge image ENV
# or the controller's Judge environment). "required" is the task contract;
# "skip-platform-incompatible" exists for platforms whose GPU virtualization hook
# stops compute-sanitizer from attaching to any CUDA process (PyroMind's HAMi
# libvgpu.so, 2026-10-05); results then record that the pass did not run.
SANITIZER_MODES = ("required", "skip-platform-incompatible")
SANITIZER_ARGUMENTS = (
    "--tool=memcheck",
    f"--error-exitcode={SANITIZER_EXIT_CODE}",
    "--target-processes=application-only",
    "--destroy-on-device-error=context",
    "--force-blocking-launches",
    "--check-api-memory-access=no",
    "--print-limit=20",
    "--kernel-name-exclude",
    "kns=nvjet",
)

# The frozen scored suite; keep identical to scripts/generate_final_manifest.py
# and the runtime (tests assert the pairs).
FINAL_CASE_COUNT = 1082
FAMILY_CASE_COUNTS = {"large": 504, "many_groups": 189, "small_batch": 307, "sequence": 82}
FINAL_SEED_MINIMUM = 1 << 32  # public seeds are all below this
MAX_SEED = (1 << 63) - 1
# The frozen edge instance; keep identical to scripts/generate_edge_manifest.py.
EDGE_CASE_COUNT = 166
EDGE_SANITIZE_COUNT = 8
EDGE_DOMAIN = {"g": (1, 128), "mn": (1, 8192), "hint": (1, 1 << 20)}
EDGE_MAX_OUTPUT_ELEMENTS = 1 << 26
EDGE_MAX_INPUT_BYTES = 1 << 28
# Byte caps of the frozen manifests (the v3 final suite is about 340 KB).
FINAL_MANIFEST_MAX_BYTES = 2 * 1024 * 1024
EDGE_MANIFEST_MAX_BYTES = 1024 * 1024

GATE_NAMES = (
    "profile_ok",
    "suite_ok",
    "source_ok",
    "hardware_ok",
    "edge_correctness_ok",
    "correctness_ok",
    "memory_safety_ok",
    "harness_integrity_ok",
)
RUNTIME_GATES = (
    "edge_correctness_ok",
    "correctness_ok",
    "memory_safety_ok",
    "harness_integrity_ok",
)

# Physical plausibility bounds; keep identical to the runtime (tests assert).
FP8_DENSE_PEAK_BOUND_TFLOPS: dict[tuple[int, int], float] = {
    (8, 9): 1_500.0,
    (9, 0): 2_500.0,
    (10, 0): 7_000.0,
    (10, 3): 7_000.0,
    (12, 0): 2_500.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive")
    return number


def geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("geometric mean requires values")
    numbers = [finite_positive(value, "geometric mean input") for value in values]
    return math.exp(math.fsum(math.log(value) for value in numbers) / len(numbers))


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is malformed") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return parsed


def _read_root_only_json(
    path: Path,
    label: str,
    *,
    max_bytes: int,
    expected_uid: int = 0,
    exact_mode: int | None = None,
) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable") from exc
    permissions = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != expected_uid
        or permissions & 0o077
        or not permissions & 0o400
        or (exact_mode is not None and permissions != exact_mode)
        or metadata.st_size <= 0
        or metadata.st_size > max_bytes
    ):
        raise RuntimeError(f"{label} does not satisfy the root-only file contract")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"{label} is unreadable") from exc
    if len(raw) != metadata.st_size:
        raise RuntimeError(f"{label} changed while being read")
    return _strict_json_bytes(raw, label), raw


def load_evaluation_profile(
    path: Path = EVALUATION_CONTEXT,
    *,
    expected_uid: int = 0,
) -> tuple[str, str, str, int | None]:
    context, _raw = _read_root_only_json(
        path,
        "SForge evaluation context",
        max_bytes=16 * 1024,
        expected_uid=expected_uid,
        exact_mode=0o600,
    )
    round_id = context.get("round_id")
    if not isinstance(round_id, str) or not round_id:
        raise RuntimeError("SForge evaluation context lacks round_id")
    # Host-side per-version replay may override the submission's own
    # FINAL_VERSION so one archive can be scored per retained implementation.
    # The context file is root-only and written by the trusted host, and the
    # override is refused on interactive rounds so an agent-visible result can
    # never come from a version other than the one its submission selected.
    version_override = context.get("candidate_version")
    if version_override is not None and (
        isinstance(version_override, bool)
        or not isinstance(version_override, int)
        or not 1 <= version_override <= 999
    ):
        raise RuntimeError("SForge evaluation candidate_version is invalid")
    # Every admitted round uses the complete frozen final suite.  The original
    # round identity still controls whether a host candidate override is legal.
    for interactive in ("auto-", "agent-"):
        if round_id.startswith(interactive) and len(round_id) > len(interactive):
            if version_override is not None:
                raise RuntimeError(
                    "candidate_version override is host-side only"
                )
            return "final", FINAL_SUITE_ID, round_id, None
    # Host replay may select a retained implementation from the frozen archive.
    for host_side in ("manual-", "candidate-"):
        if round_id.startswith(host_side) and len(round_id) > len(host_side):
            return "final", FINAL_SUITE_ID, round_id, version_override
    raise RuntimeError("SForge evaluation round_id has an unknown profile")



def _integer(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{label} must be an integer >= {minimum}")
    return value


def _align(value: int, alignment: int = 128) -> int:
    return (value + alignment - 1) // alignment * alignment


def _max_dynamic_k(expected_k: int) -> int:
    return _align(int(expected_k * 1.3))


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive")
    return number


def case_hint(case: Mapping[str, Any]) -> int:
    mean = int(case["total_k"]) / int(case["g"])
    return max(1, min(MAX_HINT, int(1.2 * mean * float(case["hint_scale"]))))


def derive_seed(case_seed: int, role: str, trial: int) -> int:
    """Keep identical to the runtime: 63 bits of SHA-256(case seed, role, trial)."""
    digest = hashlib.sha256(f"kgw-v2:{int(case_seed)}:{role}:{int(trial)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & MAX_SEED


def draw_ks(case: Mapping[str, Any], seed: int) -> list[int]:
    """Independent copy of the runtime draw; the Judge recomputes every call."""
    generator = random.Random(seed)
    g, total = int(case["g"]), int(case["total_k"])
    if case["dist"] == "uniform":
        expected = total // g
        return [_align(max(1, int(expected * generator.uniform(0.7, 1.3)))) for _ in range(g)]
    blocks = total // K_ALIGNMENT
    weights = [generator.gammavariate(float(case["alpha"]), 1.0) for _ in range(g)]
    scale = math.fsum(weights)
    if scale <= 0.0:
        weights, scale = [1.0] * g, float(g)
    shares = [weight / scale * blocks for weight in weights]
    counts = [int(share) for share in shares]
    order = sorted(range(g), key=lambda index: (counts[index] - shares[index], index))
    for index in order[: blocks - sum(counts)]:
        counts[index] += 1
    return [count * K_ALIGNMENT for count in counts]


def sequence_schedule(case: Mapping[str, Any], round_seed: int) -> list[tuple[int, int, list[int], int]]:
    generator = random.Random(round_seed)
    calls: list[tuple[int, int, list[int], int]] = []
    for layer in reversed(range(SEQUENCE_LAYERS)):
        for linear in (1, 0):
            ks = draw_ks(case, generator.getrandbits(62))
            calls.append((linear, layer, ks, generator.randrange(SEQUENCE_OUT_POOL)))
    return calls


def _check_shape(case_id: str, g: int, m: int, n: int) -> None:
    if not EDGE_DOMAIN["g"][0] <= g <= EDGE_DOMAIN["g"][1]:
        raise RuntimeError(f"{case_id}: g outside the domain")
    for value in (m, n):
        if not EDGE_DOMAIN["mn"][0] <= value <= EDGE_DOMAIN["mn"][1]:
            raise RuntimeError(f"{case_id}: m or n outside the domain")
    if g * m * n > MAX_OUTPUT_ELEMENTS:
        raise RuntimeError(f"{case_id}: output capacity outside the domain")


def validate_final_manifest(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if set(payload) != {"schema_version", "suite_id", "cases"}:
        raise RuntimeError("final manifest fields differ from schema v4")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("suite_id") != FINAL_SUITE_ID:
        raise RuntimeError("final manifest identity differs")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != FINAL_CASE_COUNT:
        raise RuntimeError(f"final manifest must contain exactly {FINAL_CASE_COUNT} cases")
    ids: set[str] = set()
    seeds: set[int] = set()
    configurations: set[tuple[Any, ...]] = set()
    counts = {family: 0 for family in FAMILIES}
    cases: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping) or set(raw_case) != SCORED_KEYS:
            raise RuntimeError("final case fields differ from schema v4")
        case = dict(raw_case)
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None or case_id in ids:
            raise RuntimeError("final case_id is invalid or repeated")
        ids.add(case_id)
        family, dist = case.get("family"), case.get("dist")
        if family not in FAMILIES or dist not in DISTRIBUTIONS:
            raise RuntimeError(f"{case_id}: family or distribution is invalid")
        if not isinstance(case.get("model"), str) or not case["model"]:
            raise RuntimeError(f"{case_id}: model is invalid")
        counts[str(family)] += 1
        for name in ("g", "m", "n", "total_k"):
            case[name] = _integer(case.get(name), f"{case_id}.{name}")
        _check_shape(case_id, case["g"], case["m"], case["n"])
        if case["total_k"] > MAX_TOTAL_K or case["total_k"] % K_ALIGNMENT:
            raise RuntimeError(f"{case_id}: total_k outside the domain")
        _positive_float(case.get("hint_scale"), f"{case_id}.hint_scale")
        if family == "sequence":
            for name in ("m2", "n2"):
                case[name] = _integer(case.get(name), f"{case_id}.{name}")
            _check_shape(case_id, case["g"], case["m2"], case["n2"])
        elif case.get("m2") is not None or case.get("n2") is not None:
            raise RuntimeError(f"{case_id}: only sequence cases carry a second shape")
        if dist == "dirichlet":
            _positive_float(case.get("alpha"), f"{case_id}.alpha")
        else:
            raise RuntimeError(f"{case_id}: Judge cases use the Dirichlet distribution")
        case["seed"] = _integer(case.get("seed"), f"{case_id}.seed", minimum=FINAL_SEED_MINIMUM)
        if case["seed"] > MAX_SEED or case["seed"] in seeds:
            raise RuntimeError("final case seeds are not unique 63-bit integers")
        seeds.add(case["seed"])
        configuration = tuple(case[name] for name in sorted(SCORED_KEYS - {"case_id", "seed"}))
        if configuration in configurations:
            raise RuntimeError(f"{case_id}: repeats the configuration of another case")
        configurations.add(configuration)
        cases.append(case)
    if counts != FAMILY_CASE_COUNTS:
        raise RuntimeError(f"final manifest family counts differ: {counts}")
    return tuple(cases)


def validate_edge_manifest(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if set(payload) != {"schema_version", "suite_id", "cases"}:
        raise RuntimeError("edge manifest fields differ from schema v3")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("suite_id") != EDGE_SUITE_ID:
        raise RuntimeError("edge manifest identity differs")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != EDGE_CASE_COUNT:
        raise RuntimeError(f"edge manifest must contain exactly {EDGE_CASE_COUNT} cases")
    ids: set[str] = set()
    cases: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping) or set(raw_case) != EDGE_KEYS:
            raise RuntimeError("edge case fields differ from schema v3")
        case = dict(raw_case)
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None or case_id in ids:
            raise RuntimeError("edge case_id is invalid or repeated")
        ids.add(case_id)
        if not isinstance(case.get("category"), str) or not case["category"]:
            raise RuntimeError("edge case category is invalid")
        g = _integer(case.get("g"), f"{case_id}.g")
        m = _integer(case.get("m"), f"{case_id}.m")
        n = _integer(case.get("n"), f"{case_id}.n")
        hint = _integer(case.get("hint"), f"{case_id}.hint")
        seed = _integer(case.get("seed"), f"{case_id}.seed", minimum=0)
        if seed > MAX_SEED:
            raise RuntimeError(f"{case_id}: seed is not a 63-bit integer")
        if not EDGE_DOMAIN["g"][0] <= g <= EDGE_DOMAIN["g"][1]:
            raise RuntimeError(f"{case_id}: g outside the domain")
        for value in (m, n):
            if not EDGE_DOMAIN["mn"][0] <= value <= EDGE_DOMAIN["mn"][1]:
                raise RuntimeError(f"{case_id}: m or n outside the domain")
        if not EDGE_DOMAIN["hint"][0] <= hint <= EDGE_DOMAIN["hint"][1]:
            raise RuntimeError(f"{case_id}: hint outside the domain")
        ks = case.get("ks")
        if (
            not isinstance(ks, list)
            or len(ks) != g
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0 or v % K_ALIGNMENT
                for v in ks
            )
            or sum(ks) > MAX_TOTAL_K
        ):
            raise RuntimeError(f"{case_id}: ks outside the domain")
        if g * m * n > EDGE_MAX_OUTPUT_ELEMENTS or sum(ks) * (m + n) > EDGE_MAX_INPUT_BYTES:
            raise RuntimeError(f"{case_id}: exceeds the edge-suite size bound")
        if not isinstance(case.get("sanitize"), bool):
            raise RuntimeError(f"{case_id}: sanitize flag is invalid")
        cases.append(case)
    if sum(1 for case in cases if case["sanitize"]) != EDGE_SANITIZE_COUNT:
        raise RuntimeError(f"edge manifest must flag exactly {EDGE_SANITIZE_COUNT} sanitizer cases")
    if not any(0 in case["ks"] for case in cases) or not any(sum(case["ks"]) == 0 for case in cases):
        raise RuntimeError("edge manifest must exercise empty groups and K_total = 0")
    return tuple(cases)


def _load_manifest(
    path: Path, label: str, validator, *, expected_uid: int, max_bytes: int
) -> tuple[tuple[dict[str, Any], ...], str]:
    payload, raw = _read_root_only_json(path, label, max_bytes=max_bytes, expected_uid=expected_uid)
    return validator(payload), hashlib.sha256(raw).hexdigest()


def load_final_manifest(
    path: Path = FINAL_MANIFEST, *, expected_uid: int = 0
) -> tuple[tuple[dict[str, Any], ...], str]:
    return _load_manifest(
        path, "final holdout manifest", validate_final_manifest,
        expected_uid=expected_uid, max_bytes=FINAL_MANIFEST_MAX_BYTES,
    )


def load_edge_manifest(
    path: Path = EDGE_MANIFEST, *, expected_uid: int = 0
) -> tuple[tuple[dict[str, Any], ...], str]:
    return _load_manifest(
        path, "edge holdout manifest", validate_edge_manifest,
        expected_uid=expected_uid, max_bytes=EDGE_MANIFEST_MAX_BYTES,
    )


def locate_submission(app_dir: Path) -> Path:
    roots = (
        app_dir / "workdir/k_grouped_wgrad_lab",
        app_dir / "environment/workdir/k_grouped_wgrad_lab",
        app_dir,
    )
    matches: list[Path] = []
    for root in roots:
        try:
            root_mode = root.lstat().st_mode
        except OSError:
            continue
        if (
            stat.S_ISDIR(root_mode)
            and not stat.S_ISLNK(root_mode)
            and (root / "candidates").is_dir()
            and (root / "FINAL_VERSION").exists()
        ):
            absolute = root.absolute()
            if absolute not in matches:
                matches.append(absolute)
    if len(matches) != 1:
        raise RuntimeError(f"expected one submission root, found {len(matches)}")
    return matches[0]


def _protected_path(kind: str, relative: str) -> Path:
    if kind == "judge":
        return JUDGE_ROOT / relative
    if kind == "worker":
        return RUNTIME_ROOT / relative
    raise ValueError(f"unknown protected path kind {kind!r}")


def verify_protected_hashes(manifest: Path) -> list[str]:
    expected_labels = {
        "judge:candidate_worker.py",
        "judge:run_judge.py",
        "judge:source_admission.py",
        "judge:verify_wgrad.py",
        "worker:k_grouped_wgrad_runtime.py",
    }
    problems: list[str] = []
    seen: set[str] = set()
    if not manifest.is_file() or manifest.is_symlink():
        return ["protected hash manifest is missing"]
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return ["protected hash manifest is unreadable"]
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            expected, label = line.split(None, 1)
            kind, relative = label.split(":", 1)
        except ValueError:
            problems.append("malformed protected hash record")
            continue
        if label in seen:
            problems.append(f"duplicate protected hash record: {label}")
            continue
        seen.add(label)
        if label not in expected_labels:
            problems.append(f"unexpected protected hash record: {label}")
            continue
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            problems.append(f"invalid protected digest for {label}")
            continue
        path = _protected_path(kind, relative)
        if not path.is_file() or path.is_symlink():
            problems.append(f"missing protected file: {label}")
        elif sha256_file(path) != expected:
            problems.append(f"protected file changed: {label}")
    missing = sorted(expected_labels - seen)
    if missing:
        problems.append(f"protected hash records are missing: {missing}")
    return problems


def _agent_account() -> pwd.struct_passwd:
    return pwd.getpwnam("agent")


def _drop_to_agent() -> None:
    account = _agent_account()
    os.setgroups([])
    os.setgid(account.pw_gid)
    os.setuid(account.pw_uid)
    os.umask(0o077)


def _signal_session(session_id: int, requested_signal: signal.Signals) -> None:
    process_groups: set[int] = set()
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        entries = ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getsid(pid) == session_id:
                process_groups.add(os.getpgid(pid))
        except (OSError, ProcessLookupError):
            continue
    for process_group in sorted(process_groups, reverse=True):
        try:
            os.killpg(process_group, requested_signal)
        except (OSError, ProcessLookupError):
            pass


def _terminate_session(process: subprocess.Popen[bytes]) -> None:
    _signal_session(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    _signal_session(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _candidate_environment(cache_dir: Path) -> dict[str, str]:
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
            "TMPDIR": str(cache_dir),
            "TRITON_CACHE_DIR": str(cache_dir / "triton"),
        }
    )
    return env


def _strict_worker_result(path: Path, *, expected_uid: int) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError("worker result file is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != expected_uid
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or metadata.st_size <= 0
        or metadata.st_size > 16 * 1024 * 1024
    ):
        raise RuntimeError("worker result file is invalid")
    return _strict_json_bytes(path.read_bytes(), "worker result")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


# Capability envelope: what the task contract actually needs, not the GPU model
# it was qualified on.  The GPU name and UUID are recorded as provenance only;
# they never gate admission.  Keep these two constants identical in the Judge
# verifier (tests assert the pair).
MIN_COMPUTE_CAPABILITY = (8, 9)
MIN_DEVICE_MEMORY_BYTES = 48 * 1024**3


def _trusted_hardware() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    count = int(torch.cuda.device_count())
    if count != 1:
        raise RuntimeError(f"exactly one visible GPU is required, found {count}")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    name = str(torch.cuda.get_device_name(0))
    properties = torch.cuda.get_device_properties(0)
    total_memory_bytes = int(properties.total_memory)
    if capability < MIN_COMPUTE_CAPABILITY:
        raise RuntimeError(
            f"GPU compute capability {capability} is below the task minimum "
            f"{MIN_COMPUTE_CAPABILITY}: {name}"
        )
    if total_memory_bytes < MIN_DEVICE_MEMORY_BYTES:
        raise RuntimeError(
            f"GPU memory {total_memory_bytes} bytes is below the task minimum "
            f"{MIN_DEVICE_MEMORY_BYTES} bytes: {name}"
        )
    return {
        "name": name,
        "count": count,
        "capability": list(capability),
        "uuid": str(getattr(properties, "uuid", "")),
        "total_memory_bytes": total_memory_bytes,
        "min_compute_capability": list(MIN_COMPUTE_CAPABILITY),
        "min_device_memory_bytes": MIN_DEVICE_MEMORY_BYTES,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
    }




def _prepare_run_dirs(run_root: Path, name: str) -> tuple[pwd.struct_passwd, Path, Path]:
    account = _agent_account()
    result_dir = run_root / f"{name}_result"
    result_dir.mkdir(mode=0o700)
    cache_dir = run_root / f"{name}_cache"
    cache_dir.mkdir(mode=0o700)
    (cache_dir / "triton").mkdir(mode=0o700)
    for path in (result_dir, cache_dir, cache_dir / "triton"):
        os.chown(path, account.pw_uid, account.pw_gid)
    return account, result_dir, cache_dir


class UnprivilegedTimeout(RuntimeError):
    """A trusted child process ran past its wall-clock budget and was killed."""


def _run_unprivileged(
    command: list[str],
    *,
    stdin_payload: bytes,
    run_root: Path,
    log_name: str,
    env: dict[str, str],
    timeout: float,
) -> int:
    log_path = run_root / log_name
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=run_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_drop_to_agent,
        )
        try:
            process.communicate(input=stdin_payload, timeout=timeout)
            returncode = process.returncode
            _signal_session(process.pid, signal.SIGKILL)
        except subprocess.TimeoutExpired:
            _terminate_session(process)
            raise UnprivilegedTimeout(f"{log_name.split('.')[0]} timed out") from None
    return int(returncode)


def run_worker(
    candidate_path: Path,
    run_root: Path,
    cases: Sequence[Mapping[str, Any]],
    edge_cases: Sequence[Mapping[str, Any]],
    *,
    trials: int,
    warmups: int,
    rounds: int,
    timeout: float,
) -> dict[str, Any]:
    account, result_dir, cache_dir = _prepare_run_dirs(run_root, "worker")
    manifest_bytes = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "mode": "suite",
            "suite_id": FINAL_SUITE_ID,
            "cases": list(cases),
            "edge_suite_id": EDGE_SUITE_ID,
            "edge_cases": list(edge_cases),
        },
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    result_path = result_dir / "result.json"
    command = [
        sys.executable,
        str(WORKER),
        "--candidate",
        str(candidate_path),
        "--output",
        str(result_path),
        "--trials",
        str(trials),
        "--warmups",
        str(warmups),
        "--rounds",
        str(rounds),
    ]
    returncode = _run_unprivileged(
        command,
        stdin_payload=manifest_bytes,
        run_root=run_root,
        log_name="candidate_worker.log",
        env=_candidate_environment(cache_dir),
        timeout=timeout,
    )
    if returncode != 0:
        raise RuntimeError("candidate worker exited without a structured result")
    return _strict_worker_result(result_path, expected_uid=account.pw_uid)


def run_sanitizer(
    candidate_path: Path,
    run_root: Path,
    sanitize_cases: Sequence[Mapping[str, Any]],
    *,
    timeout: float,
) -> dict[str, Any]:
    """Run the flagged edge cases once each under compute-sanitizer memcheck."""
    if not SANITIZER.is_file():
        raise RuntimeError("compute-sanitizer is missing from the Judge image")
    account, result_dir, cache_dir = _prepare_run_dirs(run_root, "sanitizer")
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "mode": "sanitize",
            "edge_suite_id": EDGE_SUITE_ID,
            "edge_cases": list(sanitize_cases),
        },
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    result_path = result_dir / "result.json"
    env = _candidate_environment(cache_dir)
    env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    command = [
        str(SANITIZER),
        *SANITIZER_ARGUMENTS,
        sys.executable,
        str(WORKER),
        "--mode",
        "sanitize",
        "--candidate",
        str(candidate_path),
        "--output",
        str(result_path),
    ]
    try:
        returncode = _run_unprivileged(
            command,
            stdin_payload=payload,
            run_root=run_root,
            log_name="sanitizer.log",
            env=env,
            timeout=timeout,
        )
    except UnprivilegedTimeout:
        # A pass that cannot finish establishes nothing about memory safety.
        return {"returncode": None, "cases": len(sanitize_cases), "status": "timeout",
                "report": _sanitizer_report(run_root / "sanitizer.log")}
    record: dict[str, Any] = {"returncode": returncode, "cases": len(sanitize_cases)}
    report = _sanitizer_report(run_root / "sanitizer.log")
    try:
        raw_log = (run_root / "sanitizer.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        raw_log = ""
    if SANITIZER_UNAVAILABLE_MARKER in raw_log:
        # Required mode fails closed: an uninstrumented pass is not a memcheck.
        record.update(status="unavailable", report=report)
        return record
    # An invalid access can also surface as the application's own CUDA error
    # exit, so the report text decides as well as the exit code.
    invalid_access = any(
        marker in line
        for line in report
        for marker in ("Invalid __global__", "Invalid __shared__", "Invalid __local__", "Out-of-range")
    )
    if returncode == SANITIZER_EXIT_CODE or invalid_access:
        record.update(status="memory_error", report=report)
        return record
    if returncode != 0:
        record.update(status="failed", report=report)
        return record
    result = _strict_worker_result(result_path, expected_uid=account.pw_uid)
    record["status"] = "passed" if result.get("status") == "passed" else "failed"
    if record["status"] == "failed":
        gate = result.get("failed_gate")
        record["failed_gate"] = gate if gate in RUNTIME_GATES else "memory_safety_ok"
        record["private_error"] = str(result.get("private_error") or "")
    record["result"] = result
    return record


def _sanitizer_report(log_path: Path) -> list[str]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    keep = [line for line in lines if line.startswith("=========")]
    return keep[:40] if keep else lines[-20:]


def _validate_hardware_record(result: Mapping[str, Any]) -> None:
    hardware = result.get("hardware")
    if not isinstance(hardware, Mapping):
        raise RuntimeError("runtime result lacks hardware inventory")
    capability = hardware.get("capability")
    total_memory = hardware.get("total_memory_bytes")
    if (
        hardware.get("count") != 1
        or not isinstance(capability, list)
        or len(capability) != 2
        or tuple(int(v) for v in capability) < MIN_COMPUTE_CAPABILITY
        or isinstance(total_memory, bool)
        or not isinstance(total_memory, int)
        or total_memory < MIN_DEVICE_MEMORY_BYTES
    ):
        raise RuntimeError("runtime hardware inventory violates the capability envelope")


def plausibility_bound(capability: Sequence[int] | None) -> float | None:
    if not capability or len(capability) != 2:
        return None
    return FP8_DENSE_PEAK_BOUND_TFLOPS.get((int(capability[0]), int(capability[1])))


def _finite_error(value: Any, label: str, limit: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    number = float(value)
    # The normalized error can be a tiny negative value from rounding.
    if not math.isfinite(number) or number >= limit:
        raise RuntimeError(f"{label} exceeds the numerical contract")
    return number


def _numeric_list(value: Any, label: str, expected_length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise RuntimeError(f"{label} has the wrong length")
    return [finite_positive(item, label) for item in value]


def _check_identity(row: Mapping[str, Any], expected: Mapping[str, Any], names: Sequence[str]) -> None:
    case_id = str(expected["case_id"])
    for name in names:
        if row.get(name) != expected[name]:
            raise RuntimeError(f"case identity differs for {case_id}")
    if row.get("entrypoint") != WGRAD_ENTRYPOINT:
        raise RuntimeError(f"entrypoint identity differs for {case_id}")
    if row.get("expected_k_hint") != case_hint(expected):
        raise RuntimeError(f"expected-K hint differs for {case_id}")


def _validate_single_result(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_trials: int,
    bound: float | None,
) -> float:
    case_id = str(expected["case_id"])
    _check_identity(row, expected, ("case_id", "family", "model", "g", "m", "n", "total_k", "dist"))
    if row.get("decoy_precalls") != expected_trials:
        raise RuntimeError(f"decoy pre-call protocol differs for {case_id}")
    trial_us = _numeric_list(row.get("trial_us"), f"{case_id}.trial_us", expected_trials)
    host_us = _numeric_list(row.get("trial_host_us"), f"{case_id}.trial_host_us", expected_trials)
    event_us = _numeric_list(row.get("trial_event_us"), f"{case_id}.trial_event_us", expected_trials)
    trial_tflops = _numeric_list(row.get("trial_tflops"), f"{case_id}.trial_tflops", expected_trials)
    trial_ks = row.get("trial_ks")
    trial_total_k = row.get("trial_total_k")
    if (
        not isinstance(trial_ks, list)
        or not isinstance(trial_total_k, list)
        or len(trial_ks) != expected_trials
        or len(trial_total_k) != expected_trials
    ):
        raise RuntimeError(f"dynamic-K telemetry differs for {case_id}")
    for ordinal, ks in enumerate(trial_ks):
        if ks != draw_ks(expected, derive_seed(int(expected["seed"]), "ks", ordinal)):
            raise RuntimeError(f"group sizes differ from the seeded draw for {case_id}")
        total_k = trial_total_k[ordinal]
        if isinstance(total_k, bool) or not isinstance(total_k, int) or total_k != sum(ks):
            raise RuntimeError(f"total-K count differs for {case_id}")
        if not math.isclose(
            trial_us[ordinal], max(host_us[ordinal], event_us[ordinal]), rel_tol=1.0e-9, abs_tol=1.0e-9
        ):
            raise RuntimeError(f"trial clock differs from max(host, event) for {case_id}")
        computed = 2.0 * total_k * int(expected["m"]) * int(expected["n"]) / (trial_us[ordinal] * 1.0e-6) / 1.0e12
        if not math.isclose(trial_tflops[ordinal], computed, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise RuntimeError(f"trial TFLOPS differs for {case_id}")
    computed_median_us = float(statistics.median(trial_us))
    observed_median_us = finite_positive(row.get("median_us"), f"{case_id}.median_us")
    if not math.isclose(observed_median_us, computed_median_us, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError(f"median time differs for {case_id}")
    computed_case = float(statistics.median(trial_tflops))
    observed_case = finite_positive(row.get("tflops"), f"{case_id}.tflops")
    if not math.isclose(observed_case, computed_case, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError(f"case TFLOPS differs for {case_id}")
    if bound is not None and computed_case > bound:
        raise RuntimeError(f"case TFLOPS exceeds the physical bound for {case_id}")
    _finite_error(row.get("max_element_ratio"), f"{case_id}.max_element_ratio", ELEMENT_RATIO_LIMIT)
    _finite_error(row.get("max_exact_error"), f"{case_id}.max_exact_error", EXACT_ERROR_LIMIT)
    _finite_error(row.get("max_source_error"), f"{case_id}.max_source_error", SOURCE_ERROR_LIMIT)
    return computed_case


def _validate_sequence_result(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_rounds: int,
    bound: float | None,
) -> float:
    case_id = str(expected["case_id"])
    _check_identity(row, expected, ("case_id", "family", "model", "g", "m", "n", "m2", "n2", "total_k", "dist"))
    if row.get("calls_per_round") != SEQUENCE_LAYERS * 2:
        raise RuntimeError(f"sequence length differs for {case_id}")
    round_us = _numeric_list(row.get("round_us"), f"{case_id}.round_us", expected_rounds)
    host_us = _numeric_list(row.get("round_host_us"), f"{case_id}.round_host_us", expected_rounds)
    event_us = _numeric_list(row.get("round_event_us"), f"{case_id}.round_event_us", expected_rounds)
    round_flops = _numeric_list(row.get("round_flops"), f"{case_id}.round_flops", expected_rounds)
    round_tflops = _numeric_list(row.get("round_tflops"), f"{case_id}.round_tflops", expected_rounds)
    checked = row.get("checked_calls")
    if not isinstance(checked, list) or len(checked) != expected_rounds:
        raise RuntimeError(f"sequence telemetry differs for {case_id}")
    shapes = ((int(expected["m"]), int(expected["n"])), (int(expected["m2"]), int(expected["n2"])))
    for ordinal in range(expected_rounds):
        calls = sequence_schedule(expected, derive_seed(int(expected["seed"]), "sequence-round", ordinal))
        writers = {(linear, out) for linear, _layer, _ks, out in calls}
        if checked[ordinal] != len(writers):
            raise RuntimeError(f"sequence output check count differs for {case_id}")
        flops = math.fsum(2.0 * sum(ks) * shapes[linear][0] * shapes[linear][1] for linear, _layer, ks, _out in calls)
        if not math.isclose(round_flops[ordinal], flops, rel_tol=1.0e-12, abs_tol=0.0):
            raise RuntimeError(f"sequence work differs for {case_id}")
        if not math.isclose(
            round_us[ordinal], max(host_us[ordinal], event_us[ordinal]), rel_tol=1.0e-9, abs_tol=1.0e-9
        ):
            raise RuntimeError(f"round clock differs from max(host, event) for {case_id}")
        computed = flops / (round_us[ordinal] * 1.0e-6) / 1.0e12
        if not math.isclose(round_tflops[ordinal], computed, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise RuntimeError(f"round TFLOPS differs for {case_id}")
    observed_median_us = finite_positive(row.get("median_us"), f"{case_id}.median_us")
    if not math.isclose(observed_median_us, float(statistics.median(round_us)), rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError(f"median time differs for {case_id}")
    computed_case = float(statistics.median(round_tflops))
    observed_case = finite_positive(row.get("tflops"), f"{case_id}.tflops")
    if not math.isclose(observed_case, computed_case, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError(f"case TFLOPS differs for {case_id}")
    if bound is not None and computed_case > bound:
        raise RuntimeError(f"case TFLOPS exceeds the physical bound for {case_id}")
    _finite_error(row.get("max_exact_error"), f"{case_id}.max_exact_error", EXACT_ERROR_LIMIT)
    return computed_case


def _validate_case_result(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_trials: int,
    expected_rounds: int,
    bound: float | None,
) -> float:
    if expected["family"] == "sequence":
        return _validate_sequence_result(row, expected, expected_rounds, bound)
    return _validate_single_result(row, expected, expected_trials, bound)


def aggregate_families(
    values: Mapping[str, Mapping[str, Sequence[float]]],
) -> tuple[float, dict[str, float]]:
    """Per family: geometric mean per model, then across models; the primary
    score is the geometric mean of the four family scores."""
    families = {
        family: geometric_mean([geometric_mean(list(cases)) for cases in values[family].values()])
        for family in FAMILIES
    }
    return geometric_mean([families[family] for family in FAMILIES]), families


def _validate_edge_record(
    edge: Any, expected_edges: Sequence[Mapping[str, Any]]
) -> None:
    if not isinstance(edge, Mapping):
        raise RuntimeError("runtime result lacks the edge record")
    rows = edge.get("results")
    if (
        edge.get("cases_total") != len(expected_edges)
        or edge.get("cases_passed") != len(expected_edges)
        or edge.get("cases_failed") != 0
        or not isinstance(rows, list)
        or len(rows) != len(expected_edges)
    ):
        raise RuntimeError("edge-case record differs from the edge manifest")
    expected_ids = [str(case["case_id"]) for case in expected_edges]
    if [row.get("case_id") if isinstance(row, Mapping) else None for row in rows] != expected_ids:
        raise RuntimeError("edge-case results differ from the edge manifest order")
    for row in rows:
        if row.get("ok") is not True:
            raise RuntimeError("an edge case did not pass")
        _finite_error(row.get("max_element_ratio"), f"{row['case_id']}.max_element_ratio", ELEMENT_RATIO_LIMIT)
        _finite_error(row.get("max_exact_error"), f"{row['case_id']}.max_exact_error", EXACT_ERROR_LIMIT)
        _finite_error(row.get("max_source_error"), f"{row['case_id']}.max_source_error", SOURCE_ERROR_LIMIT)


def validate_runtime_result(
    result: Mapping[str, Any],
    expected_cases: Sequence[Mapping[str, Any]],
    expected_edges: Sequence[Mapping[str, Any]],
    *,
    expected_trials: int,
    expected_rounds: int,
    expected_warmups: int | None = None,
) -> tuple[float, dict[str, float]]:
    if result.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("runtime result schema differs")
    if result.get("status") != "passed":
        raise RuntimeError("runtime result is not a passing result")
    gates = result.get("gates")
    if not isinstance(gates, Mapping) or any(gates.get(name) is not True for name in RUNTIME_GATES):
        raise RuntimeError("runtime gates are not all satisfied")
    _validate_hardware_record(result)
    if result.get("trials_per_case") != expected_trials:
        raise RuntimeError("runtime trial count differs")
    if expected_warmups is not None and result.get("warmups") != expected_warmups:
        raise RuntimeError("runtime warmup count differs")
    if result.get("rounds_per_sequence") != expected_rounds:
        raise RuntimeError("runtime sequence round count differs")
    protocol = result.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("decoy_precall") is not True:
        raise RuntimeError("runtime measurement protocol differs")
    raw_entrypoints = result.get("entrypoints")
    if not isinstance(raw_entrypoints, list) or set(raw_entrypoints) != set(ENTRYPOINTS.values()):
        raise RuntimeError("runtime entrypoint set differs")
    _validate_edge_record(result.get("edge"), expected_edges)

    raw_cases = result.get("cases")
    if not isinstance(raw_cases, list):
        raise RuntimeError("runtime result lacks cases")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in raw_cases:
        if not isinstance(item, Mapping):
            raise RuntimeError("runtime case is not an object")
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in by_id:
            raise RuntimeError("runtime case IDs are invalid")
        by_id[case_id] = item
    expected_ids = {str(case["case_id"]) for case in expected_cases}
    if len(expected_ids) != len(expected_cases) or set(by_id) != expected_ids:
        raise RuntimeError("runtime case set differs from the selected manifest")

    hardware = result.get("hardware") or {}
    bound = plausibility_bound(hardware.get("capability"))
    values: dict[str, dict[str, list[float]]] = {family: {} for family in FAMILIES}
    for expected in expected_cases:
        value = _validate_case_result(
            by_id[str(expected["case_id"])], expected, expected_trials, expected_rounds, bound
        )
        values[str(expected["family"])].setdefault(str(expected["model"]), []).append(value)
    primary, families = aggregate_families(values)

    raw_families = result.get("family_tflops")
    if not isinstance(raw_families, Mapping) or set(raw_families) != set(families):
        raise RuntimeError("runtime family aggregate set differs")
    for label, expected in families.items():
        observed = finite_positive(raw_families.get(label), f"{label} runtime aggregate")
        if not math.isclose(observed, expected, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise RuntimeError(f"runtime family aggregate differs for {label}")
    observed_primary = finite_positive(result.get("primary_tflops"), "runtime primary score")
    if not math.isclose(observed_primary, primary, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError("runtime primary score differs")
    return primary, families


def _numeric_summary(detail: Mapping[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in detail.items():
        if isinstance(value, bool):
            numeric[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                numeric[key] = number
    return numeric


def write_result(log_dir: Path, detail: Mapping[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir.chmod(0o700)
    payload = dict(detail)
    score = float(payload.get("score_tflops", 0.0))
    payload["reward"] = score
    _atomic_json(log_dir / RESULT_NAME, payload)
    numeric = _numeric_summary(payload)
    numeric["reward"] = score
    _atomic_json(log_dir / "reward.json", numeric)
    temporary = log_dir / f".reward.txt.tmp.{os.getpid()}"
    temporary.write_text(f"{score}\n", encoding="utf-8")
    os.replace(temporary, log_dir / "reward.txt")


def failure_detail(
    message: str,
    *,
    final_version: int = 0,
    versions: int = 0,
    profile: str | None = None,
    suite_id: str | None = None,
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_policy": EVALUATION_POLICY,
        "selection_protocol_id": SELECTION_PROTOCOL_ID,
        **{name: 0.0 for name in GATE_NAMES},
        "failed_gate_code": 0.0,
        "profile_code": PROFILE_CODES.get(profile or "", 0.0),
        "score_tflops": 0.0,
        **{f"{family}_tflops": 0.0 for family in FAMILIES},
        "edge_cases_total": 0.0,
        "edge_cases_failed": 0.0,
        "sanitizer_ran": 0.0,
        "final_version": float(final_version),
        "num_candidate_versions": float(versions),
        "private_error": message,
    }
    if profile is not None:
        detail["evaluation_profile"] = profile
    if suite_id is not None:
        detail["suite_id"] = suite_id
    return detail


def _mark_failed_gate(detail: dict[str, Any], gate: str) -> None:
    detail[gate] = 0.0
    if gate in GATE_NAMES and not detail.get("failed_gate_code"):
        detail["failed_gate_code"] = float(GATE_NAMES.index(gate) + 1)
        detail["failed_gate"] = gate


def _apply_runtime_gates(detail: dict[str, Any], runtime_result: Mapping[str, Any]) -> None:
    gates = runtime_result.get("gates")
    if not isinstance(gates, Mapping):
        return
    for name in RUNTIME_GATES:
        detail[name] = 1.0 if gates.get(name) is True else 0.0
    edge = runtime_result.get("edge")
    if isinstance(edge, Mapping):
        for source, target in (("cases_total", "edge_cases_total"), ("cases_failed", "edge_cases_failed")):
            value = edge.get(source)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                detail[target] = float(value)


def evaluate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    protected = verify_protected_hashes(JUDGE_ROOT / "protected_hashes.txt")
    if protected:
        return 1, failure_detail("protected Judge/runtime hash failure") | {
            "protected_failures": protected
        }

    if args.sanitizer_mode not in SANITIZER_MODES:
        return 1, failure_detail("invalid compute-sanitizer mode")
    try:
        profile, suite_id, round_id, version_override = load_evaluation_profile()
    except Exception as exc:
        return 1, failure_detail(f"{type(exc).__name__}: {exc}")
    base = failure_detail("evaluation did not complete", profile=profile, suite_id=suite_id)
    base["profile_ok"] = 1.0
    base["round_id"] = round_id
    if version_override is not None:
        base["candidate_version_override"] = float(version_override)

    try:
        cases, manifest_sha256 = load_final_manifest()
        edge_cases, edge_manifest_sha256 = load_edge_manifest()
        base["suite_ok"] = 1.0
        base["manifest_sha256"] = manifest_sha256
        base["edge_manifest_sha256"] = edge_manifest_sha256
        base["case_count"] = float(len(cases))
        base["edge_cases_total"] = float(len(edge_cases))
    except Exception as exc:
        base["private_error"] = f"{type(exc).__name__}: {exc}"
        _mark_failed_gate(base, "suite_ok")
        return 1, base

    submission_root = locate_submission(args.app_dir.resolve())
    admission = validate_submission(submission_root, version_override)
    final_version = admission.get("final_version")
    versions = admission.get("num_candidate_versions")
    version_count = versions if isinstance(versions, int) and not isinstance(versions, bool) else 0
    selected = final_version if isinstance(final_version, int) and not isinstance(final_version, bool) else 0
    base["final_version"] = float(selected)
    base["num_candidate_versions"] = float(version_count)
    if not admission.get("ok") or selected <= 0:
        base["private_error"] = "submission source admission failed"
        base["admission"] = admission
        _mark_failed_gate(base, "source_ok")
        return 1, base
    base["source_ok"] = 1.0

    if args.trials <= 0 or args.warmups <= 0 or args.rounds <= 0:
        base["private_error"] = "invalid trial, warmup or round count"
        return 1, base
    total_timeout = float(args.total_timeout)
    if not math.isfinite(total_timeout) or total_timeout <= 60.0:
        base["private_error"] = "invalid verifier total timeout"
        return 1, base

    started = time.monotonic()
    deadline = started + total_timeout
    run_root = Path(tempfile.mkdtemp(prefix="standalone-kgrouped-wgrad-v3-", dir="/tmp"))
    run_root.chmod(0o755)
    try:
        try:
            hardware = _trusted_hardware()
        except Exception as exc:
            base["private_error"] = f"{type(exc).__name__}: {exc}"
            _mark_failed_gate(base, "hardware_ok")
            return 1, base
        base["hardware_ok"] = 1.0
        selected_source = Path(str(admission["selected_path"]))
        candidate_dir = run_root / "selected_candidate"
        candidate_dir.mkdir(mode=0o755)
        candidate_path = candidate_dir / "submission.py"
        shutil.copyfile(selected_source, candidate_path)
        candidate_path.chmod(0o444)
        candidate_dir.chmod(0o555)

        # The sanitizer subset runs first, before any native call: memcheck stops
        # an out-of-bounds kernel before the access reaches the hardware, where
        # a wild pointer can wedge the shared host driver.
        if args.sanitizer_mode == "required":
            sanitize_cases = [case for case in edge_cases if case["sanitize"]]
            remaining = deadline - time.monotonic() - 60.0
            if remaining <= 0.0:
                raise RuntimeError("Judge total deadline exhausted before the sanitizer pass")
            sanitizer = run_sanitizer(
                candidate_path,
                run_root,
                sanitize_cases,
                timeout=min(float(args.sanitizer_timeout), remaining),
            )
            base["sanitizer"] = sanitizer
            if sanitizer.get("status") != "passed":
                gate = str(sanitizer.get("failed_gate") or "memory_safety_ok")
                _mark_failed_gate(base, gate if gate in GATE_NAMES else "memory_safety_ok")
                base["private_error"] = (
                    "compute-sanitizer memcheck reported invalid memory accesses"
                    if sanitizer.get("status") == "memory_error"
                    else "the compute-sanitizer pass did not finish within its time limit"
                    if sanitizer.get("status") == "timeout"
                    else "compute-sanitizer could not attach to the worker; a deployment where it cannot "
                    "attach must select the skip-platform-incompatible sanitizer mode"
                    if sanitizer.get("status") == "unavailable"
                    else str(sanitizer.get("private_error") or "the compute-sanitizer pass did not complete")
                )
                return 1, base
            base["sanitizer_ran"] = 1.0
        else:
            base["sanitizer"] = {
                "status": "not_run",
                "mode": args.sanitizer_mode,
                "reason": "compute-sanitizer cannot attach to CUDA processes under this "
                "deployment's GPU virtualization hook",
            }

        remaining = deadline - time.monotonic() - 30.0
        if remaining <= 0.0:
            raise RuntimeError("Judge total deadline exhausted")
        runtime_result = run_worker(
            candidate_path,
            run_root,
            cases,
            edge_cases,
            trials=args.trials,
            warmups=args.warmups,
            rounds=args.rounds,
            timeout=min(float(args.candidate_timeout), remaining),
        )
        _apply_runtime_gates(base, runtime_result)
        if runtime_result.get("candidate_sha256") not in (None, admission["selected_sha256"]):
            raise RuntimeError("runtime evaluated a different candidate source")
        if runtime_result.get("status") != "passed":
            gate = str(runtime_result.get("failed_gate") or "correctness_ok")
            _mark_failed_gate(base, gate if gate in GATE_NAMES else "correctness_ok")
            base["private_error"] = str(runtime_result.get("private_error") or "runtime gate failure")
            base["private_runtime_result"] = runtime_result
            return 1, base
        if runtime_result.get("candidate_sha256") != admission["selected_sha256"]:
            raise RuntimeError("runtime evaluated a different candidate source")
        score, families = validate_runtime_result(
            runtime_result,
            cases,
            edge_cases,
            expected_trials=args.trials,
            expected_rounds=args.rounds,
            expected_warmups=args.warmups,
        )

        detail = dict(base)
        detail.update(
            {
                "score_tflops": score,
                **{f"{family}_tflops": families[family] for family in FAMILIES},
                "selected_source_sha256": admission["selected_sha256"],
                "score_method": "geomean over four families of the per-model geomean within each family",
                "wall_time_seconds": time.monotonic() - started,
                "hardware": hardware,
                "private_runtime_result": runtime_result,
            }
        )
        detail["private_error"] = ""
        return 0, detail
    except Exception as exc:
        base["private_error"] = f"{type(exc).__name__}: {exc}"
        if not base.get("failed_gate_code"):
            _mark_failed_gate(base, "correctness_ok")
        return 1, base
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, default=Path(os.environ.get("APP_DIR", "/app")))
    parser.add_argument(
        "--log-dir", type=Path, default=Path(os.environ.get("LOG_DIR", "/logs/verifier"))
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=int(os.environ.get("STANDALONE_KGROUPED_WGRAD_VERIFY_TRIALS", "3")),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=int(os.environ.get("STANDALONE_KGROUPED_WGRAD_VERIFY_ROUNDS", "3")),
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=int(os.environ.get("STANDALONE_KGROUPED_WGRAD_VERIFY_WARMUPS", "1")),
    )
    parser.add_argument(
        "--candidate-timeout",
        type=int,
        default=int(os.environ.get("STANDALONE_KGROUPED_WGRAD_CANDIDATE_TIMEOUT", "3600")),
    )
    parser.add_argument(
        "--sanitizer-timeout",
        type=int,
        default=int(os.environ.get("STANDALONE_KGROUPED_WGRAD_SANITIZER_TIMEOUT", "780")),
    )
    parser.add_argument(
        "--sanitizer-mode",
        default=os.environ.get("STANDALONE_KGROUPED_WGRAD_SANITIZER_MODE", "required"),
    )
    parser.add_argument(
        "--total-timeout",
        type=int,
        default=int(os.environ.get("STANDALONE_KGROUPED_WGRAD_TOTAL_TIMEOUT", "4500")),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rc, detail = evaluate(args)
    except Exception as exc:
        detail = failure_detail(f"{type(exc).__name__}: {exc}")
        rc = 1
    write_result(args.log_dir.resolve(), detail)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
