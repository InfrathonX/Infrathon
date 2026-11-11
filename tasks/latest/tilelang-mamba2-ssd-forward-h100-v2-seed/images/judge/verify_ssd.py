#!/usr/bin/env python3
"""Private verifier for TileLang Mamba-2 SSD forward H100 v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pwd
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


JUDGE_ROOT = Path("/opt/tilelang_mamba2_ssd_judge")
RUNTIME_ROOT = Path("/opt/tilelang_mamba2_ssd_worker")
WORKER = JUDGE_ROOT / "candidate_worker.py"
EVALUATION_CONTEXT = Path("/tmp/.sforge-evaluation-context.json")
FINAL_MANIFEST = JUDGE_ROOT / "private/final_cases.json"
RESULT_NAME = "tilelang_mamba2_ssd_v2_results.json"
FINAL_SUITE_ID = "tilelang-ssd-final-v3"
MANIFEST_SCHEMA_VERSION = 3
PROFILE_CODES = {"auto": 1.0, "final": 2.0}
EVALUATION_POLICY = "full-final-shared-v2"
SELECTION_PROTOCOL_ID = "tilelang-mamba2-v2-shared-final"
SSD_ENTRYPOINT = "ssd_forward"
WORKER_GATE_FAILURE_EXIT = 3
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
LAYER_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,95}$")
CASE_KEYS = {
    "case_id", "geometry", "family", "tier", "batch", "seqlen", "regime", "seed",
    "check_trial", "layer",
}

# The workload model.  These tables are duplicated from the trusted runtime on
# purpose: the Judge must not depend on an importable copy of them, and a
# task-local test asserts the two agree.
GEOMETRIES = {
    "mamba2_2p7b": {"h": 80, "p": 64, "n": 128, "g": 1},
    "codestral_7b": {"h": 128, "p": 64, "n": 128, "g": 8},
    "falcon_h1_34b": {"h": 32, "p": 128, "n": 256, "g": 2},
    "nemotron_h_47b": {"h": 256, "p": 64, "n": 256, "g": 8},
    "granite_h_small": {"h": 128, "p": 64, "n": 128, "g": 1},
    "zamba2_7b": {"h": 112, "p": 64, "n": 64, "g": 2},
    "nemotron_3_nano": {"h": 64, "p": 64, "n": 128, "g": 8},
    "granite_h_micro": {"h": 64, "p": 64, "n": 128, "g": 1},
    "falcon_h1_7b": {"h": 24, "p": 128, "n": 256, "g": 1},
    "falcon_h1_3b": {"h": 32, "p": 128, "n": 256, "g": 1},
    "nemotron_h_47b_tp2": {"h": 128, "p": 64, "n": 256, "g": 4},
    "nemotron_h_47b_tp4": {"h": 64, "p": 64, "n": 256, "g": 2},
    "nemotron_h_47b_tp8": {"h": 32, "p": 64, "n": 256, "g": 1},
    "codestral_7b_tp4": {"h": 32, "p": 64, "n": 128, "g": 2},
    "falcon_h1_34b_tp2": {"h": 16, "p": 128, "n": 256, "g": 1},
}
SCORE_FAMILIES = ("batch", "serving", "long", "real")
TIERS = ("recipe", "sweep", "real")
FAMILY_WEIGHTS = {family: 1.0 / len(SCORE_FAMILIES) for family in SCORE_FAMILIES}
FINAL_FAMILY_COUNTS = {"batch": 42, "serving": 54, "long": 18, "real": 24}
EXPECTED_CASE_COUNT = sum(FINAL_FAMILY_COUNTS.values())
REAL_TRIALS = 1
MIN_LONG_LENGTH = 60000
MAX_BATCH = 16
MAX_TENSOR_ELEMENTS = 2**31 - 1
# Disclosed public seeds (the Work runtime's PUBLIC_SEED_BASE + index); a final
# seed must never coincide with one of them.
PUBLIC_SEED_BASE = 0x0066_0301
PUBLIC_CASE_COUNT = 24
PUBLIC_SEEDS = frozenset(PUBLIC_SEED_BASE + index for index in range(PUBLIC_CASE_COUNT))

OUT_REL_L2_TOL = 6.0e-3
OUT_SCALED_ABS_TOL = 1.5e-2
OUT_REL_TOL = 3.01 / 128
REAL_LAYER_REL_L2_TOL = 1.3e-2
WARMUP_WALL_CAP_SECONDS = 300.0
WORKSPACE_MULTIPLE = 4.0
WORKSPACE_CAP_BYTES = 24 * 1024**3


def shape_in_domain(geometry: Mapping[str, int], batch: int, seqlen: int) -> bool:
    return (
        1 <= batch <= MAX_BATCH
        and seqlen >= 1
        and batch * seqlen * geometry["h"] * geometry["p"] <= MAX_TENSOR_ELEMENTS
        and batch * seqlen * geometry["g"] * geometry["n"] <= MAX_TENSOR_ELEMENTS
    )

# ATen operators a TileLang-only candidate may dispatch: allocations and views.
# Duplicated from the trusted runtime; a task-local test asserts the two agree.
ALLOWED_TORCH_OPS = frozenset(
    {
        "aten::empty.memory_format",
        "aten::empty_strided",
        "aten::empty_like",
        "aten::new_empty",
        "aten::new_empty_strided",
        "aten::zeros",
        "aten::zeros_like",
        "aten::new_zeros",
        "aten::zero_",
        "aten::full",
        "aten::full_like",
        "aten::new_full",
        "aten::fill_.Scalar",
        "aten::alias",
        "aten::as_strided",
        "aten::detach",
        "aten::expand",
        "aten::narrow",
        "aten::permute",
        "aten::select.int",
        "aten::slice.Tensor",
        "aten::split.Tensor",
        "aten::split_with_sizes",
        "aten::squeeze",
        "aten::squeeze.dim",
        "aten::squeeze.dims",
        "aten::t",
        "aten::transpose.int",
        "aten::unbind.int",
        "aten::unsqueeze",
        "aten::view",
        "aten::view.dtype",
        "aten::_reshape_alias",
        "aten::_unsafe_view",
    }
)

# Correctness-only fixture identities; the payload lives in the trusted runtime.
EDGE_CASE_IDS = (
    "edge-len1",
    "edge-len3",
    "edge-len63",
    "edge-len64",
    "edge-len65",
    "edge-len129",
    "edge-len257",
    "edge-len513",
    "edge-slow-decay",
    "edge-slow-decay-g8",
    "edge-slow-decay-g2",
    "edge-fast-decay",
    "edge-unit-dt",
    "edge-no-residual",
    "edge-residual-only",
    "edge-long-tail-mamba2",
    "edge-long-tail-codestral",
    "edge-long-tail-falcon",
    "edge-largest-state-l777",
    "edge-one-group-b2",
    "edge-narrow-long-tail",
    "edge-long-context-tail",
    "edge-grouped-long-tail",
    "edge-small-state-b3",
)


class WorkerGateFailure(RuntimeError):
    """The candidate worker attributed its failure to one named gate."""

    def __init__(self, gate: str, message: str) -> None:
        super().__init__(message)
        self.gate = gate


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
    # Interactive authority still cannot override the submitted selector.
    # Its measurement protocol is now the complete frozen final suite;
    # agent-/auto- IDs remain unchanged for cooldowns and history.
    for interactive in ("auto-", "agent-"):
        if round_id.startswith(interactive) and len(round_id) > len(interactive):
            if version_override is not None:
                raise RuntimeError(
                    "candidate_version override is host-side only"
                )
            return "final", FINAL_SUITE_ID, round_id, None
    # Host-side rounds use the same suite and retain per-version replay
    # authority: manual- is one sforge eval and candidate- is one replay point.
    for host_side in ("manual-", "candidate-"):
        if round_id.startswith(host_side) and len(round_id) > len(host_side):
            return "final", FINAL_SUITE_ID, round_id, version_override
    raise RuntimeError("SForge evaluation round_id has an unknown profile")


def _integer(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{label} must be an integer >= {minimum}")
    return value


def _validate_final_case(case: dict[str, Any], case_id: str) -> None:
    """The workload-family rules of the trusted runtime's validate_case."""
    geometry = GEOMETRIES[case["geometry"]]
    batch, seqlen, family = case["batch"], case["seqlen"], case["family"]
    if not shape_in_domain(geometry, batch, seqlen):
        raise RuntimeError(f"final case {case_id} is outside the support domain")
    if family == "batch" and batch < 2:
        raise RuntimeError(f"final batch case {case_id} needs batch >= 2")
    if family in ("serving", "long", "real") and batch != 1:
        raise RuntimeError(f"final {family} case {case_id} must run at batch 1")
    if family == "long" and seqlen < MIN_LONG_LENGTH:
        raise RuntimeError(f"final long case {case_id} is shorter than {MIN_LONG_LENGTH}")
    if (family == "real") != (case["tier"] == "real"):
        raise RuntimeError(f"final case {case_id} mixes the real tier and family")
    layer = case["layer"]
    if family == "real":
        if not isinstance(layer, str) or LAYER_RE.fullmatch(layer) is None or case["check_trial"] != 0:
            raise RuntimeError(f"final real case {case_id} must name its layer and check trial 0")
    elif layer is not None:
        raise RuntimeError(f"final case {case_id} names a layer but is not real")


def validate_final_manifest(
    payload: Mapping[str, Any], *, trials: int = 3
) -> tuple[dict[str, Any], ...]:
    if set(payload) != {"schema_version", "suite_id", "cases"}:
        raise RuntimeError("final manifest fields differ from schema v3")
    if (
        payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or payload.get("suite_id") != FINAL_SUITE_ID
    ):
        raise RuntimeError("final manifest identity differs")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != EXPECTED_CASE_COUNT:
        raise RuntimeError(
            f"final manifest must contain exactly {EXPECTED_CASE_COUNT} cases"
        )

    cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    seeds: set[int] = set()
    shapes: set[tuple[str, int, int]] = set()
    layers: set[str] = set()
    counts: dict[str, int] = {}
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping) or set(raw_case) != CASE_KEYS:
            raise RuntimeError("final case fields differ from schema v3")
        case = dict(raw_case)
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None:
            raise RuntimeError("final case_id is invalid")
        if case_id in ids:
            raise RuntimeError("final case IDs are not unique")
        ids.add(case_id)
        if case.get("geometry") not in GEOMETRIES:
            raise RuntimeError(f"final case {case_id} geometry is invalid")
        if case.get("family") not in SCORE_FAMILIES:
            raise RuntimeError(f"final case {case_id} family is invalid")
        if case.get("tier") not in TIERS:
            raise RuntimeError(f"final case {case_id} tier is invalid")
        if case.get("regime") != "mixed":
            raise RuntimeError("final case regime differs from the scored contract")
        for field in ("batch", "seqlen"):
            case[field] = _integer(case.get(field), f"{case_id}.{field}")
        case["check_trial"] = _integer(case.get("check_trial"), f"{case_id}.check_trial", minimum=0)
        case_trials = REAL_TRIALS if case["family"] == "real" else trials
        if case["check_trial"] >= case_trials:
            raise RuntimeError(f"final case {case_id} checks a trial that does not run")
        case["seed"] = _integer(case.get("seed"), f"{case_id}.seed", minimum=0)
        if case["seed"] > 0x7FFF_FFFF_FFFF_FFFF:
            raise RuntimeError("final case seed exceeds 63 bits")
        if case["seed"] in seeds:
            raise RuntimeError("final case seeds are not unique")
        if case["seed"] in PUBLIC_SEEDS:
            raise RuntimeError("final case seed collides with a disclosed public seed")
        seeds.add(case["seed"])
        _validate_final_case(case, case_id)
        if case["family"] == "real":
            if case["layer"] in layers:
                raise RuntimeError("final cases repeat a captured layer")
            layers.add(case["layer"])
        else:
            shape = (case["geometry"], case["batch"], case["seqlen"])
            if shape in shapes:
                raise RuntimeError("final cases repeat a synthetic shape")
            shapes.add(shape)
        counts[case["family"]] = counts.get(case["family"], 0) + 1
        cases.append(case)

    if counts != FINAL_FAMILY_COUNTS:
        raise RuntimeError("final cases do not match the final suite's family counts")
    return tuple(cases)


def load_final_manifest(
    path: Path = FINAL_MANIFEST,
    *,
    expected_uid: int = 0,
    trials: int = 3,
) -> tuple[tuple[dict[str, Any], ...], str]:
    payload, raw = _read_root_only_json(
        path,
        "final holdout manifest",
        max_bytes=256 * 1024,
        expected_uid=expected_uid,
    )
    return validate_final_manifest(payload, trials=trials), hashlib.sha256(raw).hexdigest()


def locate_submission(app_dir: Path) -> Path:
    roots = (
        app_dir / "workdir/mamba2_ssd_lab",
        app_dir / "environment/workdir/mamba2_ssd_lab",
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
        "judge:verify_ssd.py",
        "worker:mamba2_ssd_runtime.py",
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


def _candidate_environment(private_root: Path) -> dict[str, str]:
    allowed = (
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
    )
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CUDA_PATH": "/usr/local/cuda",
            "HOME": "/home/agent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/cuda/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(private_root / "tmp"),
            "TRITON_CACHE_DIR": str(private_root / "triton"),
            "TILELANG_CACHE_DIR": str(private_root / "tilelang-cache"),
            "TILELANG_TMP_DIR": str(private_root / "tilelang-tmp"),
            "TILELANG_DISABLE_CACHE": "1",
            "TILELANG_AUTO_TUNING_DISABLE_CACHE": "1",
            "TILELANG_KERNEL_CACHE_USE_LIB_STAMP": "1",
            "TILELANG_CLEANUP_TEMP_FILES": "1",
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
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 73 * 1024**3


def _trusted_hardware() -> dict[str, Any]:
    import torch
    import tilelang

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    count = int(torch.cuda.device_count())
    if count != 1:
        raise RuntimeError(f"exactly one visible GPU is required, found {count}")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    name = str(torch.cuda.get_device_name(0))
    total_memory_bytes = int(torch.cuda.get_device_properties(0).total_memory)
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
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "tilelang_version": str(tilelang.__version__),
    }


def run_worker(
    candidate_path: Path,
    run_root: Path,
    cases: Sequence[Mapping[str, Any]],
    suite_id: str,
    *,
    trials: int,
    warmups: int,
    timeout: float,
) -> dict[str, Any]:
    account = _agent_account()
    manifest_bytes = json.dumps(
        {"schema_version": MANIFEST_SCHEMA_VERSION, "suite_id": suite_id, "cases": list(cases)},
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    result_dir = run_root / "result"
    result_dir.mkdir(mode=0o700)
    private_root = run_root / "candidate-private"
    private_root.mkdir(mode=0o700)
    private_dirs = tuple(
        private_root / name
        for name in ("tmp", "triton", "tilelang-cache", "tilelang-tmp")
    )
    for path in private_dirs:
        path.mkdir(mode=0o700)
    for path in (result_dir, private_root, *private_dirs):
        os.chown(path, account.pw_uid, account.pw_gid)
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
    ]
    log_path = run_root / "candidate_worker.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=run_root,
            env=_candidate_environment(private_root),
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_drop_to_agent,
        )
        try:
            process.communicate(input=manifest_bytes, timeout=timeout)
            returncode = process.returncode
            _signal_session(process.pid, signal.SIGKILL)
        except subprocess.TimeoutExpired:
            _terminate_session(process)
            raise RuntimeError("candidate evaluation timed out") from None
    if returncode == WORKER_GATE_FAILURE_EXIT:
        record = _strict_worker_result(result_path, expected_uid=account.pw_uid)
        failure = record.get("gate_failure")
        if (
            isinstance(failure, Mapping)
            and failure.get("gate") == "tilelang_only_ok"
            and isinstance(failure.get("detail"), str)
        ):
            raise WorkerGateFailure("tilelang_only_ok", failure["detail"][:2000])
        raise RuntimeError("candidate worker reported a malformed gate failure")
    if returncode != 0:
        raise RuntimeError("candidate import, JIT, launch, or correctness evaluation failed")
    return _strict_worker_result(result_path, expected_uid=account.pw_uid)


def _capability_meets_envelope(hardware: Mapping[str, Any]) -> bool:
    """Parse the sm_XY capability string and check it plus the memory floor."""
    match = re.fullmatch(r"sm_(\d+)(\d)", str(hardware.get("capability", "")))
    if match is None:
        return False
    capability = (int(match.group(1)), int(match.group(2)))
    total_memory = hardware.get("total_memory_bytes")
    if isinstance(total_memory, bool) or not isinstance(total_memory, int):
        return False
    return capability >= MIN_COMPUTE_CAPABILITY and total_memory >= MIN_DEVICE_MEMORY_BYTES


def _validate_hardware_record(result: Mapping[str, Any]) -> None:
    hardware = result.get("hardware")
    if not isinstance(hardware, Mapping):
        raise RuntimeError("runtime result lacks hardware inventory")
    if not _capability_meets_envelope(hardware):
        raise RuntimeError("runtime hardware inventory violates the capability envelope")
    if not isinstance(hardware.get("device_name"), str) or not hardware["device_name"]:
        raise RuntimeError("runtime hardware inventory lacks a device name")
    if not isinstance(hardware.get("tilelang_version"), str) or not hardware["tilelang_version"]:
        raise RuntimeError("runtime hardware inventory lacks tilelang_version")


def _bounded_error(value: Any, label: str, limit: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > limit:
        raise RuntimeError(f"{label} is outside the numerical contract")
    return number


def _numeric_list(value: Any, label: str, expected_length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise RuntimeError(f"{label} has the wrong length")
    return [finite_positive(item, label) for item in value]


def _validate_accuracy_metrics(row: Mapping[str, Any], case_id: str) -> None:
    _bounded_error(row.get("rel_l2"), f"{case_id}.rel_l2", OUT_REL_L2_TOL)
    _bounded_error(row.get("max_abs"), f"{case_id}.max_abs", math.inf)
    # The elementwise rule is scaled-absolute OR relative, and the runtime fails
    # the case on any element that violates both.  Its two maxima normally come
    # from different elements, so neither is an independent threshold: a large
    # element with a small relative error passes (real layers do this), and a
    # cancellation makes max_rel large for a correct output.  Both are finite
    # telemetry here; the verdict is the runtime's violation count plus rel_l2.
    _bounded_error(row.get("max_abs_scaled"), f"{case_id}.max_abs_scaled", math.inf)
    _bounded_error(row.get("max_rel"), f"{case_id}.max_rel", math.inf)


def _validate_torch_ops(row: Mapping[str, Any], label: str) -> None:
    """Every row must carry the TileLang-only record, and it must hold only
    allocation and view operators."""
    if row.get("tilelang_only_ok") is not True:
        raise RuntimeError(f"{label} lacks a passing TileLang-only operator record")
    ops = row.get("torch_ops")
    if not isinstance(ops, list) or not all(isinstance(op, str) for op in ops):
        raise RuntimeError(f"{label} operator record is malformed")
    forbidden = sorted(set(ops) - ALLOWED_TORCH_OPS)
    if forbidden:
        raise RuntimeError(f"{label} dispatched host torch operators {forbidden}")


def _validate_edge_results(result: Mapping[str, Any]) -> None:
    rows = result.get("edge_fixtures")
    if not isinstance(rows, list) or len(rows) != len(EDGE_CASE_IDS):
        raise RuntimeError("edge fixture result set differs")
    reported_ids = tuple(
        row.get("case_id") if isinstance(row, Mapping) else None for row in rows
    )
    if reported_ids != EDGE_CASE_IDS:
        raise RuntimeError("edge fixture identity differs from the frozen gate")
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("edge fixture result is not an object")
        case_id = str(row["case_id"])
        if row.get("scored") is not False:
            raise RuntimeError("an edge fixture must never be scored")
        if row.get("family") != "edge" or row.get("geometry") not in GEOMETRIES:
            raise RuntimeError(f"edge fixture {case_id} reports an invalid identity")
        _validate_accuracy_metrics(row, case_id)
        _validate_torch_ops(row, f"edge fixture {case_id}")


def _effective_flops(case: Mapping[str, Any]) -> float:
    geometry = GEOMETRIES[str(case["geometry"])]
    return (
        4.0
        * float(case["batch"])
        * float(case["seqlen"])
        * float(geometry["h"])
        * float(geometry["p"])
        * float(geometry["n"])
    )


def _io_bytes(case: Mapping[str, Any]) -> int:
    geometry = GEOMETRIES[str(case["geometry"])]
    batch = int(case["batch"])
    seqlen = int(case["seqlen"])
    heads = int(geometry["h"])
    headdim = int(geometry["p"])
    groups = int(geometry["g"])
    dstate = int(geometry["n"])
    # x and out are bf16, dt is fp32, and B/C are bf16.  A and D are excluded,
    # exactly as PreparedInvocation.io_bytes excludes them in the trusted runtime.
    return 4 * batch * seqlen * (heads * headdim + heads + groups * dstate)


def _validate_case_result(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_trials: int,
) -> float:
    case_id = str(expected["case_id"])
    for field in ("case_id", "family", "geometry", "tier", "batch", "seqlen"):
        if row.get(field) != expected[field]:
            raise RuntimeError(f"case identity differs for {case_id}")
    if row.get("checked_trial") != expected["check_trial"]:
        raise RuntimeError(f"checked trial differs for {case_id}")
    if row.get("scored") is not True:
        raise RuntimeError(f"case {case_id} is not marked scored")
    if row.get("regime") != "mixed":
        raise RuntimeError(f"case regime differs for {case_id}")
    if expected["family"] == "real":
        expected_trials = REAL_TRIALS

    geometry = GEOMETRIES[str(expected["geometry"])]
    for field, key in (
        ("heads", "h"),
        ("headdim", "p"),
        ("dstate", "n"),
        ("groups", "g"),
    ):
        if row.get(field) != geometry[key]:
            raise RuntimeError(f"family geometry differs for {case_id}")
    if row.get("trials") != expected_trials:
        raise RuntimeError(f"trial count differs for {case_id}")

    trial_seconds = _numeric_list(
        row.get("trial_seconds"), f"{case_id}.trial_seconds", expected_trials
    )
    host_walls = _numeric_list(
        row.get("host_wall_seconds"),
        f"{case_id}.host_wall_seconds",
        expected_trials,
    )
    event_seconds = _numeric_list(
        row.get("event_seconds"), f"{case_id}.event_seconds", expected_trials
    )
    for ordinal, seconds in enumerate(trial_seconds):
        if not math.isclose(
            seconds,
            max(host_walls[ordinal], event_seconds[ordinal]),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(f"trusted timing envelope differs for {case_id}")

    ordered = sorted(trial_seconds)
    computed_median = ordered[len(ordered) // 2]
    observed_median = finite_positive(
        row.get("median_seconds"), f"{case_id}.median_seconds"
    )
    if not math.isclose(
        observed_median, computed_median, rel_tol=1.0e-9, abs_tol=1.0e-12
    ):
        raise RuntimeError(f"median time differs for {case_id}")
    observed_min = finite_positive(row.get("min_seconds"), f"{case_id}.min_seconds")
    observed_max = finite_positive(row.get("max_seconds"), f"{case_id}.max_seconds")
    if not math.isclose(observed_min, ordered[0], rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError(f"minimum time differs for {case_id}")
    if not math.isclose(observed_max, ordered[-1], rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError(f"maximum time differs for {case_id}")

    warmup_wall = _bounded_error(
        row.get("warmup_wall_seconds"),
        f"{case_id}.warmup_wall_seconds",
        WARMUP_WALL_CAP_SECONDS,
    )
    if warmup_wall < 0.0:  # Defensive clarity; _bounded_error already enforces it.
        raise RuntimeError(f"warm-up wall time differs for {case_id}")
    workspace = row.get("workspace_bytes")
    if isinstance(workspace, bool) or not isinstance(workspace, int) or workspace < 0:
        raise RuntimeError(f"{case_id}.workspace_bytes must be a non-negative integer")
    if workspace > min(WORKSPACE_MULTIPLE * _io_bytes(expected), float(WORKSPACE_CAP_BYTES)):
        raise RuntimeError(f"workspace limit exceeded for {case_id}")

    computed_flops = _effective_flops(expected)
    observed_flops = finite_positive(
        row.get("effective_flops"), f"{case_id}.effective_flops"
    )
    if not math.isclose(
        observed_flops, computed_flops, rel_tol=1.0e-12, abs_tol=1.0e-3
    ):
        raise RuntimeError(f"effective work differs for {case_id}")
    computed_tflops = computed_flops / observed_median / 1.0e12
    observed_tflops = finite_positive(row.get("tflops"), f"{case_id}.tflops")
    if not math.isclose(
        observed_tflops, computed_tflops, rel_tol=1.0e-9, abs_tol=1.0e-12
    ):
        raise RuntimeError(f"case TFLOPS differs for {case_id}")

    _validate_accuracy_metrics(row, case_id)
    if expected["family"] == "real":
        # The captured-layer trial is gated on relative L2 against its own limit;
        # the warm-up on generated tensors keeps the generated-case rule above.
        _bounded_error(row.get("real_rel_l2"), f"{case_id}.real_rel_l2", REAL_LAYER_REL_L2_TOL)
        for key in ("real_max_abs", "real_max_abs_scaled", "real_max_rel", "real_elementwise_violations"):
            _bounded_error(row.get(key), f"{case_id}.{key}", math.inf)
    if row.get("fresh_output_ok") is not True or row.get("fresh_input_ok") is not True:
        raise RuntimeError(f"fresh-pointer gate failed for {case_id}")
    _validate_torch_ops(row, f"case {case_id}")
    return computed_tflops


def weighted_family_score(families: Mapping[str, float]) -> float:
    total = math.fsum(FAMILY_WEIGHTS[name] for name in families)
    return math.exp(
        math.fsum(
            FAMILY_WEIGHTS[name] * math.log(finite_positive(value, f"{name} family mean"))
            for name, value in families.items()
        )
        / total
    )


def validate_runtime_result(
    result: Mapping[str, Any],
    expected_cases: Sequence[Mapping[str, Any]],
    *,
    expected_trials: int,
    expected_warmups: int | None = None,
    expected_suite_id: str = FINAL_SUITE_ID,
) -> tuple[float, dict[str, float], dict[str, float]]:
    if result.get("suite_id") != expected_suite_id:
        raise RuntimeError("runtime suite identity differs")
    _validate_hardware_record(result)
    if result.get("entrypoint") != SSD_ENTRYPOINT:
        raise RuntimeError("runtime entrypoint differs")
    if result.get("trials") != expected_trials:
        raise RuntimeError("runtime trial count differs")
    if result.get("real_trials") != REAL_TRIALS:
        raise RuntimeError("runtime real-layer trial count differs")
    if expected_warmups is not None and result.get("warmups") != expected_warmups:
        raise RuntimeError("runtime warmup count differs")
    if result.get("allowed_torch_ops") != sorted(ALLOWED_TORCH_OPS):
        raise RuntimeError("runtime TileLang-only operator allowlist differs")

    tolerances = result.get("tolerances")
    expected_tolerances = {
        "rel_l2": OUT_REL_L2_TOL,
        "scaled_abs": OUT_SCALED_ABS_TOL,
        "rel": OUT_REL_TOL,
        "real_layer_rel_l2": REAL_LAYER_REL_L2_TOL,
    }
    if not isinstance(tolerances, Mapping) or set(tolerances) != set(expected_tolerances):
        raise RuntimeError("runtime tolerance record differs")
    for key, expected_value in expected_tolerances.items():
        value = tolerances.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(
                float(value), expected_value, rel_tol=0.0, abs_tol=0.0
            )
        ):
            raise RuntimeError(f"runtime tolerance differs for {key}")

    _validate_edge_results(result)
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

    family_values: dict[str, list[float]] = {family: [] for family in SCORE_FAMILIES}
    geometry_values: dict[str, list[float]] = {}
    for expected in expected_cases:
        value = _validate_case_result(
            by_id[str(expected["case_id"])], expected, expected_trials
        )
        family_values[str(expected["family"])].append(value)
        geometry_values.setdefault(str(expected["geometry"]), []).append(value)
    if any(not values for values in family_values.values()):
        raise RuntimeError("runtime cases do not cover every family")
    families = {family: geometric_mean(values) for family, values in family_values.items()}
    geometries = {
        name: geometric_mean(values) for name, values in sorted(geometry_values.items())
    }
    primary = weighted_family_score(families)

    raw_families = result.get("family_tflops")
    if not isinstance(raw_families, Mapping) or set(raw_families) != set(families):
        raise RuntimeError("runtime family aggregate set differs")
    for label, expected_value in families.items():
        observed = finite_positive(raw_families.get(label), f"{label} runtime aggregate")
        if not math.isclose(observed, expected_value, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise RuntimeError(f"runtime family aggregate differs for {label}")
    if result.get("scored_cases") != len(expected_cases):
        raise RuntimeError("runtime scored-case count differs")
    observed_primary = finite_positive(result.get("score_tflops"), "runtime primary score")
    if not math.isclose(observed_primary, primary, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError("runtime primary score differs")
    return primary, families, geometries


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
        "schema_version": 1,
        "evaluation_policy": EVALUATION_POLICY,
        "selection_protocol_id": SELECTION_PROTOCOL_ID,
        "profile_ok": 0.0,
        "suite_ok": 0.0,
        "source_ok": 0.0,
        "hardware_ok": 0.0,
        "correctness_ok": 0.0,
        "edge_ok": 0.0,
        "fresh_input_ok": 0.0,
        "fresh_output_ok": 0.0,
        "budget_ok": 0.0,
        "tilelang_only_ok": 0.0,
        "profile_code": PROFILE_CODES.get(profile or "", 0.0),
        "score_tflops": 0.0,
        **{f"{family}_tflops": 0.0 for family in SCORE_FAMILIES},
        "final_version": float(final_version),
        "num_candidate_versions": float(versions),
        "private_error": message,
    }
    if profile is not None:
        detail["evaluation_profile"] = profile
    if suite_id is not None:
        detail["suite_id"] = suite_id
    return detail


def evaluate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    protected = verify_protected_hashes(JUDGE_ROOT / "protected_hashes.txt")
    if protected:
        return 1, failure_detail("protected Judge/runtime hash failure") | {
            "protected_failures": protected
        }

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
        if profile != "final":
            raise RuntimeError("shared-final-v2 requires the complete final profile")
        cases, manifest_sha256 = load_final_manifest()
        if any(
            case["family"] != "real" and case["check_trial"] >= args.trials for case in cases
        ):
            raise RuntimeError("the final manifest checks a trial this evaluation does not run")
        base["suite_ok"] = 1.0
        base["manifest_sha256"] = manifest_sha256
        base["case_count"] = float(len(cases))
        base["edge_case_count"] = float(len(EDGE_CASE_IDS))
    except Exception as exc:
        base["private_error"] = f"{type(exc).__name__}: {exc}"
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
        return 1, base
    base["source_ok"] = 1.0

    if args.trials <= 0 or args.warmups <= 0:
        base["private_error"] = "invalid trial or warmup count"
        return 1, base
    total_timeout = float(args.total_timeout)
    if not math.isfinite(total_timeout) or total_timeout <= 60.0:
        base["private_error"] = "invalid verifier total timeout"
        return 1, base

    started = time.monotonic()
    deadline = started + total_timeout
    run_root = Path(tempfile.mkdtemp(prefix="tilelang-mamba2-ssd-v2-", dir="/tmp"))
    run_root.chmod(0o755)
    try:
        hardware = _trusted_hardware()
        base["hardware_ok"] = 1.0
        remaining = deadline - time.monotonic() - 30.0
        if remaining <= 0.0:
            raise RuntimeError("Judge total deadline exhausted")
        selected_source = Path(str(admission["selected_path"]))
        candidate_dir = run_root / "selected_candidate"
        candidate_dir.mkdir(mode=0o755)
        candidate_path = candidate_dir / "submission.py"
        shutil.copyfile(selected_source, candidate_path)
        candidate_path.chmod(0o444)
        candidate_dir.chmod(0o555)
        runtime_result = run_worker(
            candidate_path,
            run_root,
            cases,
            suite_id,
            trials=args.trials,
            warmups=args.warmups,
            timeout=min(float(args.candidate_timeout), remaining),
        )
        if runtime_result.get("submission_sha256") != admission["selected_sha256"]:
            raise RuntimeError("runtime evaluated a different candidate source")
        score, families, geometries = validate_runtime_result(
            runtime_result,
            cases,
            expected_trials=args.trials,
            expected_warmups=args.warmups,
            expected_suite_id=suite_id,
        )
        base["correctness_ok"] = 1.0
        base["edge_ok"] = 1.0
        base["fresh_input_ok"] = 1.0
        base["fresh_output_ok"] = 1.0
        base["budget_ok"] = 1.0
        base["tilelang_only_ok"] = 1.0
        detail = dict(base)
        detail.update(
            {
                "score_tflops": score,
                **{f"{family}_tflops": families[family] for family in SCORE_FAMILIES},
                "selected_source_sha256": admission["selected_sha256"],
                "score_method": (
                    "equal-weight geometric mean of the batch, serving, long and real "
                    "workload-family geometric means"
                ),
                "wall_time_seconds": time.monotonic() - started,
                "hardware": hardware,
                "geometry_tflops": geometries,
                "phase_seconds": runtime_result.get("phase_seconds"),
                "private_runtime_result": runtime_result,
            }
        )
        detail["private_error"] = ""
        return 0, detail
    except WorkerGateFailure as exc:
        base["failed_gate"] = exc.gate
        base["private_error"] = f"{exc.gate}: {exc}"
        return 1, base
    except Exception as exc:
        base["private_error"] = f"{type(exc).__name__}: {exc}"
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
        default=int(os.environ.get("TILELANG_SSD_VERIFY_TRIALS", "3")),
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=int(os.environ.get("TILELANG_SSD_VERIFY_WARMUPS", "1")),
    )
    parser.add_argument(
        "--candidate-timeout",
        type=int,
        default=int(os.environ.get("TILELANG_SSD_CANDIDATE_TIMEOUT", "4500")),
    )
    parser.add_argument(
        "--total-timeout",
        type=int,
        default=int(os.environ.get("TILELANG_SSD_TOTAL_TIMEOUT", "4900")),
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
