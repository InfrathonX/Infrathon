#!/usr/bin/env python3
"""Private verifier for the LMCache paged-KV block transfer H100 task."""

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
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_admission import validate_submission  # noqa: E402


JUDGE_ROOT = Path("/opt/lmcache_kv_transfer_judge")
RUNTIME_ROOT = Path("/opt/lmcache_kv_transfer_worker")
WORKER = JUDGE_ROOT / "candidate_worker.py"
EVALUATION_CONTEXT = Path("/tmp/.sforge-evaluation-context.json")
FINAL_MANIFEST = JUDGE_ROOT / "private/final_cases.json"
PUBLIC_CASES_FILE = JUDGE_ROOT / "public_cases.json"
RESULT_NAME = "lmcache_kv_transfer_v2_results.json"
PUBLIC_SUITE_ID = "lmcache-kv-transfer-public-v3"
FINAL_SUITE_ID = "lmcache-kv-transfer-final-holdout-v2"
EVALUATION_POLICY = "full-final-shared-v3"
SELECTION_PROTOCOL_ID = "lmcache-shared-final-v3"
PROFILE_CODES = {"auto": 1.0, "final": 2.0}
ENTRYPOINT = "lmcache_paged_kv_block_transfer"
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,96}$")

# --- frozen contract constants, duplicated from the trusted runtime on
# purpose; task-local tests assert the two agree. ---------------------------
BS = 16
HEAD_SIZES = (64, 128)
ESIZE = 2
NL_RANGE = (1, 128)
NH_RANGE = (1, 32)
NB_RANGE = (1, 65536)
OBJECTS_RANGE = (1, 4)
ARMS = ("device", "host", "trace", "overlap", "duplex")
FAMILIES = {
    "device-large": "device",
    "device-small": "device",
    "host-large": "host",
    "host-small": "host",
    "host-duplex": "duplex",
    "trace": "trace",
    "overlap": "overlap",
}
GROUP_FAMILIES = {
    "device": ("device-large", "device-small"),
    "host": ("host-large", "host-small", "host-duplex"),
    "trace": ("trace",),
    "overlap": ("overlap",),
}
GROUP_WEIGHTS = {"device": 0.35, "host": 0.25, "trace": 0.10, "overlap": 0.30}
TIERS = ("paper", "other", "public", "edge")
STRATUM_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
SMALL_CALL_BYTES = 16 * 1024 * 1024
FOREGROUNDS = ("gemm", "hbm")
MAX_DIRECT_WAVES = 128
MAX_TRACE_WAVES = 64
OVERLAP_COVER = 3.0
MAX_OVERLAP_LOSS = 0.95
# The bandwidth envelope is probed on the evaluation device in the same run by
# the trusted runtime (see kv_transfer_runtime.measure_envelopes); the Judge
# checks the probe record's identity and sanity floors, never a frozen number.
ENVELOPE_MARGIN = 1.10
ENVELOPE_PROTOCOL = "same-run-cudamemcpy-zerocopy-duplex-probe-v3"
ENVELOPE_PROBE_BYTES = 256 * 1024 * 1024
ENVELOPE_PROBE_REPEATS = 5
ENVELOPE_MIN_DEVICE_GBPS = 100.0
ENVELOPE_MIN_HOST_GBPS = 4.0
EXPECTED_CONSTANTS = {
    "bs": BS,
    "head_sizes": list(HEAD_SIZES),
    "esize": ESIZE,
    "poison": 0x7C7C,
    "scratch_bytes": 256 * 1024 * 1024,
    "envelope_margin": ENVELOPE_MARGIN,
    "envelope_protocol": ENVELOPE_PROTOCOL,
    "envelope_probe_bytes": ENVELOPE_PROBE_BYTES,
    "small_call_bytes": SMALL_CALL_BYTES,
    "overlap_cover": OVERLAP_COVER,
    "max_overlap_loss": MAX_OVERLAP_LOSS,
}
# Capability envelope; must equal the runtime's pair (tests assert it).
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 64 * 1024**3
GROUP_KEYS = ("device_gbps", "host_gbps", "trace_gbps", "overlap_gbps")
FAMILY_KEYS = tuple(f"family_{name.replace('-', '_')}_gbps" for name in FAMILIES)

EDGE_CASE_IDS = (
    "edge-skip-all",
    "edge-skip-almost-all",
    "edge-skip-partial-fused",
    "edge-single-block",
    "edge-three-objects",
    "edge-tail-chunk",
    "edge-single-layer",
    "edge-deepest-stack",
    "edge-single-head",
    "edge-most-heads",
    "edge-head-64-split",
    "edge-head-64-fused",
    "edge-pool-ends",
    "edge-whole-pool",
    "edge-repeated-blocks",
    "edge-largest-pool",
    "edge-object-beyond-scratch",
    "edge-host-single",
    "edge-host-store-fused",
    "edge-host-skip",
    "edge-host-head-64",
    "edge-host-beyond-scratch",
    "edge-trace-overwrite",
    "edge-duplex",
    "edge-duplex-skip-fused",
)

# The frozen final holdout: case count per family.
FINAL_FAMILY_COUNTS = {
    "device-large": 156,
    "device-small": 143,
    "host-duplex": 104,
    "host-large": 104,
    "host-small": 142,
    "overlap": 104,
    "trace": 90
}
EXPECTED_FINAL_COUNT = sum(FINAL_FAMILY_COUNTS.values())
MAX_MANIFEST_BYTES = 16 * 1024 * 1024


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


# --------------------------------------------------------------------------
# Wave arithmetic (duplicated from the trusted runtime; tests pin equality)
# --------------------------------------------------------------------------
def wave_geometry(wave: Mapping[str, Any]) -> dict[str, int]:
    layout = str(wave["layout"])
    if layout == "split":
        kv, width = 2, int(wave["hs"])
    elif layout == "fused":
        kv, width = 1, 2 * int(wave["hs"])
    else:
        raise RuntimeError(f"unknown layout {layout!r}")
    blocks_per_object = int(wave["chunk_tokens"]) // BS
    total_blocks = int(wave["num_objects"]) * blocks_per_object
    return {
        "kv": kv,
        "hs": width,
        "blocks_per_object": blocks_per_object,
        "total_blocks": total_blocks,
    }


def wave_logical_bytes(wave: Mapping[str, Any]) -> int:
    geometry = wave_geometry(wave)
    moved = geometry["total_blocks"] - int(wave["skip"])
    return moved * geometry["kv"] * int(wave["nl"]) * BS * int(wave["nh"]) * geometry["hs"] * ESIZE


def case_logical_bytes(case: Mapping[str, Any]) -> int:
    return sum(wave_logical_bytes(wave) for wave in case["waves"])


def case_envelope_gbps(case: Mapping[str, Any], envelopes: Mapping[str, Any]) -> float:
    arm = str(case["arm"])
    key = "device_gbps" if arm == "device" else ("duplex_gbps" if arm == "duplex" else "host_gbps")
    return float(envelopes[key])


def validate_envelopes(envelopes: Any) -> dict[str, float]:
    """Mirror of the runtime check: identity, parameters, finite values, floors."""
    if not isinstance(envelopes, Mapping) or envelopes.get("protocol") != ENVELOPE_PROTOCOL:
        raise RuntimeError("bandwidth envelope record differs")
    if envelopes.get("probe_bytes") != ENVELOPE_PROBE_BYTES or envelopes.get("repeats") != ENVELOPE_PROBE_REPEATS:
        raise RuntimeError("bandwidth envelope probe parameters differ")
    if envelopes.get("margin") != ENVELOPE_MARGIN:
        raise RuntimeError("bandwidth envelope margin differs")
    values: dict[str, float] = {}
    for key in (
        "device_gbps", "host_gbps", "h2d_gbps", "d2h_gbps",
        "zerocopy_h2d_gbps", "zerocopy_d2h_gbps", "duplex_gbps",
    ):
        values[key] = finite_positive(envelopes.get(key), f"envelopes.{key}")
    if values["host_gbps"] != max(
        values["h2d_gbps"], values["d2h_gbps"],
        values["zerocopy_h2d_gbps"], values["zerocopy_d2h_gbps"],
    ):
        raise RuntimeError("bandwidth envelope host value differs from its probes")
    if values["device_gbps"] < ENVELOPE_MIN_DEVICE_GBPS or values["host_gbps"] < ENVELOPE_MIN_HOST_GBPS:
        raise RuntimeError("bandwidth envelope probe is below the sanity floor")
    if values["duplex_gbps"] < values["host_gbps"]:
        raise RuntimeError("bandwidth envelope duplex value is below the one-way host value")
    return values


WAVE_KEYS = {
    "layout", "nl", "nh", "hs", "chunk_tokens", "num_objects", "direction",
    "skip", "nb", "ids",
}
CASE_KEYS = {"case_id", "family", "arm", "foreground", "stratum", "tier", "waves", "trials", "seed"}
POOL_KEYS = ("layout", "nl", "nh", "hs", "nb")
DIRECT_KEYS = ("layout", "nl", "nh", "hs", "chunk_tokens", "num_objects", "direction", "nb")
DUPLEX_KEYS = ("layout", "nl", "nh", "hs", "chunk_tokens", "num_objects", "nb")


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return int(value)


def _validate_wave(wave: Any, case_id: str) -> dict[str, Any]:
    if not isinstance(wave, Mapping) or set(wave) != WAVE_KEYS:
        raise RuntimeError(f"{case_id}: wave fields differ from schema")
    if str(wave["layout"]) not in ("split", "fused"):
        raise RuntimeError(f"{case_id}: invalid layout")
    if str(wave["direction"]) not in ("store", "retrieve"):
        raise RuntimeError(f"{case_id}: invalid direction")
    nl = _strict_int(wave["nl"], "nl")
    nh = _strict_int(wave["nh"], "nh")
    hs = _strict_int(wave["hs"], "hs")
    nb = _strict_int(wave["nb"], "nb")
    tokens = _strict_int(wave["chunk_tokens"], "chunk_tokens")
    objects = _strict_int(wave["num_objects"], "num_objects")
    skip = _strict_int(wave["skip"], "skip")
    if not (NL_RANGE[0] <= nl <= NL_RANGE[1] and NH_RANGE[0] <= nh <= NH_RANGE[1]
            and NB_RANGE[0] <= nb <= NB_RANGE[1] and hs in HEAD_SIZES
            and OBJECTS_RANGE[0] <= objects <= OBJECTS_RANGE[1]):
        raise RuntimeError(f"{case_id}: wave geometry outside the support domain")
    if tokens <= 0 or tokens % BS:
        raise RuntimeError(f"{case_id}: chunk_tokens must be a positive multiple of {BS}")
    total = objects * (tokens // BS)
    if total > nb or not 0 <= skip <= total:
        raise RuntimeError(f"{case_id}: wave exceeds the pool or skips outside it")
    ids = wave["ids"]
    if not isinstance(ids, list) or len(ids) != total:
        raise RuntimeError(f"{case_id}: block-id table has the wrong length")
    values = [_strict_int(value, "block id") for value in ids]
    if min(values) < 0 or max(values) >= nb or len(set(values)) != total:
        raise RuntimeError(f"{case_id}: block ids must be unique and inside the pool")
    return {key: wave[key] for key in WAVE_KEYS}


def validate_scored_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) - {"scored"} != CASE_KEYS:
        raise RuntimeError("case fields differ from schema")
    case_id = str(raw["case_id"])
    if CASE_ID_RE.fullmatch(case_id) is None:
        raise RuntimeError("case_id is invalid")
    family = str(raw["family"])
    arm = str(raw["arm"])
    if arm not in ARMS or FAMILIES.get(family) != arm:
        raise RuntimeError(f"{case_id}: family and arm differ from the schema")
    foreground = raw["foreground"]
    if (arm == "overlap") != (foreground in FOREGROUNDS) or (arm != "overlap" and foreground is not None):
        raise RuntimeError(f"{case_id}: foreground differs from the schema")
    waves_raw = raw["waves"]
    if not isinstance(waves_raw, list) or not waves_raw:
        raise RuntimeError(f"{case_id}: no waves")
    waves = [_validate_wave(wave, case_id) for wave in waves_raw]
    keys = POOL_KEYS if arm == "trace" else (DUPLEX_KEYS if arm == "duplex" else DIRECT_KEYS)
    if len({tuple(wave[key] for key in keys) for wave in waves}) != 1:
        raise RuntimeError(f"{case_id}: waves must share {keys}")
    if arm == "duplex":
        stores = {b for w in waves if w["direction"] == "store" for b in w["ids"]}
        loads = {b for w in waves if w["direction"] == "retrieve" for b in w["ids"]}
        if not stores or not loads or stores & loads:
            raise RuntimeError(f"{case_id}: duplex stores and retrieves must be present and disjoint")
    stratum = str(raw["stratum"])
    tier = str(raw["tier"])
    if STRATUM_RE.fullmatch(stratum) is None or tier not in TIERS:
        raise RuntimeError(f"{case_id}: stratum or tier differs from the schema")
    limit = MAX_TRACE_WAVES if arm == "trace" else MAX_DIRECT_WAVES
    if len(waves) > limit:
        raise RuntimeError(f"{case_id}: wave count outside range")
    if family in ("device-large", "device-small", "host-large", "host-small"):
        small = family.endswith("-small")
        for wave in waves:
            if (wave_logical_bytes(wave) < SMALL_CALL_BYTES) != small:
                raise RuntimeError(f"{case_id}: a wave's size does not match its family")
    trials = _strict_int(raw["trials"], "trials")
    if not 1 <= trials <= 32:
        raise RuntimeError(f"{case_id}: trials outside range")
    return {
        "case_id": case_id,
        "family": family,
        "arm": arm,
        "foreground": foreground,
        "stratum": stratum,
        "tier": tier,
        "waves": waves,
        "trials": trials,
        "seed": _strict_int(raw["seed"], "seed"),
        "scored": True,
    }


def shape_identity(wave: Mapping[str, Any]) -> tuple:
    return (str(wave["layout"]), int(wave["nl"]), int(wave["nh"]), int(wave["hs"]))


# --------------------------------------------------------------------------
# Root-only file plumbing
# --------------------------------------------------------------------------
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
    version_override = context.get("candidate_version")
    if version_override is not None and (
        isinstance(version_override, bool)
        or not isinstance(version_override, int)
        or not 1 <= version_override <= 999
    ):
        raise RuntimeError("SForge evaluation candidate_version is invalid")
    for interactive in ("auto-", "agent-"):
        if round_id.startswith(interactive) and len(round_id) > len(interactive):
            if version_override is not None:
                raise RuntimeError("candidate_version override is host-side only")
            return "final", FINAL_SUITE_ID, round_id, None
    for host_side in ("manual-", "candidate-"):
        if round_id.startswith(host_side) and len(round_id) > len(host_side):
            return "final", FINAL_SUITE_ID, round_id, version_override
    raise RuntimeError("SForge evaluation round_id has an unknown profile")


def _load_pinned_cases(path: Path, label: str) -> tuple[dict[str, Any], ...]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable") from exc
    payload = _strict_json_bytes(raw, label)
    if payload.get("schema_version") != 2 or not isinstance(payload.get("cases"), list):
        raise RuntimeError(f"{label} identity differs")
    return tuple(validate_scored_case(case) for case in payload["cases"])


def validate_final_manifest(
    payload: Mapping[str, Any],
    excluded_shapes: frozenset,
    excluded_seeds: frozenset,
) -> tuple[dict[str, Any], ...]:
    if set(payload) != {"schema_version", "suite_id", "cases"}:
        raise RuntimeError("final manifest fields differ from schema v2")
    if payload.get("schema_version") != 2 or payload.get("suite_id") != FINAL_SUITE_ID:
        raise RuntimeError("final manifest identity differs")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != EXPECTED_FINAL_COUNT:
        raise RuntimeError(
            f"final manifest must contain exactly {EXPECTED_FINAL_COUNT} cases"
        )
    cases = []
    ids: set[str] = set()
    seeds: set[int] = set()
    families: dict[str, int] = {}
    layouts: set[str] = set()
    directions: set[tuple[str, str]] = set()
    for raw_case in raw_cases:
        case = validate_scored_case(raw_case)
        case_id = case["case_id"]
        if case_id in ids:
            raise RuntimeError("final case IDs are not unique")
        ids.add(case_id)
        if case["seed"] in seeds:
            raise RuntimeError("final case seeds are not unique")
        if case["seed"] in excluded_seeds:
            raise RuntimeError("final case reuses a public seed")
        seeds.add(case["seed"])
        families[case["family"]] = families.get(case["family"], 0) + 1
        for wave in case["waves"]:
            if shape_identity(wave) in excluded_shapes:
                raise RuntimeError("final case reuses a public shape")
            layouts.add(str(wave["layout"]))
            directions.add((case["arm"], str(wave["direction"])))
        cases.append(case)
    if families != FINAL_FAMILY_COUNTS:
        raise RuntimeError("final manifest family counts differ")
    for name in FAMILIES:
        if {case["tier"] for case in cases if case["family"] == name} != {"paper", "other"}:
            raise RuntimeError(f"final manifest family {name} must cover both tiers")
    if layouts != {"split", "fused"}:
        raise RuntimeError("final manifest must cover both layouts")
    for arm in ARMS:
        if {(arm, "store"), (arm, "retrieve")} - directions:
            raise RuntimeError(f"final manifest must cover both directions in arm {arm}")
    return tuple(cases)


def load_final_manifest(
    path: Path = FINAL_MANIFEST,
    *,
    expected_uid: int = 0,
) -> tuple[tuple[dict[str, Any], ...], str]:
    payload, raw = _read_root_only_json(
        path,
        "final holdout manifest",
        max_bytes=MAX_MANIFEST_BYTES,
        expected_uid=expected_uid,
    )
    public = _load_pinned_cases(PUBLIC_CASES_FILE, "public case table")
    excluded_shapes = frozenset(shape_identity(wave) for case in public for wave in case["waves"])
    excluded_seeds = frozenset(int(case["seed"]) for case in public)
    cases = validate_final_manifest(payload, excluded_shapes, excluded_seeds)
    return cases, hashlib.sha256(raw).hexdigest()


def locate_submission(app_dir: Path) -> Path:
    roots = (
        app_dir / "workdir/kv_transfer_lab",
        app_dir / "environment/workdir/kv_transfer_lab",
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
        "judge:public_cases.json",
        "judge:run_judge.py",
        "judge:source_admission.py",
        "judge:verify_kv_transfer.py",
        "worker:kv_transfer_runtime.py",
        "worker:transfer_abi.h",
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
        or metadata.st_size > 64 * 1024 * 1024
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


def _trusted_hardware() -> dict[str, Any]:
    import torch

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
    }


def run_worker(
    candidate_path: Path,
    run_root: Path,
    cases: Sequence[Mapping[str, Any]],
    suite_id: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    account = _agent_account()
    manifest_bytes = json.dumps(
        {"schema_version": 2, "suite_id": suite_id, "cases": list(cases)},
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    result_dir = run_root / "result"
    result_dir.mkdir(mode=0o700)
    private_root = run_root / "candidate-private"
    private_root.mkdir(mode=0o700)
    build_dir = private_root / "build"
    tmp_dir = private_root / "tmp"
    for path in (build_dir, tmp_dir):
        path.mkdir(mode=0o700)
    for path in (result_dir, private_root, build_dir, tmp_dir):
        os.chown(path, account.pw_uid, account.pw_gid)
    result_path = result_dir / "result.json"
    command = [
        sys.executable,
        str(WORKER),
        "--candidate",
        str(candidate_path),
        "--output",
        str(result_path),
        "--build-dir",
        str(build_dir),
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
    if returncode != 0:
        raise RuntimeError(
            "candidate compilation, binary audit, launch, or correctness evaluation failed"
        )
    return _strict_worker_result(result_path, expected_uid=account.pw_uid)


# --------------------------------------------------------------------------
# Result validation
# --------------------------------------------------------------------------
def _validate_hardware_record(result: Mapping[str, Any]) -> None:
    hardware = result.get("hardware")
    if not isinstance(hardware, Mapping):
        raise RuntimeError("runtime result lacks hardware inventory")
    match = re.fullmatch(r"sm_(\d+)(\d)", str(hardware.get("capability", "")))
    total_memory = hardware.get("total_memory_bytes")
    if (
        match is None
        or (int(match.group(1)), int(match.group(2))) < MIN_COMPUTE_CAPABILITY
        or isinstance(total_memory, bool)
        or not isinstance(total_memory, int)
        or total_memory < MIN_DEVICE_MEMORY_BYTES
    ):
        raise RuntimeError("runtime hardware inventory violates the capability envelope")
    if not isinstance(hardware.get("device_name"), str) or not hardware["device_name"]:
        raise RuntimeError("runtime hardware inventory lacks a device name")


def _numeric_list(value: Any, label: str, expected_length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise RuntimeError(f"{label} has the wrong length")
    return [finite_positive(item, label) for item in value]


def _validate_case_result(
    row: Mapping[str, Any], expected: Mapping[str, Any], envelopes: Mapping[str, Any]
) -> tuple[str, float]:
    case_id = str(expected["case_id"])
    if (row.get("case_id") != case_id or row.get("arm") != expected["arm"]
            or row.get("family") != expected["family"]
            or row.get("foreground") != expected["foreground"]
            or row.get("stratum") != expected["stratum"] or row.get("tier") != expected["tier"]):
        raise RuntimeError(f"case identity differs for {case_id}")
    if row.get("scored") is not True:
        raise RuntimeError(f"case {case_id} is not marked scored")
    if row.get("waves") != len(expected["waves"]):
        raise RuntimeError(f"wave count differs for {case_id}")
    if row.get("trials") != expected["trials"]:
        raise RuntimeError(f"trial count differs for {case_id}")

    logical = case_logical_bytes(expected)
    if row.get("logical_bytes") != logical:
        raise RuntimeError(f"logical byte count differs for {case_id}")
    trials = int(expected["trials"])
    trial_seconds = _numeric_list(row.get("trial_seconds"), f"{case_id}.trial_seconds", trials)
    host_walls = _numeric_list(row.get("host_wall_seconds"), f"{case_id}.host_wall", trials)
    event_seconds = _numeric_list(row.get("event_seconds"), f"{case_id}.event", trials)
    for ordinal, seconds in enumerate(trial_seconds):
        if not math.isclose(
            seconds,
            max(host_walls[ordinal], event_seconds[ordinal]),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(f"trusted timing envelope differs for {case_id}")
    median = statistics_median(sorted(trial_seconds))
    observed_median = finite_positive(row.get("median_seconds"), f"{case_id}.median")
    if not math.isclose(observed_median, median, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError(f"median time differs for {case_id}")
    gbps = logical / median / 1.0e9
    observed_gbps = finite_positive(row.get("gbps"), f"{case_id}.gbps")
    if not math.isclose(observed_gbps, gbps, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError(f"case bandwidth differs for {case_id}")
    if gbps > case_envelope_gbps(expected, envelopes) * ENVELOPE_MARGIN:
        raise RuntimeError(f"case {case_id} exceeds the physical bandwidth envelope")
    score = gbps
    if expected["arm"] == "overlap":
        _numeric_list(row.get("alone_seconds"), f"{case_id}.alone", trials)
        fg_overlap = _numeric_list(row.get("fg_overlap_seconds"), f"{case_id}.fg_overlap", trials)
        fg_alone = _numeric_list(row.get("fg_alone_seconds"), f"{case_id}.fg_alone", trials)
        iterations = row.get("fg_iterations")
        if (not isinstance(iterations, list) or len(iterations) != trials
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in iterations)):
            raise RuntimeError(f"foreground iteration record differs for {case_id}")
        losses = [
            min(max((fg_overlap[i] - fg_alone[i]) / trial_seconds[i], 0.0), MAX_OVERLAP_LOSS)
            for i in range(trials)
        ]
        observed = row.get("fg_loss")
        if (not isinstance(observed, list) or len(observed) != trials
                or any(not math.isclose(float(a), b, rel_tol=1.0e-9, abs_tol=1.0e-12)
                       for a, b in zip(observed, losses))):
            raise RuntimeError(f"foreground loss differs for {case_id}")
        score = gbps * (1.0 - statistics_median(sorted(losses)))
    observed_score = finite_positive(row.get("case_score"), f"{case_id}.case_score")
    if not math.isclose(observed_score, score, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError(f"case score differs for {case_id}")
    return str(expected["family"]), score


def family_score(entries: Sequence[tuple[str, str, float]]) -> float:
    """Tier-balanced geometric mean of stratum geometric means (mirror of the runtime)."""
    strata: dict[tuple[str, str], list[float]] = {}
    for tier, stratum, score in entries:
        strata.setdefault((tier, stratum), []).append(score)
    tiers: dict[str, list[float]] = {}
    for (tier, _stratum), values in strata.items():
        tiers.setdefault(tier, []).append(math.log(geometric_mean(values)))
    return math.exp(math.fsum(math.fsum(logs) / len(logs) for logs in tiers.values()) / len(tiers))


def statistics_median(ordered: Sequence[float]) -> float:
    length = len(ordered)
    middle = length // 2
    if length % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _validate_edge_results(result: Mapping[str, Any]) -> None:
    rows = result.get("edge_fixtures")
    if not isinstance(rows, list) or len(rows) != len(EDGE_CASE_IDS):
        raise RuntimeError("edge fixture result set differs")
    reported = tuple(
        row.get("case_id") if isinstance(row, Mapping) else None for row in rows
    )
    if reported != EDGE_CASE_IDS:
        raise RuntimeError("edge fixture identity differs from the frozen gate")
    for row in rows:
        if not isinstance(row, Mapping) or row.get("scored") is not False:
            raise RuntimeError("an edge fixture must never be scored")
        if row.get("correct") is not True:
            raise RuntimeError("edge fixture correctness record differs")


def _close(observed: Any, expected: float, label: str) -> None:
    value = finite_positive(observed, label)
    if not math.isclose(value, expected, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise RuntimeError(f"{label} differs")


def validate_runtime_result(
    result: Mapping[str, Any],
    expected_cases: Sequence[Mapping[str, Any]],
    *,
    expected_suite_id: str,
) -> tuple[float, dict[str, float], dict[str, float]]:
    if result.get("suite_id") != expected_suite_id:
        raise RuntimeError("runtime suite identity differs")
    _validate_hardware_record(result)
    if result.get("entrypoint") != ENTRYPOINT:
        raise RuntimeError("runtime entrypoint differs")
    constants = result.get("constants")
    if not isinstance(constants, Mapping) or dict(constants) != EXPECTED_CONSTANTS:
        raise RuntimeError("runtime constants record differs")
    envelopes = validate_envelopes(result.get("envelopes"))
    binary = result.get("binary_report")
    if not isinstance(binary, Mapping) or binary.get("ok") is not True:
        raise RuntimeError("runtime binary audit record differs")

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

    family_values: dict[str, list[tuple[str, str, float]]] = {name: [] for name in FAMILIES}
    for expected in expected_cases:
        family, score = _validate_case_result(by_id[str(expected["case_id"])], expected, envelopes)
        family_values[family].append((str(expected["tier"]), str(expected["stratum"]), score))
    if any(not values for values in family_values.values()):
        raise RuntimeError("runtime cases do not cover every scored family")
    families = {name: family_score(values) for name, values in family_values.items()}
    groups = {
        group: geometric_mean([families[name] for name in names])
        for group, names in GROUP_FAMILIES.items()
    }
    primary = math.exp(
        math.fsum(GROUP_WEIGHTS[name] * math.log(groups[name]) for name in GROUP_WEIGHTS)
    )

    raw_families = result.get("family_scores")
    if not isinstance(raw_families, Mapping) or set(raw_families) != set(families):
        raise RuntimeError("runtime family aggregate set differs")
    for label, value in families.items():
        _close(raw_families.get(label), value, f"runtime family aggregate {label}")
    raw_groups = result.get("group_scores")
    if not isinstance(raw_groups, Mapping) or set(raw_groups) != set(groups):
        raise RuntimeError("runtime group aggregate set differs")
    for label, value in groups.items():
        _close(raw_groups.get(label), value, f"runtime group aggregate {label}")
    if result.get("group_weights") != GROUP_WEIGHTS:
        raise RuntimeError("runtime group weights differ")
    if result.get("scored_cases") != len(expected_cases):
        raise RuntimeError("runtime scored-case count differs")
    _close(result.get("score_gbps"), primary, "runtime primary score")
    return primary, groups, families


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
    score = float(payload.get("score_gbps", 0.0))
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
        "binary_ok": 0.0,
        "correctness_ok": 0.0,
        "edge_ok": 0.0,
        "envelope_ok": 0.0,
        "profile_code": PROFILE_CODES.get(profile or "", 0.0),
        "score_gbps": 0.0,
        "device_gbps": 0.0,
        "host_gbps": 0.0,
        "trace_gbps": 0.0,
        "overlap_gbps": 0.0,
        **{key: 0.0 for key in FAMILY_KEYS},
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
        cases, manifest_sha256 = load_final_manifest()
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

    total_timeout = float(args.total_timeout)
    if not math.isfinite(total_timeout) or total_timeout <= 60.0:
        base["private_error"] = "invalid verifier total timeout"
        return 1, base

    started = time.monotonic()
    deadline = started + total_timeout
    run_root = Path(tempfile.mkdtemp(prefix="lmcache-kv-transfer-v2-", dir="/tmp"))
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
        candidate_path = candidate_dir / "submission.cu"
        shutil.copyfile(selected_source, candidate_path)
        candidate_path.chmod(0o444)
        candidate_dir.chmod(0o555)
        runtime_result = run_worker(
            candidate_path,
            run_root,
            cases,
            suite_id,
            timeout=min(float(args.candidate_timeout), remaining),
        )
        if runtime_result.get("source_sha256") != admission["selected_sha256"]:
            raise RuntimeError("runtime evaluated a different candidate source")
        score, groups, families = validate_runtime_result(
            runtime_result,
            cases,
            expected_suite_id=suite_id,
        )
        base["binary_ok"] = 1.0
        base["correctness_ok"] = 1.0
        base["edge_ok"] = 1.0
        base["envelope_ok"] = 1.0
        detail = dict(base)
        detail.update(
            {
                "score_gbps": score,
                "device_gbps": groups["device"],
                "host_gbps": groups["host"],
                "trace_gbps": groups["trace"],
                "overlap_gbps": groups["overlap"],
                **{f"family_{name.replace('-', '_')}_gbps": value
                   for name, value in families.items()},
                "selected_source_sha256": admission["selected_sha256"],
                "score_method": (
                    "weighted geometric mean (device 0.35, host 0.25, trace 0.10, "
                    "overlap 0.30) of group scores; device is the geometric mean of its "
                    "large-call and small-call families, host of large, small and duplex; "
                    "a family is the tier-balanced geometric mean of its shape strata"
                ),
                "wall_time_seconds": time.monotonic() - started,
                "hardware": hardware,
                "private_runtime_result": runtime_result,
            }
        )
        detail["private_error"] = ""
        return 0, detail
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
        "--candidate-timeout",
        type=int,
        default=int(os.environ.get("LMCACHE_KV_CANDIDATE_TIMEOUT", "3600")),
    )
    parser.add_argument(
        "--total-timeout",
        type=int,
        default=int(os.environ.get("LMCACHE_KV_TOTAL_TIMEOUT", "4200")),
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
