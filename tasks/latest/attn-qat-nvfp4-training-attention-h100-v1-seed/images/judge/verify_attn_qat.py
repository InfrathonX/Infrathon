#!/usr/bin/env python3
"""Private verifier for the FastVideo-derived Attn-QAT training attention, BF16 H100 v1."""

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
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_admission import validate_submission  # noqa: E402

JUDGE_ROOT = Path("/opt/attn_qat_judge")
RUNTIME_ROOT = Path("/opt/attn_qat_worker")
WORKER = JUDGE_ROOT / "candidate_worker.py"
EVALUATION_CONTEXT = Path("/tmp/.sforge-evaluation-context.json")
FINAL_MANIFEST = JUDGE_ROOT / "private/final_cases.json"
RESULT_NAME = "attn_qat_v1_results.json"
FINAL_SUITE_ID = "final-holdout-v3"
PROFILE_CODES = {"auto": 1.0, "final": 2.0}
EVALUATION_POLICY = "full-final-shared-v2"
SELECTION_PROTOCOL_ID = "attn-qat-shared-final-v3"
FORWARD_ENTRYPOINT = "attn_qat_fwd"
BACKWARD_ENTRYPOINT = "attn_qat_bwd"
ENTRYPOINTS = (FORWARD_ENTRYPOINT, BACKWARD_ENTRYPOINT)
VIDEO_FAMILIES = ("wan13b", "wan14b", "ltx2", "kandinsky5", "cogvideox", "hunyuan", "real_video")
LLM_FAMILIES = ("llm_pretrain", "llm_long", "llm_sft", "llm_d256", "real_llm")
FAMILIES = VIDEO_FAMILIES + LLM_FAMILIES
DOMAINS = ("video", "llm")
FAMILY_DOMAIN = {
    **{name: "video" for name in VIDEO_FAMILIES},
    **{name: "llm" for name in LLM_FAMILIES},
}
REAL_FAMILIES = frozenset({"real_video", "real_llm"})
TIERS = ("recipe", "sweep", "real")
# The suite's shape: how many cases of each tier each family holds.  The geometries themselves
# live only in the Judge-only final manifest.
EXPECTED_TIER_COUNTS = {
    "wan13b": {"recipe": 3, "sweep": 15},
    "wan14b": {"recipe": 1, "sweep": 22},
    "ltx2": {"recipe": 1, "sweep": 11},
    "kandinsky5": {"recipe": 2, "sweep": 16},
    "cogvideox": {"recipe": 2, "sweep": 18},
    "hunyuan": {"sweep": 15},
    "real_video": {"real": 42},
    "llm_pretrain": {"recipe": 1, "sweep": 32},
    "llm_long": {"sweep": 14},
    "llm_sft": {"sweep": 36},
    "llm_d256": {"sweep": 20},
    "real_llm": {"real": 36},
}
EXPECTED_CASE_COUNT = 287
EXPECTED_EDGE_CASE_COUNT = 15
EXPECTED_TRIALS = 3
EXPECTED_WARMUPS = 1
REAL_TRIALS = 1
CASE_KEYS = {
    "case_id", "family", "tier", "b", "h", "h_kv", "d", "s_q", "s_kv", "causal", "seed",
    "gated", "layer",
}
# Video lengths are `prefix + lattice * t`.
FAMILY_LATTICES = {
    "wan13b": ((0, 1560),),
    "wan14b": ((0, 1560), (0, 3600)),
    "ltx2": ((0, 390),),
    "kandinsky5": ((0, 1536),),
    "cogvideox": ((226, 1350),),
    "hunyuan": ((256, 2040),),
    "real_video": ((0, 1560), (0, 3600), (226, 1350)),
}
LLM_LENGTH_QUANTUM = 64
FAMILY_HEAD_DIMS = {
    "wan13b": {128}, "wan14b": {128}, "ltx2": {128}, "kandinsky5": {64}, "cogvideox": {64},
    "hunyuan": {128}, "real_video": {64, 128}, "llm_pretrain": {128}, "llm_long": {128},
    "llm_sft": {128}, "llm_d256": {256}, "real_llm": {128, 256},
}
KANDINSKY5_TEXT_TOKENS = 512
PUBLIC_GEOMETRIES = frozenset(
    {(4, 4, 128), (8, 8, 128), (24, 24, 128), (16, 16, 64), (24, 24, 64),
     (28, 4, 128), (14, 2, 128), (16, 8, 128), (20, 4, 256)}
)
PUBLIC_LENGTHS = frozenset(
    {2728, 2784, 3584, 6120, 7168, 9216, 9696, 9984, 12384, 14336, 18016, 20480, 26208, 27280,
     28672, 40960, 57440}
)
REAL_LAYER_PATTERN = re.compile(
    r"(wan(13|14)b|cogvideox2b)-b[0-9]{2}-s[0-9]{3}|(qwen3-14b|qwen35-4b)-l[0-9]{2}-x[0-9]{2}"
)
EDGE_CASE_IDS = (
    "edge-e1-tiny",
    "edge-e2-unaligned",
    "edge-e3-cross-attention",
    "edge-e4-unaligned-cross",
    "edge-e5-zero-key-zero-v-groups",
    "edge-e6-rtne-ties",
    "edge-e7-wan-grad-capture",
    "edge-e8-forced-sink",
    "edge-e9-tiny-do-scale",
    "edge-e10-ltx2-minimal",
    "edge-e11-head-dim-64",
    "edge-e12-causal-gqa",
    "edge-e13-causal-sink",
    "edge-e14-head-dim-256",
    "edge-e15-head-dim-256-full",
)
# Candidate statistics every scored row must carry, one group per gated output.  The
# worst-`(b, h)` statistic is reported by the runtime and re-read here, but it is not a
# gate: the fixed global limits are what DESIGN section 5.5 closes on.
GATE_TENSORS = ("o", "dq", "dk", "dv")
METRICS = tuple(f"{name}_rel_l2" for name in GATE_TENSORS)
REPORTED_METRICS = tuple(
    f"{name}_{statistic}"
    for name in GATE_TENSORS
    for statistic in ("head_rel_l2", "max_abs")
)
LIMITS = tuple(f"{name}_rel_l2_limit" for name in GATE_TENSORS)
IDENTITY_KEYS = (
    "family",
    "tier",
    "b",
    "s_q",
    "s_kv",
    "h",
    "h_kv",
    "d",
    "causal",
    "softmax_scale",
    "scored",
)
RUNTIME_GATE_KEYS = (
    "correctness_ok",
    "edge_ok",
    "fresh_input_ok",
    "fresh_output_ok",
    "saved_cap_ok",
)
WORKER_ENVIRONMENT_KEYS = (
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_positive(value: Any, label: str) -> float:
    """Return `value` as a float, rejecting non-numeric, non-finite and non-positive input."""
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be finite positive")
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be finite positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise RuntimeError(f"{label} must be finite positive")
    return number


def geometric_mean(values: Sequence[float]) -> float:
    """Geometric mean of strictly positive values."""
    logs = [math.log(finite_positive(value, "geomean")) for value in values]
    return math.exp(math.fsum(logs) / len(values))


def attention_pairs(s_q: int, s_kv: int, causal: bool = False) -> int:
    """Query-key pairs the operator scores: `s_q * s_kv`, or the causal triangle."""
    return s_q * (s_q + 1) // 2 if causal else s_q * s_kv


def forward_flops(b: int, h: int, s_q: int, s_kv: int, d: int, causal: bool = False) -> int:
    """Two QK-type passes and two PV-type products; the runtime's forward denominator."""
    return 8 * b * h * attention_pairs(s_q, s_kv, causal) * d


def backward_flops(b: int, h: int, s_q: int, s_kv: int, d: int, causal: bool = False) -> int:
    """The FlashAttention-2 backward's five matrix products."""
    return 10 * b * h * attention_pairs(s_q, s_kv, causal) * d


def effective_flops(b: int, h: int, s_q: int, s_kv: int, d: int, causal: bool = False) -> int:
    """`26 * base`: two forwards to one backward, the production checkpointing weighting."""
    return 2 * forward_flops(b, h, s_q, s_kv, d, causal) + backward_flops(b, h, s_q, s_kv, d, causal)


def _positive_int(value: Any, label: str) -> int:
    """Reject anything but a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} differs")
    return value


def validate_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one scored case against the suite rules and derive its geometry."""
    if set(raw) != CASE_KEYS:
        raise RuntimeError("case fields differ")
    case = dict(raw)
    family = str(case["family"])
    tier = str(case["tier"])
    if family not in FAMILIES or tier not in TIERS or (family in REAL_FAMILIES) != (tier == "real"):
        raise RuntimeError("family/tier differs")
    b = _positive_int(case["b"], "batch")
    h = _positive_int(case["h"], "heads")
    h_kv = _positive_int(case["h_kv"], "key/value heads")
    d = _positive_int(case["d"], "head dimension")
    s_q = _positive_int(case["s_q"], "sequence length")
    s_kv = _positive_int(case["s_kv"], "key length")
    causal = case["causal"]
    if not isinstance(causal, bool):
        raise RuntimeError("causal flag differs")
    if d not in FAMILY_HEAD_DIMS[family] or h % h_kv:
        raise RuntimeError("head geometry differs")
    if (h, h_kv, d) in PUBLIC_GEOMETRIES or s_q in PUBLIC_LENGTHS:
        raise RuntimeError("case overlaps a public identity")
    if FAMILY_DOMAIN[family] == "video":
        if causal or h_kv != h or b != 1:
            raise RuntimeError("video geometry differs")
        if not any(s_q > p and (s_q - p) % t == 0 for p, t in FAMILY_LATTICES[family]):
            raise RuntimeError("sequence length differs")
        if s_kv != s_q and not (family == "kandinsky5" and s_kv == KANDINSKY5_TEXT_TOKENS):
            raise RuntimeError("cross attention differs")
    elif not causal or s_kv != s_q or s_q % LLM_LENGTH_QUANTUM:
        raise RuntimeError("LLM geometry differs")
    seed = case["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RuntimeError("seed differs")
    gated = case["gated"]
    if not isinstance(gated, bool) or (tier != "sweep" and not gated):
        raise RuntimeError("gated flag differs")
    layer = case["layer"]
    if tier == "real":
        if not isinstance(layer, str) or not REAL_LAYER_PATTERN.fullmatch(layer) or b != 1:
            raise RuntimeError("real layer differs")
    elif layer is not None:
        raise RuntimeError("real layer differs")
    case.update(
        softmax_scale=float(d) ** -0.5,
        scored=True,
        effective_flops=effective_flops(b, h, s_q, s_kv, d, causal),
    )
    return case


def validate_final_manifest(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Validate the Judge-only manifest: the declared shape, unique identities, half gated."""
    if set(payload) != {"schema_version", "suite_id", "cases"}:
        raise RuntimeError("manifest identity differs")
    if payload.get("schema_version") != 2 or payload.get("suite_id") != FINAL_SUITE_ID:
        raise RuntimeError("manifest identity differs")
    raw = payload.get("cases")
    if not isinstance(raw, list) or len(raw) != EXPECTED_CASE_COUNT:
        raise RuntimeError("manifest case count differs")
    ids: set[Any] = set()
    seeds: set[Any] = set()
    identities: set[Any] = set()
    counts: dict[str, dict[str, int]] = {}
    gated_sweep: dict[str, int] = {}
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise RuntimeError("case is not object")
        case = validate_case(item)
        identity = case["layer"] or (
            case["family"], case["b"], case["h"], case["h_kv"], case["d"], case["s_q"],
            case["s_kv"], case["causal"],
        )
        if case["case_id"] in ids or case["seed"] in seeds or identity in identities:
            raise RuntimeError("manifest identities are not unique")
        ids.add(case["case_id"])
        seeds.add(case["seed"])
        identities.add(identity)
        family_counts = counts.setdefault(case["family"], {})
        family_counts[case["tier"]] = family_counts.get(case["tier"], 0) + 1
        if case["tier"] == "sweep" and case["gated"]:
            gated_sweep[case["family"]] = gated_sweep.get(case["family"], 0) + 1
        result.append({key: case[key] for key in CASE_KEYS})
    if counts != EXPECTED_TIER_COUNTS:
        raise RuntimeError("manifest shape differs")
    for family, tiers in EXPECTED_TIER_COUNTS.items():
        if "sweep" in tiers and gated_sweep.get(family, 0) < tiers["sweep"] // 2:
            raise RuntimeError("manifest gates too few sweep cases")
    return tuple(result)


def _strict_json(
    path: Path,
    label: str,
    max_bytes: int = 16 << 20,
    expected_uid: int = 0,
    mode: int | None = None,
) -> dict[str, Any]:
    """Read a JSON object from a file that must be a plain, unshared, privately-owned file."""
    st = path.lstat()
    permissions = stat.S_IMODE(st.st_mode)
    ok = (
        stat.S_ISREG(st.st_mode)
        and st.st_nlink == 1
        and st.st_uid == expected_uid
        and not permissions & 0o077
        and (mode is None or permissions == mode)
        and 0 < st.st_size <= max_bytes
    )
    if not ok:
        raise RuntimeError(f"{label} file contract failed")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)),
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} not object")
    return value


def load_evaluation_profile(
    path: Path = EVALUATION_CONTEXT, *, expected_uid: int = 0
) -> tuple[str, str, str, int | None]:
    """Derive the evaluation profile, suite id and candidate override from the round id."""
    context = _strict_json(path, "evaluation context", expected_uid=expected_uid, mode=0o600)
    round_id = context.get("round_id")
    override = context.get("candidate_version")
    if not isinstance(round_id, str):
        raise RuntimeError("round_id missing")
    if override is not None:
        bad_override = (
            isinstance(override, bool)
            or not isinstance(override, int)
            or not 1 <= override <= 999
        )
        if bad_override:
            raise RuntimeError("candidate version invalid")
    if any(round_id.startswith(prefix) and len(round_id) > len(prefix) for prefix in ("auto-", "agent-")):
        if override is not None:
            raise RuntimeError("interactive override forbidden")
        # Keep interactive authority separate from host-side replay authority,
        # while both phases evaluate the same frozen complete final suite.
        return "final", FINAL_SUITE_ID, round_id, None
    if any(round_id.startswith(prefix) and len(round_id) > len(prefix) for prefix in ("manual-", "candidate-")):
        return "final", FINAL_SUITE_ID, round_id, override
    raise RuntimeError("unknown profile")


def load_final_manifest(
    path: Path = FINAL_MANIFEST, *, expected_uid: int = 0
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Load and validate the private holdout manifest, returning its cases and file digest."""
    payload = _strict_json(path, "final manifest", expected_uid=expected_uid)
    raw = path.read_bytes()
    return validate_final_manifest(payload), hashlib.sha256(raw).hexdigest()


def locate_submission(app_dir: Path) -> Path:
    """Find the one candidate lab root under the application directory."""
    roots = (
        app_dir / "workdir/attn_qat_lab",
        app_dir / "environment/workdir/attn_qat_lab",
        app_dir,
    )
    found: list[Path] = []
    for root in roots:
        is_lab_root = (
            root.is_dir()
            and not root.is_symlink()
            and (root / "candidates").is_dir()
            and (root / "FINAL_VERSION").exists()
        )
        if is_lab_root and root.absolute() not in found:
            found.append(root.absolute())
    if len(found) != 1:
        raise RuntimeError(f"expected one submission, found {len(found)}")
    return found[0]


def verify_protected_hashes(manifest: Path) -> list[str]:
    """Check the Judge's own trusted files against the shipped SHA-256 manifest."""
    expected = {
        "judge:candidate_worker.py": JUDGE_ROOT / "candidate_worker.py",
        "judge:run_judge.py": JUDGE_ROOT / "run_judge.py",
        "judge:source_admission.py": JUDGE_ROOT / "source_admission.py",
        "judge:verify_attn_qat.py": JUDGE_ROOT / "verify_attn_qat.py",
        "worker:attn_qat_runtime.py": RUNTIME_ROOT / "attn_qat_runtime.py",
        # The e7 capture is bound by this recipe rather than by a byte digest, so the
        # recipe is as trusted as the runtime that enforces it.
        "worker:RECIPE.json": RUNTIME_ROOT / "RECIPE.json",
        "worker:REAL_LAYERS.json": RUNTIME_ROOT / "REAL_LAYERS.json",
        "worker:REAL_LAYERS_V2.json": RUNTIME_ROOT / "REAL_LAYERS_V2.json",
    }
    problems: list[str] = []
    seen: set[str] = set()
    try:
        lines = manifest.read_text().splitlines()
    except OSError:
        return ["protected manifest missing"]
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        digest, label = line.split(None, 1)
        seen.add(label)
        matches = (
            label in expected
            and expected[label].is_file()
            and sha256_file(expected[label]) == digest
        )
        if not matches:
            problems.append(f"protected file failure: {label}")
    if seen != set(expected):
        problems.append("protected records differ")
    return problems


def _agent_account() -> pwd.struct_passwd:
    """The unprivileged account the worker runs as."""
    return pwd.getpwnam("agent")


def _drop_to_agent() -> None:
    """Drop every privilege before exec'ing the worker."""
    account = _agent_account()
    os.setgroups([])
    os.setgid(account.pw_gid)
    os.setuid(account.pw_uid)
    os.umask(0o077)


def _signal_session(sid: int, sig: signal.Signals) -> None:
    """Signal every process group in the worker's session."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getsid(int(entry.name)) == sid:
                os.killpg(os.getpgid(int(entry.name)), sig)
        except OSError:
            pass


def run_worker(
    candidate: Path,
    run_root: Path,
    cases: Sequence[Mapping[str, Any]],
    suite_id: str,
    *,
    trials: int,
    warmups: int,
    timeout: float,
) -> dict[str, Any]:
    """Run one suite in a uid-dropped worker in its own session and return its result."""
    account = _agent_account()
    result_dir = run_root / "result"
    cache = run_root / "cache"
    result_dir.mkdir(mode=0o700)
    cache.mkdir(mode=0o700)
    for path in (result_dir, cache):
        os.chown(path, account.pw_uid, account.pw_gid)
    output = result_dir / "result.json"
    env = {
        key: os.environ[key]
        for key in WORKER_ENVIRONMENT_KEYS
        if key in os.environ
    }
    env.update(
        HOME="/home/agent",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        TMPDIR=str(cache),
        TRITON_CACHE_DIR=str(cache / "triton"),
        ATTN_QAT_FIXTURE_DIR=str(JUDGE_ROOT / "fixtures"),
    )
    (cache / "triton").mkdir()
    os.chown(cache / "triton", account.pw_uid, account.pw_gid)
    command = [
        sys.executable,
        str(WORKER),
        "--candidate",
        str(candidate),
        "--output",
        str(output),
        "--trials",
        str(trials),
        "--warmups",
        str(warmups),
    ]
    manifest = json.dumps(
        {"schema_version": 1, "suite_id": suite_id, "cases": list(cases)}
    ).encode()
    with (run_root / "worker.log").open("wb") as log:
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
            process.communicate(manifest, timeout=timeout)
        except subprocess.TimeoutExpired:
            _signal_session(process.pid, signal.SIGKILL)
            raise RuntimeError("worker timed out")
    if process.returncode:
        raise RuntimeError("worker failed")
    return _strict_json(output, "worker result", expected_uid=account.pw_uid)


def _metric(row: Mapping[str, Any], name: str, minimum: float = 0.0) -> float:
    """Read one reported statistic, rejecting anything non-numeric, non-finite or too small."""
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"bad metric {name}")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise RuntimeError(f"bad metric {name}")
    return number


# Capability envelope: what the task contract actually needs, not the GPU model
# it was qualified on.  The GPU name and UUID are recorded as provenance only;
# they never gate admission.  Keep these two constants identical in the Judge
# verifier (tests assert the pair).
MIN_COMPUTE_CAPABILITY = (8, 9)
MIN_DEVICE_MEMORY_BYTES = 40 * 1024**3



def _validate_hardware_record(result: Mapping[str, Any]) -> None:
    """Reject a worker inventory that does not satisfy the capability envelope."""
    hardware = result.get("hardware") if isinstance(result, Mapping) else None
    capability = hardware.get("capability") if isinstance(hardware, Mapping) else None
    total_memory = hardware.get("total_memory_bytes") if isinstance(hardware, Mapping) else None
    if (
        not isinstance(hardware, Mapping)
        or hardware.get("count") != 1
        or not isinstance(capability, list)
        or len(capability) != 2
        or tuple(int(v) for v in capability) < MIN_COMPUTE_CAPABILITY
        or isinstance(total_memory, bool)
        or not isinstance(total_memory, int)
        or total_memory < MIN_DEVICE_MEMORY_BYTES
    ):
        raise RuntimeError("runtime hardware inventory violates the capability envelope")


def validate_runtime_result(
    result: Mapping[str, Any],
    expected_cases: Sequence[Mapping[str, Any]],
    *,
    expected_trials: int,
    expected_warmups: int | None = None,
) -> tuple[float, dict[str, float]]:
    """Re-derive every scored number from the worker's result and return the aggregate."""
    header_ok = (
        result.get("schema_version") == 1
        and result.get("entrypoints") == list(ENTRYPOINTS)
        and result.get("trials_per_case") == expected_trials
    )
    if not header_ok:
        raise RuntimeError("runtime header differs")
    if expected_warmups is not None and result.get("warmups") != expected_warmups:
        raise RuntimeError("warmups differ")
    if any(result.get(key) is not True for key in RUNTIME_GATE_KEYS):
        raise RuntimeError("runtime gate failed")
    _validate_hardware_record(result)
    if tuple(result.get("edge_case_ids", ())) != EDGE_CASE_IDS:
        raise RuntimeError("edge set differs")
    if len(result.get("edge_cases", ())) != EXPECTED_EDGE_CASE_COUNT:
        raise RuntimeError("edge set differs")
    expected = tuple(validate_case(case) for case in expected_cases)
    rows = result.get("cases")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise RuntimeError("case set differs")
    by_id = {row.get("case_id"): row for row in rows}
    values: dict[str, dict[str, list[float]]] = {}
    for case in expected:
        row = by_id.get(case["case_id"])
        if not isinstance(row, Mapping):
            raise RuntimeError("case missing")
        for key in ("family", "tier", "b", "s_q", "s_kv", "h", "h_kv", "d", "causal", "gated",
                    "effective_flops"):
            if row.get(key) != case[key]:
                raise RuntimeError(f"case {key} differs")
        trial_us = row.get("trial_us")
        trial_tf = row.get("trial_tflops")
        trial_forward = row.get("trial_forward_us")
        trial_backward = row.get("trial_backward_us")
        case_trials = REAL_TRIALS if case["tier"] == "real" else expected_trials
        for series in (trial_us, trial_tf, trial_forward, trial_backward):
            if not isinstance(series, list) or len(series) != case_trials:
                raise RuntimeError("trials differ")
        for index in range(case_trials):
            forward = finite_positive(trial_forward[index], "trial forward us")
            backward = finite_positive(trial_backward[index], "trial backward us")
            # The weighted time is re-derived here, not trusted: two forwards to one
            # backward is the production checkpointing ratio the score is built on.
            weighted = 2.0 * forward + backward
            if not math.isclose(weighted, float(trial_us[index]), rel_tol=1e-9):
                raise RuntimeError("trial weighting differs")
            computed = case["effective_flops"] / (weighted * 1e-6) / 1e12
            if not math.isclose(computed, float(trial_tf[index]), rel_tol=1e-9):
                raise RuntimeError("trial accounting differs")
        value = finite_positive(row.get("tflops"), "case TFLOPS")
        if not math.isclose(value, statistics.median(trial_tf), rel_tol=1e-9):
            raise RuntimeError("median differs")
        for name in (METRICS + REPORTED_METRICS + LIMITS) if case["gated"] else ():
            _metric(row, name)
        for name in GATE_TENSORS if case["gated"] else ():
            limit = _metric(row, f"{name}_rel_l2_limit")
            error = _metric(row, f"{name}_rel_l2")
            reference_norm = _metric(row, f"{name}_ref_norm")
            # A scored case always has a non-zero golden, so the relative gate is the one
            # that applied and the Judge re-checks the inequality itself.
            if reference_norm <= 0.0 or error > limit:
                raise RuntimeError(f"case {name} accuracy differs")
        saved_bytes = _metric(row, "saved_bytes")
        saved_limit = _metric(row, "saved_bytes_limit")
        expected_limit = 2 * case["b"] * case["d"] * (
            2 * case["h"] * case["s_q"] + 2 * case["h_kv"] * case["s_kv"]
        ) + 4 * case["b"] * case["h"] * case["s_q"]
        if saved_limit != float(expected_limit) or saved_bytes > saved_limit:
            raise RuntimeError("saved cap differs")
        if row.get("fresh_input_ok") is not True or row.get("fresh_output_ok") is not True:
            raise RuntimeError("fresh pointer differs")
        values.setdefault(case["family"], {}).setdefault(case["tier"], []).append(value)
    if set(values) != set(FAMILIES):
        raise RuntimeError("family set differs")
    families = {
        name: geometric_mean([geometric_mean(group) for group in tiers.values()])
        for name, tiers in values.items()
    }
    domains = {
        domain: geometric_mean([families[name] for name in FAMILIES if FAMILY_DOMAIN[name] == domain])
        for domain in DOMAINS
    }
    primary = geometric_mean([domains[name] for name in DOMAINS])
    if result.get("family_tflops") is None:
        raise RuntimeError("aggregate differs")
    if not math.isclose(primary, float(result.get("primary_tflops", 0)), rel_tol=1e-9):
        raise RuntimeError("aggregate differs")
    return primary, families


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON object to `path` through a temporary file in the same directory."""
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def failure_detail(
    message: str, *, profile: str | None = None, suite_id: str | None = None
) -> dict[str, Any]:
    """The all-gates-closed record every failure path starts from."""
    detail: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_policy": EVALUATION_POLICY,
        "selection_protocol_id": SELECTION_PROTOCOL_ID,
        "profile_ok": 0.0,
        "suite_ok": 0.0,
        "source_ok": 0.0,
        "worker_ok": 0.0,
        "hardware_ok": 0.0,
        "correctness_ok": 0.0,
        "edge_ok": 0.0,
        "fresh_input_ok": 0.0,
        "fresh_output_ok": 0.0,
        "saved_cap_ok": 0.0,
        "profile_code": PROFILE_CODES.get(profile or "", 0.0),
        "score_tflops": 0.0,
        **{f"{name}_tflops": 0.0 for name in FAMILIES},
        "video_tflops": 0.0,
        "llm_tflops": 0.0,
        "final_version": 0.0,
        "num_candidate_versions": 0.0,
        "private_error": message,
    }
    if profile:
        detail["evaluation_profile"] = profile
    if suite_id:
        detail["suite_id"] = suite_id
    return detail


def evaluate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Run one evaluation end to end and return its exit code and private record."""
    if verify_protected_hashes(JUDGE_ROOT / "protected_hashes.txt"):
        return 1, failure_detail("protected hash failure")
    try:
        profile, suite, round_id, override = load_evaluation_profile()
    except Exception as exc:
        return 1, failure_detail(str(exc))
    base = failure_detail("incomplete", profile=profile, suite_id=suite)
    base["profile_ok"] = 1.0
    base["round_id"] = round_id
    try:
        if profile != "final":
            raise RuntimeError("shared-final-v2 requires the complete final profile")
        cases, digest = load_final_manifest()
        base.update(
            suite_ok=1.0,
            manifest_sha256=digest,
            case_count=float(EXPECTED_CASE_COUNT),
            edge_case_count=float(EXPECTED_EDGE_CASE_COUNT),
        )
        admission = validate_submission(locate_submission(args.app_dir.resolve()), override)
        base["final_version"] = float(admission.get("final_version") or 0)
        base["num_candidate_versions"] = float(admission.get("num_candidate_versions") or 0)
        if not admission.get("ok"):
            raise RuntimeError("source admission failed")
        base["source_ok"] = 1.0
        run_root = Path(tempfile.mkdtemp(prefix="attn-qat-v1-"))
        run_root.chmod(0o755)
        source = run_root / "submission.py"
        shutil.copyfile(admission["selected_path"], source)
        source.chmod(0o444)
        result = run_worker(
            source,
            run_root,
            cases,
            suite,
            trials=args.trials,
            warmups=args.warmups,
            timeout=args.candidate_timeout,
        )
        # The worker produced a trusted result file, so any gate that fails from here on is a
        # property of the candidate, not of the hardware.
        base["worker_ok"] = 1.0
        base["hardware_ok"] = 1.0
        for key in RUNTIME_GATE_KEYS:
            base[key] = 1.0 if result.get(key) is True else 0.0
        score, families = validate_runtime_result(
            result, cases, expected_trials=args.trials, expected_warmups=args.warmups
        )
        base.update(
            score_tflops=score,
            **{f"{name}_tflops": families[name] for name in FAMILIES},
            video_tflops=geometric_mean(
                [families[name] for name in FAMILIES if FAMILY_DOMAIN[name] == "video"]),
            llm_tflops=geometric_mean(
                [families[name] for name in FAMILIES if FAMILY_DOMAIN[name] == "llm"]),
            selected_source_sha256=admission["selected_sha256"],
            private_runtime_result=result,
            private_error="",
        )
        shutil.rmtree(run_root, ignore_errors=True)
        return 0, base
    except Exception as exc:
        base["private_error"] = f"{type(exc).__name__}: {exc}"
        return 1, base


def write_result(log_dir: Path, detail: Mapping[str, Any]) -> None:
    """Write the private record, the numeric reward map and the plain reward file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    score = float(detail.get("score_tflops", 0))
    payload = dict(detail)
    payload["reward"] = score
    _atomic_json(log_dir / RESULT_NAME, payload)
    numeric = {
        key: float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }
    numeric["reward"] = score
    _atomic_json(log_dir / "reward.json", numeric)
    (log_dir / "reward.txt").write_text(f"{score}\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the Judge's invocation, defaulting every knob from the environment."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--app-dir", type=Path, default=Path(os.environ.get("APP_DIR", "/app"))
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path(os.environ.get("LOG_DIR", "/logs/verifier"))
    )
    parser.add_argument(
        "--trials", type=int, default=int(os.environ.get("ATTN_QAT_VERIFY_TRIALS", "3"))
    )
    parser.add_argument(
        "--warmups", type=int, default=int(os.environ.get("ATTN_QAT_VERIFY_WARMUPS", "1"))
    )
    parser.add_argument(
        "--candidate-timeout",
        type=int,
        default=int(os.environ.get("ATTN_QAT_CANDIDATE_TIMEOUT", "3600")),
    )
    parser.add_argument(
        "--total-timeout",
        type=int,
        default=int(os.environ.get("ATTN_QAT_TOTAL_TIMEOUT", "3900")),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate, then always write a result record even when the evaluation raised."""
    args = parse_args(argv)
    try:
        rc, detail = evaluate(args)
    except Exception as exc:
        rc, detail = 1, failure_detail(str(exc))
    write_result(args.log_dir, detail)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
