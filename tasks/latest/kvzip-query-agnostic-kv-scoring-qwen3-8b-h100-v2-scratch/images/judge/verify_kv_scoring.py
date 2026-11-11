#!/usr/bin/env python3
"""Root-owned two-process verifier for query-agnostic KV scoring, sharded across the Judge's GPUs.

Every scored round evaluates the complete frozen final suite
(full-final-shared-v2).  The suite is split into one shard per visible GPU;
each shard runs the unprivileged Phase-A worker and then the trusted Phase-B
decode on its own device, pinned to CPUs local to it.  Per-row work does not
depend on the shard layout, so the score does not depend on the GPU count.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, os, pwd, shutil, signal, stat, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

JUDGE_ROOT = Path("/opt/kvzip_kv_scoring_judge")
WORKER_ROOT = Path("/opt/kvzip_kv_scoring_worker")
WORKER = JUDGE_ROOT / "candidate_worker.py"
REPLAY = JUDGE_ROOT / "trusted_replay.py"
ASSET_GATE = JUDGE_ROOT / "released_weights_gate.py"
RELEASED_FINGERPRINT = JUDGE_ROOT / "private/released_weights_fingerprint.pt"
# The host-only oracle (scripts/make_oracle_archive.sh) ships the released
# gate on purpose.  Only this exact transport asset, and only when the host
# sets KV_SCORING_HOST_ORACLE=1 for the Judge (never during agent runs), skips
# the released-weights gate.
ORACLE_TRANSPORT_SHA256 = "300de4f36e3f09c31117fe66a02512db40f9c981eda365d2d1e8213fc147d20e"
ASSET_GATE_TIMEOUT_S = 900
FINAL_MANIFEST = JUDGE_ROOT / "private/final_manifest.json"
FINAL_ITEMS = JUDGE_ROOT / "private/final_items.jsonl.gz"
PUBLIC_ITEMS = JUDGE_ROOT / "corpus/items_public.jsonl"
EVALUATION_CONTEXT = Path("/tmp/.sforge-evaluation-context.json")
RESULT_NAME = "kvzip_kv_scoring_v2_results.json"
PRIVATE_NAME = "kvzip_kv_scoring_v2_private.json.gz"
sys.path.insert(0, str(WORKER_ROOT)); sys.path.insert(0, str(JUDGE_ROOT))
from kv_scoring_runtime import (MIN_COMPUTE_CAPABILITY, MIN_DEVICE_MEMORY_BYTES, FINAL_SUITE_ID, GROUPS, RATIOS,
    OVERHEAD_FRACTION, OVERHEAD_ITEM_CAP, aggregate, context_only, handoff_bytes, item_cost, load_items_jsonl, overhead_item_floor_s, validate_final_items)
from source_admission import validate_submission
import judge_shards

PROTECTED = {
    "judge:candidate_worker.py": JUDGE_ROOT / "candidate_worker.py",
    "judge:trusted_replay.py": JUDGE_ROOT / "trusted_replay.py",
    "judge:judge_shards.py": JUDGE_ROOT / "judge_shards.py",
    "judge:released_weights_gate.py": JUDGE_ROOT / "released_weights_gate.py",
    "judge:run_judge.py": JUDGE_ROOT / "run_judge.py",
    "judge:source_admission.py": JUDGE_ROOT / "source_admission.py",
    "judge:verify_kv_scoring.py": JUDGE_ROOT / "verify_kv_scoring.py",
    "worker:kv_scoring_runtime.py": WORKER_ROOT / "kv_scoring_runtime.py",
    "worker:kv_scoring_metric.py": WORKER_ROOT / "kv_scoring_metric.py",
}
GATE_KEYS = ("profile_ok", "suite_ok", "source_ok", "assets_ok", "hardware_ok", "scores_ok", "immutable_ok", "rng_ok", "overhead_ok", "memory_ok", "metric_ok", "reference_ok")
RATIO_KEYS = {"0.1": "ratio_010", "0.2": "ratio_020", "0.3": "ratio_030"}
# Rows over the per-item overhead cap are timed once more in a fresh worker
# process (no module state survives; candidates cannot touch the filesystem),
# so a platform stall during one call cannot fail a scorer.  Beyond this many
# such rows the scorer is genuinely slow and nothing is re-timed.
RETIME_MAX_ITEMS = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def verify_protected_hashes(manifest: Path) -> list[str]:
    problems = []; seen = set()
    try: lines = manifest.read_text().splitlines()
    except OSError: return ["protected manifest missing"]
    for line in lines:
        if not line.strip() or line.startswith("#"): continue
        try: digest, label = line.split(None, 1)
        except ValueError: problems.append("malformed protected record"); continue
        seen.add(label)
        if label not in PROTECTED or not PROTECTED[label].is_file() or sha256_file(PROTECTED[label]) != digest: problems.append(f"protected file failure: {label}")
    if seen != set(PROTECTED): problems.append("protected records differ")
    return problems


def _strict_file(path: Path, label: str, *, mode: int, max_bytes: int) -> bytes:
    st = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_uid != 0 or stat.S_IMODE(st.st_mode) != mode or not 0 < st.st_size <= max_bytes: raise RuntimeError(f"{label} file contract failed")
    return path.read_bytes()


def _strict_json(path: Path, label: str, *, mode: int = 0o600, max_bytes: int = 1 << 20) -> dict[str, Any]:
    value = json.loads(_strict_file(path, label, mode=mode, max_bytes=max_bytes), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict): raise RuntimeError(f"{label} must be an object")
    return value


def evaluation_profile() -> tuple[str, str, int | None]:
    raw = _strict_json(EVALUATION_CONTEXT, "evaluation context")
    round_id = raw.get("round_id"); override = raw.get("candidate_version")
    if not isinstance(round_id, str): raise RuntimeError("round_id missing")
    if override is not None and (isinstance(override, bool) or not isinstance(override, int) or not 1 <= override <= 999): raise RuntimeError("candidate_version invalid")
    if round_id.startswith(("auto-", "agent-")):
        if override is not None: raise RuntimeError("interactive override forbidden")
        return "final", FINAL_SUITE_ID, None
    if round_id.startswith(("manual-", "candidate-")): return "final", FINAL_SUITE_ID, override
    raise RuntimeError("unknown evaluation profile")


def load_final_suite() -> tuple[tuple[dict[str, Any], ...], str]:
    manifest_raw = _strict_file(FINAL_MANIFEST, "final manifest", mode=0o400, max_bytes=64 << 20)
    _strict_file(FINAL_ITEMS, "final items", mode=0o400, max_bytes=2 << 30)
    with gzip.open(FINAL_ITEMS, "rt", encoding="utf-8") as handle: rows = [json.loads(line) for line in handle if line.strip()]
    items = validate_final_items(rows, json.loads(manifest_raw), load_items_jsonl(PUBLIC_ITEMS))
    return items, hashlib.sha256(manifest_raw).hexdigest()


def locate_submission(app_dir: Path) -> Path:
    roots = (app_dir / "workdir/kv_scoring_lab", app_dir / "environment/workdir/kv_scoring_lab", app_dir); found = []
    for root in roots:
        if root.is_dir() and not root.is_symlink() and (root / "candidates").is_dir() and (root / "FINAL_VERSION").is_file() and root.absolute() not in found: found.append(root.absolute())
    if len(found) != 1: raise RuntimeError(f"expected one submission, found {len(found)}")
    return found[0]


def _agent(): return pwd.getpwnam("agent")


class Sessions:
    """Every child is a session leader; on any failure all of them are killed."""
    def __init__(self) -> None: self._lock = threading.Lock(); self._pids: set[int] = set()
    def add(self, pid: int) -> None:
        with self._lock: self._pids.add(pid)
    def discard(self, pid: int) -> None:
        with self._lock: self._pids.discard(pid)
    def kill_all(self) -> None:
        with self._lock: sessions = set(self._pids)
        if not sessions: return
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                try:
                    if os.getsid(int(entry.name)) in sessions: os.killpg(os.getpgid(int(entry.name)), signal.SIGKILL)
                except OSError: pass


def _run_child(cmd: list[str], *, env: dict[str, str], cwd: Path, log: Path, stdin: bytes | None, timeout: float, cpus: Sequence[int],
               as_agent: bool, sessions: Sessions, label: str) -> None:
    # Shards start children from several threads at once, so no preexec_fn:
    # privileges drop through Popen's user/group arguments and the CPU slice is
    # applied by taskset before the interpreter starts any thread.
    pinned = ["taskset", "--cpu-list", ",".join(str(c) for c in cpus), *cmd]
    identity: dict[str, Any] = {}
    if as_agent:
        account = _agent(); identity = {"user": account.pw_uid, "group": account.pw_gid, "extra_groups": [], "umask": 0o077}
    with log.open("wb") as handle:
        p = subprocess.Popen(pinned, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, **identity)
        sessions.add(p.pid)
        try: p.communicate(stdin, timeout=max(1.0, timeout))
        except subprocess.TimeoutExpired: sessions.kill_all(); raise RuntimeError(f"{label} timed out")
        finally: sessions.discard(p.pid)
    if p.returncode:
        tail = log.read_text(errors="replace")[-2000:].strip()
        raise RuntimeError(f"{label} failed: {tail}")


def _base_env() -> dict[str, str]:
    keep = ("LD_LIBRARY_PATH", "PATH", "KV_SCORING_MODEL_DIR", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "NVIDIA_DRIVER_CAPABILITIES", "PYTORCH_CUDA_ALLOC_CONF")
    return {k: os.environ[k] for k in keep if k in os.environ}


def _agent_json(path: Path, label: str, *, max_bytes: int = 1 << 16) -> dict[str, Any]:
    """Telemetry written by the unprivileged worker: one regular, agent-owned JSON object."""
    st = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_uid != _agent().pw_uid or not 0 < st.st_size <= max_bytes: raise RuntimeError(f"{label} file contract failed")
    value = json.loads(path.read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict): raise RuntimeError(f"{label} must be an object")
    return value


def _run_worker(gpu: judge_shards.Gpu, cpus: Sequence[int], items: Sequence[Mapping[str, Any]], candidate: Path, asset: Path | None,
                shard_root: Path, out_name: str, deadline: float, sessions: Sessions, label: str) -> float:
    account = _agent(); out = shard_root / out_name; out.mkdir(); out.chmod(0o700); os.chown(out, account.pw_uid, account.pw_gid)
    env = judge_shards.shard_environment(_base_env(), gpu, cpus); env.update(HOME="/home/agent", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    cmd = [sys.executable, str(WORKER), "--candidate", str(candidate), "--out-dir", str(out)]
    if asset is not None: cmd.extend(("--assets", str(asset)))
    request = json.dumps({"items": [context_only(x) for x in items]}).encode()
    started = time.monotonic(); _run_child(cmd, env=env, cwd=shard_root, log=shard_root / f"{out_name}.log", stdin=request, timeout=deadline - started,
                                           cpus=cpus, as_agent=True, sessions=sessions, label=label)
    return time.monotonic() - started


def run_phase_a_shard(index: int, gpu: judge_shards.Gpu, cpus: Sequence[int], items: Sequence[Mapping[str, Any]], candidate: Path, asset: Path | None,
                      run_root: Path, deadlines: Mapping[str, float], sessions: Sessions) -> dict[str, Any]:
    shard_root = run_root / f"shard-{index}"; shard_root.mkdir(); shard_root.chmod(0o755)
    wall = _run_worker(gpu, cpus, items, candidate, asset, shard_root, "scores", deadlines["phase_a"], sessions, f"Phase A shard {index}")
    return {"index": index, "gpu": gpu.record(), "cpus": list(cpus), "items": len(items), "phase_a_wall_s": wall}


def run_phase_b_shard(index: int, gpu: judge_shards.Gpu, cpus: Sequence[int], items: Sequence[Mapping[str, Any]], run_root: Path,
                      deadlines: Mapping[str, float], sessions: Sessions) -> dict[str, Any]:
    shard_root = run_root / f"shard-{index}"; scores = shard_root / "scores"
    replay_request = shard_root / "replay-request.json"; fd = os.open(replay_request, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle: json.dump({"items": list(items)}, handle)
    out = shard_root / "replay-result.json"
    replay_env = judge_shards.shard_environment(_base_env(), gpu, cpus); replay_env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    started = time.monotonic(); _run_child([sys.executable, str(REPLAY), "--request", str(replay_request), "--scores", str(scores), "--out", str(out)],
        env=replay_env, cwd=shard_root, log=shard_root / "replay.log", stdin=None, timeout=deadlines["total"] - started,
        cpus=cpus, as_agent=False, sessions=sessions, label=f"Phase B shard {index}")
    phase_b_wall = time.monotonic() - started
    result = _strict_json(out, f"shard {index} replay result", mode=0o600, max_bytes=1 << 30)
    hardware = result.get("hardware") or {}
    # torch reports the UUID without nvidia-smi's "GPU-" prefix.
    if str(hardware.get("uuid", "")).removeprefix("GPU-") != gpu.uuid.removeprefix("GPU-"): raise RuntimeError(f"shard {index} ran on {hardware.get('uuid')}, expected {gpu.uuid}")
    return {"rows": result["rows"], "hardware": hardware, "phase_b_wall_s": phase_b_wall, "phase_b_seconds": result.get("phase_b_seconds")}


def released_weights_check(asset: Path | None, run_root: Path, sessions: Sessions) -> dict[str, Any]:
    """Compare the selected assets with the released-checkpoint fingerprint in a privilege-dropped trusted process.

    No candidate code runs there; the unprivileged process only parses the
    untrusted asset (weights_only) and receives the root-only fingerprint on
    stdin.  Any failure to produce a verdict fails the gate.
    """
    if asset is None: return {"ok": True, "status": "no-assets"}
    if os.environ.get("KV_SCORING_HOST_ORACLE") == "1" and sha256_file(asset) == ORACLE_TRANSPORT_SHA256: return {"ok": True, "status": "host-oracle-exempt"}
    reference = _strict_file(RELEASED_FINGERPRINT, "released-weights fingerprint", mode=0o400, max_bytes=64 << 20)
    log = run_root / "released-weights-gate.log"; cpus = sorted(os.sched_getaffinity(0))
    env = _base_env(); env.update(HOME="/home/agent", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", OMP_NUM_THREADS=str(min(16, len(cpus))))
    _run_child([sys.executable, "-I", str(ASSET_GATE), "check", str(asset)], env=env, cwd=run_root, log=log, stdin=reference, timeout=ASSET_GATE_TIMEOUT_S,
               cpus=cpus, as_agent=True, sessions=sessions, label="released-weights gate")
    lines = [line for line in log.read_text(errors="replace").splitlines() if line.startswith("{")]
    if not lines: raise RuntimeError("released-weights gate produced no verdict")
    verdict = json.loads(lines[-1], parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(verdict, dict) or not isinstance(verdict.get("ok"), bool): raise RuntimeError("released-weights gate verdict is malformed")
    return {**verdict, "status": "checked"}


def over_cap_items(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    floor = overhead_item_floor_s(rows)
    return sorted(str(row["item_id"]) for row in rows if float(row["score_time_sum"]) > max(OVERHEAD_ITEM_CAP * float(row.get("prefill_s", 0.0)), floor))


def apply_retime(rows: Sequence[dict[str, Any]], item_id: str, fresh: Mapping[str, Any]) -> bool:
    """Adopt a re-timed measurement for one row when it reproduces the scores bit for bit and is faster."""
    first = [row for row in rows if row["item_id"] == item_id]
    if len(first) != 1: return False
    row = first[0]; digest = row.get("scores_sha256")
    if fresh.get("item_id") != item_id or not isinstance(digest, str) or len(digest) != 64 or fresh.get("scores_sha256") != digest: return False
    if not all(fresh.get(k) is True for k in ("scores_ok", "immutable_ok", "rng_ok")): return False
    if float(fresh["score_time_sum"]) >= float(row["score_time_sum"]): return False
    row.update(score_time_sum_first=float(row["score_time_sum"]), retimed=True, score_time_sum=float(fresh["score_time_sum"]),
               score_time_max_layer=float(fresh["score_time_max_layer"]), memory_delta=max(int(row["memory_delta"]), int(fresh["memory_delta"])))
    return True


def retime_over_cap(pool: ThreadPoolExecutor, rows: Sequence[dict[str, Any]], plan: Sequence[Sequence[str]], gpus: Sequence[judge_shards.Gpu],
                    slices: Sequence[Sequence[int]], by_id: Mapping[str, Mapping[str, Any]], candidate: Path, asset: Path | None,
                    run_root: Path, deadlines: Mapping[str, float], sessions: Sessions) -> dict[str, Any]:
    over = set(over_cap_items(rows)); info = {"over_cap_items": float(len(over)), "retimed_items": 0.0, "retime_wall_s": 0.0}
    if not over or len(over) > RETIME_MAX_ITEMS: return info
    started = time.monotonic(); jobs = {i: [by_id[x] for x in plan[i] if x in over] for i in range(len(gpus))}
    futures = {i: pool.submit(_run_worker, gpus[i], slices[i], items, candidate, asset, run_root / f"shard-{i}", "retime", deadlines["total"], sessions,
                              f"Phase A re-time shard {i}") for i, items in jobs.items() if items}
    for i, future in futures.items():
        try: future.result()
        except Exception: continue  # a failed re-time keeps the first measurement
        out = run_root / f"shard-{i}" / "retime"
        for item in jobs[i]:
            try: fresh = _agent_json(out / f"{item['item_id']}.json", "re-time telemetry")
            except Exception: continue
            if apply_retime(rows, item["item_id"], fresh): info["retimed_items"] += 1.0
    info["retime_wall_s"] = time.monotonic() - started
    return info


def failure(message: str, profile: str = "") -> dict[str, Any]:
    result = {"schema_version": 2, "evaluation_profile": profile, "profile_code": 2.0 if profile == "final" else 0.0,
        "score_rel_auc": 0.0, "item_count": 0.0, "question_count": 0.0,
        "final_version": 0.0, "num_candidate_versions": 0.0, "overhead_item_violations": 0.0, "deterministic_items": 0.0, "private_error": message}
    result.update({key: 0.0 for key in RATIO_KEYS.values()}); result.update({key: 0.0 for key in GATE_KEYS})
    result.update({f"group_{g}": 0.0 for g in GROUPS})
    return result


def evaluate(app_dir: Path, candidate_timeout: int, total_timeout: int) -> tuple[int, dict[str, Any], dict[str, Any]]:
    started = time.monotonic(); private: dict[str, Any] = {}
    if verify_protected_hashes(JUDGE_ROOT / "protected_hashes.txt"): return 1, failure("protected hash failure"), private
    try: profile, suite_id, override = evaluation_profile()
    except Exception as exc: return 1, failure(str(exc)), private
    detail = failure("incomplete", profile); detail.update(suite_id=suite_id, profile_ok=1.0)
    sessions = Sessions(); run_root: Path | None = None
    try:
        items, manifest_sha = load_final_suite()
        detail.update(suite_ok=1.0, manifest_sha256=manifest_sha, item_count=float(len(items)), question_count=float(sum(len(x["questions"]) for x in items)))
        admission = validate_submission(locate_submission(app_dir), override)
        detail.update(final_version=float(admission.get("final_version") or 0), num_candidate_versions=float(admission.get("num_candidate_versions") or 0))
        if not admission["ok"]: raise RuntimeError("source admission failed: " + "; ".join(admission["errors"]))
        detail["source_ok"] = 1.0
        gpus = judge_shards.visible_gpus(); fleet = judge_shards.validate_fleet(gpus, min_capability=MIN_COMPUTE_CAPABILITY, min_memory_bytes=MIN_DEVICE_MEMORY_BYTES)
        slices, cpu_mode = judge_shards.cpu_slices(gpus)
        by_id = {x["item_id"]: x for x in items}
        plan = judge_shards.plan_shards([x["item_id"] for x in items], {x["item_id"]: item_cost(x) for x in items}, len(gpus))
        detail.update(judge_gpu_count=float(len(gpus)), judge_fleet=fleet, cpu_slices=slices, cpu_slice_mode=cpu_mode)
        run_root = Path(tempfile.mkdtemp(prefix="kv-scoring-v2-")); run_root.chmod(0o755); account = _agent()
        need = handoff_bytes(items); free = shutil.disk_usage(run_root).free; detail.update(scratch_needed_bytes=float(need), scratch_free_bytes=float(free))
        if free < need * 5 // 4 + (1 << 30): raise RuntimeError(f"Judge scratch {run_root.parent} has {free >> 30} GiB free; the Phase A -> Phase B handoff needs {need >> 30} GiB plus 25% and 1 GiB")
        candidate = run_root / "submission.py"; shutil.copyfile(admission["selected_path"], candidate); candidate.chmod(0o444); os.chown(candidate, account.pw_uid, account.pw_gid)
        asset = None
        if admission.get("selected_assets_path"):
            asset = run_root / "assets.pt"; shutil.copyfile(admission["selected_assets_path"], asset); asset.chmod(0o444); os.chown(asset, account.pw_uid, account.pw_gid)
        released = released_weights_check(asset, run_root, sessions); detail["released_weights"] = released
        if not released["ok"]: raise RuntimeError("released-weights gate: assets match a published KV-importance checkpoint: " + "; ".join(map(str, released.get("reasons", []))))
        detail["assets_ok"] = 1.0
        deadlines = {"phase_a": started + candidate_timeout, "total": started + total_timeout - 30}
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            # Every shard finishes Phase A before any Phase B starts, so no
            # process start, model load or decode runs while a shard is timing.
            futures = [pool.submit(run_phase_a_shard, i, gpus[i], slices[i], [by_id[x] for x in plan[i]], candidate, asset, run_root, deadlines, sessions) for i in range(len(gpus))]
            try: phase_a = [f.result() for f in futures]
            except BaseException: sessions.kill_all(); raise
            futures = [pool.submit(run_phase_b_shard, i, gpus[i], slices[i], [by_id[x] for x in plan[i]], run_root, deadlines, sessions) for i in range(len(gpus))]
            try: shards = [{**a, **f.result()} for a, f in zip(phase_a, futures)]
            except BaseException: sessions.kill_all(); raise
            detail["hardware_ok"] = 1.0
            rows = [row for shard in shards for row in shard["rows"]]
            if sorted(row["item_id"] for row in rows) != sorted(by_id): raise RuntimeError("shard rows do not cover the suite exactly once")
            detail.update(retime_over_cap(pool, rows, plan, gpus, slices, by_id, candidate, asset, run_root, deadlines, sessions))
        order = {x["item_id"]: i for i, x in enumerate(items)}; rows.sort(key=lambda row: order[row["item_id"]])
        summary = aggregate(rows, RATIOS)
        expected_floor = overhead_item_floor_s(rows)
        expected_violations = sum(float(row["score_time_sum"]) > max(OVERHEAD_ITEM_CAP * float(row["prefill_s"]), expected_floor) for row in rows)
        expected_suite_ok = math.fsum(float(row["score_time_sum"]) for row in rows) <= OVERHEAD_FRACTION * math.fsum(float(row["prefill_s"]) for row in rows)
        if summary["overhead_item_violations"] != expected_violations or summary["gates"]["overhead_ok"] != (expected_suite_ok and expected_violations == 0):
            raise RuntimeError("overhead gate accounting differs")
        if summary["deterministic_items"] != sum(row.get("scores_deterministic") is True for row in rows):
            raise RuntimeError("score determinism telemetry differs")
        for gate, passed in summary["gates"].items(): detail[gate] = 1.0 if passed else 0.0
        group_score = {g: math.fsum(summary["group_rel"][str(float(r))].get(g, 0.0) for r in RATIOS) / len(RATIOS) for g in GROUPS}
        detail.update(score_rel_auc=summary["primary"], **{RATIO_KEYS[k]: v for k, v in summary["ratio_score"].items()}, **{f"group_{g}": v for g, v in group_score.items()},
            overhead_ratio_suite=summary["overhead_ratio_suite"], overhead_ratio_item_max=summary["overhead_ratio_item_max"], overhead_item_violations=float(summary["overhead_item_violations"]),
            deterministic_items=float(summary["deterministic_items"]), task_phase_b_seconds=summary["task_phase_b_seconds"], groups=summary["groups"], tasks=summary["tasks"],
            task_acc=summary["task_acc"], task_acc_full=summary["task_acc_full"], rel=summary["rel"], group_rel=summary["group_rel"], ratio_score=summary["ratio_score"],
            selected_source_sha256=admission["selected_sha256"], shards=[{k: v for k, v in s.items() if k != "rows"} for s in shards],
            wall_seconds=time.monotonic() - started, private_error="")
        private["private_rows"] = rows
        ok = all(detail[key] == 1.0 for key in GATE_KEYS)
        if not ok: detail.update(score_rel_auc=0.0, **{k: 0.0 for k in RATIO_KEYS.values()}, **{f"group_{g}": 0.0 for g in GROUPS})
        return (0 if ok else 1), detail, private
    except BaseException as exc:
        sessions.kill_all(); detail["private_error"] = f"{type(exc).__name__}: {exc}"; return 1, detail, private
    finally:
        if run_root is not None: shutil.rmtree(run_root, ignore_errors=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}"); tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(tmp, path)


def write_result(log_dir: Path, detail: Mapping[str, Any], private: Mapping[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True); score = float(detail.get("score_rel_auc", 0.0)); payload = {**detail, "reward": score}; _write_json(log_dir / RESULT_NAME, payload)
    numeric = {k: float(v) for k, v in payload.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))}; numeric["reward"] = score
    _write_json(log_dir / "reward.json", numeric); (log_dir / "reward.txt").write_text(f"{score}\n")
    tmp = log_dir / f".{PRIVATE_NAME}.tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as handle: json.dump({"detail": payload, **private}, handle, sort_keys=True, allow_nan=False)
    os.replace(tmp, log_dir / PRIVATE_NAME)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--app-dir", type=Path, default=Path("/app")); parser.add_argument("--log-dir", type=Path, default=Path("/logs/verifier"))
    parser.add_argument("--candidate-timeout", type=int, default=int(os.environ.get("KV_SCORING_CANDIDATE_TIMEOUT", "5400")))
    parser.add_argument("--total-timeout", type=int, default=int(os.environ.get("KV_SCORING_TOTAL_TIMEOUT", "9000"))); args = parser.parse_args(argv)
    private: dict[str, Any] = {}
    try:
        signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(TimeoutError("total verifier timeout"))); signal.alarm(args.total_timeout)
        rc, detail, private = evaluate(args.app_dir.resolve(), args.candidate_timeout, args.total_timeout); signal.alarm(0)
    except BaseException as exc: rc, detail = 1, failure(f"{type(exc).__name__}: {exc}")
    write_result(args.log_dir, detail, private); return rc


if __name__ == "__main__": raise SystemExit(main())
