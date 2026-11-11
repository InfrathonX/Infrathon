#!/usr/bin/env python3
"""Root-owned two-process verifier for KV eviction, sharded across the Judge's GPUs.

Every scored round evaluates the complete frozen final suite
(full-final-shared-v2).  The suite is split into one shard per visible GPU;
each shard runs the unprivileged Phase-A worker and then the trusted Phase-B
replay on its own device, pinned to CPUs local to it.  Per-item work does not
depend on the shard layout, so the score does not depend on the GPU count.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, os, pwd, shutil, signal, stat, subprocess, sys, tempfile, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

JUDGE_ROOT = Path("/opt/kvpress_kv_eviction_judge")
WORKER_ROOT = Path("/opt/kvpress_kv_eviction_worker")
WORKER = JUDGE_ROOT / "candidate_worker.py"
REPLAY = JUDGE_ROOT / "trusted_replay.py"
FINAL_MANIFEST = JUDGE_ROOT / "private/final_manifest.json"
FINAL_ITEMS = JUDGE_ROOT / "private/final_items.jsonl.gz"
PUBLIC_ITEMS = JUDGE_ROOT / "corpus/items_public.jsonl"
EVALUATION_CONTEXT = Path("/tmp/.sforge-evaluation-context.json")
RESULT_NAME = "kvpress_kv_eviction_v2_results.json"
PRIVATE_NAME = "kvpress_kv_eviction_v2_private.json.gz"
sys.path.insert(0, str(WORKER_ROOT)); sys.path.insert(0, str(JUDGE_ROOT))
from kv_eviction_runtime import (MIN_COMPUTE_CAPABILITY, MIN_DEVICE_MEMORY_BYTES, FINAL_SUITE_ID, RATIOS, FAMILIES, LATENCY_ITEM_CAP,
    aggregate, context_only, handoff_bytes, item_cost, load_items_jsonl, task_breakdown, validate_final_items)
from source_admission import validate_submission
import judge_shards

PROTECTED = {
    "judge:candidate_worker.py": JUDGE_ROOT / "candidate_worker.py",
    "judge:trusted_replay.py": JUDGE_ROOT / "trusted_replay.py",
    "judge:judge_shards.py": JUDGE_ROOT / "judge_shards.py",
    "judge:run_judge.py": JUDGE_ROOT / "run_judge.py",
    "judge:source_admission.py": JUDGE_ROOT / "source_admission.py",
    "judge:verify_kv_eviction.py": JUDGE_ROOT / "verify_kv_eviction.py",
    "worker:kv_eviction_runtime.py": WORKER_ROOT / "kv_eviction_runtime.py",
}
GATE_KEYS = ("profile_ok", "suite_ok", "source_ok", "hardware_ok", "indices_ok", "budgets_ok", "immutable_ok", "rng_ok", "latency_ok", "memory_ok", "fidelity_ok")
RATIO_KEYS = {"0.5": "ratio_050", "0.75": "ratio_075", "0.875": "ratio_0875"}
PRIVATE_ROW_KEYS = ("item_id", "family", "task", "ratio", "fid", "mean_kl", "top1", "nll_delta", "n_tokens", "select_time_sum", "select_time_max_layer", "select_time_own", "memory_delta", "prefill_s", "context_tokens", "retimed", "select_time_sum_first")
# Items over the per-item latency cap are timed once more in a fresh worker
# process (no module state survives; candidates cannot touch the filesystem),
# so a platform stall during one call cannot fail a policy.  Beyond this many
# such items the policy is genuinely slow and nothing is re-timed.
RETIME_MAX_ITEMS = 64


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


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
    manifest = json.loads(manifest_raw)
    _strict_file(FINAL_ITEMS, "final items", mode=0o400, max_bytes=2 << 30)
    with gzip.open(FINAL_ITEMS, "rt", encoding="utf-8") as handle: rows = [json.loads(line) for line in handle if line.strip()]
    items = validate_final_items(rows, manifest, load_items_jsonl(PUBLIC_ITEMS))
    return items, hashlib.sha256(manifest_raw).hexdigest()


def locate_submission(app_dir: Path) -> Path:
    roots = (app_dir / "workdir/kv_eviction_lab", app_dir / "environment/workdir/kv_eviction_lab", app_dir); found = []
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
    keep = ("LD_LIBRARY_PATH", "PATH", "KV_EVICTION_MODEL_DIR", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "NVIDIA_DRIVER_CAPABILITIES")
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
    request = json.dumps({"items": [context_only(x) for x in items], "ratios": list(RATIOS)}).encode()
    started = _now(); _run_child(cmd, env=env, cwd=shard_root, log=shard_root / f"{out_name}.log", stdin=request, timeout=deadline - started,
                                 cpus=cpus, as_agent=True, sessions=sessions, label=label)
    return _now() - started


def run_phase_a_shard(index: int, gpu: judge_shards.Gpu, cpus: Sequence[int], items: Sequence[Mapping[str, Any]], candidate: Path, asset: Path | None,
                      run_root: Path, deadlines: Mapping[str, float], sessions: Sessions) -> dict[str, Any]:
    shard_root = run_root / f"shard-{index}"; shard_root.mkdir(); shard_root.chmod(0o755)
    wall = _run_worker(gpu, cpus, items, candidate, asset, shard_root, "indices", deadlines["phase_a"], sessions, f"Phase A shard {index}")
    return {"index": index, "gpu": gpu.record(), "cpus": list(cpus), "items": len(items), "phase_a_wall_s": wall}


def run_phase_b_shard(index: int, gpu: judge_shards.Gpu, cpus: Sequence[int], items: Sequence[Mapping[str, Any]], run_root: Path,
                      deadlines: Mapping[str, float], sessions: Sessions) -> dict[str, Any]:
    shard_root = run_root / f"shard-{index}"; indices = shard_root / "indices"
    replay_request = shard_root / "replay-request.json"; fd = os.open(replay_request, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle: json.dump({"items": list(items)}, handle)
    out = shard_root / "replay-result.json"
    replay_env = judge_shards.shard_environment(_base_env(), gpu, cpus); replay_env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    started = _now(); _run_child([sys.executable, str(REPLAY), "--request", str(replay_request), "--indices", str(indices), "--out", str(out)],
        env=replay_env, cwd=shard_root, log=shard_root / "replay.log", stdin=None, timeout=deadlines["total"] - started,
        cpus=cpus, as_agent=False, sessions=sessions, label=f"Phase B shard {index}")
    phase_b_wall = _now() - started
    result = _strict_json(out, f"shard {index} replay result", mode=0o600, max_bytes=512 << 20)
    hardware = result.get("hardware") or {}
    # torch reports the UUID without nvidia-smi's "GPU-" prefix.
    if str(hardware.get("uuid", "")).removeprefix("GPU-") != gpu.uuid.removeprefix("GPU-"): raise RuntimeError(f"shard {index} ran on {hardware.get('uuid')}, expected {gpu.uuid}")
    return {"rows": result["rows"], "hardware": hardware, "phase_b_wall_s": phase_b_wall, "phase_b_seconds": result.get("phase_b_seconds")}


def over_cap_items(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(row["item_id"]) for row in rows
                   if not float(row.get("prefill_s", 0.0)) > 0.0 or float(row["select_time_sum"]) / float(row["prefill_s"]) > LATENCY_ITEM_CAP})


def apply_retime(rows: Sequence[dict[str, Any]], item_id: str, fresh: Mapping[float, Mapping[str, Any]]) -> bool:
    """Adopt a re-timed measurement for one item when it reproduces every selection bit for bit and is faster."""
    first = [row for row in rows if row["item_id"] == item_id]
    if len(first) != len(RATIOS) or set(fresh) != {float(r) for r in RATIOS}: return False
    for row in first:
        t = fresh[float(row["ratio"])]; digest = row.get("masks_sha256")
        if t.get("item_id") != item_id or not isinstance(digest, str) or len(digest) != 64 or t.get("masks_sha256") != digest: return False
        if not all(t.get(k) is True for k in ("indices_ok", "immutable_ok", "rng_ok", "budgets_ok")): return False
    if max(float(t["select_time_sum"]) for t in fresh.values()) >= min(float(row["select_time_sum"]) for row in first): return False
    for row in first:
        t = fresh[float(row["ratio"])]
        row.update(select_time_sum_first=float(row["select_time_sum"]), retimed=True, select_time_sum=float(t["select_time_sum"]),
                   select_time_max_layer=float(t["select_time_max_layer"]), select_time_own=float(t["select_time_own"]),
                   memory_delta=max(int(row["memory_delta"]), int(t["memory_delta"])))
    return True


def retime_over_cap(pool: ThreadPoolExecutor, rows: Sequence[dict[str, Any]], plan: Sequence[Sequence[str]], gpus: Sequence[judge_shards.Gpu],
                    slices: Sequence[Sequence[int]], by_id: Mapping[str, Mapping[str, Any]], candidate: Path, asset: Path | None,
                    run_root: Path, deadlines: Mapping[str, float], sessions: Sessions) -> dict[str, Any]:
    over = set(over_cap_items(rows)); info = {"over_cap_items": float(len(over)), "retimed_items": 0.0, "retime_wall_s": 0.0}
    if not over or len(over) > RETIME_MAX_ITEMS: return info
    started = _now(); jobs = {i: [by_id[x] for x in plan[i] if x in over] for i in range(len(gpus))}
    futures = {i: pool.submit(_run_worker, gpus[i], slices[i], items, candidate, asset, run_root / f"shard-{i}", "retime", deadlines["total"], sessions,
                              f"Phase A re-time shard {i}") for i, items in jobs.items() if items}
    for i, future in futures.items():
        try: future.result()
        except Exception: continue  # a failed re-time keeps the first measurement
        out = run_root / f"shard-{i}" / "retime"
        for item in jobs[i]:
            try: fresh = {float(r): _agent_json(out / item["item_id"] / f"{float(r)}.json", "re-time telemetry") for r in RATIOS}
            except Exception: continue
            if apply_retime(rows, item["item_id"], fresh): info["retimed_items"] += 1.0
    info["retime_wall_s"] = _now() - started
    return info


def _now() -> float:
    import time
    return time.monotonic()


def failure(message: str, profile: str = "") -> dict[str, Any]:
    result = {"schema_version": 2, "evaluation_profile": profile, "profile_code": 2.0 if profile == "final" else 0.0, "score_fidelity_auc": 0.0,
              "item_count": 0.0, "final_version": 0.0, "num_candidate_versions": 0.0, "private_error": message}
    result.update({key: 0.0 for key in RATIO_KEYS.values()}); result.update({key: 0.0 for key in GATE_KEYS})
    result.update({f"family_{family}": 0.0 for family in FAMILIES})
    return result


def evaluate(app_dir: Path, candidate_timeout: int, total_timeout: int) -> tuple[int, dict[str, Any], dict[str, Any]]:
    started = _now(); private: dict[str, Any] = {}
    if verify_protected_hashes(JUDGE_ROOT / "protected_hashes.txt"): return 1, failure("protected hash failure"), private
    try: profile, suite_id, override = evaluation_profile()
    except Exception as exc: return 1, failure(str(exc)), private
    detail = failure("incomplete", profile); detail.update(suite_id=suite_id, profile_ok=1.0)
    sessions = Sessions(); run_root: Path | None = None
    try:
        items, manifest_sha = load_final_suite()
        detail.update(suite_ok=1.0, manifest_sha256=manifest_sha, item_count=float(len(items)))
        admission = validate_submission(locate_submission(app_dir), override)
        detail.update(final_version=float(admission.get("final_version") or 0), num_candidate_versions=float(admission.get("num_candidate_versions") or 0))
        if not admission["ok"]: raise RuntimeError("source admission failed")
        detail["source_ok"] = 1.0
        gpus = judge_shards.visible_gpus(); fleet = judge_shards.validate_fleet(gpus, min_capability=MIN_COMPUTE_CAPABILITY, min_memory_bytes=MIN_DEVICE_MEMORY_BYTES)
        slices, cpu_mode = judge_shards.cpu_slices(gpus)
        by_id = {x["item_id"]: x for x in items}
        plan = judge_shards.plan_shards([x["item_id"] for x in items], {x["item_id"]: item_cost(x) for x in items}, len(gpus))
        detail.update(judge_gpu_count=float(len(gpus)), judge_fleet=fleet, cpu_slices=slices, cpu_slice_mode=cpu_mode)
        run_root = Path(tempfile.mkdtemp(prefix="kv-eviction-v2-")); run_root.chmod(0o755); account = _agent()
        need = handoff_bytes(items); free = shutil.disk_usage(run_root).free; detail.update(scratch_needed_bytes=float(need), scratch_free_bytes=float(free))
        if free < need * 5 // 4 + (1 << 30): raise RuntimeError(f"Judge scratch {run_root.parent} has {free >> 30} GiB free; the Phase A -> Phase B handoff needs {need >> 30} GiB plus 25% and 1 GiB")
        candidate = run_root / "submission.py"; shutil.copyfile(admission["selected_path"], candidate); candidate.chmod(0o444); os.chown(candidate, account.pw_uid, account.pw_gid)
        source_asset = Path(admission["selected_path"]).with_name("assets.pt"); asset = None
        if source_asset.exists(): asset = run_root / "assets.pt"; shutil.copyfile(source_asset, asset); asset.chmod(0o444); os.chown(asset, account.pw_uid, account.pw_gid)
        deadlines = {"phase_a": started + candidate_timeout, "total": started + total_timeout - 30}
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            # Every shard finishes Phase A before any Phase B starts, so no
            # process start, model load or replay runs while a shard is timing.
            futures = [pool.submit(run_phase_a_shard, i, gpus[i], slices[i], [by_id[x] for x in plan[i]], candidate, asset, run_root, deadlines, sessions) for i in range(len(gpus))]
            try: phase_a = [f.result() for f in futures]
            except BaseException: sessions.kill_all(); raise
            futures = [pool.submit(run_phase_b_shard, i, gpus[i], slices[i], [by_id[x] for x in plan[i]], run_root, deadlines, sessions) for i in range(len(gpus))]
            try: shards = [{**a, **f.result()} for a, f in zip(phase_a, futures)]
            except BaseException: sessions.kill_all(); raise
            detail["hardware_ok"] = 1.0
            rows = [row for shard in shards for row in shard["rows"]]
            expected = {(x["item_id"], float(r)) for x in items for r in RATIOS}; seen = [(row["item_id"], float(row["ratio"])) for row in rows]
            if len(seen) != len(expected) or set(seen) != expected: raise RuntimeError("shard rows do not cover the suite exactly once")
            detail.update(retime_over_cap(pool, rows, plan, gpus, slices, by_id, candidate, asset, run_root, deadlines, sessions))
        order = {x["item_id"]: i for i, x in enumerate(items)}; rows.sort(key=lambda row: (order[row["item_id"]], float(row["ratio"])))
        summary = aggregate(rows)
        for gate, passed in summary["gates"].items(): detail[gate] = 1.0 if passed else 0.0
        detail.update(score_fidelity_auc=summary["primary"], **{RATIO_KEYS[k]: v for k, v in summary["ratio_score"].items()},
            **{f"family_{k}": v for k, v in summary["family_score"].items()},
            latency_ratio_suite=summary["latency_ratio_suite"], latency_ratio_item_max=summary["latency_ratio_item_max"],
            family_fid=summary["family_fid"], ratio_score=summary["ratio_score"], selected_source_sha256=admission["selected_sha256"],
            shards=[{k: v for k, v in s.items() if k != "rows"} for s in shards], wall_seconds=_now() - started, private_error="")
        private.update(task_fid=task_breakdown(rows), private_rows=[{k: row.get(k) for k in PRIVATE_ROW_KEYS} for row in rows])
        ok = all(detail[key] == 1.0 for key in GATE_KEYS)
        if not ok:
            detail.update(score_fidelity_auc=0.0, **{k: 0.0 for k in RATIO_KEYS.values()}, **{f"family_{f}": 0.0 for f in FAMILIES})
        return (0 if ok else 1), detail, private
    except BaseException as exc:
        sessions.kill_all(); detail["private_error"] = f"{type(exc).__name__}: {exc}"; return 1, detail, private
    finally:
        if run_root is not None: shutil.rmtree(run_root, ignore_errors=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}"); tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(tmp, path)


def write_result(log_dir: Path, detail: Mapping[str, Any], private: Mapping[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True); score = float(detail.get("score_fidelity_auc", 0.0)); payload = {**detail, "reward": score}; _write_json(log_dir / RESULT_NAME, payload)
    numeric = {k: float(v) for k, v in payload.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))}; numeric["reward"] = score; _write_json(log_dir / "reward.json", numeric); (log_dir / "reward.txt").write_text(f"{score}\n")
    tmp = log_dir / f".{PRIVATE_NAME}.tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as handle: json.dump({"detail": payload, **private}, handle, sort_keys=True, allow_nan=False)
    os.replace(tmp, log_dir / PRIVATE_NAME)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("--app-dir", type=Path, default=Path("/app")); p.add_argument("--log-dir", type=Path, default=Path("/logs/verifier"))
    p.add_argument("--candidate-timeout", type=int, default=int(os.environ.get("KV_EVICTION_CANDIDATE_TIMEOUT", "5400")))
    p.add_argument("--total-timeout", type=int, default=int(os.environ.get("KV_EVICTION_TOTAL_TIMEOUT", "9000"))); a = p.parse_args(argv)
    private: dict[str, Any] = {}
    try:
        signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(TimeoutError("total verifier timeout")))
        signal.alarm(a.total_timeout); rc, detail, private = evaluate(a.app_dir.resolve(), a.candidate_timeout, a.total_timeout); signal.alarm(0)
    except BaseException as exc: rc, detail = 1, failure(f"{type(exc).__name__}: {exc}")
    write_result(a.log_dir, detail, private); return rc


if __name__ == "__main__": raise SystemExit(main())
