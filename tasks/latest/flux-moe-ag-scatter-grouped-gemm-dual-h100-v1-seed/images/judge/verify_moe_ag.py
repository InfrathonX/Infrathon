#!/usr/bin/env python3
"""Private verifier for the eight-GPU Flux-derived MoE operator."""

from __future__ import annotations
import argparse, hashlib, json, math, os, pwd, shutil, signal, stat, statistics, subprocess, sys, tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_admission import validate_submission

JUDGE_ROOT = Path("/opt/flux_moe_ag_judge")
RUNTIME_ROOT = Path("/opt/flux_moe_ag_worker")
WORKER = JUDGE_ROOT / "candidate_worker.py"
EVALUATION_CONTEXT = Path("/tmp/.sforge-evaluation-context.json")
FINAL_MANIFEST = JUDGE_ROOT / "private/final_cases.json"
ROUTING_FIXTURES = JUDGE_ROOT / "private/routing"
RESULT_NAME = "flux_moe_ag_v2_results.json"
FINAL_SUITE_ID = "final-holdout-v2"
EVALUATION_POLICY = "full-final-shared-v3"
SELECTION_PROTOCOL_ID = "flux-moe-ag-shared-final-v3"
PROFILE_CODES = {"final": 2.0}
ENTRYPOINT = "moe_ag_scatter_gemm_bf16"
WORLD = 8
EXPECTED_CASE_COUNT = 967
EXPECTED_EDGE_CASE_COUNT = 19
EXPECTED_TRIALS = 7
EXPECTED_WARMUPS = 1
METRICS = ("rel_l1", "cossim", "row_rel_l2_p999", "row_rel_l2_max", "max_abs")
LIMITS = ("rel_l1_limit", "row_rel_l2_p999_limit", "row_rel_l2_max_limit")
IDENTITY_FIELDS = ("family", "tier", "group", "W", "TP", "EP", "K", "N", "E", "E_rank", "topk", "M", "layers",
                   "split_counts", "effective_flops", "scored")


def _runtime():
    if str(RUNTIME_ROOT) not in sys.path:
        sys.path.insert(0, str(RUNTIME_ROOT))
    import moe_ag_runtime

    return moe_ag_runtime


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_positive(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise RuntimeError(f"{label} must be finite positive")
    return float(value)


def validate_final_manifest(payload: Mapping[str, Any], rt=None):
    rt = rt or _runtime()
    if (
        set(payload) != {"schema_version", "suite_id", "cases"}
        or payload.get("schema_version") != 2
        or payload.get("suite_id") != FINAL_SUITE_ID
    ):
        raise RuntimeError("manifest identity differs")
    items = payload.get("cases")
    if not isinstance(items, list):
        raise RuntimeError("manifest cases missing")
    cases = [rt.validate_case(item) for item in items]
    ids = [c["case_id"] for c in cases]
    seeds = [c["seed"] for c in cases]
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise RuntimeError("manifest identities are not unique")
    scored = [c for c in cases if c["scored"]]
    if len(scored) != EXPECTED_CASE_COUNT or len(cases) - len(scored) != EXPECTED_EDGE_CASE_COUNT:
        raise RuntimeError("manifest case counts differ")
    if {c["family"] for c in scored} != set(rt.FAMILIES):
        raise RuntimeError("manifest must cover every family")
    return tuple(dict(item) for item in items)


def _strict_json(
    path: Path,
    label: str,
    max_bytes: int = 64 << 20,
    expected_uid: int = 0,
    mode: int | None = None,
):
    st = path.lstat()
    if (
        not stat.S_ISREG(st.st_mode)
        or st.st_nlink != 1
        or st.st_uid != expected_uid
        or stat.S_IMODE(st.st_mode) & 0o077
        or (mode is not None and stat.S_IMODE(st.st_mode) != mode)
        or not 0 < st.st_size <= max_bytes
    ):
        raise RuntimeError(f"{label} file contract failed")
    value = json.loads(
        path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} not object")
    return value


def load_evaluation_profile(path: Path = EVALUATION_CONTEXT, *, expected_uid: int = 0):
    context = _strict_json(path, "evaluation context", expected_uid=expected_uid, mode=0o600)
    round_id = context.get("round_id")
    override = context.get("candidate_version")
    if not isinstance(round_id, str):
        raise RuntimeError("round_id missing")
    if override is not None and (
        isinstance(override, bool)
        or not isinstance(override, int)
        or not 1 <= override <= 999
    ):
        raise RuntimeError("candidate version invalid")
    if round_id.startswith(("auto-", "agent-")):
        if override is not None:
            raise RuntimeError("interactive override forbidden")
        return "final", FINAL_SUITE_ID, round_id, None
    if round_id.startswith(("manual-", "candidate-")):
        return "final", FINAL_SUITE_ID, round_id, override
    raise RuntimeError("unknown profile")


def load_final_manifest(path: Path = FINAL_MANIFEST, *, expected_uid: int = 0):
    payload = _strict_json(path, "final manifest", expected_uid=expected_uid)
    return validate_final_manifest(payload), sha256_file(path)


def locate_submission(app_dir: Path):
    roots = (
        app_dir / "workdir/moe_ag_lab",
        app_dir / "environment/workdir/moe_ag_lab",
        app_dir,
    )
    found = []
    for root in roots:
        if (
            root.is_dir()
            and not root.is_symlink()
            and (root / "candidates").is_dir()
            and (root / "FINAL_VERSION").exists()
            and root.absolute() not in found
        ):
            found.append(root.absolute())
    if len(found) != 1:
        raise RuntimeError(f"expected one submission, found {len(found)}")
    return found[0]


def verify_protected_hashes(manifest: Path):
    expected = {
        "judge:candidate_worker.py": JUDGE_ROOT / "candidate_worker.py",
        "judge:run_judge.py": JUDGE_ROOT / "run_judge.py",
        "judge:source_admission.py": JUDGE_ROOT / "source_admission.py",
        "judge:verify_moe_ag.py": JUDGE_ROOT / "verify_moe_ag.py",
        "worker:moe_ag_runtime.py": RUNTIME_ROOT / "moe_ag_runtime.py",
    }
    problems = []
    seen = set()
    try:
        lines = manifest.read_text().splitlines()
    except OSError:
        return ["protected manifest missing"]
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        digest, label = line.split(None, 1)
        seen.add(label)
        if (
            label not in expected
            or not expected[label].is_file()
            or sha256_file(expected[label]) != digest
        ):
            problems.append(f"protected file failure: {label}")
    if seen != set(expected):
        problems.append("protected records differ")
    return problems


def _agent_account():
    return pwd.getpwnam("agent")


def _drop_to_agent():
    a = _agent_account()
    os.setgroups([])
    os.setgid(a.pw_gid)
    os.setuid(a.pw_uid)
    os.umask(0o077)


def _signal_session(sid, sig):
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if os.getsid(int(entry.name)) == sid:
                    os.killpg(os.getpgid(int(entry.name)), sig)
            except OSError:
                pass


def stage_routing_fixtures(run_root: Path, cases: Sequence[Mapping[str, Any]], source: Path | None = None):
    """Copy the routing fixtures the cases pin out of the root-only private layer into a
    root-owned, world-readable directory of the run, checking each digest first."""
    pinned: dict[str, str] = {}
    for case in cases:
        routing = case.get("routing")
        if isinstance(routing, Mapping) and routing.get("kind") == "captured":
            name, digest = str(routing["fixture"]), str(routing["sha256"])
            if pinned.setdefault(name, digest) != digest:
                raise RuntimeError(f"manifest pins two digests for routing fixture {name}")
    if not pinned:
        return None
    source = ROUTING_FIXTURES if source is None else source
    target = run_root / "routing"
    target.mkdir(mode=0o755)
    for name, digest in sorted(pinned.items()):
        data = (source / f"{name}.npy").read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"Judge routing fixture {name} differs from the manifest")
        fd = os.open(target / f"{name}.npy", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
    return target


def run_worker(
    candidate: Path,
    run_root: Path,
    cases: Sequence[Mapping[str, Any]],
    suite_id: str,
    *,
    trials: int,
    warmups: int,
    timeout: float,
):
    a = _agent_account()
    result_dir = run_root / "result"
    cache = run_root / "cache"
    result_dir.mkdir(mode=0o700)
    cache.mkdir(mode=0o700)
    for path in (result_dir, cache):
        os.chown(path, a.pw_uid, a.pw_gid)
    output = result_dir / "result.json"
    env = {
        k: os.environ[k]
        for k in (
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDA_VISIBLE_DEVICES",
            "LD_LIBRARY_PATH",
            "NVIDIA_VISIBLE_DEVICES",
            "PATH",
        )
        if k in os.environ
    }
    env.update(
        HOME="/home/agent",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        TMPDIR=str(cache),
        TRITON_CACHE_DIR=str(cache / "triton"),
        CUDA_VISIBLE_DEVICES=",".join(str(x) for x in range(WORLD)),
    )
    (cache / "triton").mkdir()
    os.chown(cache / "triton", a.pw_uid, a.pw_gid)
    cmd = [
        sys.executable,
        str(WORKER),
        "--candidate", str(candidate),
        "--output", str(output),
        "--runtime-root", str(RUNTIME_ROOT),
        "--cache", str(cache),
        "--trials", str(trials),
        "--warmups", str(warmups),
        "--timeout", str(timeout),
        "--expected-count", str(len(cases)),
    ]
    fixtures = stage_routing_fixtures(run_root, cases)
    if fixtures is not None:
        cmd += ["--fixtures", str(fixtures)]
    with (run_root / "worker.log").open("wb") as log:
        process = subprocess.Popen(
            cmd,
            cwd=run_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_drop_to_agent,
        )
        try:
            process.communicate(
                json.dumps({"schema_version": 2, "suite_id": suite_id, "cases": list(cases)}).encode(),
                timeout=timeout + 30,
            )
        except subprocess.TimeoutExpired:
            _signal_session(process.pid, signal.SIGKILL)
            raise RuntimeError("worker timed out")
    if process.returncode:
        raise RuntimeError("worker failed")
    return _strict_json(output, "worker result", expected_uid=a.pw_uid)


def _metric(row, name, minimum=0.0):
    value = row.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise RuntimeError(f"bad metric {name}")
    return float(value)


MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 64 * 1024**3


def _check_row(row: Mapping[str, Any], case: Mapping[str, Any], *, expected_trials: int, rt) -> float:
    for key in IDENTITY_FIELDS:
        if row.get(key) != case[key]:
            raise RuntimeError(f"case {case['case_id']} field {key} differs")
    trials = expected_trials if case["scored"] else 1
    times = row.get("trial_seconds")
    if not isinstance(times, list) or len(times) != trials:
        raise RuntimeError(f"case {case['case_id']} trials differ")
    median = statistics.median(finite_positive(x, "trial seconds") for x in times)
    computed = case["effective_flops"] / median / 1e12
    value = finite_positive(row.get("tflops"), "case TFLOPS")
    if not math.isclose(value, computed, rel_tol=1e-9):
        raise RuntimeError(f"case {case['case_id']} trial accounting differs")
    records = row.get("rank_records")
    if not isinstance(records, list) or len(records) != case["W"] or {x.get("rank") for x in records} != set(range(case["W"])):
        raise RuntimeError(f"case {case['case_id']} rank records differ")
    golden_ran = "rel_l1" in row
    if golden_ran:
        for name in METRICS + LIMITS:
            _metric(row, name)
        if row["rel_l1"] > row["rel_l1_limit"] or row["row_rel_l2_p999"] > row["row_rel_l2_p999_limit"] \
                or row["row_rel_l2_max"] > row["row_rel_l2_max_limit"] or row["cossim"] < rt.COS_FLOOR:
            raise RuntimeError(f"case {case['case_id']} accuracy gate differs")
    elif any(max(r.get("rows_per_trial") or [0]) > 0 for r in records) and case["layers"] == 1:
        raise RuntimeError(f"case {case['case_id']} golden check missing")
    if _metric(row, "freivalds_rel_l2") > rt.FREIVALDS_REL_L2 or _metric(row, "freivalds_worst_row") > rt.FREIVALDS_ROW_FRACTION:
        raise RuntimeError(f"case {case['case_id']} projection gate differs")
    hidden = row.get("hidden_comm")
    if hidden is not None:
        if isinstance(hidden, bool) or not isinstance(hidden, (int, float)) or not 0.0 <= float(hidden) <= 1.0:
            raise RuntimeError(f"case {case['case_id']} hidden_comm differs")
    return value


def validate_runtime_result(
    result: Mapping[str, Any],
    expected_cases: Sequence[Mapping[str, Any]],
    *,
    expected_trials: int,
    expected_warmups: int | None = None,
):
    rt = _runtime()
    if (
        result.get("schema_version") != 2
        or result.get("entrypoints") != [ENTRYPOINT]
        or result.get("trials_per_case") != expected_trials
    ):
        raise RuntimeError("runtime header differs")
    if expected_warmups is not None and result.get("warmups") != expected_warmups:
        raise RuntimeError("warmups differ")
    if any(result.get(k) is not True for k in ("correctness_ok", "edge_ok", "fresh_input_ok", "rank_agreement_ok")):
        raise RuntimeError("runtime gate failed")
    hardware = result.get("hardware", {})
    devices = hardware.get("devices", [])
    if (
        hardware.get("count") != WORLD
        or len(devices) != WORLD
        or any(
            not isinstance(x.get("capability"), list)
            or len(x["capability"]) != 2
            or tuple(int(v) for v in x["capability"]) < MIN_COMPUTE_CAPABILITY
            or isinstance(x.get("total_memory_bytes"), bool)
            or not isinstance(x.get("total_memory_bytes"), int)
            or x["total_memory_bytes"] < MIN_DEVICE_MEMORY_BYTES
            or x.get("peer_access_to_all") is not True
            for x in devices
        )
    ):
        raise RuntimeError("hardware differs")
    expected = [rt.validate_case(x) for x in expected_cases]
    scored = [c for c in expected if c["scored"]]
    edges = [c for c in expected if not c["scored"]]
    rows = result.get("cases")
    edge_rows = result.get("edge_cases")
    if not isinstance(rows, list) or len(rows) != len(scored):
        raise RuntimeError("case set differs")
    if not isinstance(edge_rows, list) or len(edge_rows) != len(edges):
        raise RuntimeError("edge set differs")
    by_id = {x.get("case_id"): x for x in rows}
    edge_by_id = {x.get("case_id"): x for x in edge_rows}
    for case in edges:
        row = edge_by_id.get(case["case_id"])
        if not isinstance(row, Mapping):
            raise RuntimeError("edge case missing")
        _check_row(row, case, expected_trials=expected_trials, rt=rt)
    checked = []
    for case in scored:
        row = by_id.get(case["case_id"])
        if not isinstance(row, Mapping):
            raise RuntimeError("case missing")
        _check_row(row, case, expected_trials=expected_trials, rt=rt)
        checked.append(row)
    aggregate = rt.aggregate_case_results(checked)
    families = aggregate["family_tflops"]
    if set(families) != set(rt.FAMILIES):
        raise RuntimeError("family aggregate differs")
    reported = result.get("family_tflops")
    if not isinstance(reported, Mapping) or set(reported) != set(families):
        raise RuntimeError("family aggregate differs")
    for family, value in families.items():
        if not math.isclose(finite_positive(reported.get(family), "family TFLOPS"), value, rel_tol=1e-9):
            raise RuntimeError("family aggregate differs")
    score = aggregate["score_tflops"]
    if not math.isclose(score, float(result.get("score_tflops", 0)), rel_tol=1e-9):
        raise RuntimeError("aggregate differs")
    hidden = aggregate["hidden_comm"]
    if hidden is not None and not math.isclose(hidden, float(result.get("hidden_comm") or 0.0), rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError("suite hidden communication aggregate differs")
    return score, families, hidden


def _atomic_json(path, payload):
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


FAMILY_KEYS = ("tp8_tflops", "tp4_tflops", "tp2_tflops", "ep_tflops", "decode_tflops", "sequence_tflops")


def failure_detail(message, *, profile=None, suite_id=None):
    d = {
        "schema_version": 2,
        "profile_ok": 0.0,
        "suite_ok": 0.0,
        "source_ok": 0.0,
        "worker_ok": 0.0,
        "hardware_ok": 0.0,
        "correctness_ok": 0.0,
        "edge_ok": 0.0,
        "fresh_input_ok": 0.0,
        "rank_agreement_ok": 0.0,
        "profile_code": PROFILE_CODES.get(profile or "", 0.0),
        "score_tflops": 0.0,
        "hidden_comm": 0.0,
        "final_version": 0.0,
        "num_candidate_versions": 0.0,
        "private_error": message,
        **{key: 0.0 for key in FAMILY_KEYS},
    }
    if profile:
        d["evaluation_profile"] = profile
    if suite_id:
        d["suite_id"] = suite_id
    return d


def evaluate(args):
    if verify_protected_hashes(JUDGE_ROOT / "protected_hashes.txt"):
        return 1, failure_detail("protected hash failure")
    try:
        profile, suite, round_id, override = load_evaluation_profile()
    except Exception as exc:
        return 1, failure_detail(str(exc))
    base = failure_detail("incomplete", profile=profile, suite_id=suite)
    base["evaluation_policy"] = EVALUATION_POLICY
    base["selection_protocol_id"] = SELECTION_PROTOCOL_ID
    base["profile_ok"] = 1.0
    base["round_id"] = round_id
    run_root = None
    try:
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
        run_root = Path(tempfile.mkdtemp(prefix="flux-moe-ag-v2-"))
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
        base["worker_ok"] = 1.0
        base["hardware_ok"] = 1.0
        for key in ("correctness_ok", "edge_ok", "fresh_input_ok", "rank_agreement_ok"):
            base[key] = 1.0 if result.get(key) is True else 0.0
        score, families, hidden = validate_runtime_result(
            result, cases, expected_trials=args.trials, expected_warmups=args.warmups
        )
        base.update(
            score_tflops=score,
            hidden_comm=float(hidden) if hidden is not None else 0.0,
            selected_source_sha256=admission["selected_sha256"],
            private_runtime_result=result,
            private_error="",
            **{f"{family}_tflops": float(value) for family, value in families.items()},
        )
        return 0, base
    except Exception as exc:
        base["private_error"] = f"{type(exc).__name__}: {exc}"
        return 1, base
    finally:
        if run_root is not None:
            shutil.rmtree(run_root, ignore_errors=True)


def write_result(log_dir, detail):
    log_dir.mkdir(parents=True, exist_ok=True)
    score = float(detail.get("score_tflops", 0))
    payload = dict(detail)
    payload["reward"] = score
    _atomic_json(log_dir / RESULT_NAME, payload)
    numeric = {
        k: float(v)
        for k, v in payload.items()
        if isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(float(v))
    }
    numeric["reward"] = score
    _atomic_json(log_dir / "reward.json", numeric)
    (log_dir / "reward.txt").write_text(f"{score}\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--app-dir", type=Path, default=Path(os.environ.get("APP_DIR", "/app")))
    p.add_argument("--log-dir", type=Path, default=Path(os.environ.get("LOG_DIR", "/logs/verifier")))
    p.add_argument("--trials", type=int, default=int(os.environ.get("FLUX_MOEAG_VERIFY_TRIALS", str(EXPECTED_TRIALS))))
    p.add_argument("--warmups", type=int, default=int(os.environ.get("FLUX_MOEAG_VERIFY_WARMUPS", str(EXPECTED_WARMUPS))))
    p.add_argument("--candidate-timeout", type=int, default=int(os.environ.get("FLUX_MOEAG_CANDIDATE_TIMEOUT", "5400")))
    p.add_argument("--total-timeout", type=int, default=int(os.environ.get("FLUX_MOEAG_TOTAL_TIMEOUT", "5700")))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        rc, detail = evaluate(args)
    except Exception as exc:
        rc, detail = 1, failure_detail(str(exc))
    write_result(args.log_dir, detail)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
