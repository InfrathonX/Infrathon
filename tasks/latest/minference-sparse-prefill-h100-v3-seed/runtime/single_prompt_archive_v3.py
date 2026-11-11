"""Author prototype: authenticate exactly one five-layer dense QKV bundle.

No full 180-file archive, model execution or GPU execution happens here.
The existing retained-FD reader still verifies every byte and tensor header.
This prototype is not a production qualification or delivery protocol.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import struct
import re

PROTOCOL = "minference-single-real-prompt-qkv-v1"
LAYERS = (1, 2, 4, 15, 27)
TARGET_LAYERS = (1, 4, 15, 27)
MAX_SCRATCH_BYTES = 8 * 1024**3
TASKS = ("niah_single_2", "niah_multikey_1", "vt", "fwe")
BUDGETS = (32768, 65536, 131072)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def canonical_sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def stable_stat(info):
    return [info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
            info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def directory_identity(info):
    return [info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid]


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_plan(plan):
    """Validate the existing 48-group workload without requiring stored QKV."""
    if plan.get("schema_version") != 3 or plan.get("protocol") not in (PROTOCOL, "minference-real-ruler-prefill-v3"):
        raise ValueError("unexpected fixed performance plan")
    if plan.get("partition") not in ("public", "final"):
        raise ValueError("performance plan partition differs")
    fixtures, groups = plan.get("fixtures"), plan.get("case_groups")
    if not isinstance(fixtures, dict) or len(fixtures) != 180 or not isinstance(groups, list) or len(groups) != 48:
        raise ValueError("plan requires 180 identities and 48 groups, not 180 files")
    prompts, grid, timed = {}, set(), set()
    for fid, f in fixtures.items():
        if not isinstance(fid, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,240}", fid) or not isinstance(f.get("prompt_id"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", f["prompt_id"]):
            raise ValueError("unsafe prompt/fixture identifier")
        if fid != f.get("prompt_id", "") + "-L" + str(f.get("layer")):
            raise ValueError("fixture identifier differs")
        n, budget = f.get("actual_n"), f.get("context_budget")
        if type(n) is not int or not 2048 <= n < budget or budget not in BUDGETS:
            raise ValueError("real input length differs")
        if f.get("task") not in TASKS or type(f.get("layer")) is not int or f["layer"] not in LAYERS or not is_sha(f.get("input_token_sha256")):
            raise ValueError("input/layer identity differs")
        expected_trajectory = "dense-flash-attention-2" if plan["protocol"] == PROTOCOL else "dense-sdpa"
        if f.get("file") != fid + ".safetensors" or f.get("trajectory") != expected_trajectory:
            raise ValueError("fixture path/trajectory differs")
        identity = {k: f[k] for k in ("prompt_id", "task", "context_budget", "actual_n", "input_token_sha256")}
        pid = f["prompt_id"]
        previous = prompts.setdefault(pid, identity)
        if previous != identity:
            raise ValueError("prompt identity differs across layers")
    if len(prompts) != 36 or len({p["input_token_sha256"] for p in prompts.values()}) != 36:
        raise ValueError("36 independent real prompts are required")
    group_ids = set()
    for g in groups:
        identity = (g.get("task"), g.get("context_budget"), g.get("layer"))
        if identity in grid or g.get("case_id") in group_ids:
            raise ValueError("duplicate group")
        grid.add(identity); group_ids.add(g.get("case_id"))
        if len(g.get("warmups", [])) != 3 or len(g.get("trials", [])) != 3:
            raise ValueError("three real paired trials required")
        token_shas = set()
        for wid, tid in zip(g["warmups"], g["trials"]):
            w, t = fixtures[wid], fixtures[tid]
            if tid in timed or (t["task"], t["context_budget"], t["layer"]) != identity:
                raise ValueError("timed layer identity/reuse differs")
            timed.add(tid)
            if w["layer"] != 2 or any(w[k] != t[k] for k in prompts[t["prompt_id"]]):
                raise ValueError("same-shape distinct-layer warm-up differs")
            token_shas.add(t["input_token_sha256"])
        if len(token_shas) != 3:
            raise ValueError("group inputs must be independent")
    if grid != {(t, b, l) for t in TASKS for b in BUDGETS for l in TARGET_LAYERS}:
        raise ValueError("workload grid differs")
    if not isinstance(plan.get("bindings"), dict) or not plan["bindings"]:
        raise ValueError("missing model/input/source bindings")
    return prompts


def on_demand_plan(previous_plan, bindings):
    """Reuse workload/input identities, explicitly start a new FA2 trajectory."""
    validate_plan(previous_plan)
    value = json.loads(json.dumps(previous_plan))
    value.update(protocol=PROTOCOL, bindings=bindings, capture_state="on-demand-no-persisted-qkv-tree")
    for fixture in value["fixtures"].values():
        fixture.pop("size", None); fixture.pop("sha256", None)
        fixture["trajectory"] = "dense-flash-attention-2"
    value["historical_trajectory"] = "dense-sdpa; historical file SHAs are not reused"
    validate_plan(value)
    return value


def prompt_specs(plan, prompt_id):
    prompts = validate_plan(plan)
    if prompt_id not in prompts:
        raise ValueError("unknown prompt")
    return {fid: {k: f[k] for k in ("prompt_id", "task", "context_budget", "layer", "actual_n",
              "input_token_sha256", "file", "trajectory")} for fid, f in plan["fixtures"].items()
              if f["prompt_id"] == prompt_id}


def validate_bundle(bundle, plan):
    pid = bundle.get("prompt_id")
    specs = prompt_specs(plan, pid)
    if bundle.get("schema_version") != 1 or bundle.get("protocol") != PROTOCOL:
        raise ValueError("bundle protocol differs")
    if bundle.get("plan_sha256") != canonical_sha(plan) or bundle.get("partition") != plan["partition"]:
        raise ValueError("bundle plan/partition binding differs")
    if bundle.get("bindings") != plan["bindings"]:
        raise ValueError("bundle scientific bindings differ")
    records = bundle.get("fixtures")
    if not isinstance(records, dict) or set(records) != set(specs):
        raise ValueError("bundle must contain exactly the five selected layers")
    total = 0
    for fid, spec in specs.items():
        f = records[fid]
        if any(f.get(k) != v for k, v in spec.items()):
            raise ValueError("bundle input identity differs")
        if not is_sha(f.get("sha256")) or type(f.get("size")) is not int or f["size"] <= 0:
            raise ValueError("bundle file identity missing")
        total += f["size"]
    if total > MAX_SCRATCH_BYTES or bundle.get("scratch_bytes") != total:
        raise ValueError("single-prompt scratch exceeds 8 GiB or differs")
    return bundle


def open_bundle(root, bundle, plan, archive):
    """Root calls before importing candidate; retain five authenticated FDs."""
    validate_bundle(bundle, plan)
    root = Path(root)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.geteuid():
        raise ValueError("scratch root must be a private 0700 directory")
    if bundle.get("scratch_directory_identity") != directory_identity(info):
        raise ValueError("scratch directory inode/ownership differs")
    if set(p.name for p in root.iterdir()) != {"bundle.json", *(f["file"] for f in bundle["fixtures"].values())}:
        raise ValueError("scratch directory contains unexpected files")
    handles, identities = {}, {}
    try:
        for fid, entry in sorted(bundle["fixtures"].items()):
            path = root / entry["file"]
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o400 or before.st_uid != info.st_uid:
                raise ValueError("scratch fixture must be a single-link 0400 regular file")
            handle = open_single_handle(root, fid, entry, bundle, archive)
            handles[fid] = handle
            identity = stable_stat(os.fstat(handle.fd))
            if identity != stable_stat(before) or identity != stable_stat(path.lstat()):
                raise ValueError("fixture changed during authentication")
            identities[fid] = identity
        return handles, identities
    except BaseException:
        for handle in handles.values():
            handle.close()
        raise


def open_single_handle(root, fid, entry, bundle, archive):
    """Independent small-bundle header admission; no weakening of old reader."""
    fd = os.open(Path(root) / entry["file"], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != entry["size"] or archive.digest_fd(fd, info.st_size) != entry["sha256"]:
            raise ValueError("fixture file digest/size differs")
        raw = os.pread(fd, 8, 0)
        if len(raw) != 8:
            raise ValueError("missing tensor header")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= 65536 or 8 + length >= info.st_size:
            raise ValueError("tensor header length differs")
        header = archive.strict_json(os.pread(fd, length, 8))
        if set(header) != {"q", "k", "v", "__metadata__"}:
            raise ValueError("tensor names differ")
        expected = {"case_id": fid, "prompt_id": entry["prompt_id"], "layer": str(entry["layer"]),
            "actual_n": str(entry["actual_n"]), "input_token_sha256": entry["input_token_sha256"],
            "task": entry["task"], "context_budget": str(entry["context_budget"]),
            "partition": bundle["partition"], "trajectory": entry["trajectory"], "protocol": PROTOCOL}
        if header["__metadata__"] != expected:
            raise ValueError("tensor header source/input/trajectory identity differs")
        spans = []
        for name, heads in (("q", 32), ("k", 8), ("v", 8)):
            t = header[name]; offsets = t.get("data_offsets")
            if t.get("dtype") != "BF16" or t.get("shape") != [1, heads, entry["actual_n"], 128]:
                raise ValueError("tensor shape/dtype differs")
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(i) is not int for i in offsets):
                raise ValueError("tensor offsets differ")
            start, end = offsets
            if not 0 <= start < end <= info.st_size - 8 - length or end - start != heads * entry["actual_n"] * 128 * 2:
                raise ValueError("tensor span size differs")
            spans.append((start, end))
        spans.sort()
        if spans[0][0] != 0 or spans[-1][1] != info.st_size - 8 - length or any(a[1] != b[0] for a,b in zip(spans, spans[1:])):
            raise ValueError("tensor spans overlap or leave unused data")
        return archive.ActivationHandle(fid, entry, fd, header, 8 + length)
    except BaseException:
        os.close(fd); raise


def read_checked(torch, handle, identity, archive, device="cuda"):
    if stable_stat(os.fstat(handle.fd)) != identity:
        raise ValueError("retained fixture identity changed")
    tensors = archive.read_tensors(torch, handle, device)
    if stable_stat(os.fstat(handle.fd)) != identity:
        raise ValueError("retained fixture changed while reading")
    return tensors


def aggregate_prompt_reports(plan, reports, *, candidate_sha256, numerical_policy_sha256, policy):
    """Recompute 48x3 results; a partial stream never becomes a task score."""
    expected = validate_plan(plan)
    if not is_sha(candidate_sha256) or not is_sha(numerical_policy_sha256):
        raise ValueError("candidate/policy hash missing")
    actual, all_trials = set(), {}
    boundary_checks=None
    boundary_prompt=plan["case_groups"][0]["trials"][0].rsplit("-L",1)[0]
    for report in reports:
        pid = report.get("prompt_id")
        if pid not in expected or pid in actual or report.get("plan_sha256") != canonical_sha(plan):
            raise ValueError("missing/duplicate/foreign prompt report")
        actual.add(pid)
        if pid==boundary_prompt:
            from kernel_fidelity_v3 import validate_boundary_checks
            boundary_checks=validate_boundary_checks(report.get("boundary_checks"),policy["limits"]["default"])
        elif report.get("boundary_checks") is not None:
            raise ValueError("foreign boundary report")
        if report.get("candidate_sha256") != candidate_sha256 or report.get("numerical_policy_sha256") != numerical_policy_sha256:
            raise ValueError("candidate/policy identity differs")
        trials = report.get("trials")
        if not isinstance(trials, list) or {t.get("fixture_id") for t in trials} != {pid+f"-L{l}" for l in TARGET_LAYERS} or len(trials) != 4:
            raise ValueError("each prompt needs four measured target layers")
        for trial in trials:
            fid = trial["fixture_id"]
            spec = plan["fixtures"][fid]
            if any(trial.get(k) != spec[k] for k in ("prompt_id", "actual_n", "input_token_sha256")):
                raise ValueError("trial input identity differs")
            times = [trial.get(k) for k in ("host_seconds", "event_seconds", "seconds")]
            if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in times) or times[2] != max(times[:2]):
                raise ValueError("timing must use maximum host and event time")
            if any(trial.get(k) is not True for k in ("numeric_passed", "full_write_ok", "input_immutable_ok")):
                raise ValueError("kernel numerical/integrity check failed")
            limits = policy["limits"].get(f"{spec['task']}:{spec['context_budget']}:{spec['layer']}", policy["limits"]["default"])
            if trial.get("limits") != limits:
                raise ValueError("trial numerical limits differ from trusted policy")
            for side, count, prefix in (("uniform_stats", 512, ""), ("tail_stats", 64, "tail_")):
                s = trial.get(side, {})
                if s.get("finite") is not True or type(s.get("rows")) is not int or s["rows"] != 32 * min(spec["actual_n"], count):
                    raise ValueError("reference probe row count/finite state differs")
                if any(type(s.get(k)) not in (float, int) or not math.isfinite(s[k]) or s[k] < 0 for k in ("mean", "head_max_mean", "bad", "tail", "max")):
                    raise ValueError("malformed reference error statistics")
            from kernel_fidelity_v3 import passes
            if not passes(trial["uniform_stats"],trial["tail_stats"],limits):
                raise ValueError("trusted kernel fidelity budget failed")
            all_trials[fid] = trial
    if actual != set(expected):
        raise ValueError("full stream requires all 36 prompts")
    cases = []
    for group in plan["case_groups"]:
        rows = [all_trials[fid] for fid in group["trials"]]
        # Same dense causal equivalent FLOPs as the production runtime.
        tflops = [2 * 32 * 128 * r["actual_n"] * (r["actual_n"] + 1) / r["seconds"] / 1e12 for r in rows]
        cases.append({**{k: group[k] for k in ("case_id", "task", "context_budget", "layer")},
            "trial_ids": group["trials"], "trial_ns": [r["actual_n"] for r in rows],
            "trial_seconds": [r["seconds"] for r in rows],
            "trial_host_seconds": [r["host_seconds"] for r in rows],
            "trial_event_seconds": [r["event_seconds"] for r in rows], "trials": rows,
            "median_tflops": statistics.median(tflops), "numeric_passed": True})
    score = math.exp(statistics.mean(math.log(c["median_tflops"]) for c in cases))
    return {"schema_version": 1, "protocol": PROTOCOL, "plan_sha256": canonical_sha(plan),
        "candidate_sha256": candidate_sha256, "numerical_policy_sha256": numerical_policy_sha256,
        "cases": cases, "boundary_checks":boundary_checks, "raw_primary_tflops": score, "ok": True,
        "qualified": False, "scope": "kernel-only measured result; standard Judge authenticates and recomputes score"}


def validate_purpose(purpose, partition, uid):
    if purpose == "private-judge":
        if partition != "final" or uid != 0:
            raise ValueError("private Judge requires final inputs and root preparation")
    elif purpose == "public-diagnostic":
        if partition != "public" or uid != 1000:
            raise ValueError("public diagnostic requires public inputs and agent preparation")
    else:
        raise ValueError("unknown preparation purpose")
    return purpose
