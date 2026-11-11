#!/usr/bin/env python3
"""The tools the public suites share with the Judge. The Work image holds this file instead of run_judge.py."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PY = sys.executable


def write_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=1, allow_nan=False))


def run_logged(cmd, log: Path, *, cwd=None, env=None, timeout=None, preexec=None):
    t0 = time.monotonic()
    with open(log, "w") as f:
        try:
            p = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                               preexec_fn=preexec)
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    return code, time.monotonic() - t0


def run_tool(args, log: Path, out: Path):
    code, _ = run_logged([PY, *[str(a) for a in args]], log, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    return json.loads(out.read_text()) if out.is_file() else None


def run_tools_parallel(jobs):
    """run_tool for every (args, log, out, extra_env) at the same time (one per GPU); their outputs in job order."""
    running = []
    for args, log, out, extra in jobs:
        f = open(log, "w")
        p = subprocess.Popen([PY, *[str(a) for a in args]], stdout=f, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **extra))
        running.append((p, f, out))
    for p, f, _ in running:
        p.wait()
        f.close()
    return [json.loads(out.read_text()) if out.is_file() else None for _, _, out in running]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quality_repeats(student: Path, prompts: Path, seeds: list[int], samples: int, Q: dict, out_dir: Path) -> list[dict]:
    """The Judge's held-out KL of `student` for each case seed: the seeds are dealt round-robin over the visible GPUs,
    one quality_eval.py process per GPU (it loads the encoders and DiT copies once and runs its seeds one after
    another; a seed's result does not depend on the process or GPU it ran on). {case_seed, per_route, kl_pooled} per
    seed in `seeds` order; an evaluator that does not return every seed is a Judge error."""
    devices = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",") if x.strip()] or ["0"]
    groups = [seeds[k::len(devices)] for k in range(min(len(devices), len(seeds)))]
    jobs = []
    for i, group in enumerate(groups):
        out = out_dir / f"quality_{i}.json"
        jobs.append(([HERE / "quality_eval.py", "--prompts", prompts, "--student", student, "--samples", samples,
                      "--case-seeds", *group, "--out", out, "--model", Q["model"], "--teachers", Q["teachers_dir"]],
                     out_dir / f"quality_{i}.log", out, {"CUDA_VISIBLE_DEVICES": devices[i]}))
    got = {r["case_seed"]: r for res in run_tools_parallel(jobs) for r in (res or {}).get("repeats", [])}
    if sorted(got) != sorted(seeds):
        raise RuntimeError(f"quality evaluator returned case seeds {sorted(got)}, expected {sorted(seeds)}")
    return [got[c] for c in seeds]


def positive_finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def route_gain(repeats: list[dict], base: list[dict], route: str):
    """Mean over the case seeds of ln(KL_base / KL_final) on one route (reported, not scored); None when a KL of the
    final student is not positive and finite."""
    pairs = [(b["per_route"][route]["kl"], r["per_route"][route]["kl"]) for r, b in zip(repeats, base)]
    if not all(positive_finite(x) for pair in pairs for x in pair):
        return None
    return sum(math.log(b / f) for b, f in pairs) / len(pairs)


def paired_with_base(repeats: list[dict], base: list[dict]) -> None:
    """Repeat c must be measured like the frozen base's repeat c: same case seed, same routes, same pair counts."""
    for r, b in zip(repeats, base, strict=True):
        counts = {k: v["n"] for k, v in r["per_route"].items()}
        if r["case_seed"] != b["case_seed"] or counts != {k: v["n"] for k, v in b["per_route"].items()}:
            raise RuntimeError(f"case seed {r['case_seed']}: routes / pair counts {counts} differ from the frozen "
                               f"base's (case seed {b['case_seed']})")


def task_metric(student: Path, prompts: Path, TQ: dict, Q: dict, out_dir: Path) -> dict:
    """The Judge's task metric of `student`: the held-out prompts are dealt over the visible GPUs by index (one
    task_quality_eval.py process per GPU), the shard outputs merged; {route: mean} plus the merged result. A shard
    that returns nothing is a Judge error."""
    from task_quality_eval import merge  # stdlib only

    devices = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",") if x.strip()] or ["0"]
    jobs = []
    for i, dev in enumerate(devices):
        out = out_dir / f"task_{i}.json"
        jobs.append(([HERE / "task_quality_eval.py", "eval", "--model", Q["model"],
                      "--pickscore-model", Q["scorers"]["pickscore_model"],
                      "--pickscore-processor", Q["scorers"]["pickscore_processor"],
                      "--prompts", prompts, "--lora", student, "--cfg", TQ["cfg"], "--case-seed", TQ["case_seed"],
                      "--steps", TQ["steps"], "--res", TQ["res"], "--batch", TQ["batch"],
                      "--ocr-workers", TQ["ocr_workers_per_gpu"], "--shard", i, "--num-shards", len(devices),
                      "--out", out], out_dir / f"task_{i}.log", out, {"CUDA_VISIBLE_DEVICES": dev}))
    parts = run_tools_parallel(jobs)
    if any(p is None for p in parts):
        raise RuntimeError(f"task metric shard(s) {[i for i, p in enumerate(parts) if p is None]} returned nothing")
    merged = merge([str(out) for _, _, out, _ in jobs])
    res = merged["results"][0]
    write_json(out_dir / "task_metric.json", {k: v for k, v in merged.items() if k != "results"} |
               {"result": {k: v for k, v in res.items() if k != "items"}})
    counts = {r: v["n"] for r, v in res["per_route"].items()}
    if counts != TQ["route_counts"]:
        raise RuntimeError(f"task metric route counts {counts} differ from the frozen protocol's {TQ['route_counts']}")
    return {r: v["mean"] for r, v in res["per_route"].items()}, res


def final_lora_verdict(final: Path, dump: Path) -> dict:
    """The LoRA the quality evaluation reads is the student after the run's last optimizer step, bit for bit (the
    last record's lora_post); with quality in the score, training on after the last step must not pay. It reads
    candidate-written files: anything that goes wrong reading them fails the check (it is not a Judge error)."""
    from adamw_check import final_lora_issues  # torch: only after the last candidate launch (see the imports)

    try:
        issues = final_lora_issues(str(final), str(dump))
    except Exception as e:  # noqa: BLE001
        issues = [f"the final LoRA or the last step record could not be checked ({type(e).__name__}: {e})"[:300]]
    return {"passed": not issues, "issues": issues}


def relabel(rows: list[dict], mapping: dict[str, str]) -> list[dict]:
    out = []
    for r in rows:
        r = json.loads(json.dumps(r))
        r["metadata"]["data_source"] = mapping[r["metadata"]["data_source"]]
        out.append(r)
    return out


def start_replay_server(workload: Path, log: Path, port: int):
    p = subprocess.Popen([PY, str(HERE / "replay_server.py"), "--workload", str(workload), "--port", str(port), "--log", str(log)],
                         stdout=open(log.with_suffix(".out"), "w"), stderr=subprocess.STDOUT,
                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    for _ in range(120):
        try:
            import urllib.request

            urllib.request.urlopen(f"http://127.0.0.1:{port}/replay/status", timeout=1).read()
            return p
        except Exception:
            time.sleep(0.5)
    p.kill()
    raise RuntimeError("replay server did not come up")


def probe_failures(rep: dict, TH: dict) -> list[str]:
    """Probe verdict: structural checks, G3 transition-mean errors, G4 gradient direction AND size (a gradient that
    drops or detaches part of the pairs keeps its direction but not its size), update moved."""
    if not rep:
        return ["no probe report"]
    fails = [k for k, ok in rep.get("checks", {}).items() if not ok]
    g3, g4 = rep.get("G3", {}), rep.get("G4", {})
    for key in ("e_T_median", "e_T_p90", "e_S_median", "e_S_p90"):
        if not g3.get(key, 9) <= TH[key]:
            fails.append(key)
    if not g4.get("cos", 0) >= TH["grad_cos"]:
        fails.append("grad_cos")
    if not g4.get("rel_l2", 9) <= TH["grad_rel_l2"]:
        fails.append("grad_rel_l2")
    if not g4.get("sign_agree_top_half", 0) >= TH["sign_agree"]:
        fails.append("sign_agree")
    if not g4.get("update_nonzero"):
        fails.append("update_nonzero")
    return fails


def framework_verdict(fc) -> dict:
    """framework_check.py: the session ran through miles_diffusion's driver, RolloutManager rollout path and FSDP
    training actor (engine requests and weight updates inside those calls)."""
    if not fc:
        return {"passed": False, "error": "no framework report"}
    return {"passed": not fc["violations"], **fc}


def publication_verdict(pc) -> dict:
    """publication_check.py: engines hold exactly the dumped student after every round (fp16-rounded fingerprint
    match), and in replay sessions `published` comes only after every engine applied it."""
    if not pc:
        return {"passed": False, "error": "no publication report"}
    return {"passed": not pc["violations"] and pc.get("fingerprint_kind") == "fingerprint16",
            **{k: (v[:10] if k == "violations" else v) for k, v in pc.items()}}


def adamw_verdict(ad, TH, expect_steps: int) -> dict:
    """adamw_check.py, run with --student-init and --optimizer (the contract's): a report without either check fails."""
    if not ad:
        return {"passed": False, "error": "no AdamW report"}
    ok = (not ad["violations"] and ad["steps"] == expect_steps and ad.get("init_matches_student_init", False)
          and ad.get("optimizer_matches_contract", False)
          and (ad.get("max_v_rel") or 0) <= TH["adam_v_rel"] and (ad.get("max_clip_rel") or 0) <= TH["adam_clip_rel"]
          and (ad.get("max_theta_rel_to_update") or 0) <= TH["adam_theta_rel"])
    return {"passed": ok, **{k: v for k, v in ad.items() if k != "per_step"}}
